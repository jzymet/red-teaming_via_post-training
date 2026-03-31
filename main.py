"""
main.py — entry point for the red-teaming PPO pipeline.

Architecture:
  - Attacker: Qwen2.5-1.5B, FSDP across 2 GPUs, fine-tuned via PPO
  - Target:   Groq API (free tier) — no local GPU needed
  - Evaluator: Groq API (free tier) — scores compliance 0-1
  - Generation: rank 0 only, via summon_full_params
  - Training: both ranks, via FSDP
"""

import asyncio
import json
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from pathlib import Path

from attacker import AttackerModel
from target import TargetClient
from evaluator import EvaluatorClient
from rollout import collect_rollouts, Rollout
from training.fsdp_trainer import FSDPTrainer
from analysis.diversity import diversity_score
from analysis.plotting import plot_collapse


def load_config(path: str = "config.json") -> dict:
    defaults = {
        "attacker_model": "Qwen/Qwen2.5-1.5B",
        "groq_api_key": "YOUR_GROQ_API_KEY",
        "target_model": "llama-3.1-8b-instant",
        "evaluator_model": "llama-3.1-8b-instant",
        "num_rounds": 50,
        "rollouts_per_round": 8,
        "max_concurrent": 3,
        "log_every": 5,
        "output_path": "data/rollouts.jsonl",
    }
    if Path(path).exists():
        with open(path) as f:
            overrides = json.load(f)
        defaults.update(overrides)
        print(f"loaded config from {os.path.abspath(path)}")
    else:
        print("no config.json found, using defaults")
    return defaults


def log_rollouts(rollouts: list[Rollout], round_idx: int, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rollouts:
            f.write(
                json.dumps({
                    "round": round_idx,
                    "prompt": r.prompt,
                    "response": r.response,
                    "score": r.score,
                })
                + "\n"
            )


def compute_metrics(rollouts: list[Rollout]) -> dict:
    asr = sum(r.score > 0.5 for r in rollouts) / len(rollouts)
    diversity = diversity_score([r.prompt for r in rollouts])
    return {"asr": asr, "diversity": diversity}


def log_metrics(metrics: dict, round_idx: int):
    print(
        f"round {round_idx:3d} | "
        f"asr {metrics['asr']:.3f} | "
        f"diversity {metrics['diversity']:.3f}"
    )


async def training_loop(rank: int, world_size: int, config: dict):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # all ranks load the attacker — FSDP shards across ranks
    attacker = AttackerModel(config["attacker_model"])
    trainer = FSDPTrainer(attacker, rank=rank)

    # only rank 0 needs target + evaluator (Groq API clients)
    if rank == 0:
        target = TargetClient(
            api_key=config["groq_api_key"],
            model_name=config["target_model"],
        )
        evaluator = EvaluatorClient(
            api_key=config["groq_api_key"],
            model_name=config["evaluator_model"],
        )

    for round_idx in range(config["num_rounds"]):
        # ── generation (rank 0 only, via summon_full_params) ──
        if rank == 0:
            attacker_outputs = trainer.generate_rollout_inputs(
                n=config["rollouts_per_round"]
            )
            # async: target + evaluator in parallel
            rollouts = await collect_rollouts(
                attacker_outputs,
                target,
                evaluator,
                max_concurrent=config["max_concurrent"],
            )
            if rollouts:
                log_rollouts(rollouts, round_idx, config["output_path"])
        else:
            rollouts = []

        # ── PPO update (both ranks — data broadcast inside step()) ──
        dist.barrier()
        trainer.step(rollouts)

        # ── logging ──
        if rank == 0 and round_idx % config["log_every"] == 0 and rollouts:
            metrics = compute_metrics(rollouts)
            log_metrics(metrics, round_idx)

    # ── final plot ──
    if rank == 0 and os.path.exists(config["output_path"]):
        plot_collapse(config["output_path"])

    dist.destroy_process_group()


def _worker(rank: int, world_size: int, config: dict):
    asyncio.run(training_loop(rank, world_size, config))


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    config = load_config()
    world_size = torch.cuda.device_count()
    print(f"launching on {world_size} GPUs")

    if world_size < 2:
        print("WARNING: need 2 GPUs for FSDP. falling back to 1 GPU.")
        world_size = max(world_size, 1)

    spawn(
        fn=_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
