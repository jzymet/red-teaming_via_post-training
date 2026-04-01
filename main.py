"""
main.py — entry point for the red-teaming PPO pipeline.

Control flow per round:
  1. Rank 0: generate prompts with gen_model (no FSDP, no collectives)
  2. Rank 0: async Groq target calls + heuristic scoring
  3. Rank 1: waits at barrier
  4. Both ranks: barrier → step() (broadcast data, FSDP forward/backward)
  5. Rank 0: sync gen_model weights (no collective)
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
from rollout import collect_rollouts, Rollout
from training.fsdp_trainer import FSDPTrainer
from analysis.diversity import diversity_score
from analysis.plotting import plot_collapse


def load_config(path: str = "config.json") -> dict:
    defaults = {
        "attacker_model": "Qwen/Qwen2.5-1.5B",
        "groq_api_key": "YOUR_GROQ_API_KEY",
        "target_model": "llama-3.1-8b-instant",
        "num_rounds": 50,
        "rollouts_per_round": 8,
        "max_concurrent": 3,
        "log_every": 1,
        "output_path": "data/rollouts.jsonl",
        "lr": 1e-4,
        "beta": 0.0,
        "clip_eps": 0.3,
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
    
    # diversity over successful attacks only
    successful_prompts = [r.prompt for r in rollouts if r.score > 0.5]
    diversity = diversity_score(successful_prompts) if len(successful_prompts) >= 2 else 0.0
    
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

    attacker = AttackerModel(config["attacker_model"])
    trainer = FSDPTrainer(
        attacker, rank=rank,
        lr=config.get("lr", 1e-5),
        clip_eps=config.get("clip_eps", 0.2),
        beta=config.get("beta", 0.01),
    )

    # only rank 0 needs the target API client
    if rank == 0:
        target = TargetClient(
            api_key=config["groq_api_key"],
            model_name=config["target_model"],
        )

    for round_idx in range(config["num_rounds"]):
        if rank == 0:
            print(f"\n── round {round_idx} ──")

        # ── rank 0: generate + collect rollouts (no collective ops) ──
        if rank == 0:
            attacker_outputs = trainer.generate_rollout_inputs(
                n=config["rollouts_per_round"]
            )
            print(f"  generated {len(attacker_outputs)} prompts")

            rollouts = await collect_rollouts(
                attacker_outputs,
                target,
                max_concurrent=config["max_concurrent"],
            )
            if rollouts:
                log_rollouts(rollouts, round_idx, config["output_path"])
        else:
            rollouts = []

        # ── both ranks: sync then train ──
        dist.barrier()
        trainer.step(rollouts)

        # ── rank 0: log metrics ──
        if rank == 0 and round_idx % config["log_every"] == 0 and rollouts:
            metrics = compute_metrics(rollouts)
            log_metrics(metrics, round_idx)

    # ── cleanup ──
    if rank == 0 and os.path.exists(config.get("output_path", "")):
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

    spawn(
        fn=_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
