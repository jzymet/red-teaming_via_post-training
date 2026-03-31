import os
import asyncio
import json
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from pathlib import Path

from attacker import AttackerModel
from target import TargetClient
from evaluator import EvaluatorClient
from rollout import collect_rollouts
from training.fsdp_trainer import FSDPTrainer
from analysis.diversity import diversity_score
from analysis.plotting import plot_collapse

# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    """Load from config.json if present, else use defaults."""
    defaults = {
        "attacker_model":     "Qwen/Qwen2.5-1.5B",
        "target_model":       "meta-llama/Llama-3.2-3B-Instruct",
        "evaluator_model":    "llama-3.1-8b-instant",
        "groq_api_key":       "YOUR_GROQ_API_KEY",
        "num_rounds":         50,
        "rollouts_per_round": 8,
        "max_concurrent":     3,
        "log_every":          5,
        "output_path":        "data/rollouts.jsonl",
    }
    if Path(path).exists():
        with open(path) as f:
            overrides = json.load(f)
        defaults.update(overrides)
        print(f"loaded config from {path}")
    else:
        print("no config.json found, using defaults")
    return defaults

# ── logging ───────────────────────────────────────────────────────────────────

def log_rollouts(rollouts, round_idx: int, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rollouts:
            f.write(json.dumps({
                "round":    round_idx,
                "prompt":   r.prompt,
                "response": r.response,
                "score":    r.score,
            }) + "\n")

def compute_metrics(rollouts) -> dict:
    asr       = sum(r.score > 0.5 for r in rollouts) / len(rollouts)
    diversity = diversity_score([r.prompt for r in rollouts])
    return {"asr": asr, "diversity": diversity}

def log_metrics(metrics: dict, round_idx: int):
    print(
        f"round {round_idx:3d} | "
        f"asr {metrics['asr']:.3f} | "
        f"diversity {metrics['diversity']:.3f}"
    )

# ── training loop ─────────────────────────────────────────────────────────────

async def training_loop(rank: int, world_size: int, config: dict):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    attacker  = AttackerModel(config["attacker_model"])
    target    = TargetClient(config["target_model"])
    evaluator = EvaluatorClient(
        api_key=config["groq_api_key"],
        model_name=config["evaluator_model"]
    )
    trainer = FSDPTrainer(attacker, rank=rank)

    for round_idx in range(config["num_rounds"]):

        rollouts = await collect_rollouts(
            attacker, target, evaluator,
            n=config["rollouts_per_round"],
            max_concurrent=config["max_concurrent"]
        )

        if not rollouts:
            print(f"round {round_idx}: no successful rollouts, skipping")
            continue

        if rank == 0:
            log_rollouts(rollouts, round_idx, config["output_path"])

        dist.barrier()

        trainer.step(rollouts)

        if rank == 0 and round_idx % config["log_every"] == 0:
            metrics = compute_metrics(rollouts)
            log_metrics(metrics, round_idx)

    if rank == 0:
        if os.path.exists(config["output_path"]):
            plot_collapse(config["output_path"])
        else:
            print("No rollouts written — skipping plot")

    dist.destroy_process_group()

# ── entry point ───────────────────────────────────────────────────────────────

def _worker(rank: int, world_size: int, config: dict):
    asyncio.run(training_loop(rank, world_size, config))

def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    config     = load_config()
    world_size = 1  # force single GPU for debugging
    print(f"launching on {world_size} GPU")

    spawn(
        fn=_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()