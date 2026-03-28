# analysis/plotting.py

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from diversity import diversity_score

def load_metrics(rollouts_path: str) -> dict:
    """Reconstruct per-round metrics from jsonl log."""
    rounds = defaultdict(list)

    with open(rollouts_path) as f:
        for line in f:
            record = json.loads(line)
            rounds[record["round"]].append(record)

    round_ids   = sorted(rounds.keys())
    asrs        = []
    diversities = []

    for r in round_ids:
        rollouts  = rounds[r]
        asr       = sum(x["score"] > 0.5 for x in rollouts) / len(rollouts)
        diversity = diversity_score([x["prompt"] for x in rollouts])
        asrs.append(asr)
        diversities.append(diversity)

    return {
        "rounds":     round_ids,
        "asr":        asrs,
        "diversity":  diversities,
    }

def plot_collapse(rollouts_path: str, output_path: str = "collapse.png"):
    """
    Two-panel plot: ASR and diversity over training rounds.
    Money plot: diversity collapses as ASR plateaus.
    """
    metrics = load_metrics(rollouts_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(metrics["rounds"], metrics["asr"],
             color="crimson", linewidth=2)
    ax1.set_ylabel("Attack Success Rate")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("PPO Mode Collapse: ASR vs Prompt Diversity")

    ax2.plot(metrics["rounds"], metrics["diversity"],
             color="steelblue", linewidth=2)
    ax2.set_ylabel("Prompt Diversity\n(mean pairwise cosine dist)")
    ax2.set_xlabel("Training Round")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved to {output_path}")

if __name__ == "__main__":
    plot_collapse("data/rollouts.jsonl")