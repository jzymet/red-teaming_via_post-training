# Red-Teaming via Post-Training

Demonstrates **mode collapse** in PPO-based red-teaming: an attacker LLM fine-tuned
to maximize attack success rate against a target model converges to a narrow set of
prompts, losing diversity over training rounds.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Rank 0                                                 │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐          │
│  │ Attacker │───▶│  Target  │───▶│ Evaluator │          │
│  │ (Qwen)   │    │  (Groq)  │    │  (Groq)   │          │
│  └──────────┘    └──────────┘    └───────────┘          │
│       │              async           async               │
│       │          HTTP calls      HTTP calls              │
│       ▼                                                  │
│  summon_full_params()                                    │
│  for generation                                          │
├─────────────────────────────────────────────────────────┤
│  Both Ranks                                              │
│  ┌──────────────────────────────────┐                    │
│  │  FSDP PPO Update                │                    │
│  │  broadcast rollout data → both  │                    │
│  │  identical forward pass → loss  │                    │
│  │  backward → FSDP gradient sync  │                    │
│  └──────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

## Setup (Kaggle 2x T4)

1. Create a Kaggle notebook, select **2x T4 GPU**, enable **Internet**
2. Clone and run:

```bash
!git clone https://github.com/jzymet/red-teaming_via_post-training.git
%cd red-teaming_via_post-training
!pip install -q -r requirements.txt
```

3. Create `config.json` with your Groq API key:

```python
import json
config = {
    "groq_api_key": "gsk_YOUR_KEY_HERE",
    "num_rounds": 30,
    "rollouts_per_round": 4,
    "log_every": 5,
}
with open("config.json", "w") as f:
    json.dump(config, f)
```

4. Run:

```bash
!python main.py
```

## What to expect

The "money plot" (`collapse.png`) shows ASR rising and prompt diversity
falling over training rounds — the attacker finds a few effective attack
patterns and exploits them, losing coverage of the attack surface.

This is the exact failure mode that motivates diversity-preserving
approaches like Neural Thompson Sampling (see: Adaptive Instruction
Composition, ACL 2026).

## Components

| File | Role |
|------|------|
| `attacker.py` | Qwen2.5-1.5B policy — generates adversarial prompts |
| `target.py` | Groq API client — target model to attack |
| `evaluator.py` | Groq API client — scores compliance 0-1 |
| `rollout.py` | Async rollout collection with bounded concurrency |
| `training/fsdp_trainer.py` | FSDP-wrapped PPO trainer |
| `analysis/diversity.py` | Embedding-based diversity metric |
| `analysis/plotting.py` | ASR + diversity curves |
| `main.py` | Orchestrates the full loop |
