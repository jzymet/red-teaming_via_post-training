import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from dataclasses import dataclass
from attacker import AttackerModel
from rollout import Rollout

@dataclass
class PPOBatch:
    prompts:  list[str]
    logprobs: torch.Tensor  # [batch, seq_len]
    scores:   torch.Tensor  # [batch]

class FSDPTrainer:
    """
    Wraps attacker in FSDP and runs PPO updates.
    Designed for 2-GPU setup on Kaggle/Colab.
    """
    def __init__(self,
                 attacker:  AttackerModel,
                 rank:      int,
                 lr:        float = 1e-5,
                 clip_eps:  float = 0.2,
                 beta:      float = 0.01):
        self.rank     = rank
        self.clip_eps = clip_eps
        self.beta     = beta

        self.model = FSDP(
            attacker.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=0
        )
        self.tokenizer = attacker.tokenizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

    def _rollouts_to_batch(self, rollouts: list[Rollout]) -> PPOBatch:
        """Collate rollouts into tensors, padding logprobs to same length."""
        max_len = max(r.logprobs.shape[0] for r in rollouts)
        padded  = [
            F.pad(r.logprobs, (0, max_len - r.logprobs.shape[0]))
            for r in rollouts
        ]
        return PPOBatch(
            prompts=  [r.prompt for r in rollouts],
            logprobs= torch.stack(padded).to(self.rank),
            scores=   torch.tensor(
                [r.score for r in rollouts],
                dtype=torch.float32
            ).to(self.rank)
        )

    def _compute_new_logprobs(self, batch: PPOBatch) -> torch.Tensor:
        inputs = self.tokenizer(
            batch.prompts,
            return_tensors="pt",
            padding=True
        ).to(self.rank)

        output = self.model(**inputs)  # forward pass → logits, not scores

        # output.logits is [batch, seq_len, vocab_size] directly
        log_probs    = F.log_softmax(output.logits, dim=-1)
        new_logprobs = log_probs.gather(
            dim=-1,
            index=inputs["input_ids"].unsqueeze(-1)
        ).squeeze(-1)                                    # [batch, seq_len]

        return new_logprobs

    def _ppo_loss(self,
                  new_logprobs: torch.Tensor,
                  old_logprobs: torch.Tensor,
                  advantages:   torch.Tensor) -> torch.Tensor:
        """Clipped PPO loss + KL penalty."""
        new_seq = new_logprobs.sum(dim=-1)  # [batch]
        old_seq = old_logprobs.sum(dim=-1)  # [batch]

        ratio   = torch.exp(new_seq - old_seq)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        pg_loss = -torch.min(
            ratio * advantages,
            clipped * advantages
        ).mean()

        kl = (new_seq - old_seq).mean()
        return pg_loss + self.beta * kl

    def step(self, rollouts: list[Rollout]):
        """One PPO update step."""
        batch      = self._rollouts_to_batch(rollouts)
        advantages = batch.scores - batch.scores.mean()

        self.optimizer.zero_grad()
        new_logprobs = self._compute_new_logprobs(batch)
        loss         = self._ppo_loss(
            new_logprobs, batch.logprobs, advantages
        )
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            print(
                f"loss: {loss.item():.4f} | "
                f"mean reward: {batch.scores.mean().item():.4f}"
            )