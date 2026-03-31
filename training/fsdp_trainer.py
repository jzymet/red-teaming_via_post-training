"""
training/fsdp_trainer.py — FSDP-wrapped PPO trainer.

Key design decisions:
  - FULL_SHARD (ZeRO-3) across 2 GPUs
  - summon_full_params() for generation on rank 0
  - broadcast rollout data so both ranks do identical forward passes
  - PPO with clipped ratio + KL penalty
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from dataclasses import dataclass

from attacker import AttackerModel, AttackerOutput
from rollout import Rollout


@dataclass
class PPOBatch:
    system_ids: torch.Tensor     # [1, prompt_len] — tokenized system prompt
    generated_ids: torch.Tensor  # [batch, max_gen_len] — padded generated tokens
    gen_lengths: torch.Tensor    # [batch] — actual length of each generation
    old_logprobs: torch.Tensor   # [batch, max_gen_len] — padded old log probs
    scores: torch.Tensor         # [batch]


class FSDPTrainer:
    def __init__(
        self,
        attacker: AttackerModel,
        rank: int,
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        beta: float = 0.01,
    ):
        self.rank = rank
        self.clip_eps = clip_eps
        self.beta = beta
        self.attacker = attacker

        # FSDP wrap — moves model to GPU
        self.model = FSDP(
            attacker.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.device(f"cuda:{rank}"),
        )
        self.tokenizer = attacker.tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # cache system prompt IDs on device
        self._system_ids = attacker._system_ids.to(rank)  # [1, prompt_len]

    def generate_rollout_inputs(self, n: int) -> list[AttackerOutput]:
        """
        Generate n attacker outputs using the current policy.

        MUST be called on ALL ranks — summon_full_params is a collective
        op (all-gather) that requires every rank to participate.
        Only rank 0 actually generates; rank 1 just participates in
        the collective and returns [].
        """
        outputs = []
        with FSDP.summon_full_params(self.model, writeback=False):
            if self.rank == 0:
                for _ in range(n):
                    out = self.attacker.generate_sync(
                        device=torch.device(f"cuda:{self.rank}")
                    )
                    outputs.append(out)
        return outputs

    def _broadcast_rollout_data(self, rollouts: list[Rollout]) -> PPOBatch:
        """
        Rank 0 has rollouts; rank 1 has []. Broadcast the tensor data
        so both ranks can do identical forward passes (required by FSDP).
        """
        # broadcast batch size
        if self.rank == 0:
            batch_size = torch.tensor(len(rollouts), device=f"cuda:{self.rank}")
        else:
            batch_size = torch.tensor(0, device=f"cuda:{self.rank}")
        dist.broadcast(batch_size, src=0)
        B = batch_size.item()

        if B == 0:
            return None

        # broadcast max generation length
        if self.rank == 0:
            max_gen_len = torch.tensor(
                max(r.generated_ids.shape[0] for r in rollouts),
                device=f"cuda:{self.rank}",
            )
        else:
            max_gen_len = torch.tensor(0, device=f"cuda:{self.rank}")
        dist.broadcast(max_gen_len, src=0)
        L = max_gen_len.item()

        # pack tensors on rank 0, allocate on rank 1
        if self.rank == 0:
            gen_ids = torch.zeros(B, L, dtype=torch.long, device=f"cuda:{self.rank}")
            gen_lens = torch.zeros(B, dtype=torch.long, device=f"cuda:{self.rank}")
            old_lp = torch.zeros(B, L, dtype=torch.float32, device=f"cuda:{self.rank}")
            scores = torch.zeros(B, dtype=torch.float32, device=f"cuda:{self.rank}")

            for i, r in enumerate(rollouts):
                gl = r.generated_ids.shape[0]
                gen_ids[i, :gl] = r.generated_ids.to(f"cuda:{self.rank}")
                gen_lens[i] = gl
                old_lp[i, :gl] = r.logprobs.float().to(f"cuda:{self.rank}")
                scores[i] = r.score
        else:
            gen_ids = torch.zeros(B, L, dtype=torch.long, device=f"cuda:{self.rank}")
            gen_lens = torch.zeros(B, dtype=torch.long, device=f"cuda:{self.rank}")
            old_lp = torch.zeros(B, L, dtype=torch.float32, device=f"cuda:{self.rank}")
            scores = torch.zeros(B, dtype=torch.float32, device=f"cuda:{self.rank}")

        dist.broadcast(gen_ids, src=0)
        dist.broadcast(gen_lens, src=0)
        dist.broadcast(old_lp, src=0)
        dist.broadcast(scores, src=0)

        return PPOBatch(
            system_ids=self._system_ids,
            generated_ids=gen_ids,
            gen_lengths=gen_lens,
            old_logprobs=old_lp,
            scores=scores,
        )

    def _compute_new_logprobs(self, batch: PPOBatch) -> torch.Tensor:
        """
        Forward pass: concatenate system_prompt + generated_ids,
        extract log probs of the generated portion only.

        Returns: [batch, max_gen_len] tensor (padded positions = 0).
        """
        B = batch.generated_ids.shape[0]
        L_gen = batch.generated_ids.shape[1]
        L_sys = batch.system_ids.shape[1]

        # build input: [system_prompt | generated_tokens] for each sample
        # system prompt is the same for all — expand and concat
        sys_expanded = batch.system_ids.expand(B, -1)  # [B, L_sys]
        input_ids = torch.cat([sys_expanded, batch.generated_ids], dim=1)  # [B, L_sys + L_gen]

        # attention mask: system prompt is always attended, generated up to gen_length
        attn_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i in range(B):
            gl = batch.gen_lengths[i].item()
            attn_mask[i, : L_sys + gl] = 1

        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        # logits shape: [B, L_sys + L_gen, vocab]

        # we want P(generated_token_t | system_prompt, generated_tokens_<t)
        # logits at position (L_sys + t - 1) predict token at position (L_sys + t)
        # so for generated tokens [0..L_gen-1], we need logits [L_sys-1..L_sys+L_gen-2]
        gen_logits = output.logits[:, L_sys - 1 : L_sys + L_gen - 1, :]  # [B, L_gen, vocab]
        log_probs = F.log_softmax(gen_logits, dim=-1)

        # gather log probs for the actual generated tokens
        new_logprobs = log_probs.gather(
            dim=-1, index=batch.generated_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, L_gen]

        # zero out padding positions
        mask = torch.arange(L_gen, device=new_logprobs.device).unsqueeze(0) < batch.gen_lengths.unsqueeze(1)
        new_logprobs = new_logprobs * mask.float()

        return new_logprobs

    def _ppo_loss(
        self,
        new_logprobs: torch.Tensor,  # [B, L_gen]
        old_logprobs: torch.Tensor,  # [B, L_gen]
        advantages: torch.Tensor,    # [B]
        gen_lengths: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """Clipped PPO loss + KL penalty, operating on sequence-level log probs."""
        # sum log probs over sequence (masking already applied)
        new_seq = new_logprobs.sum(dim=-1)  # [B]
        old_seq = old_logprobs.sum(dim=-1)  # [B]

        ratio = torch.exp(new_seq - old_seq)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        pg_loss = -torch.min(
            ratio * advantages,
            clipped * advantages,
        ).mean()

        kl = (old_seq - new_seq).mean()  # approx KL(old || new)
        return pg_loss + self.beta * kl

    def step(self, rollouts: list[Rollout]):
        """
        One PPO update step. Called on all ranks.
        Rank 0 passes actual rollouts; rank 1 passes [].
        Data is broadcast internally so both ranks do identical work.
        """
        torch.cuda.empty_cache()

        batch = self._broadcast_rollout_data(rollouts)
        if batch is None:
            return

        advantages = batch.scores - batch.scores.mean()

        self.optimizer.zero_grad()
        new_logprobs = self._compute_new_logprobs(batch)
        loss = self._ppo_loss(
            new_logprobs, batch.old_logprobs, advantages, batch.gen_lengths
        )
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            print(
                f"  loss: {loss.item():.4f} | "
                f"mean reward: {batch.scores.mean().item():.3f}"
            )
