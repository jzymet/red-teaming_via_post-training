"""
training/fsdp_trainer.py — FSDP-wrapped PPO trainer.

Architecture:
  - SHARD_GRAD_OP (ZeRO-2): full params on each rank, sharded grads + optim
  - Generation uses a separate gen_model (not FSDP-wrapped)
  - After each training step, weights are synced back to gen_model
  - Rollout data is broadcast from rank 0 so both ranks do identical work
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
    system_ids: torch.Tensor     # [1, prompt_len]
    generated_ids: torch.Tensor  # [batch, max_gen_len]
    gen_lengths: torch.Tensor    # [batch]
    old_logprobs: torch.Tensor   # [batch, max_gen_len]
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

        # FSDP wraps attacker.model (the training copy)
        # gen_model stays unwrapped for generation
        self.model = FSDP(
            attacker.model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.device(f"cuda:{rank}"),
        )
        self.tokenizer = attacker.tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # cache system prompt IDs on device
        self._system_ids = attacker._system_ids.to(rank)

    def generate_rollout_inputs(self, n: int) -> list[AttackerOutput]:
        """
        Generate n attacker outputs using gen_model (not FSDP model).
        Only call on rank 0 — no collective ops.
        """
        outputs = []
        for _ in range(n):
            out = self.attacker.generate_sync(
                device=torch.device(f"cuda:{self.rank}")
            )
            outputs.append(out)
        return outputs

    def _sync_gen_model(self):
        """
        After a training step, copy weights to gen_model on rank 0.

        With SHARD_GRAD_OP, full params are on each rank already,
        so we just copy them directly — no collective ops needed.
        """
        if self.rank != 0:
            return

        # extract params from FSDP module and load into gen_model
        state = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }
        # FSDP may add a prefix — strip it
        # The gen_model expects raw model keys like "model.layers.0..."
        # FSDP wraps as "_fsdp_wrapped_module.model.layers.0..."
        cleaned = {}
        for k, v in state.items():
            # strip any FSDP prefix
            clean_key = k.replace("_fsdp_wrapped_module.", "")
            cleaned[clean_key] = v

        self.attacker.gen_model.load_state_dict(cleaned, strict=False)

    def _broadcast_rollout_data(self, rollouts: list[Rollout]) -> PPOBatch:
        """
        Rank 0 has rollouts; rank 1 has []. Broadcast tensor data
        so both ranks do identical forward passes.
        """
        if self.rank == 0:
            batch_size = torch.tensor(len(rollouts), device=f"cuda:{self.rank}")
        else:
            batch_size = torch.tensor(0, device=f"cuda:{self.rank}")
        dist.broadcast(batch_size, src=0)
        B = batch_size.item()

        if B == 0:
            return None

        if self.rank == 0:
            max_gen_len = torch.tensor(
                max(r.generated_ids.shape[0] for r in rollouts),
                device=f"cuda:{self.rank}",
            )
        else:
            max_gen_len = torch.tensor(0, device=f"cuda:{self.rank}")
        dist.broadcast(max_gen_len, src=0)
        L = max_gen_len.item()

        gen_ids = torch.zeros(B, L, dtype=torch.long, device=f"cuda:{self.rank}")
        gen_lens = torch.zeros(B, dtype=torch.long, device=f"cuda:{self.rank}")
        old_lp = torch.zeros(B, L, dtype=torch.float32, device=f"cuda:{self.rank}")
        scores = torch.zeros(B, dtype=torch.float32, device=f"cuda:{self.rank}")

        if self.rank == 0:
            for i, r in enumerate(rollouts):
                gl = r.generated_ids.shape[0]
                gen_ids[i, :gl] = r.generated_ids.to(f"cuda:{self.rank}")
                gen_lens[i] = gl
                old_lp[i, :gl] = r.logprobs.float().to(f"cuda:{self.rank}")
                scores[i] = r.score

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
        Forward pass through FSDP model: [system_prompt | generated_ids],
        extract log probs of generated portion only.
        """
        B = batch.generated_ids.shape[0]
        L_gen = batch.generated_ids.shape[1]
        L_sys = batch.system_ids.shape[1]

        sys_expanded = batch.system_ids.expand(B, -1)
        input_ids = torch.cat([sys_expanded, batch.generated_ids], dim=1)

        attn_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i in range(B):
            gl = batch.gen_lengths[i].item()
            attn_mask[i, : L_sys + gl] = 1

        output = self.model(input_ids=input_ids, attention_mask=attn_mask)

        gen_logits = output.logits[:, L_sys - 1 : L_sys + L_gen - 1, :]
        log_probs = F.log_softmax(gen_logits, dim=-1)

        new_logprobs = log_probs.gather(
            dim=-1, index=batch.generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        mask = torch.arange(L_gen, device=new_logprobs.device).unsqueeze(0) < batch.gen_lengths.unsqueeze(1)
        new_logprobs = new_logprobs * mask.float()

        return new_logprobs

    def _ppo_loss(self, new_logprobs, old_logprobs, advantages, gen_lengths):
        new_seq = new_logprobs.sum(dim=-1)
        old_seq = old_logprobs.sum(dim=-1)

        ratio = torch.exp(new_seq - old_seq)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        kl = (old_seq - new_seq).mean()
        return pg_loss + self.beta * kl

    def step(self, rollouts: list[Rollout]):
        """
        One PPO step. Called on ALL ranks.
        Rank 0 passes rollouts; rank 1 passes [].
        Data broadcast internally.
        """
        torch.cuda.empty_cache()

        batch = self._broadcast_rollout_data(rollouts)
        if batch is None:
            return

        advantages = batch.scores - batch.scores.mean()

        self.model.train()
        self.optimizer.zero_grad()
        new_logprobs = self._compute_new_logprobs(batch)
        loss = self._ppo_loss(
            new_logprobs, batch.old_logprobs, advantages, batch.gen_lengths
        )
        loss.backward()
        self.optimizer.step()

        # sync updated weights back to gen_model for next round
        self._sync_gen_model()

        if self.rank == 0:
            print(
                f"  loss: {loss.item():.4f} | "
                f"mean reward: {batch.scores.mean().item():.3f}"
            )
