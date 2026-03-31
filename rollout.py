"""
rollout.py — async rollout collection with bounded concurrency.

Each rollout: attacker generates → target responds → evaluator scores.
"""

import asyncio
import traceback
import aiohttp
import torch
from dataclasses import dataclass
from attacker import AttackerModel, AttackerOutput
from target import TargetClient
from evaluator import EvaluatorClient


@dataclass
class Rollout:
    prompt: str
    response: str
    score: float
    generated_ids: torch.Tensor  # [seq_len] — needed for PPO recomputation
    logprobs: torch.Tensor       # [seq_len] — old policy log probs


async def single_rollout(
    attacker_output: AttackerOutput,
    target: TargetClient,
    evaluator: EvaluatorClient,
    session: aiohttp.ClientSession,
) -> Rollout:
    """One target→evaluator chain (attacker already generated)."""
    response = await target.respond(attacker_output.prompt, session)
    score = await evaluator.score(
        attacker_output.prompt, response, session
    )
    return Rollout(
        prompt=attacker_output.prompt,
        response=response,
        score=score,
        generated_ids=attacker_output.generated_ids,
        logprobs=attacker_output.logprobs,
    )


async def collect_rollouts(
    attacker_outputs: list[AttackerOutput],
    target: TargetClient,
    evaluator: EvaluatorClient,
    max_concurrent: int = 3,
) -> list[Rollout]:
    """
    Given pre-generated attacker outputs, run target + evaluator
    with bounded concurrency via semaphore.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(output: AttackerOutput, session: aiohttp.ClientSession):
        async with semaphore:
            return await single_rollout(output, target, evaluator, session)

    async with aiohttp.ClientSession() as session:
        tasks = [guarded(out, session) for out in attacker_outputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"rollout {i} failed: {type(r).__name__}: {r}")
            traceback.print_exception(type(r), r, r.__traceback__)

    good = [r for r in results if isinstance(r, Rollout)]
    print(f"{len(good)}/{len(results)} rollouts succeeded")
    return good
