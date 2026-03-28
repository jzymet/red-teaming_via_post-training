# rollout.py

import asyncio
import aiohttp
from dataclasses import dataclass
import torch
from attacker import AttackerModel, AttackerOutput
from target import TargetClient
from evaluator import EvaluatorClient

@dataclass
class Rollout:
    prompt:   str
    response: str
    score:    float
    logprobs: torch.Tensor  # [seq_len]

async def single_rollout(attacker:  AttackerModel,
                         target:    TargetClient,
                         evaluator: EvaluatorClient,
                         session:   aiohttp.ClientSession) -> Rollout:
    """One attacker→target→evaluator chain."""
    attacker_output = await attacker.generate()
    response        = await target.respond(attacker_output.prompt, session)
    score           = await evaluator.score(
        attacker_output.prompt, response, session
    )
    return Rollout(
        prompt=attacker_output.prompt,
        response=response,
        score=score,
        logprobs=attacker_output.logprobs
    )

async def collect_rollouts(attacker:       AttackerModel,
                           target:         TargetClient,
                           evaluator:      EvaluatorClient,
                           n:              int,
                           max_concurrent: int = 3) -> list[Rollout]:
    """N rollouts with bounded concurrency via semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded_rollout(session: aiohttp.ClientSession) -> Rollout:
        async with semaphore:
            return await single_rollout(attacker, target, evaluator, session)

    async with aiohttp.ClientSession() as session:
        tasks   = [guarded_rollout(session) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    good = [r for r in results if isinstance(r, Rollout)]
    if len(good) < n:
        print(f"warning: {n - len(good)} rollouts failed")
    return good