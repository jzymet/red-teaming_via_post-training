"""
target.py — Groq-hosted LLM as the target model.

Using Groq for the target (instead of a local HuggingFace model) means:
  - Both GPUs are free for FSDP on the attacker
  - Rollout collection is truly async (HTTP calls, not GPU compute)
  - No memory pressure from loading a second model
"""

import aiohttp


class TargetClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.1-8b-instant",
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    async def respond(
        self,
        prompt: str,
        session: aiohttp.ClientSession,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 128,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with session.post(
            self.base_url, json=payload, headers=headers
        ) as resp:
            data = await resp.json()

        if "choices" not in data:
            print(f"unexpected groq response: {data}")
            return "[target error]"

        return data["choices"][0]["message"]["content"].strip()
