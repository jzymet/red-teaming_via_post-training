import asyncio
import torch
from transformers import pipeline

class TargetClient:
    """
    HuggingFace-hosted safety-finetuned model as target.
    No Ollama needed — runs directly on GPU.
    Sometimes refuses — that's the behavior we measure.
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    async def respond(self,
                      prompt: str,
                      session=None) -> str:          # session kept for API compatibility
        """Async wrapper — offloads GPU compute to executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._respond_sync, prompt
        )

    def _respond_sync(self, prompt: str) -> str:
        result = self.pipe(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            truncation=True,        # add this
            max_length=512          # add this — overrides the model's default 20
        )
        # pipeline returns full text including prompt — strip it
        return result[0]["generated_text"][len(prompt):]