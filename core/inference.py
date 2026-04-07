from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import json


class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str: ...


class VLLMBackend(LLMBackend):
    """Self-hosted vLLM -- for cost control and data sovereignty."""

    def __init__(self, base_url: str, model: str):
        import httpx
        self._client = httpx.AsyncClient(base_url=base_url, timeout=120)
        self._model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self._client.post(
            "/v1/completions",
            json={
                "model": self._model,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
            },
        )
        return response.json()["choices"][0]["text"]


class GoogleGenAIBackend(LLMBackend):
    """Google Generative AI -- for rapid prototyping or fallback."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai
        self._client = genai.Client()
        self._model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        from google.genai import types
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.0),
                max_output_tokens=kwargs.get("max_tokens", 2048),
            ),
        )
        return response.text


class InferenceRouter:
    """Route to the right backend based on country/config policy."""

    def __init__(self, backends: dict[str, LLMBackend] | None = None):
        self._backends: dict[str, LLMBackend] = backends or {}

    async def run(
        self, prompt: str, backend_key: str = "default", **kwargs
    ) -> dict[str, Any]:
        raw = await self._backends[backend_key].generate(prompt, **kwargs)
        return json.loads(raw)

    async def run_raw(
        self, prompt: str, backend_key: str = "default", **kwargs
    ) -> str:
        """Return raw string response without JSON parsing."""
        return await self._backends[backend_key].generate(prompt, **kwargs)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        backend_key: str = "default",
        **kwargs,
    ) -> str:
        """Convenience wrapper around run_raw for use by new pipeline components."""
        return await self.run_raw(
            prompt, backend_key=backend_key, temperature=temperature, **kwargs
        )
