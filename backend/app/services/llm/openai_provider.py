import httpx
from typing import List
from app.core.config import settings
from app.services.llm.base import LLMProvider, LLMMessage


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI & OpenAI-compatible API LLM Provider.
    Works with OpenAI, Azure OpenAI, Ollama, vLLM, and any OpenAI-compatible API.
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL if settings.LLM_MODEL != "mock-trading-intelligence-v1" else "gpt-4o-mini"
        self.base_url = settings.LLM_BASE_URL.rstrip('/') if settings.LLM_BASE_URL else "https://api.openai.com/v1"

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
