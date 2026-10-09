import httpx
from typing import List
from app.core.config import settings
from app.services.llm.base import LLMProvider, LLMMessage


class AnthropicLLMProvider(LLMProvider):
    """
    Anthropic Claude Messages API Provider.
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL if settings.LLM_MODEL != "mock-trading-intelligence-v1" else "claude-3-5-sonnet-20241022"
        self.base_url = settings.LLM_BASE_URL.rstrip('/') if settings.LLM_BASE_URL else "https://api.anthropic.com/v1"

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_prompt = "\n".join([m.content for m in messages if m.role == "system"])
        if json_mode and system_prompt:
            system_prompt += "\nOutput ONLY valid JSON."

        user_messages = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in messages if m.role != "system"
        ]

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": system_prompt,
            "messages": user_messages,
        }

        url = f"{self.base_url}/messages"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content_blocks = data.get("content", [])
            return "".join([b.get("text", "") for b in content_blocks if b.get("type") == "text"])
