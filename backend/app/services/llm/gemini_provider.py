import logging
import httpx
from typing import List
from app.core.config import settings
from app.services.llm.base import LLMProvider, LLMMessage
from app.services.llm.mock_provider import MockLLMProvider

logger = logging.getLogger("tradetrace.gemini")


class GeminiLLMProvider(LLMProvider):
    """
    Google Gemini API LLM Provider.
    Invokes Gemini 1.5/2.0 models via Google Generative Language REST API.
    Gracefully falls back to local intelligent simulation if the API key is unauthorized.
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL if settings.LLM_MODEL != "mock-trading-intelligence-v1" else "gemini-1.5-flash"
        self.base_url = settings.LLM_BASE_URL.rstrip('/') if settings.LLM_BASE_URL else "https://generativelanguage.googleapis.com/v1beta"
        self._mock_fallback = MockLLMProvider()

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        # Support query param & header authentication
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        if self.api_key.startswith("AQ.") or len(self.api_key) > 50:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code in (401, 403, 404):
                    # Try without query param if bearer auth is used
                    url_no_param = f"{self.base_url}/models/{self.model}:generateContent"
                    response = await client.post(url_no_param, json=payload, headers=headers)
                
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("Gemini API returned no response candidates.")
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join([p.get("text", "") for p in parts])
        except Exception as exc:
            print(f"\n[!] Note: Gemini API call failed ({exc}). Using offline intelligent agent engine fallback.")
            return await self._mock_fallback.generate(messages, temperature=temperature, json_mode=json_mode)
