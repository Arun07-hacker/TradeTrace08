from app.core.config import settings
from app.services.llm.base import LLMProvider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.openai_provider import OpenAILLMProvider
from app.services.llm.gemini_provider import GeminiLLMProvider
from app.services.llm.anthropic_provider import AnthropicLLMProvider


_llm_provider_instance: LLMProvider = None


def get_llm_provider() -> LLMProvider:
    """Singleton factory returning the configured LLM provider."""
    global _llm_provider_instance
    if _llm_provider_instance is None:
        provider_type = settings.LLM_PROVIDER.lower().strip()
        if provider_type == "openai":
            _llm_provider_instance = OpenAILLMProvider()
        elif provider_type == "gemini":
            _llm_provider_instance = GeminiLLMProvider()
        elif provider_type == "anthropic":
            _llm_provider_instance = AnthropicLLMProvider()
        else:
            _llm_provider_instance = MockLLMProvider()
    return _llm_provider_instance
