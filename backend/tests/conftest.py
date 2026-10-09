import os
import pytest
from app.core.config import settings
from app.services.llm.mock_provider import MockLLMProvider
import app.services.llm.factory as llm_factory


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Ensure tests run against mock providers and in-memory resources."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(llm_factory, "_llm_provider_instance", MockLLMProvider())
