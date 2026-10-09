import pytest
from unittest.mock import patch, AsyncMock
from app.services.llm.base import LLMMessage
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.openai_provider import OpenAILLMProvider
from app.services.llm.gemini_provider import GeminiLLMProvider
from app.services.llm.anthropic_provider import AnthropicLLMProvider
from app.services.llm.factory import get_llm_provider


@pytest.mark.asyncio
async def test_mock_llm_provider_research():
    provider = MockLLMProvider()
    messages = [
        LLMMessage(role="system", content="You are the TradeTrace Research Agent."),
        LLMMessage(role="user", content="Analyze AAPL"),
    ]
    res = await provider.generate(messages, json_mode=True)
    assert "technical_evaluation" in res


@pytest.mark.asyncio
async def test_mock_llm_provider_devils_advocate():
    provider = MockLLMProvider()
    messages = [
        LLMMessage(role="system", content="You are the TradeTrace Devil's Advocate Agent."),
        LLMMessage(role="user", content="Challenge AAPL thesis"),
    ]
    res = await provider.generate(messages, json_mode=True)
    assert "counter_arguments" in res


@pytest.mark.asyncio
async def test_mock_llm_provider_decision():
    provider = MockLLMProvider()
    messages = [
        LLMMessage(role="system", content="You are the TradeTrace Decision Agent."),
        LLMMessage(role="user", content="Decide AAPL trade"),
    ]
    res = await provider.generate(messages, json_mode=True)
    assert "action" in res


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_openai_llm_provider(mock_post):
    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"status": "ok"}'}}]
    }
    mock_post.return_value = mock_response

    provider = OpenAILLMProvider()
    messages = [LLMMessage(role="user", content="Hello")]
    res = await provider.generate(messages, json_mode=True)
    assert res == '{"status": "ok"}'


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_gemini_llm_provider(mock_post):
    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"status": "ok"}'}]}}]
    }
    mock_post.return_value = mock_response

    provider = GeminiLLMProvider()
    messages = [LLMMessage(role="user", content="Hello")]
    res = await provider.generate(messages, json_mode=True)
    assert res == '{"status": "ok"}'


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_anthropic_llm_provider(mock_post):
    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": '{"status": "ok"}'}]
    }
    mock_post.return_value = mock_response

    provider = AnthropicLLMProvider()
    messages = [LLMMessage(role="user", content="Hello")]
    res = await provider.generate(messages, json_mode=True)
    assert res == '{"status": "ok"}'
