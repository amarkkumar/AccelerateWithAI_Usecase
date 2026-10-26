from unittest.mock import patch

import pytest

import core.llm as llm


def test_create_llm_uses_anthropic_configuration(monkeypatch):
    monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm, "ANTHROPIC_MODEL", "test-model")

    with patch("core.llm.ChatAnthropic") as chat_anthropic:
        model = llm.create_llm()

    chat_anthropic.assert_called_once_with(api_key="test-key", model="test-model")
    assert model is chat_anthropic.return_value


def test_create_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", None)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        llm.create_llm()
