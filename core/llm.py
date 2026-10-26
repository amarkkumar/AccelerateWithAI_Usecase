"""Shared Anthropic chat model factory."""

from langchain_anthropic import ChatAnthropic

from core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def create_llm() -> ChatAnthropic:
    """Create the configured Anthropic chat model."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add a valid key to the environment or project .env file."
        )
    return ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL)
