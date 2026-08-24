"""Tests for Gemini model configuration without making API calls."""

import pytest

from agentic_chatbot.config import Settings
from agentic_chatbot.model import create_gemini_model


def test_model_requires_google_api_key() -> None:
    settings = Settings(_env_file=None, google_api_key=None)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
        create_gemini_model(settings)
