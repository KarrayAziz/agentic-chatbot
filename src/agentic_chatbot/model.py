"""Google Gemini model construction."""

from langchain_google_genai import ChatGoogleGenerativeAI

from agentic_chatbot.config import Settings


def create_gemini_model(settings: Settings) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model used by the graph.

    Validation happens here because reading configuration should not require an
    API key, while creating a real external-service client should.
    """

    if settings.google_api_key is None:
        raise ValueError(
            "GOOGLE_API_KEY is required. Copy .env.example to .env and add your "
            "Gemini API key."
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        api_key=settings.google_api_key.get_secret_value(),
    )
