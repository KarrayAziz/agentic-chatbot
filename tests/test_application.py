"""Tests for the reusable API-facing application service."""

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import agentic_chatbot.application as application_module
from agentic_chatbot.application import AgentApplication
from agentic_chatbot.config import Settings


class ApplicationFakeModel(FakeMessagesListChatModel):
    """Fake chat model that accepts the graph's bound tool schemas."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ApplicationFakeModel":
        return self


class FakeRAGService:
    pass


def test_message_uses_persistent_thread_and_visible_history(
    tmp_path: Path, monkeypatch
) -> None:
    model = ApplicationFakeModel(
        responses=[AIMessage("The existing LangGraph answered.")]
    )
    monkeypatch.setattr(
        application_module,
        "create_gemini_model",
        lambda settings: model,
    )
    monkeypatch.setattr(
        application_module,
        "create_document_rag_service",
        lambda settings: FakeRAGService(),
    )
    monkeypatch.setattr(
        application_module,
        "build_tools",
        lambda settings, document_search_tool: [],
    )
    settings = Settings(
        _env_file=None,
        conversation_db_path=tmp_path / "conversations.sqlite",
        checkpoint_db_path=tmp_path / "checkpoints.sqlite",
        chroma_db_path=tmp_path / "chroma",
    )
    service = AgentApplication(settings)
    conversation = service.create_conversation()

    reply = service.send_message(conversation.thread_id, "Explain the API path")
    state = service.get_conversation_state(conversation.thread_id)

    assert reply.status == "completed"
    assert reply.assistant_message == "The existing LangGraph answered."
    assert [(message.role, message.content) for message in state.messages] == [
        ("user", "Explain the API path"),
        ("assistant", "The existing LangGraph answered."),
    ]
    assert service.get_conversation(conversation.thread_id).title == (
        "Explain the API path"
    )
    assert settings.checkpoint_db_path.exists()
