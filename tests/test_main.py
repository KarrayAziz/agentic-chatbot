"""Tests for application wiring without making an external model call."""

from contextlib import nullcontext
from types import SimpleNamespace
from uuid import UUID

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agentic_chatbot.main as main_module


class FakeConversationRepository:
    def __init__(self) -> None:
        self.created_thread_ids: list[str] = []

    def get(self, thread_id: str):
        return None

    def create(self, thread_id: str) -> None:
        self.created_thread_ids.append(thread_id)

    def record_user_message(self, thread_id: str, message: str) -> None:
        return None


def test_main_builds_graph_and_starts_cli(monkeypatch) -> None:
    fake_model = FakeListChatModel(responses=["unused"])
    fake_tools = []
    fake_graph = object()
    fake_checkpointer = object()
    fake_rag_service = object()
    fake_document_tool = object()
    fake_repository = FakeConversationRepository()
    received_conversations = []

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "create_gemini_model", lambda settings: fake_model)
    monkeypatch.setattr(
        main_module,
        "create_document_rag_service",
        lambda settings: fake_rag_service,
    )
    monkeypatch.setattr(
        main_module,
        "create_document_search_tool",
        lambda service, thread_id: fake_document_tool,
    )
    monkeypatch.setattr(
        main_module,
        "build_tools",
        lambda settings, document_search_tool: fake_tools,
    )
    monkeypatch.setattr(
        main_module,
        "open_conversation_repository",
        lambda path: nullcontext(fake_repository),
    )
    monkeypatch.setattr(
        main_module,
        "open_sqlite_checkpointer",
        lambda path: nullcontext(fake_checkpointer),
    )
    monkeypatch.setattr(
        main_module,
        "build_chat_graph",
        lambda model, tools, checkpointer: fake_graph,
    )
    monkeypatch.setattr(
        main_module,
        "run_chat_cli",
        lambda graph, thread_id, on_user_message: received_conversations.append(
            (graph, thread_id)
        ),
    )

    main_module.main([])

    assert len(received_conversations) == 1
    assert received_conversations[0][0] is fake_graph
    assert str(UUID(received_conversations[0][1])) == received_conversations[0][1]
    assert fake_repository.created_thread_ids == [received_conversations[0][1]]


def test_delete_uses_checkpointer_api_before_removing_metadata(
    monkeypatch, capsys
) -> None:
    thread_id = "d3c5134a-65f5-44a5-b41d-f3518d5c21b9"
    calls: list[tuple[str, str]] = []

    class DeleteRepository:
        def get(self, requested_thread_id: str):
            return SimpleNamespace(thread_id=requested_thread_id)

        def delete(self, requested_thread_id: str) -> bool:
            calls.append(("metadata", requested_thread_id))
            return True

    class DeleteCheckpointer:
        def delete_thread(self, requested_thread_id: str) -> None:
            calls.append(("checkpoints", requested_thread_id))

    monkeypatch.setattr(
        main_module,
        "open_conversation_repository",
        lambda path: nullcontext(DeleteRepository()),
    )
    monkeypatch.setattr(
        main_module,
        "open_sqlite_checkpointer",
        lambda path: nullcontext(DeleteCheckpointer()),
    )

    main_module.main(["--delete", thread_id])

    assert calls == [("checkpoints", thread_id), ("metadata", thread_id)]
    assert capsys.readouterr().out == (
        f"Deleted conversation {thread_id} and its checkpoints.\n"
    )


def test_ingest_pdf_command_targets_selected_conversation(
    monkeypatch, capsys
) -> None:
    thread_id = "9cc23ba3-2e63-4c21-aefa-5d960664a291"
    ingested: list[tuple[object, str]] = []

    class ExistingRepository:
        def get(self, requested_thread_id: str):
            return SimpleNamespace(thread_id=requested_thread_id)

    class FakeRAGService:
        def ingest_pdf(self, path, requested_thread_id: str):
            ingested.append((path, requested_thread_id))
            return SimpleNamespace(
                source_filename="guide.pdf",
                document_id="document-1",
                page_count=2,
                chunk_count=4,
            )

    monkeypatch.setattr(
        main_module,
        "open_conversation_repository",
        lambda path: nullcontext(ExistingRepository()),
    )
    monkeypatch.setattr(
        main_module,
        "create_document_rag_service",
        lambda settings: FakeRAGService(),
    )

    main_module.main(
        ["--thread-id", thread_id, "--ingest-pdf", "guide.pdf"]
    )

    assert ingested == [(main_module.Path("guide.pdf"), thread_id)]
    assert capsys.readouterr().out == (
        "Ingested guide.pdf as document-1 (2 pages, 4 chunks).\n"
    )
