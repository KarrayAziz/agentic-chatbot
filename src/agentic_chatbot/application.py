"""Reusable application services shared by HTTP entry points."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agentic_chatbot.config import Settings
from agentic_chatbot.conversations import (
    DEFAULT_TITLE,
    MAX_TITLE_LENGTH,
    Conversation,
    open_conversation_repository,
)
from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.hitl import paper_approval_from_interrupts
from agentic_chatbot.model import create_gemini_model
from agentic_chatbot.persistence import open_sqlite_checkpointer
from agentic_chatbot.rag import (
    DocumentRAGService,
    IngestedDocument,
    create_document_rag_service,
)
from agentic_chatbot.streaming import AgentStreamEvent, stream_agent_events
from agentic_chatbot.tools import build_tools
from agentic_chatbot.tools.document_search import create_document_search_tool

MAX_PDF_UPLOAD_BYTES = 20 * 1024 * 1024


class ApplicationError(Exception):
    """Base class for expected application-layer failures."""


class ConversationNotFoundError(ApplicationError):
    """Raised when a requested conversation does not exist."""


class ApplicationValidationError(ApplicationError):
    """Raised when an otherwise valid transport request cannot be processed."""


class ApprovalConflictError(ApplicationError):
    """Raised when an operation conflicts with the thread's approval state."""


class PendingApprovalNotFoundError(ApprovalConflictError):
    """Raised when a thread has no approval waiting to be resumed."""


@dataclass(frozen=True, slots=True)
class VisibleMessage:
    """A user-visible message from persisted graph state."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ConversationState:
    """Visible history and any restart-safe pending approval."""

    messages: list[VisibleMessage]
    pending_approval: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class AgentReply:
    """Result of a message or approval graph execution."""

    status: Literal["completed", "pending_approval"]
    assistant_message: str | None
    pending_approval: dict[str, Any] | None


class AgentApplication:
    """Coordinate repositories, LangGraph, Chroma, and PDF ingestion."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_conversation(self, title: str | None = None) -> Conversation:
        normalized_title = (
            self._validate_title(title) if title is not None else DEFAULT_TITLE
        )
        with open_conversation_repository(
            self.settings.conversation_db_path
        ) as repository:
            return repository.create(str(uuid4()), normalized_title)

    def list_conversations(self) -> list[Conversation]:
        with open_conversation_repository(
            self.settings.conversation_db_path
        ) as repository:
            return repository.list_all()

    def get_conversation(self, thread_id: str) -> Conversation:
        with open_conversation_repository(
            self.settings.conversation_db_path
        ) as repository:
            conversation = repository.get(thread_id)
        if conversation is None:
            raise ConversationNotFoundError(
                f"Conversation '{thread_id}' does not exist."
            )
        return conversation

    def rename_conversation(self, thread_id: str, title: str) -> Conversation:
        self.get_conversation(thread_id)
        try:
            with open_conversation_repository(
                self.settings.conversation_db_path
            ) as repository:
                return repository.rename(thread_id, title)
        except ValueError as error:
            raise ApplicationValidationError(str(error)) from error

    def delete_conversation(self, thread_id: str) -> None:
        self.get_conversation(thread_id)

        rag_service = create_document_rag_service(self.settings)
        rag_service.delete_thread_documents(thread_id)
        with open_sqlite_checkpointer(self.settings.checkpoint_db_path) as checkpointer:
            checkpointer.delete_thread(thread_id)
        with open_conversation_repository(
            self.settings.conversation_db_path
        ) as repository:
            repository.delete(thread_id)

    def send_message(self, thread_id: str, content: str) -> AgentReply:
        self.get_conversation(thread_id)
        normalized_content = content.strip()
        if not normalized_content:
            raise ApplicationValidationError("Message content cannot be empty.")

        with self._open_graph(thread_id) as graph:
            snapshot = graph.get_state(self._graph_config(thread_id))
            if paper_approval_from_interrupts(snapshot.interrupts) is not None:
                raise ApprovalConflictError(
                    "Resolve the pending approval before sending another message."
                )
            with open_conversation_repository(
                self.settings.conversation_db_path
            ) as repository:
                repository.record_user_message(thread_id, normalized_content)
            return self._run_graph(
                graph,
                thread_id,
                {"messages": [HumanMessage(content=normalized_content)]},
            )

    def stream_message(
        self, thread_id: str, content: str
    ) -> Iterator[AgentStreamEvent]:
        """Run one message and yield safe events as LangGraph produces them."""

        self.get_conversation(thread_id)
        normalized_content = content.strip()
        if not normalized_content:
            raise ApplicationValidationError("Message content cannot be empty.")

        with self._open_graph(thread_id) as graph:
            snapshot = graph.get_state(self._graph_config(thread_id))
            if paper_approval_from_interrupts(snapshot.interrupts) is not None:
                raise ApprovalConflictError(
                    "Resolve the pending approval before sending another message."
                )
            with open_conversation_repository(
                self.settings.conversation_db_path
            ) as repository:
                repository.record_user_message(thread_id, normalized_content)

            yield from stream_agent_events(
                graph,
                [HumanMessage(content=normalized_content)],
                thread_id,
            )

    def respond_to_approval(
        self, thread_id: str, decision: Literal["approve", "reject"]
    ) -> AgentReply:
        self.get_conversation(thread_id)
        with self._open_graph(thread_id) as graph:
            snapshot = graph.get_state(self._graph_config(thread_id))
            if paper_approval_from_interrupts(snapshot.interrupts) is None:
                raise PendingApprovalNotFoundError(
                    "This conversation has no pending approval."
                )
            return self._run_graph(
                graph,
                thread_id,
                Command(resume=decision),
            )

    def get_conversation_state(self, thread_id: str) -> ConversationState:
        self.get_conversation(thread_id)
        with self._open_graph(thread_id) as graph:
            snapshot = graph.get_state(self._graph_config(thread_id))

        visible_messages: list[VisibleMessage] = []
        for message in snapshot.values.get("messages", []):
            text = str(message.text)
            if isinstance(message, HumanMessage) and text:
                visible_messages.append(VisibleMessage("user", text))
            elif isinstance(message, AIMessage) and text:
                visible_messages.append(VisibleMessage("assistant", text))

        return ConversationState(
            messages=visible_messages,
            pending_approval=paper_approval_from_interrupts(snapshot.interrupts),
        )

    def ingest_pdf(
        self, thread_id: str, filename: str, content: bytes
    ) -> IngestedDocument:
        self.get_conversation(thread_id)
        safe_filename = Path(filename).name
        if not safe_filename.lower().endswith(".pdf"):
            raise ApplicationValidationError("Only PDF files can be uploaded.")
        if not content:
            raise ApplicationValidationError("Uploaded PDF cannot be empty.")
        if len(content) > MAX_PDF_UPLOAD_BYTES:
            raise ApplicationValidationError("PDF uploads cannot exceed 20 MiB.")

        try:
            with TemporaryDirectory(prefix="agentic-chatbot-pdf-") as temp_directory:
                pdf_path = Path(temp_directory) / safe_filename
                pdf_path.write_bytes(content)
                return create_document_rag_service(self.settings).ingest_pdf(
                    pdf_path, thread_id
                )
        except ValueError as error:
            raise ApplicationValidationError(str(error)) from error

    def list_documents(self, thread_id: str) -> list[IngestedDocument]:
        self.get_conversation(thread_id)
        return create_document_rag_service(self.settings).list_documents(thread_id)

    @contextmanager
    def _open_graph(self, thread_id: str) -> Iterator[CompiledStateGraph]:
        model = create_gemini_model(self.settings)
        rag_service = create_document_rag_service(self.settings)
        document_search_tool = create_document_search_tool(rag_service, thread_id)
        tools = build_tools(
            self.settings,
            document_search_tool=document_search_tool,
        )
        with open_sqlite_checkpointer(self.settings.checkpoint_db_path) as checkpointer:
            yield build_chat_graph(model, tools, checkpointer=checkpointer)

    @staticmethod
    def _graph_config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    def _run_graph(
        self,
        graph: CompiledStateGraph,
        thread_id: str,
        graph_input: dict[str, Any] | Command,
    ) -> AgentReply:
        output = graph.invoke(
            graph_input,
            config=self._graph_config(thread_id),
            version="v2",
        )
        pending_approval = paper_approval_from_interrupts(output.interrupts)
        if pending_approval is not None:
            return AgentReply("pending_approval", None, pending_approval)

        assistant_message = next(
            (
                str(message.text)
                for message in reversed(output.value.get("messages", []))
                if isinstance(message, AIMessage) and str(message.text)
            ),
            None,
        )
        return AgentReply("completed", assistant_message, None)

    @staticmethod
    def _validate_title(title: str) -> str:
        normalized_title = " ".join(title.split())
        if not normalized_title:
            raise ApplicationValidationError("Conversation title cannot be empty.")
        if len(normalized_title) > MAX_TITLE_LENGTH:
            raise ApplicationValidationError(
                f"Conversation title cannot exceed {MAX_TITLE_LENGTH} characters."
            )
        return normalized_title
