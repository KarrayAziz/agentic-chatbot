"""FastAPI endpoint tests using an application-service test double."""

import asyncio
from datetime import UTC, datetime
from uuid import UUID

from httpx2 import ASGITransport, AsyncClient

from agentic_chatbot.api import create_api
from agentic_chatbot.application import (
    AgentReply,
    ApplicationValidationError,
    ConversationNotFoundError,
    ConversationState,
    PendingApprovalNotFoundError,
    VisibleMessage,
)
from agentic_chatbot.config import Settings
from agentic_chatbot.conversations import Conversation
from agentic_chatbot.rag import IngestedDocument

THREAD_ID = "4fda8f49-6234-44e8-ac68-93fd80497292"
DOCUMENT_ID = "41d259b0-0aab-49cb-90b8-8666d45dba1c"
APPROVAL = {
    "kind": "approval",
    "action": "paper_buy_stock",
    "trade_mode": "paper",
    "ticker": "AAPL",
    "quantity": 5,
    "question": "Approve this simulated paper stock purchase?",
}


def run_async_test(test_function):
    """Run one async API scenario without pytest event-loop plugins."""

    def runner() -> None:
        asyncio.run(test_function())

    return runner


class FakeAgentApplication:
    """Stateful service double that keeps API tests offline."""

    def __init__(self) -> None:
        now = datetime.now(UTC).isoformat()
        self.conversations: dict[str, Conversation] = {}
        self.now = now
        self.messages: list[VisibleMessage] = []
        self.pending_approval: dict | None = None
        self.documents: list[IngestedDocument] = []
        self.approval_decisions: list[str] = []

    def create_conversation(self, title: str | None = None) -> Conversation:
        conversation = Conversation(
            THREAD_ID,
            title or "New conversation",
            self.now,
            self.now,
        )
        self.conversations[THREAD_ID] = conversation
        return conversation

    def list_conversations(self) -> list[Conversation]:
        return list(self.conversations.values())

    def get_conversation(self, thread_id: str) -> Conversation:
        try:
            return self.conversations[thread_id]
        except KeyError as error:
            raise ConversationNotFoundError("Conversation does not exist.") from error

    def rename_conversation(self, thread_id: str, title: str) -> Conversation:
        previous = self.get_conversation(thread_id)
        renamed = Conversation(
            thread_id,
            title,
            previous.created_at,
            self.now,
        )
        self.conversations[thread_id] = renamed
        return renamed

    def delete_conversation(self, thread_id: str) -> None:
        self.get_conversation(thread_id)
        del self.conversations[thread_id]

    def send_message(self, thread_id: str, content: str) -> AgentReply:
        self.get_conversation(thread_id)
        self.messages.append(VisibleMessage("user", content))
        if "buy" in content.lower():
            self.pending_approval = APPROVAL
            return AgentReply("pending_approval", None, APPROVAL)
        answer = f"Gemini answered: {content}"
        self.messages.append(VisibleMessage("assistant", answer))
        return AgentReply("completed", answer, None)

    def get_conversation_state(self, thread_id: str) -> ConversationState:
        self.get_conversation(thread_id)
        return ConversationState(list(self.messages), self.pending_approval)

    def respond_to_approval(self, thread_id: str, decision: str) -> AgentReply:
        self.get_conversation(thread_id)
        if self.pending_approval is None:
            raise PendingApprovalNotFoundError("No pending approval.")
        self.approval_decisions.append(decision)
        self.pending_approval = None
        answer = f"Paper trade {decision}d."
        self.messages.append(VisibleMessage("assistant", answer))
        return AgentReply("completed", answer, None)

    def ingest_pdf(
        self, thread_id: str, filename: str, content: bytes
    ) -> IngestedDocument:
        self.get_conversation(thread_id)
        if not filename.lower().endswith(".pdf"):
            raise ApplicationValidationError("Only PDF files can be uploaded.")
        document = IngestedDocument(DOCUMENT_ID, filename, 3, 2)
        self.documents.append(document)
        return document

    def list_documents(self, thread_id: str) -> list[IngestedDocument]:
        self.get_conversation(thread_id)
        return list(self.documents)


def _client() -> tuple[AsyncClient, FakeAgentApplication]:
    service = FakeAgentApplication()
    api = create_api(
        settings=Settings(_env_file=None),
        service=service,  # type: ignore[arg-type]
    )
    client = AsyncClient(
        transport=ASGITransport(app=api),
        base_url="http://testserver",
    )
    return client, service


@run_async_test
async def test_conversation_crud_and_interactive_docs() -> None:
    client, _ = _client()
    async with client:
        assert (await client.get("/docs")).status_code == 200
        created = await client.post(
            "/api/conversations", json={"title": "API lesson"}
        )
        assert created.status_code == 201
        assert UUID(created.json()["thread_id"]) == UUID(THREAD_ID)
        assert created.json()["title"] == "API lesson"

        conversations = await client.get("/api/conversations")
        assert conversations.json()[0]["thread_id"] == THREAD_ID
        metadata = await client.get(f"/api/conversations/{THREAD_ID}")
        assert metadata.status_code == 200

        renamed = await client.patch(
            f"/api/conversations/{THREAD_ID}",
            json={"title": "Renamed lesson"},
        )
        assert renamed.json()["title"] == "Renamed lesson"

        deleted = await client.delete(f"/api/conversations/{THREAD_ID}")
        assert deleted.status_code == 204
        missing = await client.get(f"/api/conversations/{THREAD_ID}")
        assert missing.status_code == 404


@run_async_test
async def test_message_state_and_hitl_resume_endpoints() -> None:
    client, service = _client()
    async with client:
        await client.post("/api/conversations")

        normal = await client.post(
            f"/api/conversations/{THREAD_ID}/messages",
            json={"content": "Hello"},
        )
        assert normal.status_code == 200
        assert normal.json()["status"] == "completed"
        assert normal.json()["assistant_message"] == "Gemini answered: Hello"

        interrupted = await client.post(
            f"/api/conversations/{THREAD_ID}/messages",
            json={"content": "Paper buy 5 AAPL shares"},
        )
        assert interrupted.json()["status"] == "pending_approval"
        assert interrupted.json()["pending_approval"]["ticker"] == "AAPL"

        state = await client.get(f"/api/conversations/{THREAD_ID}/state")
        assert state.status_code == 200
        assert state.json()["pending_approval"]["quantity"] == 5
        assert [message["role"] for message in state.json()["messages"]] == [
            "user",
            "assistant",
            "user",
        ]

        resumed = await client.post(
            f"/api/conversations/{THREAD_ID}/approval",
            json={"decision": "reject"},
        )
        assert resumed.json()["status"] == "completed"
        assert service.approval_decisions == ["reject"]

        no_pending = await client.post(
            f"/api/conversations/{THREAD_ID}/approval",
            json={"decision": "approve"},
        )
        assert no_pending.status_code == 409


@run_async_test
async def test_pdf_upload_and_document_listing() -> None:
    client, _ = _client()
    async with client:
        await client.post("/api/conversations")

        uploaded = await client.post(
            f"/api/conversations/{THREAD_ID}/documents",
            files={"file": ("guide.pdf", b"%PDF-test", "application/pdf")},
        )
        assert uploaded.status_code == 201
        assert uploaded.json() == {
            "document_id": DOCUMENT_ID,
            "source_filename": "guide.pdf",
            "chunk_count": 3,
            "page_count": 2,
        }

        listed = await client.get(f"/api/conversations/{THREAD_ID}/documents")
        assert listed.status_code == 200
        assert listed.json() == [uploaded.json()]

        invalid = await client.post(
            f"/api/conversations/{THREAD_ID}/documents",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )
        assert invalid.status_code == 400


@run_async_test
async def test_request_validation_rejects_invalid_uuid_and_decision() -> None:
    client, _ = _client()
    async with client:
        invalid_uuid = await client.get("/api/conversations/not-a-uuid")
        assert invalid_uuid.status_code == 422
        response = await client.post(
            f"/api/conversations/{THREAD_ID}/approval",
            json={"decision": "maybe"},
        )
        assert response.status_code == 422
