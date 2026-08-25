"""Offline tests for the Streamlit-to-FastAPI HTTP client."""

import json
from collections.abc import Iterator

import httpx
import pytest

from agentic_chatbot.api_client import (
    AgentApiClient,
    BackendClientError,
    BackendUnavailableError,
)

THREAD_ID = "4fda8f49-6234-44e8-ac68-93fd80497292"
DOCUMENT_ID = "41d259b0-0aab-49cb-90b8-8666d45dba1c"
NOW = "2026-08-25T10:00:00+00:00"


def _conversation(title: str = "API chat") -> dict:
    return {
        "thread_id": THREAD_ID,
        "title": title,
        "created_at": NOW,
        "updated_at": NOW,
    }


class FragmentedNDJSON(httpx.SyncByteStream):
    """Yield deliberately fragmented bytes to exercise incremental parsing."""

    def __iter__(self) -> Iterator[bytes]:
        yield b'{"type":"assistant_chunk","content":"Hello "}\n{"type":"tool_'
        yield b'started","tool":"calculator"}\n'
        yield b'{"type":"tool_finished","tool":"calculator"}\n'
        yield b'{"type":"assistant_chunk","content":"world"}\n'
        yield b'{"type":"complete"}\n'


def test_client_conversation_approval_and_pdf_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/api/conversations":
            return httpx.Response(201, json=_conversation())
        if request.method == "GET" and path == "/api/conversations":
            return httpx.Response(200, json=[_conversation()])
        if request.method == "PATCH":
            return httpx.Response(200, json=_conversation("Renamed"))
        if request.method == "DELETE":
            return httpx.Response(204)
        if path.endswith("/approval"):
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "assistant_message": "The paper trade was rejected.",
                    "pending_approval": None,
                },
            )
        if request.method == "POST" and path.endswith("/documents"):
            assert b'filename="guide.pdf"' in request.content
            return httpx.Response(
                201,
                json={
                    "document_id": DOCUMENT_ID,
                    "source_filename": "guide.pdf",
                    "chunk_count": 4,
                    "page_count": 2,
                },
            )
        if request.method == "GET" and path.endswith("/documents"):
            return httpx.Response(
                200,
                json=[
                    {
                        "document_id": DOCUMENT_ID,
                        "source_filename": "guide.pdf",
                        "chunk_count": 4,
                        "page_count": 2,
                    }
                ],
            )
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    with AgentApiClient(
        "http://backend", transport=httpx.MockTransport(handler)
    ) as client:
        created = client.create_conversation()
        listed = client.list_conversations()
        renamed = client.rename_conversation(THREAD_ID, "Renamed")
        decision = client.respond_to_approval(THREAD_ID, "reject")
        client.respond_to_approval(THREAD_ID, "approve")
        uploaded = client.upload_pdf(THREAD_ID, "guide.pdf", b"%PDF-test")
        documents = client.list_documents(THREAD_ID)
        client.delete_conversation(THREAD_ID)

    assert str(created.thread_id) == THREAD_ID
    assert listed[0].title == "API chat"
    assert renamed.title == "Renamed"
    assert decision.assistant_message == "The paper trade was rejected."
    assert uploaded.source_filename == "guide.pdf"
    assert documents[0].page_count == 2
    approval_requests = [
        request for request in requests if request.url.path.endswith("/approval")
    ]
    assert [json.loads(request.content) for request in approval_requests] == [
        {"decision": "reject"},
        {"decision": "approve"},
    ]


def test_client_consumes_fragmented_ndjson_incrementally() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/messages/stream")
        assert json.loads(request.content) == {"content": "Hello"}
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=FragmentedNDJSON(),
        )

    with AgentApiClient(
        "http://backend", transport=httpx.MockTransport(handler)
    ) as client:
        iterator = client.stream_message(THREAD_ID, "Hello")
        first = next(iterator)
        remaining = list(iterator)

    assert first.type == "assistant_chunk"
    assert first.content == "Hello "
    assert [event.type for event in remaining] == [
        "tool_started",
        "tool_finished",
        "assistant_chunk",
        "complete",
    ]


def test_client_reports_unavailable_backend_concisely() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with AgentApiClient(
        "http://backend", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(BackendUnavailableError, match="Is it running"):
            client.list_conversations()


def test_stream_reports_http_error_detail_without_a_traceback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Conversation does not exist."})

    with AgentApiClient(
        "http://backend", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(BackendClientError, match="Conversation does not exist"):
            list(client.stream_message(THREAD_ID, "Hello"))
