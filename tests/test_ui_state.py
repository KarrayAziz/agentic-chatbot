"""Tests for framework-independent Streamlit presentation state."""

from datetime import UTC, datetime
from uuid import UUID

from agentic_chatbot.api_client import ClientStreamEvent
from agentic_chatbot.api_models import ConversationResponse
from agentic_chatbot.ui_state import (
    StreamViewState,
    choose_active_thread,
    extract_source_references,
    tool_status_for,
)


def _conversation(thread_id: str, title: str) -> ConversationResponse:
    now = datetime.now(UTC)
    return ConversationResponse(
        thread_id=UUID(thread_id),
        title=title,
        created_at=now,
        updated_at=now,
    )


def test_conversation_selection_preserves_valid_backend_uuid() -> None:
    first = _conversation("00000000-0000-0000-0000-000000000001", "First")
    second = _conversation("00000000-0000-0000-0000-000000000002", "Second")

    assert choose_active_thread([first, second], str(second.thread_id)) == str(
        second.thread_id
    )
    assert choose_active_thread([first, second], "missing") == str(first.thread_id)
    assert choose_active_thread([], None) is None


def test_stream_state_accumulates_once_and_tracks_tool_status() -> None:
    state = StreamViewState()
    events = [
        ClientStreamEvent("tool_started", tool="get_current_weather"),
        ClientStreamEvent("tool_finished", tool="get_current_weather"),
        ClientStreamEvent("assistant_chunk", content="It is "),
        ClientStreamEvent("assistant_chunk", content="sunny."),
        ClientStreamEvent("complete"),
    ]

    state.apply(events[0])
    assert state.tool_status == "Checking weather…"
    for event in events[1:]:
        state.apply(event)

    assert state.assistant_text == "It is sunny."
    assert state.completed is True
    assert state.tool_status is None
    assert tool_status_for("search_documents") == "Searching uploaded documents…"


def test_pending_approval_is_not_normal_completion() -> None:
    approval = {"ticker": "AAPL", "quantity": 3}
    state = StreamViewState()
    state.apply(ClientStreamEvent("tool_started", tool="paper_buy_stock"))
    state.apply(ClientStreamEvent("pending_approval", approval=approval))

    assert state.pending_approval == approval
    assert state.completed is False
    assert state.tool_status is None


def test_source_references_are_unique_and_readable() -> None:
    answer = (
        "The graph restores checkpoints (agent guide.pdf, page 4). "
        "See agent guide.pdf, page 4 and appendix.pdf page 2."
    )

    sources = extract_source_references(answer)

    assert [(source.filename, source.page) for source in sources] == [
        ("agent guide.pdf", 4),
        ("appendix.pdf", 2),
    ]
