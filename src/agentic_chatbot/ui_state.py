"""Small, framework-independent helpers for the Streamlit presentation."""

import re
from dataclasses import dataclass, field
from typing import Any

from agentic_chatbot.api_client import ClientStreamEvent
from agentic_chatbot.api_models import ConversationResponse


TOOL_STATUS_MESSAGES = {
    "calculator": "Calculating…",
    "get_current_weather": "Checking weather…",
    "tavily_search": "Searching the web…",
    "search_documents": "Searching uploaded documents…",
    "paper_buy_stock": "Preparing paper-trade approval…",
}
SOURCE_PATTERN = re.compile(
    r"(?P<filename>[A-Za-z0-9][A-Za-z0-9_.-]*(?: [A-Za-z0-9_.-]+)*\.pdf)"
    r"\s*,?\s*page\s+(?P<page>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SourceReference:
    filename: str
    page: int


@dataclass(slots=True)
class StreamViewState:
    """Accumulate client stream events without duplicating final text."""

    assistant_text: str = ""
    active_tool: str | None = None
    tool_status: str | None = None
    pending_approval: dict[str, Any] | None = None
    completed: bool = False
    error: str | None = None
    seen_events: list[str] = field(default_factory=list)

    def apply(self, event: ClientStreamEvent) -> None:
        self.seen_events.append(event.type)
        if event.type == "assistant_chunk" and event.content:
            self.assistant_text += event.content
        elif event.type == "tool_started" and event.tool:
            self.active_tool = event.tool
            self.tool_status = tool_status_for(event.tool)
        elif event.type == "tool_finished":
            if not event.tool or event.tool == self.active_tool:
                self.active_tool = None
                self.tool_status = None
        elif event.type == "pending_approval":
            self.pending_approval = event.approval
            self.completed = False
            self.active_tool = None
            self.tool_status = None
        elif event.type == "complete":
            self.completed = True
            self.active_tool = None
            self.tool_status = None
        elif event.type == "error":
            self.error = event.message or "The backend stream reported an error."
            self.active_tool = None
            self.tool_status = None


def tool_status_for(tool_name: str) -> str:
    """Return a modest user-facing status without exposing tool arguments."""

    return TOOL_STATUS_MESSAGES.get(tool_name, "Using a tool…")


def choose_active_thread(
    conversations: list[ConversationResponse], current_thread_id: str | None
) -> str | None:
    """Keep a valid selection or select the most recently listed thread."""

    identifiers = [str(conversation.thread_id) for conversation in conversations]
    if current_thread_id in identifiers:
        return current_thread_id
    return identifiers[0] if identifiers else None


def extract_source_references(text: str) -> list[SourceReference]:
    """Extract unique filename/page citations when Gemini included them."""

    sources: list[SourceReference] = []
    seen: set[tuple[str, int]] = set()
    for match in SOURCE_PATTERN.finditer(text):
        filename = re.sub(
            r"^(?:(?:see|and|from|source|sources)\s+)+",
            "",
            match.group("filename").strip(),
            flags=re.IGNORECASE,
        )
        source = SourceReference(
            filename=filename,
            page=int(match.group("page")),
        )
        identity = (source.filename.casefold(), source.page)
        if identity not in seen:
            sources.append(source)
            seen.add(identity)
    return sources
