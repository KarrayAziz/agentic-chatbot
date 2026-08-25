"""Typed HTTP client used by Streamlit to reach the FastAPI backend."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agentic_chatbot.api_models import (
    AgentReplyResponse,
    ConversationResponse,
    ConversationStateResponse,
    DocumentResponse,
    HealthResponse,
)


class BackendClientError(RuntimeError):
    """Base class for concise, user-facing backend client failures."""


class BackendUnavailableError(BackendClientError):
    """Raised when FastAPI cannot be reached."""


class BackendProtocolError(BackendClientError):
    """Raised when FastAPI returns data outside the documented protocol."""


ClientEventType = Literal[
    "assistant_chunk",
    "tool_started",
    "tool_finished",
    "pending_approval",
    "complete",
    "error",
]
VALID_EVENT_TYPES: set[str] = {
    "assistant_chunk",
    "tool_started",
    "tool_finished",
    "pending_approval",
    "complete",
    "error",
}
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ClientStreamEvent:
    """A validated event from the backend's NDJSON stream."""

    type: ClientEventType
    content: str | None = None
    tool: str | None = None
    approval: dict[str, Any] | None = None
    message: str | None = None


class AgentApiClient:
    """Call the existing FastAPI API without accessing backend internals."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(connect=5, read=None, write=30, pool=5)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> "AgentApiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> HealthResponse:
        return self._parse_model(self._request("GET", "/api/health"), HealthResponse)

    def create_conversation(self, title: str | None = None) -> ConversationResponse:
        payload = {"title": title} if title is not None else {}
        return self._parse_model(
            self._request("POST", "/api/conversations", json=payload),
            ConversationResponse,
        )

    def list_conversations(self) -> list[ConversationResponse]:
        response = self._request("GET", "/api/conversations")
        try:
            return [
                ConversationResponse.model_validate(item)
                for item in response.json()
            ]
        except (TypeError, ValueError, ValidationError) as error:
            raise BackendProtocolError(
                "The backend returned an invalid conversation list."
            ) from error

    def get_conversation(self, thread_id: str) -> ConversationResponse:
        return self._parse_model(
            self._request("GET", f"/api/conversations/{thread_id}"),
            ConversationResponse,
        )

    def rename_conversation(
        self, thread_id: str, title: str
    ) -> ConversationResponse:
        return self._parse_model(
            self._request(
                "PATCH",
                f"/api/conversations/{thread_id}",
                json={"title": title},
            ),
            ConversationResponse,
        )

    def delete_conversation(self, thread_id: str) -> None:
        self._request("DELETE", f"/api/conversations/{thread_id}")

    def get_conversation_state(self, thread_id: str) -> ConversationStateResponse:
        return self._parse_model(
            self._request("GET", f"/api/conversations/{thread_id}/state"),
            ConversationStateResponse,
        )

    def stream_message(
        self, thread_id: str, content: str
    ) -> Iterator[ClientStreamEvent]:
        """Yield events as NDJSON lines arrive from FastAPI."""

        try:
            with self._client.stream(
                "POST",
                f"/api/conversations/{thread_id}/messages/stream",
                json={"content": content},
            ) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if line.strip():
                        yield self._parse_stream_event(line)
        except httpx.RequestError as error:
            raise BackendUnavailableError(
                "Cannot reach the FastAPI backend. Is it running?"
            ) from error

    def respond_to_approval(
        self,
        thread_id: str,
        decision: Literal["approve", "reject"],
    ) -> AgentReplyResponse:
        return self._parse_model(
            self._request(
                "POST",
                f"/api/conversations/{thread_id}/approval",
                json={"decision": decision},
            ),
            AgentReplyResponse,
        )

    def upload_pdf(
        self,
        thread_id: str,
        filename: str,
        content: bytes,
    ) -> DocumentResponse:
        return self._parse_model(
            self._request(
                "POST",
                f"/api/conversations/{thread_id}/documents",
                files={"file": (filename, content, "application/pdf")},
            ),
            DocumentResponse,
        )

    def list_documents(self, thread_id: str) -> list[DocumentResponse]:
        response = self._request(
            "GET", f"/api/conversations/{thread_id}/documents"
        )
        try:
            return [DocumentResponse.model_validate(item) for item in response.json()]
        except (TypeError, ValueError, ValidationError) as error:
            raise BackendProtocolError(
                "The backend returned an invalid document list."
            ) from error

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as error:
            raise BackendUnavailableError(
                "Cannot reach the FastAPI backend. Is it running?"
            ) from error
        self._raise_for_status(response)
        return response

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        if not response.is_stream_consumed:
            response.read()
        try:
            detail = response.json().get("detail")
        except (json.JSONDecodeError, AttributeError, TypeError):
            detail = None
        message = detail or (
            f"Backend request failed with status {response.status_code}."
        )
        raise BackendClientError(str(message))

    @staticmethod
    def _parse_model(
        response: httpx.Response, model_type: type[ModelT]
    ) -> ModelT:
        try:
            return model_type.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise BackendProtocolError(
                "The backend returned an unexpected response."
            ) from error

    @staticmethod
    def _parse_stream_event(line: str) -> ClientStreamEvent:
        try:
            payload = json.loads(line)
            event_type = payload["type"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise BackendProtocolError(
                "The backend returned an invalid streaming event."
            ) from error
        if event_type not in VALID_EVENT_TYPES:
            raise BackendProtocolError(
                f"The backend returned an unknown streaming event: {event_type}."
            )
        return ClientStreamEvent(
            type=event_type,
            content=_optional_string(payload, "content"),
            tool=_optional_string(payload, "tool"),
            approval=payload.get("approval"),
            message=_optional_string(payload, "message"),
        )


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value is not None else None
