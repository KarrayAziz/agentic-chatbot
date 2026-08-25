"""FastAPI transport layer for the existing agent application services."""

import asyncio
import json
import logging
from asyncio import get_running_loop
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Empty, Queue
from threading import Event, Thread
from typing import Annotated, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from agentic_chatbot.api_models import (
    AgentReplyResponse,
    ApprovalDecisionRequest,
    ConversationResponse,
    ConversationStateResponse,
    CreateConversationRequest,
    DocumentResponse,
    HealthResponse,
    RenameConversationRequest,
    SendMessageRequest,
)
from agentic_chatbot.application import (
    MAX_PDF_UPLOAD_BYTES,
    AgentApplication,
    ApplicationError,
    ApprovalConflictError,
    ApplicationValidationError,
    ConversationNotFoundError,
)
from agentic_chatbot.config import Settings, load_settings
from agentic_chatbot.logging_config import configure_logging
from agentic_chatbot.streaming import AgentStreamEvent

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
STREAM_MEDIA_TYPE = "application/x-ndjson"
_STREAM_END = object()


async def _run_blocking(function: Callable[..., T], *args: object) -> T:
    """Run one blocking service operation outside the API event loop.

    An isolated executor also avoids a CPython 3.14 default-executor reuse issue
    present in the current development environment.
    """

    event_loop = get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-service")
    try:
        return await event_loop.run_in_executor(executor, partial(function, *args))
    finally:
        # The submitted operation has completed before this finally block. Join
        # its worker so rapid Streamlit/API requests do not accumulate threads.
        executor.shutdown(wait=True)


async def _stream_message_as_ndjson(
    service: AgentApplication,
    thread_id: str,
    content: str,
) -> AsyncIterator[bytes]:
    """Bridge the synchronous LangGraph iterator to an async HTTP body."""

    queue: Queue[AgentStreamEvent | object] = Queue()
    client_disconnected = Event()

    def enqueue(item: AgentStreamEvent | object) -> None:
        queue.put(item)

    def produce_events() -> None:
        try:
            for event in service.stream_message(thread_id, content):
                if client_disconnected.is_set():
                    break
                enqueue(event)
        except ApplicationError as error:
            enqueue(AgentStreamEvent("error", message=str(error)))
        except Exception:
            LOGGER.exception("Agent stream failed for thread %s", thread_id)
            enqueue(
                AgentStreamEvent(
                    "error",
                    message="The agent stream failed unexpectedly.",
                )
            )
        finally:
            enqueue(_STREAM_END)

    producer = Thread(
        target=produce_events,
        name="agent-stream",
        daemon=True,
    )
    producer.start()

    try:
        while True:
            try:
                event = queue.get_nowait()
            except Empty:
                # Keep the response cancellable while Gemini or a tool is busy.
                await asyncio.sleep(0.01)
                continue
            if event is _STREAM_END:
                break
            assert isinstance(event, AgentStreamEvent)
            encoded = json.dumps(
                event.as_dict(),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            yield f"{encoded}\n".encode()
    finally:
        client_disconnected.set()
        producer.join(timeout=1)


async def get_agent_application(request: Request) -> AgentApplication:
    """Return the configured application service for this API instance."""

    return request.app.state.agent_application


AgentApplicationDependency = Annotated[
    AgentApplication, Depends(get_agent_application)
]

router = APIRouter(prefix="/api", tags=["agentic-chatbot"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=201,
)
async def create_conversation(
    service: AgentApplicationDependency,
    request: CreateConversationRequest | None = None,
) -> ConversationResponse:
    conversation = await _run_blocking(
        service.create_conversation, request.title if request else None
    )
    return ConversationResponse.model_validate(conversation)


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    service: AgentApplicationDependency,
) -> list[ConversationResponse]:
    conversations = await _run_blocking(service.list_conversations)
    return [
        ConversationResponse.model_validate(conversation)
        for conversation in conversations
    ]


@router.get(
    "/conversations/{thread_id}",
    response_model=ConversationResponse,
)
async def get_conversation(
    thread_id: UUID,
    service: AgentApplicationDependency,
) -> ConversationResponse:
    conversation = await _run_blocking(service.get_conversation, str(thread_id))
    return ConversationResponse.model_validate(conversation)


@router.patch(
    "/conversations/{thread_id}",
    response_model=ConversationResponse,
)
async def rename_conversation(
    thread_id: UUID,
    request: RenameConversationRequest,
    service: AgentApplicationDependency,
) -> ConversationResponse:
    conversation = await _run_blocking(
        service.rename_conversation, str(thread_id), request.title
    )
    return ConversationResponse.model_validate(conversation)


@router.delete("/conversations/{thread_id}", status_code=204)
async def delete_conversation(
    thread_id: UUID,
    service: AgentApplicationDependency,
) -> Response:
    await _run_blocking(service.delete_conversation, str(thread_id))
    return Response(status_code=204)


@router.post(
    "/conversations/{thread_id}/messages",
    response_model=AgentReplyResponse,
)
async def send_message(
    thread_id: UUID,
    request: SendMessageRequest,
    service: AgentApplicationDependency,
) -> AgentReplyResponse:
    reply = await _run_blocking(
        service.send_message, str(thread_id), request.content
    )
    return AgentReplyResponse.model_validate(reply)


@router.post(
    "/conversations/{thread_id}/messages/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {STREAM_MEDIA_TYPE: {}},
            "description": "Newline-delimited agent stream events.",
        }
    },
)
async def stream_message(
    thread_id: UUID,
    request: SendMessageRequest,
    service: AgentApplicationDependency,
) -> StreamingResponse:
    """Stream one graph execution as sanitized newline-delimited JSON."""

    # Return an ordinary 404 before response headers if the UUID is unknown.
    await _run_blocking(service.get_conversation, str(thread_id))
    return StreamingResponse(
        _stream_message_as_ndjson(service, str(thread_id), request.content),
        media_type=STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/conversations/{thread_id}/state",
    response_model=ConversationStateResponse,
)
async def get_conversation_state(
    thread_id: UUID,
    service: AgentApplicationDependency,
) -> ConversationStateResponse:
    state = await _run_blocking(service.get_conversation_state, str(thread_id))
    return ConversationStateResponse(
        thread_id=thread_id,
        messages=[
            {"role": message.role, "content": message.content}
            for message in state.messages
        ],
        pending_approval=state.pending_approval,
    )


@router.post(
    "/conversations/{thread_id}/documents",
    response_model=DocumentResponse,
    status_code=201,
)
async def upload_pdf(
    thread_id: UUID,
    file: Annotated[UploadFile, File(description="PDF to ingest")],
    service: AgentApplicationDependency,
) -> DocumentResponse:
    content = await file.read(MAX_PDF_UPLOAD_BYTES + 1)
    document = await _run_blocking(
        service.ingest_pdf,
        str(thread_id),
        file.filename or "upload.pdf",
        content,
    )
    return DocumentResponse.model_validate(document)


@router.get(
    "/conversations/{thread_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    thread_id: UUID,
    service: AgentApplicationDependency,
) -> list[DocumentResponse]:
    documents = await _run_blocking(service.list_documents, str(thread_id))
    return [
        DocumentResponse.model_validate(document)
        for document in documents
    ]


@router.post(
    "/conversations/{thread_id}/approval",
    response_model=AgentReplyResponse,
)
async def respond_to_approval(
    thread_id: UUID,
    request: ApprovalDecisionRequest,
    service: AgentApplicationDependency,
) -> AgentReplyResponse:
    reply = await _run_blocking(
        service.respond_to_approval, str(thread_id), request.decision
    )
    return AgentReplyResponse.model_validate(reply)


def create_api(
    *,
    settings: Settings | None = None,
    service: AgentApplication | None = None,
) -> FastAPI:
    """Create the FastAPI application with injectable application services."""

    configured_settings = settings or load_settings()
    configure_logging(configured_settings.log_level)
    api = FastAPI(
        title="Agentic AI Chatbot API",
        version="0.1.0",
        description=(
            "FastAPI transport for the existing LangGraph chatbot, persistent "
            "threads, paper-trade approvals, and PDF retrieval."
        ),
    )
    api.state.agent_application = service or AgentApplication(configured_settings)
    api.include_router(router)

    @api.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        request: Request, error: ConversationNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @api.exception_handler(ApplicationValidationError)
    async def validation_error_handler(
        request: Request, error: ApplicationValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @api.exception_handler(ApprovalConflictError)
    async def pending_approval_handler(
        request: Request, error: ApprovalConflictError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    return api


app = create_api()
