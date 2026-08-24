"""Pydantic transport models for the FastAPI backend."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AttributeModel(BaseModel):
    """Base response model that can validate application dataclasses."""

    model_config = ConfigDict(from_attributes=True)


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=60)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=60)


class ConversationResponse(AttributeModel):
    thread_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]


class PendingApprovalResponse(BaseModel):
    kind: Literal["approval"]
    action: Literal["paper_buy_stock"]
    trade_mode: Literal["paper"]
    ticker: str
    quantity: int
    question: str


class AgentReplyResponse(AttributeModel):
    status: Literal["completed", "pending_approval"]
    assistant_message: str | None
    pending_approval: PendingApprovalResponse | None


class HistoryMessageResponse(AttributeModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationStateResponse(BaseModel):
    thread_id: UUID
    messages: list[HistoryMessageResponse]
    pending_approval: PendingApprovalResponse | None


class DocumentResponse(AttributeModel):
    document_id: UUID
    source_filename: str
    chunk_count: int
    page_count: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
