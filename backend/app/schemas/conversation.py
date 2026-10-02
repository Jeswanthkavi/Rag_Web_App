"""Pydantic schemas for Conversation and Message management."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from backend.app.models.message import MessageRole


class ConversationCreate(BaseModel):
    """Request schema for creating a new conversation."""
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(
        default="New Conversation",
        max_length=255,
        description="Optional title for the conversation",
    )


class ConversationUpdate(BaseModel):
    """Request schema for updating a conversation."""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="New title for the conversation",
    )


class MessageResponse(BaseModel):
    """Response schema for a single conversational message turn."""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique message UUID")
    conversation_id: str = Field(..., description="Parent conversation UUID")
    role: MessageRole = Field(..., description="Role: USER, ASSISTANT, or SYSTEM")
    content: str = Field(..., description="Message text content")
    citations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Authoritative citations associated with this turn")
    token_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Token usage telemetry")
    created_at: datetime = Field(..., description="Timestamp when the message was created")


class ConversationResponse(BaseModel):
    """Response schema for conversation list items."""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique conversation UUID")
    user_id: str = Field(..., description="Owner user UUID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last updated timestamp")


class ConversationDetailResponse(BaseModel):
    """Detailed response schema for a conversation including chronological message turns."""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique conversation UUID")
    user_id: str = Field(..., description="Owner user UUID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last updated timestamp")
    messages: List[MessageResponse] = Field(default_factory=list, description="Chronological list of message turns")
