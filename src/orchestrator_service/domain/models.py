"""Orchestrator domain models.

Core concepts:
- Session: isolated conversation context (can be saved/deleted)
- Plan: LLM-generated execution plan before tool use
- ToolCall: a single tool invocation (maps to a microservice endpoint)
- ConversationMessage: one turn in the session
- ConfirmationGate: blocks sensitive operations until user confirms
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class PlanStatus(str, Enum):
    DRAFT = "draft"           # LLM generated, not yet approved
    APPROVED = "approved"     # User confirmed
    EXECUTING = "executing"   # Currently running
    COMPLETED = "completed"   # All steps done
    FAILED = "failed"         # One or more steps failed
    CANCELLED = "cancelled"   # User cancelled


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


class ConfirmationType(str, Enum):
    SENSITIVE_ACCESS = "sensitive_access"       # Accessing SENSITIVE docs (Aadhaar, PAN)
    FILE_RETRIEVAL = "file_retrieval"           # Downloading/retrieving actual files
    EXTERNAL_ACTION = "external_action"          # Web search, download
    PRIVACY_OVERRIDE = "privacy_override"        # Accessing PRIVATE docs


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ServiceTarget(str, Enum):
    DOCUMENT = "document_service"
    MODEL = "model_service"
    VALIDATOR = "validator_service"
    WEB_SEARCH = "web_search_service"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Isolated conversation context."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    user_id: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[ConversationMessage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

@dataclass
class ConversationMessage:
    """One turn in a session conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # For TOOL role messages
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    service: ServiceTarget | None = None
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    confirmation_type: ConfirmationType | None = None
    rationale: str = ""  # Why this step is needed


@dataclass
class Plan:
    """LLM-generated execution plan."""
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str = ""
    goal: str = ""                     # What the user wants
    status: PlanStatus = PlanStatus.DRAFT
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    summary: str = ""                  # Final summary after execution


# ---------------------------------------------------------------------------
# Tool Calling
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool invocation."""
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    service: ServiceTarget | None = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationRequest:
    """Request for user confirmation before sensitive operations."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    confirmation_type: ConfirmationType = ConfirmationType.SENSITIVE_ACCESS
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    message: str = ""                  # Human-readable explanation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    responded: bool = False
    approved: bool = False


# ---------------------------------------------------------------------------
# Orchestrator Response
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResponse:
    """Response from the orchestrator to the user."""
    session_id: str
    message: str                       # Final response text
    plan: Plan | None = None           # If a plan was created
    confirmation_required: ConfirmationRequest | None = None
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
