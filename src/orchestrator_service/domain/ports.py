"""Orchestrator port protocols.

These define the interfaces for infrastructure adapters.
The orchestrator depends on these abstractions, not concrete implementations.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from orchestrator_service.domain.models import (
    ConfirmationRequest,
    ConfirmationType,
    ConversationMessage,
    Plan,
    Session,
    SessionStatus,
    ServiceTarget,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

@runtime_checkable
class SessionStore(Protocol):
    """Persist and retrieve sessions (conversation contexts)."""

    def save(self, session: Session) -> None: ...
    def get(self, session_id: str) -> Session | None: ...
    def delete(self, session_id: str) -> bool: ...
    def list_sessions(
        self, *, user_id: str | None = None, status: SessionStatus | None = None
    ) -> list[Session]: ...
    def archive(self, session_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# Confirmation store
# ---------------------------------------------------------------------------

@runtime_checkable
class ConfirmationStore(Protocol):
    """Track pending confirmation requests."""

    def save(self, request: ConfirmationRequest) -> None: ...
    def get(self, request_id: str) -> ConfirmationRequest | None: ...
    def get_pending(self, session_id: str) -> list[ConfirmationRequest]: ...
    def respond(self, request_id: str, approved: bool) -> None: ...
    def get_by_tool_call(self, call_id: str) -> ConfirmationRequest | None: ...


# ---------------------------------------------------------------------------
# Service clients (HTTP calls to microservices)
# ---------------------------------------------------------------------------

@runtime_checkable
class ServiceClient(Protocol):
    """Client that calls a downstream microservice."""

    @property
    def target(self) -> ServiceTarget: ...

    def health(self) -> dict[str, Any]: ...

    def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """LLM inference with tool calling support."""

    def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    def health(self) -> dict[str, Any]: ...


class LLMResponse:
    """Structured response from the LLM."""

    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning: str = "",
        model: str = "",
        token_usage: dict[str, int] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning
        self.model = model
        self.token_usage = token_usage


# ---------------------------------------------------------------------------
# Tool registry (maps tool names → service targets)
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolRegistry(Protocol):
    """Maps tool names to their owning service and exposes tool schemas for the LLM."""

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None: ...
    def get_all_tool_schemas(self) -> list[dict[str, Any]]: ...
    def get_service(self, tool_name: str) -> ServiceTarget | None: ...
    def requires_confirmation(self, tool_name: str, args: dict[str, Any]) -> ConfirmationType | None: ...
