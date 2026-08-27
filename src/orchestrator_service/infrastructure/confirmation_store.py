"""In-memory confirmation store.

Tracks pending confirmation requests for sensitive operations.
V1: in-memory only. For production, persist to database.
"""
from __future__ import annotations

import logging
from typing import Any

from orchestrator_service.domain.models import ConfirmationRequest

logger = logging.getLogger("orchestrator.confirmation_store")


class InMemoryConfirmationStore:
    """In-memory store for confirmation requests."""

    def __init__(self) -> None:
        self._requests: dict[str, ConfirmationRequest] = {}
        self._by_tool_call: dict[str, str] = {}  # call_id → request_id
        self._by_session: dict[str, list[str]] = {}  # session_id → [request_ids]

    def save(self, request: ConfirmationRequest) -> None:
        self._requests[request.request_id] = request
        # Track by session
        self._by_session.setdefault(request.session_id, []).append(request.request_id)
        # Track by tool call if available
        if request.tool_args.get("call_id"):
            self._by_tool_call[request.tool_args["call_id"]] = request.request_id
        logger.info(
            "Confirmation request %s created: %s",
            request.request_id,
            request.confirmation_type.value,
        )

    def get(self, request_id: str) -> ConfirmationRequest | None:
        return self._requests.get(request_id)

    def get_pending(self, session_id: str) -> list[ConfirmationRequest]:
        req_ids = self._by_session.get(session_id, [])
        return [
            self._requests[rid]
            for rid in req_ids
            if rid in self._requests and not self._requests[rid].responded
        ]

    def respond(self, request_id: str, approved: bool) -> None:
        req = self._requests.get(request_id)
        if req is None:
            logger.warning("Confirmation request %s not found", request_id)
            return
        req.responded = True
        req.approved = approved
        logger.info(
            "Confirmation %s: %s",
            request_id,
            "APPROVED" if approved else "DENIED",
        )

    def get_by_tool_call(self, call_id: str) -> ConfirmationRequest | None:
        req_id = self._by_tool_call.get(call_id)
        return self._requests.get(req_id) if req_id else None
