"""File-based confirmation store.

Persistent replacement for InMemoryConfirmationStore.
Stores pending confirmation requests as JSON files under data/confirmations/.
Survives service restarts.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator_service.domain.models import ConfirmationRequest

logger = logging.getLogger("orchestrator.confirmation_store")


class FileConfirmationStore:
    """File-based persistent confirmation store.

    Structure:
        data/confirmations/{request_id}.json

    Each file is a single ConfirmationRequest serialized as JSON.
    Responded confirmations are marked with responded=True.
    """

    def __init__(self, base_dir: str = "./data/confirmations") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, request: ConfirmationRequest) -> None:
        path = self._path(request.request_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serialize(request)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(
            "Confirmation request %s saved: %s",
            request.request_id,
            request.confirmation_type.value,
        )

    def get(self, request_id: str) -> ConfirmationRequest | None:
        path = self._path(request_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._deserialize(data)
        except Exception:
            logger.exception("Failed to load confirmation %s", request_id)
            return None

    def get_pending(self, session_id: str) -> list[ConfirmationRequest]:
        pending: list[ConfirmationRequest] = []
        for f in self._base.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                req = self._deserialize(data)
                if req.session_id == session_id and not req.responded:
                    pending.append(req)
            except Exception:
                logger.warning("Skipping corrupt confirmation file: %s", f.name)
        return pending

    def respond(self, request_id: str, approved: bool, correction: str | None = None) -> None:
        req = self.get(request_id)
        if req is None:
            logger.warning("Confirmation request %s not found", request_id)
            return
        req.responded = True
        req.approved = approved
        req.correction = correction
        # Overwrite the file with updated state
        path = self._path(request_id)
        data = self._serialize(req)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(
            "Confirmation %s: %s",
            request_id,
            "APPROVED" if approved else "DENIED",
        )

    def get_by_tool_call(self, call_id: str) -> ConfirmationRequest | None:
        """Find a confirmation request by its associated tool call ID."""
        for f in self._base.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("tool_call_id") == call_id:
                    return self._deserialize(data)
            except Exception:
                continue
        return None

    def _path(self, request_id: str) -> Path:
        return self._base / f"{request_id}.json"

    def _serialize(self, request: ConfirmationRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "confirmation_type": request.confirmation_type.value,
            "tool_name": request.tool_name,
            "tool_args": request.tool_args,
            "message": request.message,
            "created_at": request.created_at.isoformat() if request.created_at else None,
            "responded": request.responded,
            "approved": request.approved,
            "correction": getattr(request, "correction", None),
            "tool_call_id": getattr(request, "tool_call_id", None),
        }

    def _deserialize(self, data: dict[str, Any]) -> ConfirmationRequest:
        from orchestrator_service.domain.models import ConfirmationType
        req = ConfirmationRequest(
            request_id=data["request_id"],
            session_id=data["session_id"],
            confirmation_type=ConfirmationType(data["confirmation_type"]),
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args", {}),
            message=data.get("message", ""),
        )
        if "created_at" in data and data["created_at"]:
            req.created_at = datetime.fromisoformat(data["created_at"])
        req.responded = data.get("responded", False)
        req.approved = data.get("approved", False)
        if data.get("correction"):
            req.correction = data["correction"]
        if data.get("tool_call_id"):
            req.tool_call_id = data["tool_call_id"]
        return req
