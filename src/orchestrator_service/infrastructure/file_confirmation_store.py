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

    Structure (new writes):
        data/confirmations/{session_id}/{request_id}.json

    Sharding by session keeps the per-turn ``get_pending`` scan to one
    small directory instead of every confirmation ever stored. Files
    written by older versions live flat as ``{request_id}.json`` and are
    still found via the legacy fallback in :meth:`_locate`.

    Each file is a single ConfirmationRequest serialized as JSON.
    Responded confirmations are marked with responded=True.
    """

    def __init__(self, base_dir: str = "./data/confirmations") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, request: ConfirmationRequest) -> None:
        path = self._shard_path(request.session_id, request.request_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serialize(request)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(
            "Confirmation request %s saved: %s",
            request.request_id,
            request.confirmation_type.value,
        )

    def get(self, request_id: str) -> ConfirmationRequest | None:
        path = self._locate(request_id)
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._deserialize(data)
        except Exception:
            logger.exception("Failed to load confirmation %s", request_id)
            return None

    def get_pending(self, session_id: str) -> list[ConfirmationRequest]:
        pending: list[ConfirmationRequest] = []
        candidates = list(self._base.glob(f"{self._safe_segment(session_id)}/*.json"))
        # Legacy fallback: pre-sharding files stored flat.
        candidates += [f for f in self._base.glob("*.json") if f.is_file()]
        for f in candidates:
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
        # Overwrite the file with updated state, wherever it lives
        # (sharded dir for new files, flat for legacy ones).
        path = self._locate(request_id)
        if path is None:
            logger.warning("Confirmation file %s vanished before respond", request_id)
            return
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

    @staticmethod
    def _safe_segment(value: str) -> str:
        """Confine a session id to a single directory level."""
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value) or "_"

    def _shard_path(self, session_id: str, request_id: str) -> Path:
        return self._base / self._safe_segment(session_id) / f"{request_id}.json"

    def _locate(self, request_id: str) -> Path | None:
        """Find a confirmation file: sharded dirs first, flat legacy last."""
        for shard in self._base.glob("*/"):
            if not shard.is_dir():
                continue
            candidate = shard / f"{request_id}.json"
            if candidate.is_file():
                return candidate
        legacy = self._base / f"{request_id}.json"
        return legacy if legacy.is_file() else None

    def _path(self, request_id: str) -> Path:
        """Legacy flat path. Kept for backward compatibility; prefer _locate."""
        located = self._locate(request_id)
        return located if located is not None else self._base / f"{request_id}.json"

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
            "thought_signature": getattr(request, "thought_signature", None),
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
        if data.get("thought_signature"):
            req.thought_signature = data["thought_signature"]
        return req
