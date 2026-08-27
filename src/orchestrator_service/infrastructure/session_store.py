"""JSON file-based session storage.

V1 implementation: sessions stored as individual JSON files under data/sessions/.
Can be swapped for PostgreSQL/Redis later without changing domain logic.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator_service.domain.models import (
    ConversationMessage,
    MessageRole,
    Session,
    SessionStatus,
)

logger = logging.getLogger("orchestrator.session_store")


class FileSessionStore:
    """File-based session persistence.

    Structure:
        data/sessions/{session_id}.json
    """

    def __init__(self, base_dir: str = "./data/sessions") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        session.updated_at = datetime.now(timezone.utc)
        data = self._serialize(session)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.debug("Saved session %s", session.session_id)

    def get(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._deserialize(data)
        except Exception:
            logger.exception("Failed to load session %s", session_id)
            return None

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted session %s", session_id)
            return True
        return False

    def list_sessions(
        self, *, user_id: str | None = None, status: SessionStatus | None = None
    ) -> list[Session]:
        sessions: list[Session] = []
        for f in sorted(self._base.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                s = self._deserialize(data)
                if user_id and s.user_id != user_id:
                    continue
                if status and s.status != status:
                    continue
                sessions.append(s)
            except Exception:
                logger.warning("Skipping corrupt session file: %s", f.name)
        return sessions

    def archive(self, session_id: str) -> bool:
        session = self.get(session_id)
        if session is None:
            return False
        session.status = SessionStatus.ARCHIVED
        self.save(session)
        return True

    def _path(self, session_id: str) -> Path:
        return self._base / f"{session_id}.json"

    # -- Serialization --

    def _serialize(self, session: Session) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "title": session.title,
            "metadata": session.metadata,
            "messages": [self._serialize_message(m) for m in session.messages],
        }

    def _serialize_message(self, msg: ConversationMessage) -> dict[str, Any]:
        return {
            "role": msg.role.value,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "tool_call_id": msg.tool_call_id,
            "metadata": msg.metadata,
            # tool_calls serialized as list of dicts
            "tool_calls": [
                {
                    "call_id": tc.call_id,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "status": tc.status.value,
                    "result": tc.result,
                    "error": tc.error,
                }
                for tc in msg.tool_calls
            ],
        }

    def _deserialize(self, data: dict[str, Any]) -> Session:
        from orchestrator_service.domain.models import ToolCall, ToolCallStatus
        session = Session(
            session_id=data["session_id"],
            user_id=data.get("user_id", ""),
            status=SessionStatus(data.get("status", "active")),
            title=data.get("title", ""),
            metadata=data.get("metadata", {}),
        )
        if "created_at" in data:
            session.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            session.updated_at = datetime.fromisoformat(data["updated_at"])

        for m_data in data.get("messages", []):
            tool_calls = []
            for tc_data in m_data.get("tool_calls", []):
                tool_calls.append(
                    ToolCall(
                        call_id=tc_data.get("call_id", ""),
                        tool_name=tc_data.get("tool_name", ""),
                        arguments=tc_data.get("arguments", {}),
                        status=ToolCallStatus(tc_data.get("status", "pending")),
                        result=tc_data.get("result"),
                        error=tc_data.get("error"),
                    )
                )
            msg = ConversationMessage(
                role=MessageRole(m_data["role"]),
                content=m_data.get("content", ""),
                tool_call_id=m_data.get("tool_call_id"),
                metadata=m_data.get("metadata", {}),
                tool_calls=tool_calls,
            )
            if "timestamp" in m_data:
                msg.timestamp = datetime.fromisoformat(m_data["timestamp"])
            session.messages.append(msg)

        return session
