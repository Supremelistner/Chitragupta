"""File-based user store for V1 multi-user (orchestrator side).

Layout (mirrors FileSessionStore):
    <base_dir>/users/<user_id>.json      — {user_id, email, pwd_hash, created_at}
    <base_dir>/users/.index.json         — {email_lower: user_id}

Why files and not Postgres (V1): the orchestrator currently has no PG
wiring (sessions/confirmations are files); adding a PG dependency here
would pull connection config through three services. The document
service remains the PG owner (migration 006 adds the shared ``users``
table for the production global-Postgres phase). V2 will move this
store onto that table with no endpoint changes.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from shared.auth import hash_password, verify_password

logger = logging.getLogger("orchestrator.auth_store")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthStoreError(ValueError):
    pass


class FileAuthStore:
    def __init__(self, base_dir: str = "./data/users") -> None:
        self._base = Path(base_dir)
        self._dir = self._base / "users" if self._base.name != "users" else self._base
        self._dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self._dir / ".index.json"

    def _load_index(self) -> dict:
        try:
            return json.loads(self._index_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_index(self, index: dict) -> None:
        self._index_path().write_text(json.dumps(index, indent=2), encoding="utf-8")

    def register(self, email: str, password: str) -> dict:
        email_norm = (email or "").strip().lower()
        if not _EMAIL_RE.match(email_norm):
            raise AuthStoreError("invalid email")
        if len(password or "") < 8:
            raise AuthStoreError("password must be at least 8 characters")
        index = self._load_index()
        if email_norm in index:
            raise AuthStoreError("email already registered")
        user_id = uuid.uuid4().hex[:16]
        record = {
            "user_id": user_id,
            "email": email_norm,
            "pwd_hash": hash_password(password),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self._dir / f"{user_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        index[email_norm] = user_id
        self._save_index(index)
        logger.info("Registered user %s", email_norm)
        return {"user_id": user_id, "email": email_norm}

    def authenticate(self, email: str, password: str) -> dict:
        email_norm = (email or "").strip().lower()
        user_id = self._load_index().get(email_norm)
        if not user_id:
            raise AuthStoreError("invalid credentials")
        try:
            record = json.loads((self._dir / f"{user_id}.json").read_text(encoding="utf-8"))
        except Exception:
            raise AuthStoreError("invalid credentials")
        if not verify_password(password or "", record.get("pwd_hash", "")):
            raise AuthStoreError("invalid credentials")
        return {"user_id": user_id, "email": email_norm}

    def get(self, user_id: str) -> dict | None:
        try:
            return json.loads((self._dir / f"{user_id}.json").read_text(encoding="utf-8"))
        except Exception:
            return None
