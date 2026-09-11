"""Shared auth primitives for multi-user Chitragupta (V1).

Local-first, no new dependencies by design (student project, local Docker):
- passwords: PBKDF2-HMAC-SHA256 via stdlib ``hashlib`` (290k iterations).
- tokens: JWT HS256 via stdlib ``hmac``/``base64``/``json`` — no PyJWT needed.

Contract:
- ``hash_password`` / ``verify_password`` — phc-style ``$pbkdf2$<iter>$<salt_b64>$<hash_b64>``.
- ``issue_token(user_id, email, secret, ttl_seconds=7d)`` — ``sub`` + ``email`` + ``iat``/``exp``.
- ``verify_token`` — returns payload dict or raises ``AuthError`` (expired/invalid).
- ``needs_refresh`` — True when <48h of life remains (UI calls /api/auth/refresh).

Secret comes from ``CHITRAGUPTA_JWT_SECRET``; a dev fallback is used only
when unset (logged as a warning by callers, never printed).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

TOKEN_TTL_SECONDS = 7 * 24 * 3600
REFRESH_WINDOW_SECONDS = 48 * 3600
_PBKDF2_ITERATIONS = 290_000


class AuthError(ValueError):
    pass


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return (
        f"$pbkdf2${_PBKDF2_ITERATIONS}"
        f"${base64.b64encode(salt).decode()}"
        f"${base64.b64encode(dk).decode()}"
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        _, _, iter_s, salt_b64, hash_b64 = stored.split("$", 4)
        iters = int(iter_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def issue_token(*, user_id: str, email: str, secret: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(
        {"sub": user_id, "email": email, "iat": now, "exp": now + ttl_seconds},
        separators=(",", ":"),
    ).encode())
    sig = _b64url(hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def verify_token(token: str, *, secret: str) -> dict:
    try:
        header_b64, body_b64, sig_b64 = token.split(".")
        expected = _b64url(hmac.new(secret.encode(), f"{header_b64}.{body_b64}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, sig_b64):
            raise AuthError("invalid signature")
        payload = json.loads(_b64url_decode(body_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise AuthError("token expired")
        if not payload.get("sub"):
            raise AuthError("missing sub")
        return payload
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(f"invalid token: {exc}") from exc


def needs_refresh(payload: dict) -> bool:
    try:
        return int(payload.get("exp", 0)) - int(time.time()) < REFRESH_WINDOW_SECONDS
    except Exception:
        return True


def bearer_user(token_header: str | None, *, secret: str) -> dict:
    """Extract + verify ``Authorization: Bearer <jwt>``. Raises AuthError."""
    if not token_header or not token_header.startswith("Bearer "):
        raise AuthError("missing bearer token")
    return verify_token(token_header[len("Bearer "):].strip(), secret=secret)
