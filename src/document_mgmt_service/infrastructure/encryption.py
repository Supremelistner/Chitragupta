"""Payload encryption for Qdrant vector store.

Encrypts sensitive payload fields (text, description, metadata) before
storing in Qdrant. Filtering fields (document_id, version, privacy, etc.)
remain in plaintext so Qdrant can still perform filtered search.

Key derivation:
    key = HMAC-SHA256(description + upload_date + master_key)
    → 32 bytes → Fernet-compatible key

This means:
    - Same document + same date + same master key = same encryption key
    - Different documents get different keys (description is part of the hash)
    - Changing the master key invalidates all existing encrypted data
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

logger = logging.getLogger("document_mgmt_service.encryption")

# Fernet requires a 32-byte key base64-encoded
_FERNET_PREFIX = b"AAAA"

# Fields that are encrypted (sensitive content). `original_filename` is
# included: reads always decrypt before display (see qdrant.search), so
# filenames like "aadhaar card.jpeg" no longer sit in plaintext.
_ENCRYPTED_FIELDS = frozenset({"text", "description", "metadata", "original_filename"})

# Fields that remain in plaintext (needed for filtering/provenance).
# This set must match the keys _build_payload actually emits; anything
# NOT listed here or in _ENCRYPTED_FIELDS is encrypted by default (with
# a warning) so a future field can never slip through unencrypted.
_PLAINTEXT_FIELDS = frozenset({
    "chunk_id", "document_id", "version", "chunk_index",
    "page_number", "start_char", "end_char",
    "privacy", "content_type", "created_at",
    "field_pointers", "owner_type", "relation", "user_id",
})


def derive_user_master_key(master_key: str, user_id: str | None) -> str:
    """V2 per-user master: HMAC(master, "user|id") hex.

    Legacy rows (no user / "__local__") return the global master unchanged
    so existing data keeps decrypting. Every other user gets an isolated
    key domain: the global hub operator holding one user's key learns
    nothing about another user's chunks.
    """
    uid = (user_id or "").strip()
    if not master_key or not uid or uid == "__local__":
        return master_key
    return hmac.new(
        key=master_key.encode("utf-8"),
        msg=f"chitragupta-user-v1|{uid}".encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()


def derive_document_key(
    *,
    document_id: str,
    version: int,
    upload_date: str,
    master_key: str,
) -> bytes:
    """Derive a Fernet-compatible data key without PII inputs (scheme v2).

    Key = HMAC-SHA256(key=master_key, msg="chitragupta-qdrant-v2|{doc}|{ver}|{date}").
    ``document_id`` is a random UUID, so — unlike the legacy
    description-derived scheme — the derivation inputs carry no PII and
    need no plaintext hint stored alongside the ciphertext.
    """
    import base64
    message = f"chitragupta-qdrant-v2|{document_id}|{version}|{upload_date}".encode("utf-8")
    raw_key = hmac.new(
        key=master_key.encode("utf-8"),
        msg=message,
        digestmod=hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(raw_key)


def derive_encryption_key(
    description: str,
    upload_date: str,
    master_key: str,
) -> bytes:
    """Legacy (v1) derivation — description-based. Kept ONLY to decrypt
    points written before the v2 scheme. Do not use for new writes."""
    """Derive a Fernet-compatible encryption key from document metadata.

    Args:
        description: The document description (e.g., "Aadhaar identity card")
        upload_date: ISO-format upload date (e.g., "2026-08-28T16:33:50+00:00")
        master_key: The master encryption key from .env

    Returns:
        32-byte key suitable for Fernet encryption
    """
    # Combine inputs into a single message
    message = f"{description}|{upload_date}|{master_key}".encode("utf-8")
    # HMAC-SHA256 produces 32 bytes
    raw_key = hmac.new(
        key=b"chitragupta-qdrant-v1",
        msg=message,
        digestmod=hashlib.sha256,
    ).digest()
    # Fernet expects base64-encoded 32-byte key
    import base64
    return base64.urlsafe_b64encode(raw_key)


def encrypt_payload(
    payload: dict[str, Any],
    document_id: str,
    version: int,
    upload_date: str,
    master_key: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Encrypt sensitive fields in a Qdrant payload (scheme v2).

    Returns a new dict with:
    - Sensitive fields encrypted as base64 strings (prefixed with "enc:")
    - Plaintext fields passed through unchanged
    - A non-sensitive `_enc_date` hint + `_enc_v` scheme marker. No
      description (or any PII) is stored — the key re-derives from
      document_id/version/date already present in the payload.
    """
    if not master_key:
        return payload  # No encryption configured

    from cryptography.fernet import Fernet

    key = derive_document_key(
        document_id=document_id, version=version,
        upload_date=upload_date,
        master_key=derive_user_master_key(master_key, user_id or payload.get("user_id")),
    )
    fernet = Fernet(key)

    encrypted = {}
    for field, value in payload.items():
        if field in _ENCRYPTED_FIELDS or (
            field not in _PLAINTEXT_FIELDS and not field.startswith("_enc_")
        ):
            if field not in _ENCRYPTED_FIELDS:
                logger.warning(
                    "Unknown payload field '%s' — encrypting by default",
                    field,
                )
            if value is None:
                encrypted[field] = None
            else:
                plaintext = json.dumps(value, default=str).encode("utf-8")
                ciphertext = fernet.encrypt(plaintext)
                encrypted[field] = f"enc:{ciphertext.decode('ascii')}"
        else:
            encrypted[field] = value

    # Non-sensitive hints for key re-derivation during decryption.
    encrypted["_enc_date"] = upload_date
    encrypted["_enc_v"] = 2

    logger.debug(
        "Encrypted %d fields for chunk %s",
        sum(1 for f in payload if f in _ENCRYPTED_FIELDS),
        payload.get("chunk_id", "?"),
    )
    return encrypted


def decrypt_payload(
    payload: dict[str, Any],
    description: str,
    upload_date: str,
    master_key: str,
) -> dict[str, Any]:
    """Decrypt sensitive fields in a Qdrant payload.

    Fields prefixed with "enc:" are decrypted. Other fields pass through.
    If decryption fails (wrong key, corrupted data), returns the field as-is
    with a warning logged.
    """
    if not master_key:
        return payload  # No encryption configured

    from cryptography.fernet import Fernet

    key = derive_encryption_key(description, upload_date, master_key)
    result = _decrypt_with_key(payload, Fernet(key))
    # Remove internal hints
    result.pop("_enc_desc", None)
    result.pop("_enc_date", None)
    result.pop("_enc_v", None)
    return result


def _decrypt_with_key(payload: dict[str, Any], fernet) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Decrypt every ``enc:``-prefixed field with an already-built Fernet."""
    from cryptography.fernet import InvalidToken

    decrypted = {}
    for field, value in payload.items():
        if isinstance(value, str) and value.startswith("enc:"):
            try:
                ciphertext = value[4:].encode("ascii")
                plaintext_bytes = fernet.decrypt(ciphertext)
                decrypted[field] = json.loads(plaintext_bytes.decode("utf-8"))
            except InvalidToken:
                logger.warning(
                    "Decryption failed for field '%s' on chunk %s — wrong key or corrupted data",
                    field, payload.get("chunk_id", "?"),
                )
                decrypted[field] = value  # Return encrypted string as-is
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(
                    "Decoded but unparseable field '%s' on chunk %s: %s",
                    field, payload.get("chunk_id", "?"), e,
                )
                decrypted[field] = value
        else:
            decrypted[field] = value
    return decrypted


def decrypt_payload_with_date(
    payload: dict[str, Any],
    master_key: str,
) -> dict[str, Any]:
    """Decrypt a payload using only data carried in the payload itself.

    Scheme v2 (``_enc_v == 2``): re-derives the key from the payload's own
    document_id/version plus ``_enc_date`` (or ``created_at``). No PII hint
    needed. Legacy v1 points (with a plaintext ``_enc_desc`` hint) fall
    back to description-derived decryption so old data keeps working.
    Falls back to returning the payload as-is when nothing matches.
    """
    if not master_key:
        return payload

    from cryptography.fernet import Fernet

    upload_date = payload.get("_enc_date", "") or payload.get("created_at", "") or ""
    effective_master = derive_user_master_key(master_key, payload.get("user_id"))
    if payload.get("_enc_v") == 2:
        key = derive_document_key(
            document_id=str(payload.get("document_id", "")),
            version=int(payload.get("version", 0) or 0),
            upload_date=upload_date,
            master_key=effective_master,
        )
        result = _decrypt_with_key(payload, Fernet(key))
    elif payload.get("_enc_desc"):
        # Legacy v1 point: description hint still present.
        description = payload.get("_enc_desc", "") or ""
        key = derive_encryption_key(description, upload_date, effective_master)
        result = _decrypt_with_key(payload, Fernet(key))
    else:
        return payload
    # Remove internal hints from the result
    result.pop("_enc_desc", None)
    result.pop("_enc_date", None)
    result.pop("_enc_v", None)
    return result
