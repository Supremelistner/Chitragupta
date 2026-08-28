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

# Fields that are encrypted (sensitive content)
_ENCRYPTED_FIELDS = frozenset({"text", "description", "metadata"})

# Fields that remain in plaintext (needed for filtering/provenance)
_PLAINTEXT_FIELDS = frozenset({
    "chunk_id", "document_id", "version", "chunk_index",
    "page_number", "start_char", "end_char",
    "privacy", "content_type", "created_at",
})


def derive_encryption_key(
    description: str,
    upload_date: str,
    master_key: str,
) -> bytes:
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
    description: str,
    upload_date: str,
    master_key: str,
) -> dict[str, Any]:
    """Encrypt sensitive fields in a Qdrant payload.

    Returns a new dict with:
    - Sensitive fields encrypted as base64 strings (prefixed with "enc:")
    - Plaintext fields passed through unchanged
    - A plaintext `_enc_desc` hint stored for key derivation on decrypt
    """
    if not master_key:
        return payload  # No encryption configured

    from cryptography.fernet import Fernet

    key = derive_encryption_key(description, upload_date, master_key)
    fernet = Fernet(key)

    encrypted = {}
    for field, value in payload.items():
        if field in _ENCRYPTED_FIELDS:
            if value is None:
                encrypted[field] = None
            else:
                plaintext = json.dumps(value, default=str).encode("utf-8")
                ciphertext = fernet.encrypt(plaintext)
                encrypted[field] = f"enc:{ciphertext.decode('ascii')}"
        else:
            encrypted[field] = value

    # Store plaintext hints for key derivation during decryption
    encrypted["_enc_desc"] = description
    encrypted["_enc_date"] = upload_date

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

    from cryptography.fernet import Fernet, InvalidToken

    key = derive_encryption_key(description, upload_date, master_key)
    fernet = Fernet(key)

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

    # Remove internal hints
    decrypted.pop("_enc_desc", None)
    decrypted.pop("_enc_date", None)
    return decrypted


def decrypt_payload_with_date(
    payload: dict[str, Any],
    master_key: str,
) -> dict[str, Any]:
    """Decrypt using the created_at field and _enc_desc hint from the payload.

    Used during search results where we don't have the original upload date.
    The _enc_desc hint stores the original description used for key derivation.
    Falls back to empty strings if fields are missing.
    """
    if not master_key:
        return payload

    upload_date = payload.get("_enc_date", "") or payload.get("created_at", "") or ""
    description = payload.get("_enc_desc", "") or ""

    result = decrypt_payload(payload, description, upload_date, master_key)
    # Remove the internal hint from the result
    result.pop("_enc_desc", None)
    return result
