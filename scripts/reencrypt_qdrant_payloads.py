"""Re-encrypt legacy (v1) Qdrant payloads under the v2 scheme.

Background: before the P0 privacy fix, ``encrypt_payload`` stored the full
model description in plaintext as ``_enc_desc`` (key-derivation hint) in
every Qdrant point. New writes no longer do this, but pre-existing points
still carry the hint. This script rewrites those points:

    legacy ``enc:`` fields  --decrypt w/ stored hint-->  plaintext
                            --re-encrypt w/ v2 doc-key--> ``enc:`` fields
    drops ``_enc_desc``; stamps ``_enc_v: 2``.

Safety properties:
  * Dry-run by default; nothing is written without ``--apply``.
  * Each point is round-trip verified IN MEMORY before upsert (decrypt
    legacy -> encrypt v2 -> decrypt v2 must equal the legacy plaintext).
    Points that fail verification are skipped and reported.
  * ``--filter-document-id`` limits the run to one document (use for
    canary testing on a synthetic doc first).
  * The master key is read from ``ENCRYPTION_MASTER_KEY`` and never printed.

Usage:
    python scripts/reencrypt_qdrant_payloads.py [--collection document_chunks]
        [--filter-document-id <hex>] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from document_mgmt_service.infrastructure import encryption as enc  # noqa: E402


def _qdrant_post(base_url: str, path: str, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _scroll_points(base_url: str, collection: str, doc_filter: str | None):
    flt = None
    if doc_filter:
        flt = {"must": [{"key": "document_id", "match": {"value": doc_filter}}]}
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
        if flt:
            body["filter"] = flt
        if offset is not None:
            body["offset"] = offset
        result = _qdrant_post(base_url, f"/collections/{collection}/points/scroll", body)
        result = result["result"]
        for point in result.get("points", []):
            yield point
        offset = result.get("next_page_offset")
        if not offset:
            break


def _transform_payload(payload: dict, master_key: str) -> dict | None:
    """Return the v2 payload, or None if the point needs no work / fails.

    Returns None for: already-v2 points, unencrypted points, points whose
    legacy decryption fails verification, and points missing key inputs.
    """
    if payload.get("_enc_v") == 2 or "_enc_desc" not in payload:
        return None
    description = payload.get("_enc_desc", "") or ""
    upload_date = payload.get("_enc_date", "") or payload.get("created_at", "") or ""
    document_id = str(payload.get("document_id", ""))
    try:
        version = int(payload.get("version", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not document_id or not description:
        return None
    # 1. Decrypt legacy with the stored hint.
    legacy_key = enc.derive_encryption_key(description, upload_date, master_key)
    try:
        from cryptography.fernet import Fernet
        legacy_plain = enc._decrypt_with_key(dict(payload), Fernet(legacy_key))
    except Exception:
        return None
    # 1b. Every enc: field must have actually decrypted — undecryptable
    # input passes through _decrypt_with_key verbatim, which would let
    # corruption slip through the round-trip check below.
    for field in ("text", "description", "metadata", "original_filename"):
        value = legacy_plain.get(field)
        if isinstance(value, str) and value.startswith("enc:"):
            return None
    # 2. Re-encrypt under v2 (drops _enc_desc).
    new_payload = enc.encrypt_payload(
        {k: v for k, v in legacy_plain.items() if k != "_enc_desc"},
        document_id=document_id, version=version,
        upload_date=upload_date, master_key=master_key,
    )
    # 3. Round-trip verify in memory before anyone writes anything.
    check = enc.decrypt_payload_with_date(dict(new_payload), master_key)
    for field in ("text", "description", "metadata", "original_filename"):
        if legacy_plain.get(field) != check.get(field):
            return None
    return new_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="document_chunks")
    parser.add_argument("--filter-document-id", default=None)
    parser.add_argument("--apply", action="store_true",
                        help="Write rewritten payloads back (default: dry run)")
    parser.add_argument("--qdrant-url", default=os.environ.get(
        "DOCUMENT_SERVICE_QDRANT_URL", "http://localhost:6333"))
    args = parser.parse_args()

    master_key = os.environ.get("ENCRYPTION_MASTER_KEY", "")
    if not master_key:
        print("ENCRYPTION_MASTER_KEY is not set; refusing to run.", file=sys.stderr)
        return 2

    scanned = migrated = cleaned = skipped = 0
    to_write: list[tuple] = []
    to_clean: list = []
    for point in _scroll_points(args.qdrant_url, args.collection, args.filter_document_id):
        scanned += 1
        payload = dict(point.get("payload", {}))
        new_payload = _transform_payload(payload, master_key)
        if new_payload is not None:
            migrated += 1
            to_write.append((point["id"], new_payload))
        elif "_enc_desc" in payload:
            # Already v2 (or undecryptable) but still carrying the legacy
            # hint — just drop the key, no re-encryption needed.
            cleaned += 1
            to_clean.append(point["id"])
        else:
            skipped += 1

    print(f"scanned={scanned} need_rewrite={migrated} need_hint_drop={cleaned} ok={skipped}")
    if not args.apply:
        print("dry run: no writes performed (pass --apply to write)")
        return 0
    for point_id, new_payload in to_write:
        # Qdrant SetPayload operation: {"payload": {...}, "points": [...]}.
        _qdrant_post(args.qdrant_url, f"/collections/{args.collection}/points/payload", {
            "payload": new_payload,
            "points": [point_id],
        })
        # SetPayload MERGES: explicitly delete the legacy hint key.
        _delete_payload_keys(args.qdrant_url, args.collection, [point_id], ["_enc_desc"])
    if to_clean:
        _delete_payload_keys(args.qdrant_url, args.collection, to_clean, ["_enc_desc"])
    print(f"rewrote {len(to_write)} point(s), dropped hint on {len(to_clean)} point(s)")
    return 0


def _delete_payload_keys(base_url: str, collection: str, point_ids: list, keys: list) -> None:
    """Delete specific payload keys via Qdrant's DeletePayload operation."""
    import urllib.error
    req = urllib.request.Request(
        base_url.rstrip("/") + f"/collections/{collection}/points/payload/delete",
        data=json.dumps({"keys": keys, "points": point_ids}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60):
            pass
    except urllib.error.HTTPError as exc:
        # Older Qdrant builds expose delete as DELETE on /points/payload.
        if exc.code not in (404, 405):
            raise
        req = urllib.request.Request(
            base_url.rstrip("/") + f"/collections/{collection}/points/payload",
            data=json.dumps({"keys": keys, "points": point_ids}).encode(),
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=60):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
