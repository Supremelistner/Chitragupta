"""Device sync for V1 multi-user: local-primary, global-hub (docs only).

Model (per user decisions):
- Local Postgres+Qdrant+files are primary/active. Global is an opaque hub
  enabling multi-device use + on-demand backup — NOT a cache.
- Scope: file docs only. Chat sessions/confirmations never sync.
- Trigger: push on add/remove/modify; pull = wipe local user data +
  full fetch on login/user-switch (higher ``version`` wins on clash).
- Blobs: local gzip-compresses on push, gunzip-extracts on pull. Global
  stores bytes opaque — no server-side transform.

Docker phase (now): global hub = filesystem namespace
``data/global-blobs/<user_id>/{blobs/*.gz,manifest.json}`` on a Docker
volume. Production phase swaps ``GlobalHub`` for Postgres+Qdrant
endpoints behind the same ``push/pull/wipe`` interface — callers
(orchestrator upload/delete/login) do not change.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("shared.device_sync")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOBAL_ROOT = Path(os.environ.get(
    "CHITRAGUPTA_GLOBAL_ROOT",
    str(_PROJECT_ROOT / "data" / "global-blobs"),
))


def _user_dir(root: Path, user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in ("-", "_")) or "unknown"
    d = root / safe
    (d / "blobs").mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(root: Path, user_id: str) -> Path:
    return _user_dir(root, user_id) / "manifest.json"


def load_manifest(user_id: str, *, root: Path = DEFAULT_GLOBAL_ROOT) -> dict:
    try:
        return json.loads(_manifest_path(root, user_id).read_text(encoding="utf-8"))
    except Exception:
        return {"user_id": user_id, "documents": {}}


def push_document(
    *,
    user_id: str,
    document_id: str,
    version: int,
    blob: bytes,
    metadata: dict | None = None,
    root: Path = DEFAULT_GLOBAL_ROOT,
) -> dict:
    """Gzip blob to global hub + upsert manifest entry (version-wins)."""
    if not user_id:
        raise ValueError("user_id is required for sync push")
    udir = _user_dir(root, user_id)
    digest = hashlib.sha256(blob).hexdigest()[:16]
    blob_name = f"{document_id}_v{version}_{digest}.bin.gz"
    (udir / "blobs" / blob_name).write_bytes(gzip.compress(blob))
    manifest = load_manifest(user_id, root=root)
    docs = manifest.setdefault("documents", {})
    prev = docs.get(document_id, {})
    if int(prev.get("version", 0)) > int(version):
        return {"pushed": False, "reason": "stale version"}
    docs[document_id] = {
        "version": int(version),
        "blob": blob_name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "metadata": metadata or {},
        "pushed_at": datetime.now(timezone.utc).isoformat(),
    }
    _manifest_path(root, user_id).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Sync push %s v%s for user %s", document_id, version, user_id)
    # V2-3: best-effort mirror to shared infra (never fails the push).
    try:
        mirror_manifest_pg(user_id=user_id, document_id=document_id, entry=docs[document_id])
    except Exception:
        pass
    return {"pushed": True, "blob": blob_name}


def remove_document(*, user_id: str, document_id: str, root: Path = DEFAULT_GLOBAL_ROOT) -> dict:
    """Push a deletion: drop manifest entry + delete its blobs."""
    udir = _user_dir(root, user_id)
    manifest = load_manifest(user_id, root=root)
    docs = manifest.get("documents", {})
    entry = docs.pop(document_id, None)
    if entry and entry.get("blob"):
        try:
            (udir / "blobs" / entry["blob"]).unlink(missing_ok=True)
        except Exception:
            pass
    _manifest_path(root, user_id).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"removed": entry is not None}


def pull_manifest(user_id: str, *, root: Path = DEFAULT_GLOBAL_ROOT) -> dict:
    return load_manifest(user_id, root=root)


def read_blob(*, user_id: str, blob_name: str, root: Path = DEFAULT_GLOBAL_ROOT) -> bytes:
    data = (_user_dir(root, user_id) / "blobs" / blob_name).read_bytes()
    try:
        return gzip.decompress(data)
    except OSError:
        return data  # backward compat: plain bytes stored pre-gzip


def wipe_global_user(user_id: str, *, root: Path = DEFAULT_GLOBAL_ROOT) -> bool:
    d = root / user_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


# ---------------------------------------------------------------------------
# V2-3: real global Postgres + Qdrant mirror (Docker phase, best-effort)
# ---------------------------------------------------------------------------
# Filesystem hub above is always written (Docker volume). When the env below
# is set, pushes additionally mirror to shared infra so a second device (or
# the production phase) can sync without the volume:
#   GLOBAL_POSTGRES_DSN  — e.g. postgresql://chitragupta:pw@localhost:5432/chitragupta_global
#   GLOBAL_QDRANT_URL    — e.g. http://localhost:6333
#   GLOBAL_QDRANT_COLLECTION — default "global_document_chunks"
# All mirrors are best-effort: failures log and the filesystem record wins.

GLOBAL_POSTGRES_DSN = os.environ.get("GLOBAL_POSTGRES_DSN", "")
GLOBAL_QDRANT_URL = os.environ.get("GLOBAL_QDRANT_URL", "").rstrip("/")
GLOBAL_QDRANT_COLLECTION = os.environ.get("GLOBAL_QDRANT_COLLECTION", "global_document_chunks")


def mirror_manifest_pg(*, user_id: str, document_id: str, entry: dict) -> bool:
    """Upsert one manifest entry into the global Postgres mirror."""
    if not GLOBAL_POSTGRES_DSN:
        return False
    try:
        import psycopg
        with psycopg.connect(GLOBAL_POSTGRES_DSN, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS global_sync_manifest (
                        user_id TEXT NOT NULL, document_id TEXT NOT NULL,
                        version INTEGER NOT NULL, blob_name TEXT,
                        sha256 TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        pushed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (user_id, document_id))"""
                )
                cur.execute(
                    """INSERT INTO global_sync_manifest
                       (user_id, document_id, version, blob_name, sha256, metadata, pushed_at)
                       VALUES (%s,%s,%s,%s,%s,%s::jsonb,NOW())
                       ON CONFLICT (user_id, document_id) DO UPDATE SET
                         version=EXCLUDED.version, blob_name=EXCLUDED.blob_name,
                         sha256=EXCLUDED.sha256, metadata=EXCLUDED.metadata,
                         pushed_at=NOW()
                       WHERE EXCLUDED.version >= global_sync_manifest.version""",
                    (user_id, document_id, int(entry.get("version", 0)),
                     entry.get("blob"), entry.get("sha256"),
                     json.dumps(entry.get("metadata", {}))),
                )
            conn.commit()
        return True
    except Exception:
        logger.warning("Global PG mirror failed for %s (filesystem kept)", document_id)
        return False


def mirror_qdrant_user(*, user_id: str, local_base_url: str = "http://localhost:6333",
                       local_collection: str = "document_chunks", limit: int = 500) -> dict:
    """Copy one user's points local → global collection (payloads already per-user encrypted)."""
    if not GLOBAL_QDRANT_URL:
        return {"mirrored": False, "reason": "no GLOBAL_QDRANT_URL"}
    try:
        import urllib.request
        scroll_url = f"{local_base_url.rstrip('/')}/collections/{local_collection}/points/scroll"
        body = json.dumps({
            "filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]},
            "limit": limit, "with_payload": True, "with_vector": True,
        }).encode()
        req = urllib.request.Request(scroll_url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            points = json.loads(resp.read().decode()).get("result", {}).get("points", [])
        if not points:
            return {"mirrored": True, "count": 0}
        put_url = (f"{GLOBAL_QDRANT_URL}/collections/{GLOBAL_QDRANT_COLLECTION}"
                   f"/points?wait=true")
        upsert = {"points": [
            {"id": p["id"], "vector": p.get("vector"), "payload": p.get("payload", {})}
            for p in points if p.get("vector") is not None
        ]}
        req2 = urllib.request.Request(put_url, data=json.dumps(upsert).encode(), method="PUT",
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req2, timeout=30):
            pass
        return {"mirrored": True, "count": len(upsert["points"])}
    except Exception as exc:
        logger.warning("Global Qdrant mirror failed: %s", exc)
        return {"mirrored": False, "reason": str(exc)[:120]}
