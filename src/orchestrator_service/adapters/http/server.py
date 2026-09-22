"""RESTful HTTP server for the orchestrator service using FastAPI + Uvicorn.

Endpoints:
- POST   /api/chat              — send a message, get a response
- POST   /api/confirm           — respond to a confirmation request
- POST   /api/sessions          — create a new session
- GET    /api/sessions          — list sessions
- GET    /api/sessions/{id}     — get a session
- DELETE /api/sessions/{id}     — delete a session
- POST   /api/sessions/{id}/archive — archive a session
- POST   /api/upload            — upload a file (image/PDF) for ingestion
- GET    /api/healthz           — health check
- GET    /                      — UI
- GET    /static/{path}         — static assets (CSS, JS)
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from orchestrator_service.application.orchestrator import OrchestrationEngine
from orchestrator_service.config import OrchestratorConfig

logger = logging.getLogger("orchestrator.http")


def create_app(
    *,
    config: OrchestratorConfig,
    engine: OrchestrationEngine,
) -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(
        title="Chitragupta Orchestrator",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # Resolve paths
    ui_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "ui"
    doc_service_url = config.document_service_url.rstrip("/")

    # ─── CORS middleware (for dev) ─────────────────────────────────
    try:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except ImportError:
        pass

    # ─── No-store for UI assets ──────────────────────────────────
    # Stale cached app.js/i18n JSON once left users running fixed bugs
    # (and fixed bugs looking broken — including a literal "[error_mic]"
    # alert from a cached string table). Single-user LAN app: always
    # refetch the shell, scripts, and string tables (kilobytes,
    # negligible cost). Other API JSON responses are unaffected.
    try:
        from starlette.middleware.base import BaseHTTPMiddleware

        class _NoCacheStaticMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):  # noqa: ANN001, ANN202
                response = await call_next(request)
                path = request.url.path
                if (
                    path == "/"
                    or path.startswith("/static/")
                    or path.startswith("/api/i18n/")
                ):
                    response.headers["Cache-Control"] = "no-store"
                return response

        app.add_middleware(_NoCacheStaticMiddleware)
    except ImportError:
        pass

    # ─── Health ────────────────────────────────────────────────────
    @app.get("/api/healthz")
    async def healthz():
        return {"status": "ok", "service": "orchestrator-service"}

    # ─── Auth (V1 multi-user: email+password + JWT 7d sliding) ────
    def _jwt_secret() -> str:
        import os as _os
        secret = _os.environ.get("CHITRAGUPTA_JWT_SECRET", "")
        if not secret:
            logger.warning("CHITRAGUPTA_JWT_SECRET unset — using dev fallback (set it in .env)")
            return "chitragupta-dev-secret-change-me"
        return secret

    def _auth_store():
        from orchestrator_service.infrastructure.auth_store import FileAuthStore
        import os as _os
        base = _os.environ.get("CHITRAGUPTA_AUTH_DIR", "./data/users")
        return FileAuthStore(base)

    def _current_user(request: Request) -> dict:
        from shared.auth import bearer_user
        try:
            return bearer_user(request.headers.get("authorization"), secret=_jwt_secret())
        except Exception as exc:
            raise HTTPException(status_code=401, detail=str(exc))

    @app.post("/api/auth/register")
    async def auth_register(request: Request):
        from shared.auth import issue_token
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="email and password are required")
        try:
            user = _auth_store().register(body.get("email", ""), body.get("password", ""))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        token = issue_token(user_id=user["user_id"], email=user["email"], secret=_jwt_secret())
        return {"user_id": user["user_id"], "email": user["email"], "token": token, "expires_days": 7}

    @app.post("/api/auth/login")
    async def auth_login(request: Request):
        from shared.auth import issue_token
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=401, detail="invalid credentials")
        try:
            user = _auth_store().authenticate(body.get("email", ""), body.get("password", ""))
        except Exception:
            raise HTTPException(status_code=401, detail="invalid credentials")
        token = issue_token(user_id=user["user_id"], email=user["email"], secret=_jwt_secret())
        return {"user_id": user["user_id"], "email": user["email"], "token": token, "expires_days": 7}

    @app.post("/api/auth/refresh")
    async def auth_refresh(request: Request):
        from shared.auth import issue_token, verify_token
        try:
            body = await request.json()
        except Exception:
            body = {}
        token = (body.get("token") if isinstance(body, dict) else None) or (
            request.headers.get("authorization", "").replace("Bearer ", "")
        )
        try:
            payload = verify_token(token, secret=_jwt_secret())
        except Exception as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        fresh = issue_token(user_id=payload["sub"], email=payload.get("email", ""), secret=_jwt_secret())
        return {"token": fresh, "expires_days": 7, "user_id": payload["sub"]}

    @app.get("/api/me")
    async def auth_me(request: Request):
        payload = _current_user(request)
        return {"user_id": payload["sub"], "email": payload.get("email", "")}

    @app.post("/api/auth/logout")
    async def auth_logout(request: Request):
        """Wipe prev-user local device data on user-switch (docs only, no sessions).

        Body: {"user_id": "...", "wipe": true}. Purges that user's local
        file blobs staged under data/files for the device; the global hub
        copy is untouched so the next login pulls from scratch.
        """
        payload = _current_user(request)
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        if not body.get("wipe"):
            return {"logged_out": True, "user_id": payload["sub"]}
        import shutil
        from pathlib import Path as _Path
        wiped = 0
        files_root = _Path("./data/files/documents")
        if files_root.exists():
            for child in files_root.iterdir():
                # Storage keys embed document_id, not user_id; full purge of
                # the device cache is the V1 wipe (single active user).
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                    wiped += 1
                except Exception:
                    pass
        return {"logged_out": True, "user_id": payload["sub"], "wiped_entries": wiped}

    # ─── Device sync (V1: docs-only push/pull against global hub) ─
    @app.get("/api/sync/manifest")
    async def sync_manifest(request: Request):
        payload = _current_user(request)
        from shared.device_sync import pull_manifest
        return pull_manifest(payload["sub"])

    @app.post("/api/sync/pull")
    async def sync_pull(request: Request):
        """Pull = list global manifest for re-download (client re-ingests).

        V1 restores by re-uploading blobs through the normal ingest path
        (new versions acceptable); version-preserving restore lands with
        the production global-Postgres phase.
        """
        payload = _current_user(request)
        from shared.device_sync import pull_manifest
        manifest = pull_manifest(payload["sub"])
        docs = manifest.get("documents", {})
        return {"user_id": payload["sub"], "count": len(docs), "documents": docs}

    @app.post("/api/sync/restore")
    async def sync_restore(request: Request):
        """V2-4 version-preserving pull: replay global hub blobs into the
        doc service via POST /documents/restore (same IDs + versions)."""
        import base64 as _b64
        import requests as _req
        payload = _current_user(request)
        user_id = payload["sub"]
        from shared.device_sync import pull_manifest, read_blob
        manifest = pull_manifest(user_id)
        docs = manifest.get("documents", {})
        restored, errors = [], []
        headers = {"X-User-Id": user_id}
        try:
            body = await request.json()
        except Exception:
            body = {}
        only = body.get("document_id")
        for doc_id, entry in docs.items():
            if only and doc_id != only:
                continue
            try:
                blob = read_blob(user_id=user_id, blob_name=entry["blob"])
                resp = _req.post(
                    f"{doc_service_url}/documents/restore",
                    json={"document_id": doc_id, "version": int(entry.get("version", 1)),
                          "user_id": user_id,
                          "filename": (entry.get("metadata") or {}).get("filename", doc_id),
                          "content_base64": _b64.b64encode(blob).decode()},
                    headers=headers, timeout=120,
                )
                if resp.status_code >= 400:
                    errors.append({"document_id": doc_id, "error": resp.text[:100]})
                else:
                    restored.append({"document_id": doc_id, **resp.json()})
            except Exception as exc:
                errors.append({"document_id": doc_id, "error": str(exc)[:100]})
        return {"user_id": user_id, "restored": restored, "errors": errors}

    # ─── Sessions ──────────────────────────────────────────────────
    @app.get("/api/sessions")
    async def list_sessions(request: Request):
        try:
            user_id = _current_user(request)["sub"]
        except HTTPException:
            user_id = None
        sessions = engine.list_sessions(user_id=user_id)
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "title": s.title,
                    "status": s.status.value,
                    "created_at": s.created_at.isoformat(),
                    "message_count": len(s.messages),
                }
                for s in sessions
            ]
        }

    @app.post("/api/sessions")
    async def create_session(request: Request):
        body = await request.json()
        user_id = body.get("user_id", "")
        title = body.get("title", "")
        session = engine.create_session(user_id=user_id, title=title)
        return JSONResponse(
            status_code=201,
            content={
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
            },
        )

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        session = engine.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        try:
            user_id = _current_user(request)["sub"]
        except HTTPException:
            user_id = None
        if user_id and session.user_id and session.user_id != user_id:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {
            "session_id": session.session_id,
            "title": session.title,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "messages": [
                {"role": m.role.value, "content": m.content[:500]}
                for m in session.messages
            ],
        }

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        try:
            user_id = _current_user(request)["sub"]
        except HTTPException:
            user_id = None
        if user_id:
            session = engine.get_session(session_id)
            if session is not None and session.user_id and session.user_id != user_id:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        ok = engine.delete_session(session_id)
        return {"deleted": ok}

    # ─── Document management (list + delete for the My Documents panel) ─
    @app.get("/api/documents")
    async def list_documents(request: Request):
        """List the signed-in user's stored documents for the UI panel."""
        try:
            docs_user_id = _current_user(request)["sub"]
        except HTTPException:
            docs_user_id = None
        return engine.list_documents(user_id=docs_user_id)

    @app.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str, request: Request, version: int | None = None):
        """Delete a stored document (or one version) for the signed-in user.

        ``?version=N`` deletes only that version; omit it to delete the
        whole document (all versions). Scoped to the caller's user_id so
        no account can delete another's document. This removes the stored
        document data itself — distinct from deleting a chat session,
        which never touches documents.
        """
        try:
            docs_user_id = _current_user(request)["sub"]
        except HTTPException:
            docs_user_id = None
        result = engine.delete_document(document_id, version=version, user_id=docs_user_id)
        if not result.get("deleted"):
            # Not found (or not owned) reads as 404; other failures as 502.
            err = (result.get("error") or "").lower()
            if "not found" in err:
                raise HTTPException(status_code=404, detail=result.get("error"))
            raise HTTPException(status_code=502, detail=result.get("error") or "delete failed")
        return result

    @app.get("/api/documents/{document_id}/versions/{version}/retrieve")
    async def retrieve_document(document_id: str, version: int, request: Request, approve: bool = False):
        """Retrieve a stored document's original file through the approval gate.

        Reuses the same access-policy gate as chat, driven by code (the id
        comes from the user's own document list, never an LLM). Without
        ``?approve=true`` a sensitive/private document returns HTTP 202
        with ``{"status": "requires_confirmation"}`` so the UI can show the
        two-step confirm. With ``approve=true`` (after the user confirms)
        the original file streams back as a download.
        """
        try:
            docs_user_id = _current_user(request)["sub"]
        except HTTPException:
            docs_user_id = None
        result = engine.retrieve_document(
            document_id, version, user_id=docs_user_id, approval_granted=approve,
        )
        status = result.get("status")
        if status == "requires_confirmation":
            return JSONResponse(status_code=202, content={
                "status": "requires_confirmation",
                "document_id": document_id,
                "version": version,
            })
        if status != "ok":
            msg = result.get("message") or "retrieve failed"
            code = 404 if "not found" in msg.lower() else 502
            raise HTTPException(status_code=code, detail=msg)
        content_b64 = result.get("content_base64") or ""
        if not content_b64:
            raise HTTPException(status_code=502, detail="document file is empty")
        import base64 as _b64
        from fastapi.responses import Response as _Response
        blob = _b64.b64decode(content_b64)
        filename = result.get("filename") or f"{document_id}_v{version}"
        media_type = result.get("content_type") or "application/octet-stream"
        return _Response(
            content=blob,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/sessions/{session_id}/archive")
    async def archive_session(session_id: str, request: Request):
        try:
            user_id = _current_user(request)["sub"]
        except HTTPException:
            user_id = None
        if user_id:
            session = engine.get_session(session_id)
            if session is not None and session.user_id and session.user_id != user_id:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        ok = engine.archive_session(session_id)
        return {"archived": ok}

    # ─── Chat ──────────────────────────────────────────────────────
    @app.post("/api/chat")
    async def chat(request: Request):
        content_type = request.headers.get("content-type", "")
        session_id = ""
        message = ""
        file = None
        if "multipart/form-data" in content_type:
            form = await request.form()
            session_id = form.get("session_id", "")
            message = form.get("message", "")
            file = form.get("file")
            if hasattr(file, "read"):
                file_bytes = await file.read()
                file = {"filename": file.filename, "content_type": file.content_type, "bytes": file_bytes}
        else:
            body = await request.json()
            session_id = body.get("session_id", "")
            message = body.get("message", "")
        if not session_id or (not message and file is None):
            raise HTTPException(status_code=400, detail="session_id and message (or file) required")
        try:
            chat_user_id = _current_user(request)["sub"]
        except HTTPException:
            chat_user_id = None
        if file is not None:
            message = await _ingest_chat_attachment(
                message,
                filename=file.get("filename") or "attachment",
                content=file.get("bytes") or b"",
                content_type=file.get("content_type") or "application/octet-stream",
                user_id=chat_user_id or "",
            )
        response = engine.process_message(session_id, message, user_id=chat_user_id)
        return {
            "session_id": response.session_id,
            "message": response.message,
            "confirmation_required": (
                {
                    "request_id": response.confirmation_required.request_id,
                    "type": response.confirmation_required.confirmation_type.value,
                    "message": response.confirmation_required.message,
                    "tool_args": response.confirmation_required.tool_args,
                }
                if response.confirmation_required
                else None
            ),
            "tool_calls_count": len(response.tool_calls_made),
            "metadata": response.metadata,
        }

    # ─── Language preference (V1 multilingual) ────────────────────
    @app.get("/api/language")
    async def get_language(session_id: str = ""):
        """Return the current language preference for a session.

        Falls back to the V1 default (en→hi) when the session has not
        yet set a preference. ``session_id`` is optional: when omitted
        the response carries only the default so the UI can render
        the toggle state on first paint.
        """
        if not session_id:
            from model_service.infrastructure.language_preferences import (
                DEFAULT_PREFERENCE,
            )
            return {
                "source": DEFAULT_PREFERENCE.source,
                "target": DEFAULT_PREFERENCE.target,
                "default": True,
            }
        pref = engine.get_language_preference(session_id)
        return {
            "session_id": session_id,
            "source": pref.source,
            "target": pref.target,
            "default": False,
        }

    @app.post("/api/language")
    async def set_language(request: Request):
        """Update the language preference for a session.

        Body: ``{"session_id": "...", "source": "hi", "target": "en"}``.
        ``source`` and ``target`` are BCP-47 codes (en, hi, ta, bn).
        Either or both may be omitted to read the current value. The
        response echoes the new preference.
        """
        body = await request.json()
        session_id = body.get("session_id", "")
        if not session_id:
            raise HTTPException(
                status_code=400, detail="session_id is required"
            )
        try:
            pref = engine.set_language_preference(
                session_id,
                source=body.get("source"),
                target=body.get("target"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "session_id": session_id,
            "source": pref.source,
            "target": pref.target,
        }

    # ─── Expiring documents (proactive banner, no chat turn) ─────
    @app.get("/api/expiring")
    async def expiring(request: Request, within_days: int = 30):
        """Documents expiring within N days (overdue included).

        Powers the welcome-screen banner. Session-independent; failures
        collapse to an empty list so the banner just stays hidden.
        """
        try:
            exp_user_id = _current_user(request)["sub"]
        except HTTPException:
            exp_user_id = None
        return engine.list_expiring_documents(within_days=within_days, user_id=exp_user_id)

    # ─── Profile (display name for the persona) ───────────────────
    @app.get("/api/profile")
    async def get_profile():
        """Return the persona's display name for the user.

        Prefers the live engine value, falling back to the persisted
        profile file (covers fresh restarts before any UI sync).
        """
        from orchestrator_service.onboarding import load_profile
        name = engine.user_name
        if not name:
            profile = load_profile(config.profile_path or None)
            name = profile.display_name if profile else None
        return {"display_name": name}

    @app.post("/api/profile")
    async def set_profile(request: Request):
        """Set the display name (from the UI profile modal / settings).

        Persists to the profile file and applies to the live engine so
        the persona addresses the user by name immediately. Only the
        name is stored — never contact details.
        """
        from orchestrator_service.onboarding import save_profile
        body = await request.json()
        name = (body.get("display_name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="display_name is required")
        if len(name) > 100:
            raise HTTPException(status_code=400, detail="display_name too long (max 100)")
        save_profile(name, path=config.profile_path or None)
        engine.set_user_name(name)
        return {"display_name": name}

    # ─── Confirm ───────────────────────────────────────────────────
    @app.post("/api/confirm")
    async def confirm(request: Request):
        body = await request.json()
        session_id = body.get("session_id", "")
        request_id = body.get("request_id", "")
        approved = body.get("approved", False)
        correction = body.get("correction")
        if not session_id or not request_id:
            raise HTTPException(status_code=400, detail="session_id and request_id required")
        try:
            confirm_user_id = _current_user(request)["sub"]
        except HTTPException:
            confirm_user_id = None
        response = engine.handle_confirmation(
            session_id, request_id, approved, correction=correction,
            user_id=confirm_user_id,
        )
        return {
            "session_id": response.session_id,
            "message": response.message,
            "tool_calls_count": len(response.tool_calls_made),
        }

    # ─── TTS proxy ──────────────────────────────────────────────
    @app.get("/api/i18n/{lang}")
    async def i18n(lang: str):
        """Serve an i18n string table.

        The UI loads translations from a static file in the original
        design, but FastAPI's StaticFiles doesn't traverse subdirectories
        by default, so we proxy the JSON through this endpoint. The
        JSON files live under ``ui/i18n/`` and are loaded from disk.
        """
        import pathlib
        safe = ''.join(c for c in lang if c.isalnum() or c == '-')
        if not safe:
            raise HTTPException(status_code=400, detail="invalid lang")
        candidate = pathlib.Path(__file__).resolve().parents[4] / "ui" / "i18n" / f"{safe}.json"
        if not candidate.exists():
            raise HTTPException(status_code=404, detail="not found")
        from fastapi.responses import FileResponse
        return FileResponse(candidate, media_type="application/json")

    @app.post("/api/tts")
    async def tts(request: Request):
        """Text-to-speech proxy.

        Forwards the request to the model service's
        ``/infer/synthesize-audio`` endpoint and returns the audio
        bytes as base64-encoded WAV in a JSON envelope. The UI's
        audio.js decodes that into a Blob for playback.
        """
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        language = body.get("language") or body.get("lang") or "en-US"
        # Cap the text length to avoid sending huge payloads to the
        # model service. 4 KB of text is roughly 5-7 minutes of TTS.
        text = text[:4000]
        upstream_url = f"{config.model_service_url.rstrip('/')}/infer/synthesize-audio"
        upstream_payload = {"text": text, "language": language}
        if body.get("voice"):
            upstream_payload["voice"] = body["voice"]
        # `requests` is a hard runtime dependency (also used by the
        # upload proxy above); no httpx/urllib fallback chain needed.
        import requests as req_lib

        try:
            resp = req_lib.post(upstream_url, json=upstream_payload, timeout=30)
        except req_lib.exceptions.ConnectionError as exc:
            raise HTTPException(
                status_code=502,
                detail="Model service is not running.",
            ) from exc
        except req_lib.exceptions.Timeout as exc:
            raise HTTPException(
                status_code=502,
                detail="Upstream TTS timed out.",
            ) from exc
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"upstream TTS failed: HTTP {resp.status_code}",
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Upstream TTS returned invalid JSON.",
            ) from exc
        return {
            "text": text,
            "language": language,
            "audio_base64": data.get("output", ""),
        }

    def _forward_to_document_service(
        *,
        filename: str,
        content: bytes,
        content_type: str,
        document_id: str = "",
        privacy: str = "",
        description: str = "",
        user_id: str = "",
    ) -> dict:
        """Forward raw file bytes to the document service for ingestion.

        Shared by /api/upload and chat attachments so both paths get
        identical error handling. Raises HTTPException on failure,
        otherwise returns the parsed JSON response.
        """
        import requests as req_lib

        files = {"file": (filename, content, content_type)}
        data: dict[str, str] = {}
        if document_id:
            data["document_id"] = document_id
        if privacy:
            data["privacy"] = privacy
        if description:
            data["description"] = description
        headers: dict[str, str] = {}
        if user_id:
            data["user_id"] = user_id
            headers["X-User-Id"] = user_id

        try:
            resp = req_lib.post(
                f"{doc_service_url}/documents",
                files=files,
                data=data,
                headers=headers or None,
                timeout=120,
            )
        except req_lib.exceptions.ConnectionError as exc:
            raise HTTPException(
                status_code=503,
                detail="Document service is not running. Start it on port 8080.",
            ) from exc
        except Exception as exc:
            logger.exception("Upload failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if resp.status_code >= 400:
            error_msg = resp.text[:500]
            logger.error("Document upload failed (%d): %s", resp.status_code, error_msg)
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Document service error: {error_msg}",
            )
        return resp.json()

    async def _ingest_chat_attachment(
        message: str,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        user_id: str = "",
    ) -> str:
        """Ingest a file attached to a chat message, then annotate the message.

        Returns the user message with a system fact line carrying the
        ingestion outcome, so the LLM answers with document IDs and
        statuses in context. Never raises for ingestion failures — the
        failure becomes a fact the LLM can explain instead.
        """
        if not content:
            note = f"[System: an empty file named '{filename}' was attached; nothing was ingested.]"
            return f"{message}\n{note}".strip()
        try:
            result = _forward_to_document_service(
                filename=filename,
                content=content,
                content_type=content_type,
                description=(message or "")[:500],
                user_id=user_id,
            )
        except HTTPException as exc:
            note = f"[System: attached file '{filename}' could not be ingested ({exc.detail}).]"
            return f"{message}\n{note}".strip()
        docs = result.get("documents") if isinstance(result, dict) else None
        if docs:
            parts = ", ".join(
                f"{d.get('document_id')} (status {d.get('processing_status')})"
                for d in docs
            )
            fact = f"[System: attached file '{filename}' ingested as {len(docs)} document(s): {parts}.]"
        else:
            fact = (
                f"[System: attached file '{filename}' ingested as document "
                f"{result.get('document_id')} (status {result.get('processing_status')}).]"
            )
        prompt = message or "Please process the attached file."
        return f"{prompt}\n{fact}".strip()

    # ─── File Upload ───────────────────────────────────────────────
    @app.post("/api/upload")
    async def upload_file(
        request: Request,
        file: UploadFile = File(...),
        session_id: str = Form(default=""),
        privacy: str = Form(default=""),
        description: str = Form(default=""),
    ):
        """Upload a file (image or PDF) and forward to the document service for ingestion.

        For multi-page PDFs, the document service handles splitting — each page
        gets its own document_id, metadata, and storage record.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Validate file type
        allowed_types = {
            "image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff",
            "application/pdf",
        }
        content_type = file.content_type or "application/octet-stream"
        if content_type not in allowed_types:
            # Check extension as fallback
            ext = Path(file.filename).suffix.lower()
            ext_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".tiff": "image/tiff",
                ".pdf": "application/pdf",
            }
            content_type = ext_map.get(ext, content_type)
            if content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {content_type}. "
                           f"Allowed: images (JPEG, PNG, WebP, GIF, TIFF) and PDFs.",
                )

        # Read file content
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Optional auth: Bearer → user_id, else legacy __local__ (keeps
        # existing UI/tests working pre-login).
        upload_user_id = "__local__"
        try:
            upload_user_id = _current_user(request)["sub"]
        except HTTPException:
            pass

        # Forward to document service (shared helper; raises on failure)
        try:
            result = _forward_to_document_service(
                filename=file.filename,
                content=file_bytes,
                content_type=content_type,
                document_id=session_id,
                privacy=privacy,
                description=description,
                user_id=upload_user_id,
            )
            # Event-driven sync: push raw bytes to the global hub so a new
            # device can pull. Best-effort — ingest already succeeded.
            try:
                from shared.device_sync import push_document
                if "documents" in result:
                    for d in result["documents"]:
                        push_document(
                            user_id=upload_user_id,
                            document_id=d.get("document_id", ""),
                            version=int(d.get("version", 1)),
                            blob=file_bytes,
                            metadata={"filename": file.filename},
                        )
                elif result.get("document_id"):
                    push_document(
                        user_id=upload_user_id,
                        document_id=result["document_id"],
                        version=int(result.get("version", 1)),
                        blob=file_bytes,
                        metadata={"filename": file.filename},
                    )
            except Exception:
                logger.exception("Global sync push failed (ingest kept)")

            # Upload responses are intentionally terse; generated descriptions
            # stay available through metadata/description endpoints.
            if "documents" in result:
                docs = [
                    {
                        "document_id": d.get("document_id"),
                        "version": d.get("version"),
                        "processing_status": d.get("processing_status"),
                        "page_number": d.get("page_number"),
                        "total_pages": d.get("total_pages"),
                    }
                    for d in result["documents"]
                ]
                return {
                    "status": "successful",
                    "message": "Document uploaded successfully.",
                    "documents": docs,
                    "count": len(docs),
                }

            return {
                "status": "successful",
                "message": "Document uploaded successfully.",
                "document_id": result.get("document_id"),
                "version": result.get("version"),
                "processing_status": result.get("processing_status"),
            }

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Upload failed")
            raise HTTPException(status_code=500, detail=str(exc))

    # ─── Proxy to document service (optional info endpoints) ────────
    @app.get("/api/documents/{document_id}/versions/{version}/status")
    async def document_status(document_id: str, version: int):
        """Proxy to document service status endpoint."""
        import requests as req_lib
        try:
            resp = req_lib.get(
                f"{doc_service_url}/documents/{document_id}/versions/{version}/status",
                timeout=10,
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            raise HTTPException(status_code=503, detail="Document service unavailable")

    @app.get("/api/documents/{document_id}/versions/{version}/ocr")
    async def document_ocr(document_id: str, version: int):
        """Proxy to document service OCR endpoint."""
        import requests as req_lib
        try:
            resp = req_lib.get(
                f"{doc_service_url}/documents/{document_id}/versions/{version}/ocr",
                timeout=10,
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            raise HTTPException(status_code=503, detail="Document service unavailable")

    @app.get("/api/documents/{document_id}/versions/{version}/metadata")
    async def document_metadata(document_id: str, version: int):
        """Proxy to document service metadata endpoint."""
        import requests as req_lib
        try:
            resp = req_lib.get(
                f"{doc_service_url}/documents/{document_id}/versions/{version}/metadata",
                timeout=10,
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            raise HTTPException(status_code=503, detail="Document service unavailable")

    # ─── Bulk document download ────────────────────────────────────
    @app.post("/api/documents/bulk-download")
    async def bulk_download(request: Request):
        """Download multiple documents at once as a zip."""
        import zipfile

        import requests as req_lib

        body = await request.json()
        document_ids = body.get("document_ids", [])  # [{document_id, version}]
        if not document_ids:
            raise HTTPException(status_code=400, detail="document_ids list required")

        zip_buffer = io.BytesIO()
        downloaded = []
        errors = []

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for doc in document_ids:
                doc_id = doc.get("document_id", "")
                version = doc.get("version", 1)
                try:
                    resp = req_lib.get(
                        f"{doc_service_url}/documents/{doc_id}/versions/{version}/file",
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content_b64 = data.get("content_base64", "")
                        if content_b64:
                            import base64
                            content = base64.b64decode(content_b64)
                            filename = data.get("filename", f"{doc_id}_v{version}")
                            zf.writestr(filename, content)
                            downloaded.append({"document_id": doc_id, "filename": filename})
                    elif resp.status_code == 403:
                        errors.append({"document_id": doc_id, "error": "Requires approval"})
                    else:
                        errors.append({"document_id": doc_id, "error": f"HTTP {resp.status_code}"})
                except Exception as exc:
                    errors.append({"document_id": doc_id, "error": str(exc)[:100]})

        if not downloaded:
            raise HTTPException(status_code=404, detail="No documents could be downloaded")

        zip_buffer.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=documents_{len(downloaded)}.zip",
            },
        )

    @app.post("/api/documents/bulk-metadata")
    async def bulk_metadata(request: Request):
        """Get metadata for multiple documents at once."""
        import requests as req_lib

        body = await request.json()
        document_ids = body.get("document_ids", [])
        if not document_ids:
            raise HTTPException(status_code=400, detail="document_ids list required")

        results = []
        for doc in document_ids:
            doc_id = doc.get("document_id", "")
            version = doc.get("version", 1)
            try:
                resp = req_lib.get(
                    f"{doc_service_url}/documents/{doc_id}/versions/{version}/metadata",
                    timeout=10,
                )
                if resp.status_code == 200:
                    m = resp.json()
                    results.append({
                        "document_id": doc_id,
                        "name": m.get("document_sub_type") or m.get("description", "Document"),
                        "description": m.get("description", ""),
                        "type": m.get("document_type"),
                        "sub_type": m.get("document_sub_type"),
                        "privacy": m.get("privacy"),
                    })
                else:
                    results.append({"document_id": doc_id, "error": f"HTTP {resp.status_code}"})
            except Exception as exc:
                results.append({"document_id": doc_id, "error": str(exc)[:100]})

        return {"documents": results, "count": len(results)}

    # ─── Static UI files ───────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        index_path = ui_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Chitragupta</h1><p>UI not found.</p>", status_code=404)

    # Mount static directory
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    return app


class HTTPServer:
    """FastAPI-based HTTP server wrapper."""

    def __init__(
        self,
        *,
        config: OrchestratorConfig,
        engine: OrchestrationEngine,
    ) -> None:
        self._config = config
        self._engine = engine
        self._app = create_app(config=config, engine=engine)

    def serve(self) -> None:
        """Start the Uvicorn server."""
        logger.info(
            "Orchestrator HTTP server starting on %s:%d",
            self._config.http_host,
            self._config.http_port,
        )
        logger.info(
            "API docs at http://%s:%d/api/docs",
            self._config.http_host,
            self._config.http_port,
        )
        logger.info(
            "UI available at http://%s:%d/",
            self._config.http_host,
            self._config.http_port,
        )
        uvicorn.run(
            self._app,
            host=self._config.http_host,
            port=self._config.http_port,
            log_level="info",
            access_log=True,
        )

    @property
    def app(self) -> FastAPI:
        """Expose the FastAPI app for testing."""
        return self._app
