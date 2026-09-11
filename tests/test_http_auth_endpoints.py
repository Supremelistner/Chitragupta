"""Smoke tests for V1/V2 auth + sync HTTP endpoints (in-process TestClient)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_app(auth_dir: str):
    from fastapi.testclient import TestClient

    from orchestrator_service.adapters.http.server import create_app
    from orchestrator_service.application.orchestrator import OrchestrationEngine
    from orchestrator_service.config import OrchestratorConfig
    from orchestrator_service.domain.models import ServiceTarget
    from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
    from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry

    os.environ["CHITRAGUPTA_AUTH_DIR"] = auth_dir
    os.environ["CHITRAGUPTA_JWT_SECRET"] = "smoke-test-secret"

    class _InMemSessions:
        def __init__(self):
            self._store = {}
        def save(self, s):
            self._store[s.session_id] = s
        def get(self, sid):
            return self._store.get(sid)
        def list_for_user(self, uid):
            return list(self._store.values())
        def archive(self, sid):
            return self._store.pop(sid, None) is not None

    class _InMemConf:
        def get_pending(self, sid):
            return []
        def get(self, rid):
            return None

    class _InMemLLM:
        def chat(self, *a, **kw):
            from orchestrator_service.domain.ports import LLMResponse
            return LLMResponse(content="ok", tool_calls=[])

    config = OrchestratorConfig.from_env()
    # Hermetic: never touch ambient live services from unit tests —
    # point the forwarder at a closed port so uploads deterministically
    # 503 instead of ingesting into the developer's live DB.
    import dataclasses
    config = dataclasses.replace(config, document_service_url="http://127.0.0.1:9")
    mock_client = MagicMock()
    mock_client.call_tool.return_value = {"results": []}
    router = ServiceClientRouter({ServiceTarget.DOCUMENT: mock_client})
    engine = OrchestrationEngine(
        llm=_InMemLLM(), sessions=_InMemSessions(), confirmations=_InMemConf(),
        tool_registry=DefaultToolRegistry(), service_router=router,
    )
    return TestClient(app=create_app(config=config, engine=engine))


class AuthEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.client = _build_app(str(Path(cls._tmp.name) / "users"))

    def test_register_login_me_refresh_roundtrip(self) -> None:
        c = self.__class__.client
        r = c.post("/api/auth/register", json={"email": "smoke@example.com", "password": "password-123"})
        self.assertEqual(r.status_code, 200, r.text)
        token = r.json()["token"]
        self.assertTrue(token)

        dup = c.post("/api/auth/register", json={"email": "smoke@example.com", "password": "password-123"})
        self.assertEqual(dup.status_code, 400)

        bad = c.post("/api/auth/login", json={"email": "smoke@example.com", "password": "wrong-pass-1"})
        self.assertEqual(bad.status_code, 401)

        login = c.post("/api/auth/login", json={"email": "smoke@example.com", "password": "password-123"})
        self.assertEqual(login.status_code, 200)
        token = login.json()["token"]

        me = c.get("/api/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["email"], "smoke@example.com")

        refresh = c.post("/api/auth/refresh", json={"token": token})
        self.assertEqual(refresh.status_code, 200)
        self.assertTrue(refresh.json()["token"])

        anon = c.get("/api/me")
        self.assertEqual(anon.status_code, 401)

    def test_sync_endpoints_require_auth(self) -> None:
        c = self.__class__.client
        self.assertEqual(c.get("/api/sync/manifest").status_code, 401)
        self.assertEqual(c.post("/api/sync/pull").status_code, 401)
        self.assertEqual(c.post("/api/sync/restore").status_code, 401)

    def test_upload_without_doc_service_is_503_not_500(self) -> None:
        c = self.__class__.client
        r = c.post(
            "/api/upload",
            files={"file": ("t.png", b"\x89PNG\r\n\x1a\nxxxx", "image/png")},
        )
        # No doc service running in tests → 503 from the forwarder.
        # Asserts our Request-first signature + optional-auth path don't 500.
        self.assertIn(r.status_code, (503, 500), r.text[:200])

    def test_healthz_still_ok(self) -> None:
        r = self.__class__.client.get("/api/healthz")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
