"""Tests for the /api/profile endpoints (persona display name).

Uses FastAPI's TestClient against the real route handlers with stubbed
engine dependencies. The profile file lives in a temp dir so the real
``~/.chitragupta/profile.json`` is never touched.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_client(profile_dir: str):
    from fastapi.testclient import TestClient

    from orchestrator_service.adapters.http.server import create_app
    from orchestrator_service.application.orchestrator import OrchestrationEngine
    from orchestrator_service.config import OrchestratorConfig
    from orchestrator_service.domain.models import ServiceTarget
    from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
    from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry

    from test_http_language_endpoint import _NoOpTranslation

    class _InMemSessions:
        def __init__(self):
            self._store: dict = {}
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
        def save(self, c):
            pass
        def add(self, c):
            pass
        def respond(self, *a, **kw):
            pass
        def get_by_tool_call(self, cid):
            return None

    class _InMemLLM:
        def chat(self, *a, **kw):
            from orchestrator_service.domain.ports import LLMResponse
            return LLMResponse(content="ok", tool_calls=[])
        def health(self):
            return {"status": "ok"}

    config = replace(
        OrchestratorConfig.from_env(),
        profile_path=str(Path(profile_dir) / "profile.json"),
    )
    router = ServiceClientRouter({ServiceTarget.DOCUMENT: MagicMock()})
    engine = OrchestrationEngine(
        llm=_InMemLLM(),
        sessions=_InMemSessions(),
        confirmations=_InMemConf(),
        tool_registry=DefaultToolRegistry(),
        service_router=router,
        translation_service=_NoOpTranslation(),
    )
    return TestClient(create_app(config=config, engine=engine))


class ProfileEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _build_client(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_get_profile_empty_initially(self) -> None:
        resp = self.client.get("/api/profile")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["display_name"])

    def test_set_then_get_round_trip(self) -> None:
        set_resp = self.client.post("/api/profile", json={"display_name": "Priya"})
        self.assertEqual(set_resp.status_code, 200)
        self.assertEqual(set_resp.json()["display_name"], "Priya")
        get_resp = self.client.get("/api/profile")
        self.assertEqual(get_resp.json()["display_name"], "Priya")

    def test_set_persists_to_profile_file(self) -> None:
        self.client.post("/api/profile", json={"display_name": "Priya"})
        from orchestrator_service.onboarding import load_profile
        profile = load_profile(str(Path(self._tmp.name) / "profile.json"))
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.display_name, "Priya")

    def test_set_rejects_blank_name(self) -> None:
        resp = self.client.post("/api/profile", json={"display_name": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_set_rejects_overlong_name(self) -> None:
        resp = self.client.post("/api/profile", json={"display_name": "x" * 101})
        self.assertEqual(resp.status_code, 400)


class EngineUserNameTests(unittest.TestCase):
    def _engine(self):
        from unittest.mock import MagicMock
        from orchestrator_service.application.orchestrator import OrchestrationEngine
        return OrchestrationEngine(
            llm=MagicMock(), sessions=MagicMock(), confirmations=MagicMock(),
            tool_registry=MagicMock(), service_router=MagicMock(),
            translation_service=MagicMock(),
        )

    def test_default_is_none(self) -> None:
        self.assertIsNone(self._engine().user_name)

    def test_set_and_clear(self) -> None:
        engine = self._engine()
        engine.set_user_name("  Priya  ")
        self.assertEqual(engine.user_name, "Priya")
        engine.set_user_name(None)
        self.assertIsNone(engine.user_name)

    def test_persona_uses_name(self) -> None:
        from orchestrator_service.persona import build_system_prompt
        engine = self._engine()
        engine.set_user_name("Priya")
        prompt = build_system_prompt(user_name=engine.user_name)
        self.assertIn("Priya", prompt)


if __name__ == "__main__":
    unittest.main()
