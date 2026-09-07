"""End-to-end smoke test of the /api/language HTTP endpoints.

Verifies the FastAPI route wires to the orchestrator engine correctly
and that round-trips through GET and POST preserve the language
preference across calls.

Uses FastAPI's TestClient (which spins up the app in-process) so the
test exercises the actual route handlers — no separate uvicorn needed.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class _NoOpTranslation:
    """Stand-in for the real translation service.

    The /api/language tests don't exercise translation itself; they
    just verify the HTTP layer correctly reads and writes the
    preference store. The orchestrator wraps `process_message` with
    translation, but for these tests we drive ``set_language_preference``
    and ``get_language_preference`` directly.
    """

    def __init__(self) -> None:
        from model_service.infrastructure.language_preferences import (
            LanguagePreferenceStore,
        )
        self._tmp = tempfile.TemporaryDirectory()
        # The engine's set_preference / get_preference methods
        # delegate to ``self._store.set`` / ``self._store.get`` on the
        # real TranslationService. The HTTP endpoint tests exercise
        # this path so the fake needs a working store.
        self._store = LanguagePreferenceStore(
            Path(self._tmp.name) / "prefs.json"
        )

    def __del__(self) -> None:
        try:
            self._tmp.cleanup()
        except (OSError, FileNotFoundError):
            pass

    def translate_user_input(self, text, session_id):
        from model_service.infrastructure.translation_provider import (
            TranslationResult,
        )
        return TranslationResult(
            text=text, source="en", target="en",
            cached=True, latency_ms=0,
        )

    def translate_bot_reply(self, text, session_id):
        from model_service.infrastructure.translation_provider import (
            TranslationResult,
        )
        return TranslationResult(
            text=text, source="en", target="en",
            cached=True, latency_ms=0,
        )

    def get_preference(self, session_id):
        return self._store.get(session_id)

    def set_preference(self, session_id, *, source=None, target=None):
        return self._store.set(session_id, source=source, target=target)


def _build_app():
    """Build the orchestrator's FastAPI app with a no-op translation stub."""
    from fastapi.testclient import TestClient

    from orchestrator_service.adapters.http.server import create_app
    from orchestrator_service.application.orchestrator import OrchestrationEngine
    from orchestrator_service.config import OrchestratorConfig
    from orchestrator_service.domain.models import ServiceTarget
    from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
    from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry

    # Minimal in-memory store stubs.
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

    config = OrchestratorConfig.from_env()
    sessions = _InMemSessions()
    confirmations = _InMemConf()
    registry = DefaultToolRegistry()
    mock_client = MagicMock()
    mock_client.call_tool.return_value = {"results": []}
    router = ServiceClientRouter({ServiceTarget.DOCUMENT: mock_client})
    llm = _InMemLLM()

    engine = OrchestrationEngine(
        llm=llm,
        sessions=sessions,
        confirmations=confirmations,
        tool_registry=registry,
        service_router=router,
        translation_service=_NoOpTranslation(),
    )
    app = create_app(config=config, engine=engine)
    return TestClient(app)


class LanguageEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _build_app()

    def test_get_default_when_no_session(self) -> None:
        """GET /api/language with no session_id returns the V1 default."""
        resp = self.client.get("/api/language")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["source"], "en")
        self.assertEqual(body["target"], "hi")
        self.assertTrue(body["default"])

    def test_get_session_default_is_v1_default(self) -> None:
        """An unknown session_id still gets the V1 default."""
        resp = self.client.get(
            "/api/language?session_id=brand-new-session"
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["source"], "en")
        self.assertEqual(body["target"], "hi")
        self.assertFalse(body["default"])

    def test_set_and_get_round_trip(self) -> None:
        """POST updates, GET reads back."""
        sid = "session-roundtrip"
        set_resp = self.client.post(
            "/api/language",
            json={"session_id": sid, "source": "hi", "target": "en"},
        )
        self.assertEqual(set_resp.status_code, 200)
        body = set_resp.json()
        self.assertEqual(body["source"], "hi")
        self.assertEqual(body["target"], "en")
        # Read it back.
        get_resp = self.client.get(
            f"/api/language?session_id={sid}"
        )
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["source"], "hi")

    def test_set_session_isolation(self) -> None:
        """Two sessions keep independent preferences."""
        s1 = "session-aaa"
        s2 = "session-bbb"
        self.client.post(
            "/api/language",
            json={"session_id": s1, "source": "hi", "target": "en"},
        )
        self.client.post(
            "/api/language",
            json={"session_id": s2, "source": "ta", "target": "en"},
        )
        a = self.client.get(f"/api/language?session_id={s1}").json()
        b = self.client.get(f"/api/language?session_id={s2}").json()
        self.assertEqual(a["source"], "hi")
        self.assertEqual(b["source"], "ta")

    def test_set_rejects_unsupported_language(self) -> None:
        resp = self.client.post(
            "/api/language",
            json={"session_id": "x", "target": "klingon"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("klingon", resp.json()["detail"])

    def test_set_requires_session_id(self) -> None:
        resp = self.client.post(
            "/api/language", json={"source": "hi"}
        )
        self.assertEqual(resp.status_code, 400)

    def test_partial_update_keeps_existing(self) -> None:
        """POSTing only source keeps the existing target."""
        sid = "session-partial"
        # Set both first.
        self.client.post(
            "/api/language",
            json={"session_id": sid, "source": "hi", "target": "en"},
        )
        # Then update only source.
        resp = self.client.post(
            "/api/language",
            json={"session_id": sid, "source": "ta"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["source"], "ta")
        self.assertEqual(body["target"], "en")  # unchanged


if __name__ == "__main__":
    unittest.main()