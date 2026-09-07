"""MCP-level smoke tests for the translate_text and set_language_preference tools.

These tests instantiate an ``MCPServer`` directly, build a minimal
``InferenceService`` stub, and exercise the full ``tools/list`` and
``tools/call`` dispatch paths. They use a ``FakeTranslationProvider`` so
no real Gemini API key is needed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model_service.adapters.mcp.server import MCPServer
from model_service.application.translation_service import (
    TranslationService,
    reset_service_for_tests,
)
from model_service.config import ModelServiceConfig
from model_service.infrastructure.language_preferences import (
    LanguagePreferenceStore,
    reset_store_for_tests,
)
from model_service.infrastructure.translation_provider import (
    TranslationRequest,
    TranslationResult,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _FakeInference:
    """Minimal InferenceService stub — translate tools don't touch it."""
    def list_providers(self):
        return []

    def get_active_provider(self):
        return "fake"

    def set_active_provider(self, name):
        return name

    def health(self):
        return {"status": "ok"}


class _FakeTranslationProvider:
    """Returns deterministic translations, records every call."""

    def __init__(self) -> None:
        self.calls: list[TranslationRequest] = []
        self.responses: dict[tuple[str, str], str] = {}

    def translate(self, request: TranslationRequest) -> TranslationResult:
        self.calls.append(request)
        key = (request.source, request.target)
        text = self.responses.get(
            key, f"[{request.source}->{request.target}] {request.text}"
        )
        return TranslationResult(
            text=text,
            source=request.source,
            target=request.target,
            cached=False,
            latency_ms=5,
        )

    def supported_pairs(self):
        return [("en", "hi"), ("hi", "en")]

    def health(self):
        return True

    def close(self):
        pass


class _ServerFixture:
    """Builds a fully-wired MCPServer with fakes."""

    def __enter__(self) -> tuple[MCPServer, _FakeTranslationProvider]:
        reset_service_for_tests()
        reset_store_for_tests()

        self._tmpdir = tempfile.TemporaryDirectory()
        pref_path = Path(self._tmpdir.name) / "prefs.json"
        cache_path = Path(self._tmpdir.name) / "cache.json"
        store = LanguagePreferenceStore(pref_path)

        # Build a translation service wired to our fake provider + tmp store.
        self._provider = _FakeTranslationProvider()
        self._service = TranslationService(
            provider=self._provider,  # type: ignore[arg-type]
            preferences=store,
        )

        cfg = ModelServiceConfig.from_env()
        self._server = MCPServer(
            config=cfg,
            inference_service=_FakeInference(),
            translation_service=self._service,
        )
        return self._server, self._provider

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmpdir.cleanup()


def _tools_list(server: MCPServer) -> dict[str, Any]:
    """Helper: invoke tools/list via the server's private dispatcher."""
    return server._dispatch(  # type: ignore[attr-defined]
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )


def _call_tool(server: MCPServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return server._dispatch(  # type: ignore[attr-defined]
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )


def _extract_payload(mcp_response: dict[str, Any]) -> dict[str, Any]:
    """MCP tool results are wrapped in ``content[0].text`` (JSON string)."""
    content = mcp_response["result"]["content"]
    return json.loads(content[0]["text"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TranslateTextToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fix = _ServerFixture()
        self._server, self._provider = self._fix.__enter__()

    def tearDown(self) -> None:
        self._fix.__exit__(None, None, None)

    def test_tools_list_includes_translation(self) -> None:
        resp = _tools_list(self._server)
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertIn("translate_text", names)
        self.assertIn("set_language_preference", names)

    def test_translate_text_en_to_hi(self) -> None:
        self._provider.responses[("en", "hi")] = "नमस्ते"
        resp = _call_tool(
            self._server,
            "translate_text",
            {"text": "hello", "source": "en", "target": "hi"},
        )
        self.assertFalse(resp["result"].get("isError", False))
        payload = _extract_payload(resp)
        self.assertEqual(payload["text"], "नमस्ते")
        self.assertEqual(payload["source"], "en")
        self.assertEqual(payload["target"], "hi")
        self.assertEqual(len(self._provider.calls), 1)

    def test_translate_text_hi_to_en(self) -> None:
        self._provider.responses[("hi", "en")] = "hello"
        resp = _call_tool(
            self._server,
            "translate_text",
            {"text": "नमस्ते", "source": "hi", "target": "en"},
        )
        payload = _extract_payload(resp)
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(self._provider.calls[0].text, "नमस्ते")
        self.assertEqual(self._provider.calls[0].source, "hi")
        self.assertEqual(self._provider.calls[0].target, "en")

    def test_translate_text_uses_session_preference(self) -> None:
        """When source/target are omitted and session_id is given, the
        session's stored preference drives the direction."""
        # Set preference for session 's1' to en <-> hi.
        _call_tool(
            self._server,
            "set_language_preference",
            {"session_id": "s1", "source": "hi", "target": "en"},
        )
        self._provider.responses[("hi", "en")] = "hello"
        resp = _call_tool(
            self._server,
            "translate_text",
            {"text": "नमस्ते", "session_id": "s1"},
        )
        payload = _extract_payload(resp)
        # The tool should have read preference (source=hi, target=en) and
        # translated hi -> en.
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["source"], "hi")
        self.assertEqual(payload["target"], "en")

    def test_translate_text_short_circuits_same_language(self) -> None:
        """source == target echoes the input and skips the provider."""
        resp = _call_tool(
            self._server,
            "translate_text",
            {"text": "hello", "source": "en", "target": "en"},
        )
        payload = _extract_payload(resp)
        self.assertEqual(payload["text"], "hello")
        self.assertTrue(payload["cached"])
        self.assertEqual(self._provider.calls, [])

    def test_translate_text_missing_text_returns_error(self) -> None:
        resp = _call_tool(self._server, "translate_text", {})
        # The dispatcher converts the raised ValueError into a JSON-RPC
        # error envelope. The MCP layer surfaces errors via the
        # ``isError`` flag in the result; the exception itself bubbles up
        # as a JSON-RPC error message.
        self.assertIn("error", resp)


class SetLanguagePreferenceToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fix = _ServerFixture()
        self._server, self._provider = self._fix.__enter__()

    def tearDown(self) -> None:
        self._fix.__exit__(None, None, None)

    def test_get_default_when_session_unknown(self) -> None:
        resp = _call_tool(
            self._server,
            "set_language_preference",
            {"session_id": "fresh-session"},
        )
        payload = _extract_payload(resp)
        self.assertEqual(payload["source"], "en")
        self.assertEqual(payload["target"], "hi")

    def test_set_and_read_round_trip(self) -> None:
        resp = _call_tool(
            self._server,
            "set_language_preference",
            {"session_id": "s2", "source": "hi", "target": "en"},
        )
        payload = _extract_payload(resp)
        self.assertEqual(payload["source"], "hi")
        self.assertEqual(payload["target"], "en")
        # Read back.
        resp2 = _call_tool(
            self._server,
            "set_language_preference",
            {"session_id": "s2"},
        )
        self.assertEqual(_extract_payload(resp2)["source"], "hi")

    def test_set_rejects_unsupported_language(self) -> None:
        resp = _call_tool(
            self._server,
            "set_language_preference",
            {"session_id": "s3", "target": "klingon"},
        )
        self.assertIn("error", resp)


if __name__ == "__main__":
    unittest.main()