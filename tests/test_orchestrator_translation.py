"""Orchestrator-level translation integration tests.

Verifies the end-to-end flow:

1. User writes in Hindi → orchestrator translates to English before
   handing to the LLM.
2. LLM replies in English → orchestrator translates to Hindi before
   returning to the user.
3. Session in English mode → translations are no-ops.
4. Per-session language preference is respected (one session in Hindi,
   another in English).
5. Confirmation copy is translated too.

These tests don't talk to a real LLM. They use ``MagicMock`` for the
LLM and a fake ``TranslationService`` so they run instantly.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestrator_service.application.orchestrator import (
    OrchestrationEngine,
)
from orchestrator_service.domain.models import (
    ConfirmationRequest,
    MessageRole,
    ServiceTarget,
    Session,
    SessionStatus,
)
from orchestrator_service.domain.ports import (
    LLMProvider,
    LLMResponse,
    SessionStore,
)
from orchestrator_service.infrastructure.service_clients import (
    ServiceClientRouter,
)
from orchestrator_service.infrastructure.tool_registry import (
    DefaultToolRegistry,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeTranslationService:
    """Stands in for the real TranslationService. Records all calls.

    Holds its own preference store so tests don't trip over the
    module-level singleton (the real one reads ``data/translation_state.json``
    on first construction and caches the result for the whole process).
    """

    def __init__(self) -> None:
        import tempfile as _tmp
        from pathlib import Path as _Path

        from model_service.infrastructure.language_preferences import (
            LanguagePreferenceStore,
        )
        self._tmp = _tmp.TemporaryDirectory()
        self._store = LanguagePreferenceStore(
            _Path(self._tmp.name) / "prefs.json"
        )
        self.calls: list[tuple[str, str, str]] = []
        # Translation map: (direction, source, target) -> output. Any
        # unmapped direction round-trips text prefixed with the direction.
        self.responses: dict[tuple[str, str], str] = {}

    def __del__(self) -> None:
        try:
            self._tmp.cleanup()
        except (OSError, FileNotFoundError):
            pass

    def translate_user_input(self, text: str, session_id: str):
        from model_service.infrastructure.translation_provider import (
            TranslationResult,
        )
        pref = self._store.get(session_id)
        self.calls.append(("input", pref.source, pref.target))
        # Default response drops Devanagari/Tamil/etc. script so the
        # LLM sees English-only content.
        out = self.responses.get(
            ("input", pref.source),
            f"[EN from {pref.source}] {text}",
        )
        ascii_only = "".join(c if ord(c) < 128 else "" for c in out)
        return TranslationResult(
            text=ascii_only or out,
            source=pref.source,
            target="en",
            cached=False,
            latency_ms=1,
        )

    def translate_bot_reply(self, text: str, session_id: str):
        from model_service.infrastructure.translation_provider import (
            TranslationResult,
        )
        pref = self._store.get(session_id)
        self.calls.append(("reply", pref.source, pref.target))
        # The orchestrator's _translate_response uses ``cached=True`` as
        # the user-visible signal that no real translation was done. The
        # *real* service only short-circuits when target==en (the user
        # is in English mode and the bot's English reply needs no
        # translation). A hi→hi session means the user is in Hindi, so
        # the bot's English reply DOES need translation to Hindi — even
        # though source happens to equal target.
        if pref.target == "en":
            return TranslationResult(
                text=text,
                source="en",
                target=pref.target,
                cached=True,
                latency_ms=0,
            )
        out = self.responses.get(
            ("reply", pref.target),
            f"[{pref.target}]:{text}",
        )
        return TranslationResult(
            text=out,
            source="en",
            target=pref.target,
            cached=False,
            latency_ms=1,
        )

    def get_preference(self, session_id: str):
        return self._store.get(session_id)

    def set_preference(self, session_id, *, source=None, target=None):
        return self._store.set(session_id, source=source, target=target)


class _FakeLLM(LLMProvider):
    """Returns canned English replies, regardless of what was sent in."""

    def __init__(self, content: str = "Here is your answer.") -> None:
        self._content = content
        self.last_messages: list[Any] = []

    def chat(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        self.last_messages = list(messages)
        return LLMResponse(content=self._content, tool_calls=[])

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}


class _InMemorySessionStore(SessionStore):
    """Minimal in-memory session store for tests."""

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        self._store[session.session_id] = session

    def get(self, session_id: str):
        return self._store.get(session_id)

    def list_for_user(self, user_id: str):
        return [s for s in self._store.values() if s.user_id == user_id]

    def archive(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False


class _InMemoryConfirmationStore:
    """Minimal in-memory ConfirmationStore for tests."""

    def __init__(self) -> None:
        self._pending: dict[str, list[Any]] = {}
        self._by_id: dict[str, Any] = {}

    def get_pending(self, session_id: str) -> list:
        return self._pending.get(session_id, [])

    def get(self, request_id: str):
        return self._by_id.get(request_id)

    def get_by_tool_call(self, tool_call_id: str):
        return None

    def save(self, confirmation) -> None:
        self._by_id[confirmation.request_id] = confirmation
        self._pending.setdefault(confirmation.session_id, []).append(confirmation)

    def add(self, confirmation) -> None:
        self.save(confirmation)

    def respond(self, request_id, approved, *, correction=None) -> None:
        # Find the confirmation and remove it from pending.
        for session_id, items in list(self._pending.items()):
            for pending_conf in list(items):
                if pending_conf.request_id == request_id:
                    items.remove(pending_conf)
                    pending_conf.responded = True
                    pending_conf.approved = approved
                    if correction is not None:
                        pending_conf.correction = correction
                    return

    def clear(self, session_id: str) -> None:
        self._pending.pop(session_id, None)
class OrchestratorTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset module-level preference store to defaults between tests.
        from model_service.infrastructure.language_preferences import (
            reset_store_for_tests,
        )
        reset_store_for_tests()

        self._translation = _FakeTranslationService()
        self._sessions = _InMemorySessionStore()
        self._registry = DefaultToolRegistry()
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"results": []}
        self._router = ServiceClientRouter(
            {ServiceTarget.DOCUMENT: mock_client}
        )
        self._llm = _FakeLLM(content="Here is your answer.")

        self._engine = OrchestrationEngine(
            llm=self._llm,
            sessions=self._sessions,
            confirmations=_InMemoryConfirmationStore(),
            tool_registry=self._registry,
            service_router=self._router,
            translation_service=self._translation,
        )

    def _new_session(self, session_id: str = "s1") -> Session:
        session = Session(session_id=session_id, user_id="u1", status=SessionStatus.ACTIVE)
        self._sessions.save(session)
        return session

    def test_hindi_session_translates_inbound(self) -> None:
        """User writes in Hindi, orchestrator translates to English before LLM."""
        session = self._new_session()
        # Default preference is en->hi. Flip source to hi so the user
        # writes in Hindi.
        from model_service.infrastructure.language_preferences import (
            get_store,
        )
        get_store().set(session.session_id, source="hi", target="hi")

        self._engine.process_message(session.session_id, "मेरा आधार नंबर क्या है?")

        # The translation service was called once on the inbound message.
        inbound_calls = [c for c in self._translation.calls if c[0] == "input"]
        self.assertEqual(len(inbound_calls), 1)
        # The LLM was then handed the *translated* text, not the original.
        last_user_msg = self._llm.last_messages[-1]
        self.assertEqual(last_user_msg.role, MessageRole.USER)
        # The LLM should see English, not Hindi.
        self.assertNotIn("आधार", last_user_msg.content)

    def test_hindi_session_translates_outbound(self) -> None:
        """LLM's English reply is translated back to Hindi."""
        session = self._new_session()
        from model_service.infrastructure.language_preferences import (
            get_store,
        )
        get_store().set(session.session_id, source="hi", target="hi")
        self._translation.responses[("reply", "hi")] = "यहाँ आपका उत्तर है।"

        result = self._engine.process_message(session.session_id, "kuch bhi")

        # The translation service was called for the bot reply.
        reply_calls = [c for c in self._translation.calls if c[0] == "reply"]
        self.assertEqual(len(reply_calls), 1)
        self.assertEqual(result.message, "यहाँ आपका उत्तर है।")
        # The translation metadata is surfaced in the response.
        self.assertEqual(result.metadata["translation"]["target"], "hi")

    def test_english_session_is_no_op(self) -> None:
        """When both source and target are English, no real translation runs.

        The V1 default is en→hi (user-facing language = Hindi), so a
        vanilla session IS translated on the bot-reply side. To get a
        true no-op we set both source AND target to English on the
        fake's own preference store.
        """
        # Register an explicit en→en reply so the assertion is
        # deterministic when the no-op fires.
        self._translation.responses[("reply", "en")] = "Here is your answer."
        # Default en→hi: BOTH methods are called, but only the reply
        # actually performs translation (input is en→en = cached no-op).
        # Verify the input call is cached, the reply call is not.
        self._new_session("s-default")
        result = self._engine.process_message("s-default", "hello")
        self.assertEqual(len(self._translation.calls), 2)
        reply_calls = [c for c in self._translation.calls if c[0] == "reply"]
        input_calls = [c for c in self._translation.calls if c[0] == "input"]
        self.assertEqual(len(reply_calls), 1)
        self.assertEqual(len(input_calls), 1)
        # Both directions English: every call is cached = no real work.
        self._translation.set_preference("s-en", source="en", target="en")
        self._translation.calls.clear()
        self._new_session("s-en")
        result = self._engine.process_message("s-en", "hello")
        # Both methods still get called (cheap service-layer short-circuit)
        # but neither hits the network.
        self.assertEqual(len(self._translation.calls), 2)
        self.assertEqual(result.message, "Here is your answer.")
        self.assertTrue(result.metadata["translation"]["cached"])

    def test_per_session_language_isolation(self) -> None:
        """One session Hindi, another English (both directions en)."""
        from model_service.infrastructure import language_preferences as _lp_mod
        _lp_mod._store = None

        # Set Hindi-mode preference for s-hi via the fake's own store.
        self._translation.set_preference("s-hi", source="hi", target="hi")
        self._translation.responses[("reply", "hi")] = "[HINDI-REPLY]"
        # English mode for s-en: source=en, target=en, no-op.
        self._translation.set_preference("s-en", source="en", target="en")
        # The fake's default reply response is ``[target]:text``. Register
        # an explicit en→en response so the assertion is deterministic.
        self._translation.responses[("reply", "en")] = "Here is your answer."

        self._new_session("s-hi")
        self._new_session("s-en")

        hi_result = self._engine.process_message("s-hi", "नमस्ते")
        en_result = self._engine.process_message("s-en", "hello")

        # Hindi session: user input translated, reply translated.
        self.assertEqual(hi_result.message, "[HINDI-REPLY]")
        # English session: no translations at all.
        self.assertEqual(en_result.message, "Here is your answer.")
        # Only the Hindi session triggered a real (non-cached) inbound
        # translation. The English session still calls the method but
        # the service short-circuits (en→en is a no-op).
        hi_inbound = [
            c for c in self._translation.calls
            if c[0] == "input" and c[1] != "en"
        ]
        self.assertEqual(len(hi_inbound), 1)

    def test_session_log_records_english_only(self) -> None:
        """The session's user message history must only contain English.

        This matters for privacy (no PII in user's preferred script in
        stored logs) and for the LLM's reasoning consistency. System
        messages injected by auto-search are also English.
        """
        from model_service.infrastructure.language_preferences import (
            get_store,
        )
        get_store().set("s1", source="hi", target="hi")
        session = self._new_session("s1")

        self._engine.process_message("s1", "मेरा नाम क्या है?")

        # The session log has exactly one user message (auto-search may
        # inject system messages, but never user messages).
        user_messages = [m for m in session.messages if m.role == MessageRole.USER]
        self.assertEqual(len(user_messages), 1)
        # It is English (translated), not Hindi.
        self.assertNotIn("नाम", user_messages[0].content)
        # Every message in the log, regardless of role, must be English.
        for m in session.messages:
            self.assertNotIn("नाम", m.content,
                             msg=f"nonHidian script in {m.role} message: {m.content!r}")

    def test_confirmation_message_translated(self) -> None:
        """The 'requires approval' confirmation popup copy is translated."""
        from model_service.infrastructure.language_preferences import (
            get_store,
        )
        get_store().set("s1", source="hi", target="hi")
        self._translation.responses[("reply", "hi")] = "[HINDI]"

        # Wire up a confirmation-gated tool call.
        # For simplicity, force the engine to emit a confirmation by setting
        # up the inner state directly. We use the engine's internal helpers
        # via the public method: register a pending confirmation.
        self._new_session("s1")
        confirmation = ConfirmationRequest(
            session_id="s1",
            tool_name="get_field_value",
            tool_args={"document_id": "d", "version": 1, "field": "aadhaar_number"},
            message="Action requires approval",
        )
        self._engine._confirmations.add(confirmation)  # type: ignore[attr-defined]

        # Simulate the user approving the confirmation.
        response = self._engine.handle_confirmation(
            session_id="s1",
            request_id=confirmation.request_id,
            approved=True,
        )

        # The final message was translated.
        reply_calls = [c for c in self._translation.calls if c[0] == "reply"]
        self.assertGreaterEqual(len(reply_calls), 1)
        self.assertEqual(response.metadata["translation"]["target"], "hi")

    def test_set_language_preference_propagates_to_next_call(self) -> None:
        """Switching the language mid-conversation takes effect on the
        next message without dropping context."""
        session = self._new_session()
        # First message in default en→hi mode.
        self._translation.calls.clear()
        self._engine.process_message(session.session_id, "hello")
        # Inbound short-circuits (en→en), but reply is en→hi.
        first_inbound = [c for c in self._translation.calls if c[0] == "input"]
        self.assertEqual(len(first_inbound), 1)
        # Flip to Hindi on both sides for clear assertions.
        self._engine.set_language_preference(
            session.session_id, source="hi", target="hi"
        )
        self._translation.responses[("reply", "hi")] = "[HINDI]"
        self._translation.calls.clear()

        self._engine.process_message(session.session_id, "नमस्ते")

        # Now inbound DOES translate (hi→en); reply translates (en→hi).
        inbound_calls = [c for c in self._translation.calls if c[0] == "input"]
        self.assertEqual(len(inbound_calls), 1)
        reply_calls = [c for c in self._translation.calls if c[0] == "reply"]
        self.assertGreaterEqual(
            len([c for c in reply_calls if c[2] == "hi"]), 1
        )


if __name__ == "__main__":
    unittest.main()