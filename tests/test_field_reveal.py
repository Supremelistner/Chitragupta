"""Regression: approved field reveal must return the VALUE, not loop step 1.

Replays the reported UI loop ("cached... awaiting approval" -> "Done."
-> repeat): the hardened doc service ignores legacy confirm=true and
demands confirmation_token, so the orchestrator auto-completes step 2
post-approval instead of relying on the LLM to parrot the token.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _engine_with_router(router):
    from orchestrator_service.application.orchestrator import OrchestrationEngine
    from orchestrator_service.domain.models import ServiceTarget
    from orchestrator_service.domain.ports import LLMResponse
    from orchestrator_service.infrastructure.confirmation_store import InMemoryConfirmationStore
    from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
    from orchestrator_service.infrastructure.session_store import FileSessionStore
    from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry
    from model_service.infrastructure.language_preferences import LanguagePreferenceStore
    from model_service.infrastructure.translation_provider import TranslationResult

    class _NoOp:
        def __init__(self):
            self._store = LanguagePreferenceStore(Path(tempfile.mkdtemp()) / "prefs.json")

        def translate_user_input(self, text, session_id):
            return TranslationResult(text=text, source="en", target="en", cached=True, latency_ms=0)

        def translate_bot_reply(self, text, session_id):
            return TranslationResult(text=text, source="en", target="en", cached=True, latency_ms=0)

        def get_preference(self, session_id):
            return self._store.get(session_id)

        def set_preference(self, session_id, **kw):
            return self._store.set(session_id, **kw)

    llm = MagicMock()
    llm.chat.return_value = LLMResponse(content="summary", tool_calls=[])
    tmp = tempfile.mkdtemp()
    return OrchestrationEngine(
        llm=llm, sessions=FileSessionStore(base_dir=f"{tmp}/s"),
        confirmations=InMemoryConfirmationStore(),
        tool_registry=DefaultToolRegistry(),
        service_router=router,
        translation_service=_NoOp(),
    )


def _hardened_server_router():
    """Mimics the token-hardened /tools/get_field_value endpoint."""
    from orchestrator_service.domain.models import ServiceTarget
    from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
    client = MagicMock()

    def call_tool(name, args):
        assert args.get("user_id") == "u-alice", f"tenant missing: {args}"
        if args.get("confirmation_token") == "tok-123":
            return {"status": "ok", "field": "aadhaar_number", "value": "XXXX-XXXX-1234"}
        return {"status": "requires_confirmation", "field": "aadhaar_number",
                "confirmation_token": "tok-123"}

    client.call_tool.side_effect = call_tool
    return ServiceClientRouter({ServiceTarget.DOCUMENT: client}), client


def test_confirmed_step_auto_completes_with_token():
    from orchestrator_service.domain.models import ConfirmationRequest, ConfirmationType
    router, client = _hardened_server_router()
    engine = _engine_with_router(router)
    session = engine.create_session(user_id="u-alice", title="t")

    confirmation = ConfirmationRequest(
        session_id=session.session_id,
        confirmation_type=ConfirmationType.SENSITIVE_ACCESS,
        tool_name="get_field_value",
        tool_args={"document_id": "d1", "version": 1, "field": "aadhaar_number"},
        message="proceed?",
    )
    engine._confirmations.save(confirmation)
    resp = engine._execute_confirmed_step(session, confirmation.request_id)

    calls = [c.args[1] for c in client.call_tool.call_args_list]
    assert len(calls) == 2, f"expected step1+step2, got {calls}"
    assert calls[1].get("confirmation_token") == "tok-123"
    tool_result = resp.tool_calls_made[0].result
    assert tool_result.get("status") == "ok", tool_result
    assert tool_result.get("value") == "XXXX-XXXX-1234"


def test_step2_failure_keeps_step1_result():
    router = MagicMock()
    router.call_tool.side_effect = [
        {"status": "requires_confirmation", "confirmation_token": "tok-1"},
        {"error": True, "message": "boom"},
    ]
    engine = _engine_with_router(router)
    out = engine._maybe_complete_field_value(
        "document_service", "get_field_value", {"document_id": "d1"},
        {"status": "requires_confirmation", "confirmation_token": "tok-1"},
    )
    assert out["status"] == "requires_confirmation"


def test_no_token_no_extra_call():
    router = MagicMock()
    router.call_tool.return_value = {"status": "ok", "value": "v"}
    engine = _engine_with_router(router)
    out = engine._maybe_complete_field_value(
        "document_service", "get_field_value", {},
        {"status": "requires_confirmation"},
    )
    assert out["status"] == "requires_confirmation"
    router.call_tool.assert_not_called()
