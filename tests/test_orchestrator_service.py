"""Tests for the Orchestrator Service.

Covers:
- Session management (create, save, delete, context isolation)
- Tool registry (schemas, service routing, confirmation rules)
- Confirmation flow (sensitive access, file retrieval)
- Orchestration engine (plan → confirm → execute loop)
- LLM provider (message building)
"""
from __future__ import annotations

import base64
import json
import os
import shutil

# -- Test setup --
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator_service.application.orchestrator import OrchestrationEngine
from orchestrator_service.domain.models import (
    ConfirmationRequest,
    ConfirmationType,
    ConversationMessage,
    MessageRole,
    Plan,
    PlanStatus,
    ServiceTarget,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from orchestrator_service.domain.ports import LLMResponse
from orchestrator_service.infrastructure.confirmation_store import (
    InMemoryConfirmationStore,
)
from orchestrator_service.infrastructure.service_clients import (
    HttpServiceClient,
    ServiceClientRouter,
)
from orchestrator_service.infrastructure.session_store import FileSessionStore
from orchestrator_service.infrastructure.tool_registry import (
    LLM_VISIBLE_TOOLS,
    DefaultToolRegistry,
)

# =========================================================================
# Session Store Tests
# =========================================================================

class TestFileSessionStore(unittest.TestCase):
    """Test file-based session storage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = FileSessionStore(base_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get(self):
        session = Session(session_id="test123", user_id="user1", title="Test Session")
        self.store.save(session)

        loaded = self.store.get("test123")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, "test123")
        self.assertEqual(loaded.user_id, "user1")
        self.assertEqual(loaded.title, "Test Session")

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_delete(self):
        session = Session(session_id="del123")
        self.store.save(session)
        self.assertTrue(self.store.delete("del123"))
        self.assertIsNone(self.store.get("del123"))
        self.assertFalse(self.store.delete("del123"))  # Already deleted

    def test_list_sessions(self):
        self.store.save(Session(session_id="s1", user_id="u1"))
        self.store.save(Session(session_id="s2", user_id="u2"))
        self.store.save(Session(session_id="s3", user_id="u1"))

        all_sessions = self.store.list_sessions()
        self.assertEqual(len(all_sessions), 3)

        u1_sessions = self.store.list_sessions(user_id="u1")
        self.assertEqual(len(u1_sessions), 2)

    def test_list_with_status_filter(self):
        self.store.save(Session(session_id="active1", status=SessionStatus.ACTIVE))
        self.store.save(Session(session_id="archived1", status=SessionStatus.ARCHIVED))

        active = self.store.list_sessions(status=SessionStatus.ACTIVE)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].session_id, "active1")

    def test_archive(self):
        session = Session(session_id="arch1")
        self.store.save(session)
        self.assertTrue(self.store.archive("arch1"))
        loaded = self.store.get("arch1")
        self.assertEqual(loaded.status, SessionStatus.ARCHIVED)

    def test_messages_persisted(self):
        session = Session(session_id="msg1")
        session.messages.append(
            ConversationMessage(role=MessageRole.USER, content="Hello")
        )
        session.messages.append(
            ConversationMessage(role=MessageRole.ASSISTANT, content="Hi there!")
        )
        self.store.save(session)

        loaded = self.store.get("msg1")
        self.assertEqual(len(loaded.messages), 2)
        self.assertEqual(loaded.messages[0].role, MessageRole.USER)
        self.assertEqual(loaded.messages[0].content, "Hello")
        self.assertEqual(loaded.messages[1].content, "Hi there!")

    def test_tool_calls_persisted(self):
        session = Session(session_id="tc1")
        tc = ToolCall(tool_name="search_documents", arguments={"query": "Aadhaar"})
        tc.status = ToolCallStatus.SUCCESS
        tc.result = {"results": []}
        session.messages.append(
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[tc],
            )
        )
        self.store.save(session)

        loaded = self.store.get("tc1")
        self.assertEqual(len(loaded.messages[0].tool_calls), 1)
        self.assertEqual(loaded.messages[0].tool_calls[0].tool_name, "search_documents")
        self.assertEqual(loaded.messages[0].tool_calls[0].status, ToolCallStatus.SUCCESS)


# =========================================================================
# Context Isolation Tests
# =========================================================================

class TestContextIsolation(unittest.TestCase):
    """Verify that sessions are completely isolated from each other."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = FileSessionStore(base_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_session_contexts_dont_mix(self):
        s1 = Session(session_id="iso1", title="Session 1")
        s1.messages.append(ConversationMessage(role=MessageRole.USER, content="I have an Aadhaar card"))

        s2 = Session(session_id="iso2", title="Session 2")
        s2.messages.append(ConversationMessage(role=MessageRole.USER, content="I have a passport"))

        self.store.save(s1)
        self.store.save(s2)

        loaded1 = self.store.get("iso1")
        loaded2 = self.store.get("iso2")

        self.assertEqual(loaded1.messages[0].content, "I have an Aadhaar card")
        self.assertEqual(loaded2.messages[0].content, "I have a passport")
        self.assertEqual(len(loaded1.messages), 1)
        self.assertEqual(len(loaded2.messages), 1)

    def test_delete_session_removes_context(self):
        s1 = Session(session_id="del_ctx", title="To Delete")
        s1.messages.append(ConversationMessage(role=MessageRole.USER, content="Secret data"))
        self.store.save(s1)

        self.store.delete("del_ctx")
        self.assertIsNone(self.store.get("del_ctx"))


# =========================================================================
# Tool Registry Tests
# =========================================================================

class TestToolRegistry(unittest.TestCase):
    """Test tool registry: schemas, routing, confirmation rules."""

    def setUp(self):
        self.registry = DefaultToolRegistry()

    def test_all_visible_tools_have_schemas(self):
        for tool_name in LLM_VISIBLE_TOOLS:
            schema = self.registry.get_tool_schema(tool_name)
            self.assertIsNotNone(schema, f"Missing schema for {tool_name}")
            self.assertEqual(schema["type"], "function")
            self.assertIn("function", schema)
            self.assertEqual(schema["function"]["name"], tool_name)

    def test_get_all_tool_schemas(self):
        schemas = self.registry.get_all_tool_schemas()
        self.assertGreater(len(schemas), 10)  # At least 10 tools exposed
        for s in schemas:
            self.assertEqual(s["type"], "function")

    def test_service_routing(self):
        self.assertEqual(
            self.registry.get_service("search_documents"), ServiceTarget.DOCUMENT
        )
        self.assertEqual(
            self.registry.get_service("classify_document"), ServiceTarget.MODEL
        )
        self.assertEqual(
            self.registry.get_service("validate_document"), ServiceTarget.VALIDATOR
        )
        self.assertEqual(
            self.registry.get_service("web_search"), ServiceTarget.WEB_SEARCH
        )

    def test_unknown_tool_returns_none(self):
        self.assertIsNone(self.registry.get_service("nonexistent_tool"))
        self.assertIsNone(self.registry.get_tool_schema("nonexistent_tool"))

    def test_file_retrieval_requires_confirmation(self):
        for tool in ["get_document", "get_page", "download_file"]:
            ct = self.registry.requires_confirmation(tool, {})
            self.assertEqual(
                ct, ConfirmationType.FILE_RETRIEVAL, f"{tool} should require FILE_RETRIEVAL"
            )

    def test_sensitive_access_requires_confirmation(self):
        for tool in ["get_evidence", "request_sensitive_access"]:
            ct = self.registry.requires_confirmation(tool, {})
            self.assertEqual(
                ct, ConfirmationType.SENSITIVE_ACCESS, f"{tool} should require SENSITIVE_ACCESS"
            )

    def test_search_does_not_require_confirmation(self):
        for tool in ["search_documents", "search_document_content", "list_documents"]:
            ct = self.registry.requires_confirmation(tool, {})
            self.assertIsNone(ct, f"{tool} should NOT require confirmation")

    def test_metadata_access_does_not_require_confirmation(self):
        """Information retrieval should not require confirmation."""
        for tool in ["get_document_metadata", "get_document_description"]:
            ct = self.registry.requires_confirmation(tool, {})
            self.assertIsNone(ct, f"{tool} should NOT require confirmation")

    def test_health_checks_no_confirmation(self):
        for tool in ["health_document", "health_model", "health_validator", "health_web_search"]:
            ct = self.registry.requires_confirmation(tool, {})
            self.assertIsNone(ct)


# =========================================================================
# Confirmation Store Tests
# =========================================================================

class TestConfirmationStore(unittest.TestCase):
    """Test confirmation store: save, get, respond, by-tool-call lookup."""

    def setUp(self):
        self.store = InMemoryConfirmationStore()

    def test_save_and_get(self):
        req = ConfirmationRequest(
            request_id="conf1",
            session_id="s1",
            confirmation_type=ConfirmationType.SENSITIVE_ACCESS,
            message="Access Aadhaar data?",
        )
        self.store.save(req)
        loaded = self.store.get("conf1")
        self.assertIsNotNone(loaded)
        self.assertFalse(loaded.responded)

    def test_respond(self):
        req = ConfirmationRequest(request_id="conf2", session_id="s1")
        self.store.save(req)
        self.store.respond("conf2", approved=True)
        loaded = self.store.get("conf2")
        self.assertTrue(loaded.responded)
        self.assertTrue(loaded.approved)

    def test_get_pending(self):
        self.store.save(ConfirmationRequest(request_id="p1", session_id="s1"))
        self.store.save(ConfirmationRequest(request_id="p2", session_id="s1"))
        self.store.save(ConfirmationRequest(request_id="p3", session_id="s2"))

        self.store.respond("p1", approved=True)  # Not pending anymore

        pending = self.store.get_pending("s1")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].request_id, "p2")

    def test_get_by_tool_call(self):
        req = ConfirmationRequest(
            request_id="conf_tc",
            session_id="s1",
            tool_args={"call_id": "tc_abc"},
        )
        self.store.save(req)
        found = self.store.get_by_tool_call("tc_abc")
        self.assertIsNotNone(found)
        self.assertEqual(found.request_id, "conf_tc")


# =========================================================================
# Service Client Tests
# =========================================================================

class TestServiceClientRouter(unittest.TestCase):
    """Test service client routing (with mocked HTTP)."""

    def setUp(self):
        mock_client = MagicMock(spec=HttpServiceClient)
        mock_client.target = ServiceTarget.DOCUMENT
        mock_client.call_tool.return_value = {"results": ["doc1", "doc2"]}
        mock_client.health.return_value = {"status": "ok"}

        self.router = ServiceClientRouter({ServiceTarget.DOCUMENT: mock_client})
        self.mock_client = mock_client

    def test_routes_to_correct_service(self):
        result = self.router.call_tool(
            ServiceTarget.DOCUMENT, "search_documents", {"query": "test"}
        )
        self.mock_client.call_tool.assert_called_once_with(
            "search_documents", {"query": "test"}
        )
        self.assertEqual(result["results"], ["doc1", "doc2"])

    def test_missing_service_returns_error(self):
        result = self.router.call_tool(
            ServiceTarget.MODEL, "classify_document", {}
        )
        self.assertTrue(result.get("error"))
        self.assertIn("No client", result["message"])

    def test_health_all(self):
        health = self.router.health_all()
        self.assertIn("document_service", health)


# =========================================================================
# Orchestration Engine Tests (with mocked LLM + services)
# =========================================================================

class TestOrchestrationEngine(unittest.TestCase):
    """Test the core orchestration engine with mocked dependencies."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sessions = FileSessionStore(base_dir=os.path.join(self.tmpdir, "sessions"))
        self.confirmations = InMemoryConfirmationStore()
        self.registry = DefaultToolRegistry()

        # Mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = LLMResponse(
            content="I found your documents. You have 2 documents stored.",
            tool_calls=[],
        )

        # Mock service router
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"results": []}
        mock_client.health.return_value = {"status": "ok"}
        self.router = ServiceClientRouter({ServiceTarget.DOCUMENT: mock_client})

        self.engine = OrchestrationEngine(
            llm=self.mock_llm,
            sessions=self.sessions,
            confirmations=self.confirmations,
            tool_registry=self.registry,
            service_router=self.router,
            translation_service=self._no_op_translation(),
        )

    def _no_op_translation(self):
        """Return a translation stub that does nothing.

        Pre-V1 orchestrator tests don't care about language plumbing;
        they just need a translation service to exist so the engine
        doesn't lazily try to construct a real Gemini-backed one.
        """
        import tempfile as _tmp
        from pathlib import Path as _Path

        from model_service.infrastructure.language_preferences import (
            LanguagePreferenceStore,
        )
        from model_service.infrastructure.translation_provider import (
            TranslationResult,
        )

        class _NoOp:
            def __init__(self):
                self._store = LanguagePreferenceStore(
                    _Path(_tmp.mkdtemp()) / "prefs.json"
                )

            def translate_user_input(self, text, session_id):
                return TranslationResult(
                    text=text, source="en", target="en",
                    cached=True, latency_ms=0,
                )

            def translate_bot_reply(self, text, session_id):
                return TranslationResult(
                    text=text, source="en", target="en",
                    cached=True, latency_ms=0,
                )

            def get_preference(self, session_id):
                return self._store.get(session_id)

            def set_preference(self, session_id, *, source=None, target=None):
                return self._store.set(
                    session_id, source=source, target=target
                )

        return _NoOp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_session(self):
        session = self.engine.create_session(user_id="u1", title="Test")
        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.user_id, "u1")

    def test_process_message_direct_response(self):
        session = self.engine.create_session()
        response = self.engine.process_message(session.session_id, "List my documents")

        self.assertEqual(response.session_id, session.session_id)
        self.assertIn("documents", response.message.lower())

        # Verify user message was added
        loaded = self.sessions.get(session.session_id)
        self.assertEqual(len(loaded.messages), 2)  # user + assistant

    def test_process_message_with_tool_calls(self):
        """LLM decides to call a tool, engine executes it."""
        session = self.engine.create_session()

        # First call: LLM wants to call search_documents
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_123",
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "arguments": '{"query": "Aadhaar card"}',
                },
            }],
        )
        # Second call: LLM summarizes the tool result
        summary_response = LLMResponse(
            content="I found 1 Aadhaar card in your documents.",
        )

        self.mock_llm.chat.side_effect = [tool_call_response, summary_response]

        response = self.engine.process_message(session.session_id, "Find my Aadhaar card")

        self.assertIn("Aadhaar", response.message)
        self.assertEqual(len(response.tool_calls_made), 1)
        self.assertEqual(response.tool_calls_made[0].tool_name, "search_documents")
        self.assertEqual(response.tool_calls_made[0].status, ToolCallStatus.SUCCESS)

    def test_sensitive_tool_requires_confirmation(self):
        """Sensitive tools pause and ask for confirmation."""
        session = self.engine.create_session()

        # LLM wants to get protected evidence (sensitive)
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_sensitive",
                "type": "function",
                "function": {
                    "name": "get_evidence",
                    "arguments": '{"document_id": "abc123", "version": 1, "query": "protected detail"}',
                },
            }],
        )
        self.mock_llm.chat.return_value = tool_call_response

        response = self.engine.process_message(session.session_id, "Show me the OCR text")

        # Should return confirmation request, not execute
        self.assertIsNotNone(response.confirmation_required)
        self.assertEqual(
            response.confirmation_required.confirmation_type,
            ConfirmationType.SENSITIVE_ACCESS,
        )

    def test_file_retrieval_requires_confirmation(self):
        """File retrieval tools pause and ask for confirmation."""
        session = self.engine.create_session()

        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_file",
                "type": "function",
                "function": {
                    "name": "get_document",
                    "arguments": '{"document_id": "abc123", "version": 1}',
                },
            }],
        )
        self.mock_llm.chat.return_value = tool_call_response

        response = self.engine.process_message(session.session_id, "Download my document")

        self.assertIsNotNone(response.confirmation_required)
        self.assertEqual(
            response.confirmation_required.confirmation_type,
            ConfirmationType.FILE_RETRIEVAL,
        )

    def test_confirmation_approved_executes(self):
        """After user approves, the gated tool executes."""
        session = self.engine.create_session()

        # Step 1: Get confirmation
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_confirm",
                "type": "function",
                "function": {
                    "name": "get_evidence",
                    "arguments": '{"document_id": "abc123", "version": 1, "query": "protected detail"}',
                },
            }],
        )
        summary_response = LLMResponse(content="Here is the evidence from your document.")

        self.mock_llm.chat.return_value = tool_call_response
        response1 = self.engine.process_message(session.session_id, "Show OCR text")
        self.assertIsNotNone(response1.confirmation_required)

        # Step 2: User approves
        self.mock_llm.chat.return_value = summary_response
        response2 = self.engine.handle_confirmation(
            session.session_id,
            response1.confirmation_required.request_id,
            approved=True,
        )
        self.assertIn("evidence", response2.message)

    def test_confirmation_denied_skips_tool(self):
        """After user denies, the tool is skipped."""
        session = self.engine.create_session()

        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_deny",
                "type": "function",
                "function": {
                    "name": "get_document",
                    "arguments": '{"document_id": "abc123", "version": 1}',
                },
            }],
        )
        denied_response = LLMResponse(content="OK, I won't download the file.")

        self.mock_llm.chat.return_value = tool_call_response
        response1 = self.engine.process_message(session.session_id, "Download doc")
        self.assertIsNotNone(response1.confirmation_required)

        # User denies
        self.mock_llm.chat.return_value = denied_response
        response2 = self.engine.handle_confirmation(
            session.session_id,
            response1.confirmation_required.request_id,
            approved=False,
        )
        self.assertIn("OK", response2.message)

    def test_deny_with_correction_records_correction_on_request(self):
        """A plain deny + correction stores the correction on the"""
        """confirmation request so the LLM can read it later."""
        session = self.engine.create_session()
        tool_call = {
            "id": "tc_wrong",
            "type": "function",
            "function": {
                "name": "get_field_value",
                'arguments': '{"document_id": "doc-1", "version": 1, "field": "phone_number"}',
            },
        }
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tool_call])
        r1 = self.engine.process_message(session.session_id, "phone")
        self.assertIsNotNone(r1.confirmation_required)
        request_id = r1.confirmation_required.request_id
        # User denies with correction
        self.engine.handle_confirmation(session.session_id, request_id, approved=False, correction="pan_number")
        loaded = self.confirmations.get(request_id)
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.responded)
        self.assertFalse(loaded.approved)
        self.assertEqual(loaded.correction, "pan_number")

    def test_deny_with_correction_injects_feedback_into_session(self):
        """A denied confirmation with a correction appends a SYSTEM"""
        """message to the session so the LLM retries the right field."""
        session = self.engine.create_session()
        tool_call = {
            "id": "tc_x",
            "type": "function",
            "function": {
                "name": "get_field_value",
                'arguments': '{"document_id": "doc-1", "version": 1, "field": "phone_number"}',
            },
        }
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tool_call])
        r1 = self.engine.process_message(session.session_id, "phone")
        self.assertIsNotNone(r1.confirmation_required)
        # The denial triggers _plan_and_execute again. The mock LLM
        # needs to return something to keep the loop from re-asking.
        self.mock_llm.chat.return_value = LLMResponse(content="OK")
        self.engine.handle_confirmation(
            session.session_id, r1.confirmation_required.request_id,
            approved=False, correction="pan_number",
        )
        # The SYSTEM note pointing the LLM at the correct field should
        # be somewhere in the session messages (it gets followed by
        # an assistant turn produced by _plan_and_execute).
        msgs = self.engine.get_session(session.session_id).messages
        system_msgs = [m for m in msgs if m.role.value == "system"]
        self.assertTrue(system_msgs, "expected a system message with the correction")
        last_system = system_msgs[-1]
        self.assertIn("phone_number", last_system.content)
        self.assertIn("pan_number", last_system.content)

    def test_deny_without_correction_uses_plain_user_message(self):
        """A plain deny (no correction) falls back to the user-style"""
        """No, do not do that. message."""
        session = self.engine.create_session()
        tool_call = {
            "id": "tc_d",
            "type": "function",
            "function": {
                "name": "get_document",
                'arguments': '{"document_id": "doc-1", "version": 1}',
            },
        }
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tool_call])
        r1 = self.engine.process_message(session.session_id, "download")
        self.assertIsNotNone(r1.confirmation_required)
        self.mock_llm.chat.return_value = LLMResponse(content="OK")
        self.engine.handle_confirmation(
            session.session_id, r1.confirmation_required.request_id,
            approved=False,
        )
        msgs = self.engine.get_session(session.session_id).messages
        user_msgs = [m for m in msgs if m.role.value == "user" and "do that" in m.content]
        self.assertTrue(user_msgs, "expected a user-style deny message")

    def test_field_approval_cache_skips_second_prompt(self):
        """After approving get_field_value for a specific (doc, ver, field),
        the second call for the same key should execute without prompting.
        """
        session = self.engine.create_session()
        tool_call = {
            "id": "tc_fv_1",
            "type": "function",
            "function": {
                "name": "get_field_value",
                "arguments": chr(123) + chr(34) + "document_id" + chr(34) + ": " + chr(34) + "doc-1" + chr(34) + ", " + chr(34) + "version" + chr(34) + ": 1, " + chr(34) + "field" + chr(34) + ": " + chr(34) + "aadhaar_number" + chr(34) + ", " + chr(34) + "confirm" + chr(34) + ": false" + chr(125),
            },
        }
        # First call: should prompt
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tool_call])
        r1 = self.engine.process_message(session.session_id, "what is my aadhaar number?")
        self.assertIsNotNone(r1.confirmation_required, "first call should still prompt")
        # User approves
        self.mock_llm.chat.return_value = LLMResponse(content="Here is the value")
        self.engine.handle_confirmation(
            session.session_id, r1.confirmation_required.request_id, approved=True,
        )
        # Second call (same doc, same field) within TTL: should NOT prompt
        tool_call_2 = dict(tool_call)
        tool_call_2["id"] = "tc_fv_2"
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tool_call_2])
        r2 = self.engine.process_message(session.session_id, "show me again")
        self.assertIsNone(r2.confirmation_required, "second call within TTL should NOT prompt")
    def test_field_approval_cache_different_field_still_prompts(self):
        """Approval for one field does not carry to a different field."""
        session = self.engine.create_session()

        def make_call(call_id, field):
            args = (
                chr(123) + "document_id: doc-1, version: 1, field: "
                + field
                + ", confirm: false" + chr(125)
            )
            args = "{" + args + "}"
            return LLMResponse(
                content="",
                tool_calls=[{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "get_field_value",
                        "arguments": args,
                    },
                }],
            )

        self.mock_llm.chat.return_value = make_call("t1", "aadhaar_number")
        r1 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r1.confirmation_required)
        self.mock_llm.chat.return_value = LLMResponse(content="done")
        self.engine.handle_confirmation(
            session.session_id, r1.confirmation_required.request_id, approved=True
        )
        self.mock_llm.chat.return_value = make_call("t2", "name")
        r2 = self.engine.process_message(session.session_id, "name?")
        self.assertIsNotNone(
            r2.confirmation_required, "different field should still prompt"
        )

    def test_field_approval_cache_different_doc_still_prompts(self):
        """Approval for one document does not carry to a different document."""
        session = self.engine.create_session()
        def make_call(call_id, doc_id):
            args = '{"document_id": "' + doc_id + '", "version": 1, "field": "aadhaar_number", "confirm": false}'
            return LLMResponse(content="", tool_calls=[{"id": call_id, "type": "function", "function": {"name": "get_field_value", "arguments": args}}])
        self.mock_llm.chat.return_value = make_call("t1", "doc-A")
        r1 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r1.confirmation_required)
        self.mock_llm.chat.return_value = LLMResponse(content="done")
        self.engine.handle_confirmation(session.session_id, r1.confirmation_required.request_id, approved=True)
        self.mock_llm.chat.return_value = make_call("t2", "doc-B")
        r2 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r2.confirmation_required, "different document should still prompt")

    def test_field_approval_cache_deny_does_not_populate(self):
        """Denying a confirmation does not cache an approval."""
        session = self.engine.create_session()
        tc = {"id": "tc_fv", "type": "function", "function": {"name": "get_field_value", "arguments": '{"document_id": "doc-1", "version": 1, "field": "aadhaar_number", "confirm": false}'}}
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tc])
        r1 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r1.confirmation_required)
        self.mock_llm.chat.return_value = LLMResponse(content="OK")
        self.engine.handle_confirmation(session.session_id, r1.confirmation_required.request_id, approved=False)
        tc2 = dict(tc); tc2["id"] = "tc_fv_2"
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tc2])
        r2 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r2.confirmation_required, "denial should not populate cache")

    def test_field_approval_cache_ttl_expiry(self):
        """After TTL expires, the next call should re-prompt."""
        import time as _t
        self.engine._field_approval_ttl_seconds = 0
        session = self.engine.create_session()
        tc = {"id": "tc_fv", "type": "function", "function": {"name": "get_field_value", "arguments": '{"document_id": "doc-1", "version": 1, "field": "aadhaar_number", "confirm": false}'}}
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tc])
        r1 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r1.confirmation_required)
        self.mock_llm.chat.return_value = LLMResponse(content="done")
        self.engine.handle_confirmation(session.session_id, r1.confirmation_required.request_id, approved=True)
        _t.sleep(0.05)
        tc2 = dict(tc); tc2["id"] = "tc_fv_2"
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tc2])
        r2 = self.engine.process_message(session.session_id, "aadhaar?")
        self.assertIsNotNone(r2.confirmation_required, "expired TTL should re-prompt")

    def test_field_approval_cache_only_for_get_field_value(self):
        """Other gated tools (get_evidence, get_document) should still prompt on every call. Only get_field_value is cacheable."""
        session = self.engine.create_session()
        tc = {"id": "tc_evidence", "type": "function", "function": {"name": "get_evidence", "arguments": '{"document_id": "doc-1", "version": 1, "query": "ocr"}'}}
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tc])
        r1 = self.engine.process_message(session.session_id, "show evidence")
        self.assertIsNotNone(r1.confirmation_required)
        self.mock_llm.chat.return_value = LLMResponse(content="done")
        self.engine.handle_confirmation(session.session_id, r1.confirmation_required.request_id, approved=True)
        tc2 = dict(tc); tc2["id"] = "tc_evidence_2"
        self.mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[tc2])
        r2 = self.engine.process_message(session.session_id, "show evidence again")
        self.assertIsNotNone(r2.confirmation_required, "get_evidence should still prompt")


    def test_delete_session(self):
        session = self.engine.create_session()
        self.assertTrue(self.engine.delete_session(session.session_id))
        self.assertIsNone(self.engine.get_session(session.session_id))

    def test_session_not_found(self):
        response = self.engine.process_message("nonexistent", "Hello")
        self.assertIn("not found", response.message.lower())

    def test_list_sessions(self):
        self.engine.create_session(title="S1")
        self.engine.create_session(title="S2")
        sessions = self.engine.list_sessions()
        self.assertEqual(len(sessions), 2)


# =========================================================================
# LLM Provider Message Building Tests
# =========================================================================

class TestLLMMessageBuilding(unittest.TestCase):
    """Test message conversion for the LLM API."""

    def setUp(self):
        # We test the message builder without actually calling the API
        from orchestrator_service.infrastructure.llm_provider import QwenLLMProvider
        self.provider = QwenLLMProvider(token="fake-token")

    def test_build_api_messages(self):
        messages = [
            ConversationMessage(role=MessageRole.SYSTEM, content="You are helpful."),
            ConversationMessage(role=MessageRole.USER, content="Hello"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Hi!"),
        ]
        api_msgs = self.provider._build_api_messages(messages)
        self.assertEqual(len(api_msgs), 3)
        self.assertEqual(api_msgs[0]["role"], "system")
        self.assertEqual(api_msgs[1]["role"], "user")
        self.assertEqual(api_msgs[2]["role"], "assistant")

    def test_tool_result_messages(self):
        tc = ToolCall(call_id="tc1", tool_name="search", arguments={})
        messages = [
            ConversationMessage(
                role=MessageRole.ASSISTANT, content="", tool_calls=[tc]
            ),
            ConversationMessage(
                role=MessageRole.TOOL,
                content="{}",
                tool_call_id="tc1",
            ),
        ]
        api_msgs = self.provider._build_api_messages(messages)
        self.assertEqual(len(api_msgs), 2)
        self.assertEqual(api_msgs[0]["tool_calls"][0]["id"], "tc1")
        self.assertEqual(api_msgs[1]["role"], "tool")
        self.assertEqual(api_msgs[1]["tool_call_id"], "tc1")


# =========================================================================
# Domain Model Tests
# =========================================================================

# Shared google.genai.types fake (see tests/_genai_stub.py). This used to be
# a local stub incompatible with test_model_service.py's copy, which made
# these tests fail in full-suite runs while passing solo.
from _genai_stub import ensure_genai_stub as _ensure_genai_stub

_ensure_genai_stub()


class TestGeminiLLMProvider(unittest.TestCase):
    """Tests for the Gemini LLM provider. No network. The google.genai SDK
    is faked via a direct _client swap on the adapter."""

    def _make_provider(self, **overrides):
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            GeminiLLMProvider,
        )
        defaults = dict(api_key="test-key", model_id="gemini-2.5-flash-lite")
        defaults.update(overrides)
        return GeminiLLMProvider(**defaults)

    def _install_fake_client(self, provider, fake_models):
        class _Client:
            pass
        c = _Client()
        c.models = fake_models
        provider._client = c
        return c

    def test_provider_imports(self) -> None:
        provider = self._make_provider()
        self.assertEqual(provider._model_id, "gemini-2.5-flash-lite")

    def test_chat_text_only_response(self) -> None:
        class _Part:
            def __init__(self, text):
                self.text = text

        class _Content:
            def __init__(self, parts):
                self.parts = parts

        class _Candidate:
            def __init__(self, parts):
                self.content = _Content(parts)

        class _Usage:
            prompt_token_count = 10
            candidates_token_count = 4

        class _Response:
            candidates = [_Candidate([_Part("hi back")])]
            usage_metadata = _Usage()
            model = "gemini-2.5-flash-lite"

        class _Models:
            def __init__(self):
                self.last = None

            def generate_content(self, *, model, contents, config):
                self.last = {"model": model, "contents": contents, "config": config}
                return _Response()

        fake = _Models()
        provider = self._make_provider()
        self._install_fake_client(provider, fake)

        messages = [
            ConversationMessage(role=MessageRole.SYSTEM, content="be terse"),
            ConversationMessage(role=MessageRole.USER, content="hello"),
        ]
        resp = provider.chat(messages)
        self.assertEqual(resp.content, "hi back")
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.model, "gemini-2.5-flash-lite")
        self.assertEqual(resp.token_usage, {"input": 10, "output": 4})
        self.assertEqual(fake.last["config"].system_instruction, "be terse")
        user_contents = [c for c in fake.last["contents"] if c.role == "user"]
        self.assertTrue(user_contents)

    def test_chat_with_function_call(self) -> None:
        class _Part:
            def __init__(self, fc=None, text=None):
                self.text = text
                self.function_call = fc

        class _Content:
            def __init__(self, parts):
                self.parts = parts

        class _Candidate:
            def __init__(self, parts):
                self.content = _Content(parts)

        class _FC:
            name = "search_documents"
            args = {"query": "aadhaar"}

        class _Response:
            candidates = [_Candidate([_Part(fc=_FC())])]
            usage_metadata = None
            model = "gemini-2.5-flash-lite"

        class _Models:
            def generate_content(self, *, model, contents, config):
                return _Response()

        provider = self._make_provider()
        self._install_fake_client(provider, _Models())

        tools = [{
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }]
        messages = [ConversationMessage(role=MessageRole.USER, content="find aadhaar")]
        resp = provider.chat(messages, tools=tools)
        self.assertEqual(len(resp.tool_calls), 1)
        tc = resp.tool_calls[0]
        self.assertEqual(tc["function"]["name"], "search_documents")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"query": "aadhaar"})
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertEqual(tc["type"], "function")

    def test_chat_without_tools_omits_tool_config(self) -> None:
        captured = {}

        class _Models:
            def generate_content(self, *, model, contents, config):
                captured["config"] = config
                class _R:
                    candidates = []
                    usage_metadata = None
                    model = "gemini-2.5-flash-lite"
                return _R()

        provider = self._make_provider()
        self._install_fake_client(provider, _Models())
        provider.chat([ConversationMessage(role=MessageRole.USER, content="hi")])
        # The GenerateContentConfig only carries fields we actually set.
        self.assertFalse(hasattr(captured["config"], "tools"))

    def test_health_returns_status_dict(self) -> None:
        class _Models:
            def generate_content(self, *, model, contents, config):
                class _R:
                    candidates = []
                    usage_metadata = None
                    model = "gemini-2.5-flash-lite"
                return _R()

        provider = self._make_provider()
        self._install_fake_client(provider, _Models())
        h = provider.health()
        self.assertEqual(h["status"], "healthy")
        self.assertEqual(h["provider"], "gemini")


class TestGeminiThoughtSignature(unittest.TestCase):
    """Gemini 2.5+ requires replayed function calls to carry their
    thought_signature, else follow-up turns fail with 400
    INVALID_ARGUMENT. These tests pin the capture → persist → replay
    chain plus the one-time degraded retry."""

    def _make_provider(self):
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            GeminiLLMProvider,
        )
        return GeminiLLMProvider(api_key="test-key", model_id="gemini-2.5-flash-lite")

    def _install_fake_client(self, provider, fake_models):
        class _Client:
            pass
        c = _Client()
        c.models = fake_models
        provider._client = c
        return c

    def _fc_response(self, sig=b"opaque-sig-bytes"):
        class _FC:
            name = "get_field_value"
            args = {"document_id": "d1", "version": 1, "field": "aadhaar_number"}

        class _Part:
            def __init__(self):
                self.text = None
                self.function_call = _FC()
                self.thought_signature = sig

        class _Content:
            parts = [_Part()]

        class _Candidate:
            content = _Content()

        class _Response:
            candidates = [_Candidate()]
            usage_metadata = None
            model = "gemini-2.5-flash-lite"

        return _Response()

    def _text_response(self, text="done"):
        class _Part:
            def __init__(self):
                self.text = text
                self.function_call = None

        class _Content:
            parts = [_Part()]

        class _Candidate:
            content = _Content()

        class _Response:
            candidates = [_Candidate()]
            usage_metadata = None
            model = "gemini-2.5-flash-lite"

        return _Response()

    def test_capture_thought_signature(self):
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            _gemini_response_to_llm,
        )
        resp = _gemini_response_to_llm(self._fc_response(), "m")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(
            resp.tool_calls[0]["thought_signature"],
            base64.b64encode(b"opaque-sig-bytes").decode("ascii"),
        )

    def test_missing_signature_captures_none(self):
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            _gemini_response_to_llm,
        )

        class _FC:
            name = "get_field_value"
            args = {}

        class _Part:
            text = None
            function_call = _FC()
            # no thought_signature attribute at all

        class _Content:
            parts = [_Part()]

        class _Candidate:
            content = _Content()

        class _Response:
            candidates = [_Candidate()]
            usage_metadata = None
            model = "m"

        resp = _gemini_response_to_llm(_Response(), "m")
        self.assertIsNone(resp.tool_calls[0]["thought_signature"])

    def test_history_replays_signature(self):
        from orchestrator_service.domain.models import MessageRole, ToolCall
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            _domain_messages_to_gemini_contents,
        )
        sig_b64 = base64.b64encode(b"opaque-sig-bytes").decode("ascii")
        messages = [
            ConversationMessage(role=MessageRole.USER, content="show number"),
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[ToolCall(
                    call_id="c1", tool_name="get_field_value",
                    arguments={"field": "aadhaar_number"},
                    thought_signature=sig_b64,
                )],
            ),
        ]
        _, contents = _domain_messages_to_gemini_contents(messages)
        model_parts = [
            p for c in contents if c.role == "model" for p in c.parts
        ]
        fc_parts = [p for p in model_parts if p.function_call is not None]
        self.assertEqual(len(fc_parts), 1)
        self.assertEqual(fc_parts[0].thought_signature, b"opaque-sig-bytes")

    def test_strip_function_calls(self):
        from orchestrator_service.domain.models import MessageRole, ToolCall
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            _domain_messages_to_gemini_contents,
        )
        messages = [
            ConversationMessage(role=MessageRole.USER, content="show number"),
            ConversationMessage(
                role=MessageRole.ASSISTANT, content="",
                tool_calls=[ToolCall(call_id="c1", tool_name="get_field_value")],
            ),
            ConversationMessage(
                role=MessageRole.TOOL, content='{"status": "ok"}',
                tool_call_id="c1",
            ),
        ]
        _, contents = _domain_messages_to_gemini_contents(
            messages, strip_function_calls=True
        )
        for content in contents:
            for part in content.parts:
                # getattr: the test-suite genai stub only sets attrs that
                # were explicitly passed to the constructor.
                self.assertIsNone(getattr(part, "function_call", None))
                self.assertIsNone(getattr(part, "function_response", None))

    def test_retry_on_thought_signature_400(self):
        calls = {"n": 0}
        ok_response = self._text_response()

        class _Models:
            def generate_content(self, *, model, contents, config):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise Exception(
                        "400 INVALID_ARGUMENT. {'error': {'message': "
                        "'Function call is missing a thought_signature'}}"
                    )
                return ok_response

        provider = self._make_provider()
        self._install_fake_client(provider, _Models())
        resp = provider.chat([ConversationMessage(role=MessageRole.USER, content="hi")])
        self.assertEqual(calls["n"], 2)
        self.assertEqual(resp.content, "done")

    def test_no_retry_on_other_errors(self):
        calls = {"n": 0}

        class _Models:
            def generate_content(self, *, model, contents, config):
                calls["n"] += 1
                raise ValueError("boom")

        provider = self._make_provider()
        self._install_fake_client(provider, _Models())
        resp = provider.chat([ConversationMessage(role=MessageRole.USER, content="hi")])
        self.assertEqual(calls["n"], 1)
        self.assertTrue(resp.content.startswith("Error:"))

    def test_session_store_round_trip(self):
        import tempfile
        from orchestrator_service.domain.models import (
            MessageRole, ToolCall, ToolCallStatus,
        )
        from orchestrator_service.infrastructure.session_store import (
            FileSessionStore,
        )
        tmp = tempfile.mkdtemp()
        try:
            store = FileSessionStore(base_dir=tmp)
            session = self._session_with_call()
            store.save(session)
            loaded = store.get(session.session_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            tcs = loaded.messages[0].tool_calls
            self.assertEqual(len(tcs), 1)
            self.assertEqual(
                tcs[0].thought_signature,
                base64.b64encode(b"opaque-sig-bytes").decode("ascii"),
            )
            self.assertEqual(tcs[0].status, ToolCallStatus.PENDING)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _session_with_call(self):
        from orchestrator_service.domain.models import (
            MessageRole, Session, ToolCall,
        )
        session = Session(user_id="u", title="t")
        session.messages.append(ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[ToolCall(
                call_id="c1", tool_name="get_field_value",
                thought_signature=base64.b64encode(b"opaque-sig-bytes").decode("ascii"),
            )],
        ))
        return session

    def test_is_thought_signature_error(self):
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            _is_thought_signature_error,
        )
        self.assertTrue(_is_thought_signature_error(
            Exception("400 INVALID_ARGUMENT: missing a thought_signature")))
        self.assertTrue(_is_thought_signature_error(
            Exception("missing a thought signature in parts")))
        self.assertFalse(_is_thought_signature_error(ValueError("boom")))
        self.assertFalse(_is_thought_signature_error(Exception("429 rate limit")))


class TestDomainModels(unittest.TestCase):
    """Test domain model defaults and values."""

    def test_session_defaults(self):
        s = Session()
        self.assertEqual(s.status, SessionStatus.ACTIVE)
        self.assertEqual(len(s.messages), 0)

    def test_plan_defaults(self):
        p = Plan()
        self.assertEqual(p.status, PlanStatus.DRAFT)
        self.assertEqual(len(p.steps), 0)

    def test_tool_call_defaults(self):
        tc = ToolCall()
        self.assertEqual(tc.status, ToolCallStatus.PENDING)

    def test_confirmation_request_defaults(self):
        cr = ConfirmationRequest()
        self.assertFalse(cr.responded)
        self.assertFalse(cr.approved)

    def test_session_status_values(self):
        self.assertEqual(SessionStatus.ACTIVE.value, "active")
        self.assertEqual(SessionStatus.ARCHIVED.value, "archived")
        self.assertEqual(SessionStatus.DELETED.value, "deleted")

    def test_confirmation_type_values(self):
        self.assertEqual(ConfirmationType.SENSITIVE_ACCESS.value, "sensitive_access")
        self.assertEqual(ConfirmationType.FILE_RETRIEVAL.value, "file_retrieval")
        self.assertEqual(ConfirmationType.EXTERNAL_ACTION.value, "external_action")


class TestSessionRetitle(unittest.TestCase):
    """First user message titles a default-named session (and only then)."""

    def setUp(self):
        import shutil
        import tempfile as _tmp

        from orchestrator_service.domain.models import ServiceTarget
        from orchestrator_service.infrastructure.file_confirmation_store import (
            FileConfirmationStore,
        )
        from orchestrator_service.infrastructure.session_store import (
            FileSessionStore,
        )
        from orchestrator_service.infrastructure.service_clients import (
            ServiceClientRouter,
        )
        from orchestrator_service.infrastructure.tool_registry import (
            DefaultToolRegistry,
        )

        self.tmpdir = _tmp.mkdtemp()
        sessions = FileSessionStore(base_dir=os.path.join(self.tmpdir, "sessions"))
        confirmations = FileConfirmationStore(
            base_dir=os.path.join(self.tmpdir, "confirmations"))
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(content="ok", tool_calls=[])
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"results": []}
        self.engine = OrchestrationEngine(
            llm=mock_llm,
            sessions=sessions,
            confirmations=confirmations,
            tool_registry=DefaultToolRegistry(),
            service_router=ServiceClientRouter({ServiceTarget.DOCUMENT: mock_client}),
            translation_service=TestOrchestrationEngine._no_op_translation(self),
        )
        self.sessions = sessions
        self._shutil = shutil

    def tearDown(self):
        self._shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_first_message_retitles_default_session(self):
        session = self.engine.create_session(title="New chat")
        self.engine.process_message(session.session_id, "What is my aadhaar number?")
        reloaded = self.sessions.get(session.session_id)
        self.assertEqual(reloaded.title, "What is my aadhaar number?")

    def test_custom_title_is_preserved(self):
        session = self.engine.create_session(title="Tax stuff")
        self.engine.process_message(session.session_id, "What is my aadhaar number?")
        reloaded = self.sessions.get(session.session_id)
        self.assertEqual(reloaded.title, "Tax stuff")

    def test_second_message_does_not_retitle(self):
        session = self.engine.create_session(title="New chat")
        self.engine.process_message(session.session_id, "First question here")
        self.engine.process_message(session.session_id, "Second question here")
        reloaded = self.sessions.get(session.session_id)
        self.assertEqual(reloaded.title, "First question here")

    def test_long_title_is_truncated(self):
        session = self.engine.create_session(title="")
        self.engine.process_message(session.session_id, "x" * 100)
        reloaded = self.sessions.get(session.session_id)
        self.assertTrue(len(reloaded.title) <= 48)
        self.assertTrue(reloaded.title.endswith("…"))


if __name__ == "__main__":
    unittest.main()
