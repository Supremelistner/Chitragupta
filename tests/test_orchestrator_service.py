"""Tests for the Orchestrator Service.

Covers:
- Session management (create, save, delete, context isolation)
- Tool registry (schemas, service routing, confirmation rules)
- Confirmation flow (sensitive access, file retrieval)
- Orchestration engine (plan → confirm → execute loop)
- LLM provider (message building)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

# -- Test setup --
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator_service.domain.models import (
    ConfirmationRequest,
    ConfirmationType,
    ConversationMessage,
    MessageRole,
    Plan,
    PlanStatus,
    PlanStep,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from orchestrator_service.infrastructure.session_store import FileSessionStore
from orchestrator_service.infrastructure.confirmation_store import InMemoryConfirmationStore
from orchestrator_service.infrastructure.tool_registry import (
    DefaultToolRegistry,
    LLM_VISIBLE_TOOLS,
)
from orchestrator_service.infrastructure.service_clients import (
    HttpServiceClient,
    ServiceClientRouter,
)
from orchestrator_service.domain.models import ServiceTarget
from orchestrator_service.application.orchestrator import OrchestrationEngine
from orchestrator_service.domain.ports import LLMResponse


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
        )

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


if __name__ == "__main__":
    unittest.main()
