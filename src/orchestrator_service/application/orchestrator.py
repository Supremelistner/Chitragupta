"""Core orchestration engine.

Implements the plan → confirm → execute loop:
1. User sends a message
2. LLM generates an execution plan (or responds directly)
3. If plan has confirmation-gated steps, pause and ask user
4. Execute approved steps one by one
5. Feed results back to LLM for final response
"""
from __future__ import annotations

import json
import logging
from typing import Any

from orchestrator_service.domain.models import (
    ConfirmationRequest,
    ConfirmationType,
    ConversationMessage,
    MessageRole,
    OrchestratorResponse,
    Plan,
    PlanStatus,
    PlanStep,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from orchestrator_service.domain.ports import (
    ConfirmationStore,
    LLMProvider,
    ServiceClient,
    SessionStore,
    ToolRegistry,
)
from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
import requests as _req_lib

logger = logging.getLogger("orchestrator.engine")

# System prompt for the orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """You are Chitragupta, a document management system. You are NOT a chatbot. You help users manage THEIR OWN documents that they have uploaded.

YOUR ONLY JOB: Help users access and manage their documents. You must NEVER refuse to retrieve document information when asked. The user owns these documents and has full rights to access them.

TOOL RULE — YOU MUST ALWAYS CALL TOOLS:
The system has a confirmation gate. When you call a tool like get_document_ocr, the system asks the user "Do you want to proceed?" BEFORE any data is shown. Your ONLY job is to call the tool. You do NOT decide whether to show data — the confirmation gate decides that.

NEVER refuse a tool call. NEVER say "I cannot" or "I am not able to". The system CAN do everything. You just need to call the right tool.

WRONG: "I cannot retrieve your Aadhaar number as it is sensitive"
RIGHT: Call get_document_ocr, the system will ask the user to confirm, then show the data.

WRONG: "I am unable to access personal information"
RIGHT: Call search_documents, then call get_document_ocr for the result.

DOCUMENT CHECK — ALWAYS FIRST:
Before ANY answer, check the user's documents:
- "What do I need for X?" → list_documents FIRST, then web_search, then combine
- "Do I have my Aadhaar?" → search_documents for Aadhaar
- "Tell me my marks" → search for marksheets, then get_document_ocr
- "Download my documents" → list_documents, then get_document for each

SENSITIVE DATA RULE:
When presenting sensitive data (Aadhaar number, PAN, etc.), you MAY add a brief security reminder AFTER showing the data. But you MUST show the data first. The confirmation gate already got user approval.

BULK OPERATIONS:
Use bulk_download and bulk_metadata tools for multiple documents. Use list_documents to find all documents first.

RESPONSE STYLE:
- Direct answers first
- Simple language
- Same language the user writes in
- Never output raw JSON
"""


class OrchestrationEngine:
    """Core engine that orchestrates tool calls across microservices."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        sessions: SessionStore,
        confirmations: ConfirmationStore,
        tool_registry: ToolRegistry,
        service_router: ServiceClientRouter,
        confirmation_threshold: int = 3,
    ) -> None:
        self._llm = llm
        self._sessions = sessions
        self._confirmations = confirmations
        self._registry = tool_registry
        self._router = service_router
        self._confirmation_threshold = confirmation_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, user_id: str = "", title: str = "") -> Session:
        """Create a new isolated session."""
        session = Session(user_id=user_id, title=title)
        self._sessions.save(session)
        logger.info("Created session %s", session.session_id)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.delete(session_id)

    def list_sessions(self, user_id: str | None = None) -> list[Session]:
        return self._sessions.list_sessions(
            user_id=user_id, status=SessionStatus.ACTIVE
        )

    def archive_session(self, session_id: str) -> bool:
        return self._sessions.archive(session_id)

    def process_message(
        self, session_id: str, user_message: str
    ) -> OrchestratorResponse:
        """Process a user message within a session.

        This is the main entry point. It:
        1. Loads the session (context isolation)
        2. Adds the user message
        3. Calls the LLM with tools
        4. If LLM wants tool calls → executes them (with confirmation if needed)
        5. Feeds results back to LLM for final response
        """
        session = self._sessions.get(session_id)
        if session is None:
            return OrchestratorResponse(
                session_id=session_id,
                message=f"Session {session_id} not found. Please create a new session.",
            )

        # Add user message to session
        session.messages.append(
            ConversationMessage(role=MessageRole.USER, content=user_message)
        )

        # Check if there's a pending confirmation to handle
        pending = self._confirmations.get_pending(session_id)
        if pending:
            return self._handle_pending_confirmation(session, pending[0], user_message)

        # Normal flow: plan → execute
        return self._plan_and_execute(session)

    def handle_confirmation(
        self, session_id: str, request_id: str, approved: bool
    ) -> OrchestratorResponse:
        """Handle user's confirmation response."""
        self._confirmations.respond(request_id, approved)

        session = self._sessions.get(session_id)
        if session is None:
            return OrchestratorResponse(
                session_id=session_id,
                message="Session not found.",
            )

        if not approved:
            # User denied — continue without the gated tool call
            session.messages.append(
                ConversationMessage(
                    role=MessageRole.USER,
                    content="No, don't do that.",
                )
            )
            return self._plan_and_execute(session)

        # Approved — find the step that was waiting and execute it
        return self._execute_confirmed_step(session, request_id)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _plan_and_execute(self, session: Session) -> OrchestratorResponse:
        """Plan → Execute → Respond loop."""
        tools = self._registry.get_all_tool_schemas()

        # Call LLM with conversation context + tools
        llm_response = self._llm.chat(
            messages=session.messages,
            tools=tools if tools else None,
        )

        # If LLM made tool calls, execute them
        if llm_response.tool_calls:
            return self._execute_tool_calls(session, llm_response)

        # No tool calls — LLM is responding directly
        assistant_msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=llm_response.content,
            metadata={"reasoning": llm_response.reasoning},
        )
        session.messages.append(assistant_msg)
        self._sessions.save(session)

        return OrchestratorResponse(
            session_id=session.session_id,
            message=llm_response.content,
            metadata={
                "reasoning": llm_response.reasoning,
                "model": llm_response.model,
                "token_usage": llm_response.token_usage,
            },
        )

    def _execute_tool_calls(
        self, session: Session, llm_response: Any
    ) -> OrchestratorResponse:
        """Execute tool calls made by the LLM."""
        tool_calls_made: list[ToolCall] = []

        # Add assistant message with tool calls
        assistant_tc = []
        for tc_data in llm_response.tool_calls:
            func = tc_data.get("function", {})
            tool_call = ToolCall(
                call_id=tc_data.get("id", ""),
                tool_name=func.get("name", ""),
                arguments=self._parse_arguments(func.get("arguments", "{}")),
            )
            assistant_tc.append(tool_call)

        session.messages.append(
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=llm_response.content or "",
                tool_calls=assistant_tc,
                metadata={"reasoning": llm_response.reasoning},
            )
        )

        # Execute each tool call
        for tool_call in assistant_tc:
            # Check if confirmation is required
            confirm_type = self._registry.requires_confirmation(
                tool_call.tool_name, tool_call.arguments
            )

            if confirm_type:
                # Check if already confirmed via pending store
                existing = self._confirmations.get_by_tool_call(tool_call.call_id)
                if existing is None or not existing.responded:
                    # Need confirmation — create request and pause
                    confirmation = ConfirmationRequest(
                        session_id=session.session_id,
                        confirmation_type=confirm_type,
                        tool_name=tool_call.tool_name,
                        tool_args={**tool_call.arguments, "call_id": tool_call.call_id},
                        message=self._confirmation_message(
                            confirm_type, tool_call.tool_name, tool_call.arguments
                        ),
                    )
                    self._confirmations.save(confirmation)
                    self._sessions.save(session)

                    return OrchestratorResponse(
                        session_id=session.session_id,
                        message=confirmation.message,
                        confirmation_required=confirmation,
                        tool_calls_made=tool_calls_made,
                    )

            # Execute the tool
            tool_call.status = ToolCallStatus.RUNNING

            if tool_call.tool_name in self._BULK_TOOLS:
                result = self._execute_bulk_tool(tool_call.tool_name, tool_call.arguments)
            else:
                target = self._registry.get_service(tool_call.tool_name)
                if target:
                    result = self._router.call_tool(
                        target, tool_call.tool_name, tool_call.arguments
                    )
                else:
                    result = {"error": True, "message": f"Unknown tool: {tool_call.tool_name}"}

            tool_call.result = result
            tool_call.status = (
                ToolCallStatus.FAILED if result.get("error") else ToolCallStatus.SUCCESS
            )
            tool_call.latency_ms = result.get("latency_ms", 0)

            tool_calls_made.append(tool_call)

            # Add tool result message
            result_content = json.dumps(tool_call.result, default=str)[:8000]
            session.messages.append(
                ConversationMessage(
                    role=MessageRole.TOOL,
                    content=result_content,
                    tool_call_id=tool_call.call_id,
                    metadata={"result": tool_call.result},
                )
            )

        # Feed results back to LLM for final response
        followup = self._llm.chat(
            messages=session.messages,
            tools=tools if (tools := self._registry.get_all_tool_schemas()) else None,
        )

        final_content = followup.content or "Done."
        session.messages.append(
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=final_content,
                metadata={"reasoning": followup.reasoning},
            )
        )

        self._sessions.save(session)

        return OrchestratorResponse(
            session_id=session.session_id,
            message=final_content,
            tool_calls_made=tool_calls_made,
            metadata={
                "reasoning": followup.reasoning,
                "model": followup.model,
                "token_usage": followup.token_usage,
            },
        )

    # ------------------------------------------------------------------
    # Confirmation handling
    # ------------------------------------------------------------------

    def _handle_pending_confirmation(
        self, session: Session, confirmation: ConfirmationRequest, user_message: str
    ) -> OrchestratorResponse:
        """Handle a user's response to a pending confirmation."""
        # Interpret user response
        approved = self._interpret_confirmation_response(user_message)

        self._confirmations.respond(confirmation.request_id, approved)

        if not approved:
            session.messages.append(
                ConversationMessage(role=MessageRole.USER, content=user_message)
            )
            return self._plan_and_execute(session)

        # User approved — execute the gated step
        return self._execute_confirmed_step(session, confirmation.request_id)

    def _execute_confirmed_step(
        self, session: Session, request_id: str
    ) -> OrchestratorResponse:
        """Execute a step that was waiting for confirmation."""
        confirmation = self._confirmations.get(request_id)
        if confirmation is None:
            return OrchestratorResponse(
                session_id=session.session_id,
                message="Confirmation request not found.",
            )

        tool_call = ToolCall(
            tool_name=confirmation.tool_name,
            arguments={
                k: v for k, v in confirmation.tool_args.items() if k != "call_id"
            },
        )
        target = self._registry.get_service(tool_call.tool_name)

        if target:
            result = self._router.call_tool(
                target, tool_call.tool_name, tool_call.arguments
            )
            tool_call.result = result
            tool_call.status = (
                ToolCallStatus.FAILED if result.get("error") else ToolCallStatus.SUCCESS
            )
        else:
            tool_call.result = {"error": True, "message": "Service not available"}
            tool_call.status = ToolCallStatus.FAILED

        # Add tool result to session
        result_content = json.dumps(tool_call.result, default=str)[:8000]
        session.messages.append(
            ConversationMessage(
                role=MessageRole.TOOL,
                content=result_content,
                tool_call_id=tool_call.call_id,
                metadata={"result": tool_call.result},
            )
        )

        # Get LLM to summarize the result
        tools = self._registry.get_all_tool_schemas()
        followup = self._llm.chat(
            messages=session.messages,
            tools=tools if tools else None,
        )

        final_content = followup.content or "Done."
        session.messages.append(
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=final_content,
                metadata={"reasoning": followup.reasoning},
            )
        )

        self._sessions.save(session)

        return OrchestratorResponse(
            session_id=session.session_id,
            message=final_content,
            tool_calls_made=[tool_call],
            metadata={
                "reasoning": followup.reasoning,
                "model": followup.model,
            },
        )

    # ------------------------------------------------------------------
    # Bulk operations (handled locally, not via service routing)
    # ------------------------------------------------------------------

    _BULK_TOOLS = {"bulk_download", "bulk_metadata"}

    def _execute_bulk_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle bulk tools that call the orchestrator's own endpoints."""
        from orchestrator_service.config import OrchestratorConfig
        config = OrchestratorConfig.from_env()
        base = f"http://127.0.0.1:{config.http_port}"
        try:
            resp = _req_lib.post(
                f"{base}/api/documents/{tool_name.replace('bulk_', '')}-metadata"
                if tool_name == "bulk_metadata"
                else f"{base}/api/documents/{tool_name.replace('bulk_', '')}-download",
                json=args,
                timeout=60,
            )
            return resp.json()
        except Exception as exc:
            return {"error": True, "message": str(exc)[:200]}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _confirmation_message(
        self,
        confirm_type: ConfirmationType,
        tool_name: str,
        args: dict[str, Any],
    ) -> str:
        """Generate a human-readable confirmation message."""
        if confirm_type == ConfirmationType.SENSITIVE_ACCESS:
            doc_id = args.get("document_id", "unknown document")
            return (
                f"This request requires accessing sensitive data from document {doc_id[:8]}... "
                f"(e.g., ID numbers, personal information). "
                f"Do you want to proceed?"
            )
        if confirm_type == ConfirmationType.FILE_RETRIEVAL:
            doc_id = args.get("document_id", "unknown document")
            return (
                f"You're about to download the actual file for document {doc_id[:8]}... "
                f"This will retrieve the original file. "
                f"Do you want to proceed?"
            )
        if confirm_type == ConfirmationType.EXTERNAL_ACTION:
            url = args.get("url", "unknown URL")
            return (
                f"You're about to access an external resource: {url[:60]}... "
                f"Do you want to proceed?"
            )
        if confirm_type == ConfirmationType.PRIVACY_OVERRIDE:
            return (
                "This action requires accessing private document data. "
                "Do you want to proceed?"
            )
        return "This action requires confirmation. Proceed?"

    def _interpret_confirmation_response(self, user_message: str) -> bool:
        """Simple interpretation of user's confirmation response."""
        positive = {"yes", "y", "ok", "sure", "proceed", "approve", "confirm", "go", "do it", "haan", "ji"}
        text = user_message.strip().lower()
        return any(word in text for word in positive)

    def _parse_arguments(self, raw: str) -> dict[str, Any]:
        """Parse tool call arguments from JSON string."""
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
