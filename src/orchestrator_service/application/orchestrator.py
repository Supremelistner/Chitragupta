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
import os
import re
import time
from typing import Any

import requests as _req_lib

from orchestrator_service.domain.models import (
    ConfirmationRequest,
    ConfirmationType,
    ConversationMessage,
    MessageRole,
    OrchestratorResponse,
    ServiceTarget,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from orchestrator_service.domain.ports import (
    ConfirmationStore,
    LLMProvider,
    SessionStore,
    ToolRegistry,
)
from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
from orchestrator_service.persona import build_system_prompt
from orchestrator_service.persona import humanize as _humanize_reply

# Translation service for V1 multilingual support. Imported lazily so the
# orchestrator can be unit-tested without the model-service stack present.
try:
    from model_service.application.translation_service import (
        TranslationService,
    )
    from model_service.application.translation_service import (
        get_translation_service as _get_translation_service,
    )
except ImportError:  # pragma: no cover -- model service may not be on path
    TranslationService = None  # type: ignore[assignment,misc]
    _get_translation_service = None  # type: ignore[assignment]

logger = logging.getLogger("orchestrator.engine")

# System prompt for the orchestrator. The actual value is set per-request
# via build_system_prompt() in :mod:`orchestrator_service.persona` so the
# LLM can address the user by name. Kept here for backward compatibility.
ORCHESTRATOR_SYSTEM_PROMPT = """You are Chitragupta, a personal document agent.

CORE BEHAVIOR

1. Treat every user question as a document task when possible.
2. Search or list documents before making claims about what the user has.
3. Use web_search for outside requirements, templates, application rules, and current process guidance.
4. Compare external requirements with the user's available documents.
5. Tell the user what is available, what is missing, and what next step makes sense.

DOCUMENT WORKFLOWS

- "What documents do I have?" -> list_documents.
- "Do I have my passport?" -> search_documents("passport").
- "What are my marks?" -> search_documents("marksheet") then search_document_content or get_evidence.
- "How can I apply for insurance?" -> web_search for requirements, search_documents/list_documents for matching documents, then explain available and missing documents.
- "Download/open/view this file" -> find the document first, then call the gated file retrieval tool.

ACCESS AND PRIVACY

The Document Management Service owns access policy. You must respect access_action values exactly:

- ALLOW: use the returned information.
- REDACT: explain only the redacted/safe result.
- REQUIRE_APPROVAL: tell the user approval is required and wait for the confirmation flow.
- DENY: say the service blocked access.

Do not claim direct access to Qdrant, PostgreSQL, OCR, or storage. Use only document-level tools. Never ask the user for document IDs unless a tool result is ambiguous and you need them to choose among returned documents.

UPLOAD RESPONSE STYLE

For uploads, respond only with success or failure and any document IDs/statuses returned. Do not summarize the document immediately after upload unless the user asks for metadata or description.

RESPONSE STYLE

- Direct answer first.
- Use simple language.
- Match the user's language when practical.
- Do not output raw JSON.
- For document requirement questions, separate "Available" and "Missing" clearly.
"""


def session_fallback_user_id(sessions) -> str:
    """Best-effort user_id when none is set on the engine.

    Used as the namespace for field-approval caching when the
    orchestrator is configured without a user_id (e.g. local dev).
    Falls back to a static local user marker.
    """
    return "__local__"


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
        translation_service: Any | None = None,
    ) -> None:
        self._llm = llm
        self._sessions = sessions
        self._confirmations = confirmations
        self._registry = tool_registry
        self._router = service_router
        self._confirmation_threshold = confirmation_threshold
        # Per-user persona (set via set_user_name). When set, the system
        # message is generated from the user name; otherwise the default
        # ORCHESTRATOR_SYSTEM_PROMPT is used.
        self._user_name: str | None = None

        # Translation service for V1 multilingual support. Optional so
        # tests can run without the model service on the path. When not
        # injected, falls back to the module-level singleton on first use.
        self._translation: Any = translation_service

        # Per-user, per-field approval cache. After a user approves
        # get_field_value for a specific (document_id, version, field),
        # subsequent calls for the same key within TTL skip the gate.
        # Default TTL is 2 hours; override with
        # CHITRAGUPTA_FIELD_APPROVAL_TTL_SECONDS env var.
        self._field_approval_ttl_seconds = int(
            os.getenv("CHITRAGUPTA_FIELD_APPROVAL_TTL_SECONDS", "7200")
        )
        # cache_key -> expires_at_monotonic
        self._field_approvals: dict[str, float] = {}
        # user_id cache (set via set_user_id). Used as the first segment
        # of every cache key so approvals are scoped per user.
        self._user_id: str | None = None

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

    def list_expiring_documents(self, within_days: int = 30, user_id: str | None = None) -> dict:
        """Documents expiring soon, for the proactive banner (no LLM turn).

        Calls the document service directly. Never raises for transport
        trouble — the banner simply stays hidden instead.
        """
        try:
            within_days = min(365, max(1, int(within_days)))
        except (TypeError, ValueError):
            within_days = 30
        try:
            target = self._registry.get_service("list_expiring_documents")
            if target is None:
                return {"results": []}
            args: dict[str, Any] = {"within_days": within_days}
            if user_id:
                args["user_id"] = user_id
            result = self._router.call_tool(
                target, "list_expiring_documents", args
            )
            if not isinstance(result, dict) or result.get("error"):
                return {"results": []}
            return {"results": result.get("results", [])}
        except Exception as exc:
            logger.warning("list_expiring_documents failed: %s", exc)
            return {"results": []}

    def process_message(
        self, session_id: str, user_message: str, file: dict | None = None,
        user_id: str | None = None,
    ) -> OrchestratorResponse:
        """Process a user message within a session.

        This is the main entry point. It:
        1. Loads the session (context isolation)
        2. Adds the user message
        3. Calls the LLM with tools
        4. If LLM wants tool calls → executes them (with confirmation if needed)
        5. Feeds results back to LLM for final response
        6. Translates the response back into the user's UI language

        Translation is applied at the boundary: the user's message is
        translated to English before it lands in the session history
        (so the LLM only ever sees English), and the final response
        message is translated into the user's preferred UI language
        before returning. When the session is already in English mode
        these are no-ops with no API cost.
        """
        # Translate inbound: the LLM thinks in English regardless of
        # what the user typed. We swap the local variable so the original
        # user message is never persisted in the session log (privacy +
        # reproducibility of the model's internal narrative).
        inbound = self.translate_user_input(user_message, session_id)
        english_message = inbound.text

        if file is not None:
            english_message = self._attach_file_context(english_message, file, user_id=user_id)

        response = self._process_message_english(
            session_id, english_message, original_message=user_message,
            file_context=bool(file), user_id=user_id,
        )
        # Translate outbound: present the response in the user's
        # preferred UI language.
        return self._translate_response(response)

    # Titles the UI sends at creation ("New chat", ...) plus legacy
    # artifacts. Only these are ever auto-replaced by _maybe_retitle.
    _DEFAULT_SESSION_TITLES = frozenset({
        "", "New chat", "New conversation", "नई बातचीत",
        "புதிய அரட்டை", "नई चैट", "[nav_new_chat]",
    })

    def _maybe_retitle_session(
        self, session: Session, original_message: str, file_context: bool = False
    ) -> None:
        """Title a fresh session from its first user message.

        Only fires when this is the session's first user message AND the
        current title is a known default. Uses the ORIGINAL (untranslated)
        text so Hindi users get Hindi titles. Never raises — titling is
        cosmetic and must not break the chat turn.
        """
        try:
            user_turns = sum(1 for m in session.messages if m.role is MessageRole.USER)
            if user_turns != 1:
                return
            if (session.title or "") not in self._DEFAULT_SESSION_TITLES:
                return
            text = " ".join((original_message or "").split())
            if not text and file_context:
                text = "Photo attachment"
            if not text:
                return
            session.title = text if len(text) <= 48 else text[:47].rstrip() + "…"
            self._sessions.save(session)
        except Exception as exc:
            logger.warning("Session retitle failed: %s", exc)

    def _attach_file_context(self, english_message: str, file: dict, user_id: str | None = None) -> str:
        """Ingest a chat-attached file via the ``upload_document`` tool and
        annotate the message with the outcome.

        Never raises: ingestion failures become a system fact line so the
        LLM can explain them instead of the request dying mid-pipeline.
        """
        import base64

        filename = str(file.get("filename") or "attachment")
        content = file.get("bytes") or b""
        if not content:
            note = (
                f"[System: an empty file named '{filename}' was attached; "
                f"nothing was ingested.]"
            )
            return f"{english_message}\n{note}".strip()
        try:
            target = self._registry.get_service("upload_document")
            if target is None:
                raise RuntimeError("upload_document tool is not registered")
            args: dict[str, Any] = {
                "filename": filename,
                "content_base64": base64.b64encode(content).decode("ascii"),
                "content_type": file.get("content_type") or "application/octet-stream",
                "description": english_message[:500],
            }
            result = self._router.call_tool(
                target, "upload_document",
                self._scoped_args(None, target, args, fallback_user_id=user_id),
            )
        except Exception as exc:
            logger.warning("Chat attachment ingest failed: %s", exc)
            note = (
                f"[System: attached file '{filename}' could not be ingested "
                f"({exc}). Ask the user to retry.]"
            )
            return f"{english_message}\n{note}".strip()
        if not isinstance(result, dict) or result.get("error"):
            detail = result.get("message", "unknown error") if isinstance(result, dict) else "unknown error"
            note = f"[System: attached file '{filename}' ingest failed: {detail}.]"
            return f"{english_message}\n{note}".strip()
        prompt = english_message or "Please process the attached file."
        fact = (
            f"[System: attached file '{filename}' ingested as document "
            f"{result.get('document_id')} (status {result.get('processing_status')}).]"
        )
        return f"{prompt}\n{fact}".strip()

    def _process_message_english(
            self, session_id: str, user_message: str, file: dict = None,
            original_message: str | None = None, file_context: bool = False,
            user_id: str | None = None,
        ) -> OrchestratorResponse:
        """Inner ``process_message`` that operates in English only.

        Split out so the public ``process_message`` can wrap it with
        translation without polluting the existing logic with language
        concerns. This function assumes ``user_message`` is in English;
        ``original_message`` carries the pre-translation text for
        display purposes (e.g. session titles).
        """
        session = self._sessions.get(session_id)
        if session is None:
            return OrchestratorResponse(
                session_id=session_id,
                message=f"Session {session_id} not found. Please create a new session.",
            )

        # Multi-user binding: a token user takes ownership of unowned
        # sessions; touching another account's session is refused.
        if user_id:
            if session.user_id and session.user_id != user_id:
                return OrchestratorResponse(
                    session_id=session_id,
                    message="This chat belongs to another account. Please start a new chat.",
                )
            if not session.user_id:
                session.user_id = user_id
                self._sessions.save(session)

        # Add user message to session
        session.messages.append(
            ConversationMessage(role=MessageRole.USER, content=user_message)
        )

        # Fresh session with a default title? Name it after the message.
        self._maybe_retitle_session(
            session, original_message if original_message is not None else user_message,
            file_context=file_context,
        )

        # Check if there's a pending confirmation to handle
        pending = self._confirmations.get_pending(session_id)
        if pending:
            return self._handle_pending_confirmation(session, pending[0], user_message)

        # Normal flow: plan → execute
        return self._plan_and_execute(session)

    def set_user_id(self, user_id: str | None) -> None:
        """Set the active user id for per-user approval scoping."""
        self._user_id = user_id

    @property
    def user_name(self) -> str | None:
        """Display name used by the persona prompt (None → 'friend')."""
        return self._user_name

    def set_user_name(self, user_name: str | None) -> None:
        """Set the display name greetings address the user by.

        Blank/None resets to the 'friend' fallback. Only the name is
        stored — never age, gender, or contact details.
        """
        cleaned = (user_name or "").strip()
        self._user_name = cleaned[:100] or None

    # ------------------------------------------------------------------
    # Translation (V1 multilingual)
    # ------------------------------------------------------------------

    def _get_translation(self) -> Any:
        """Resolve the translation service, falling back to the singleton."""
        if self._translation is None:
            if _get_translation_service is None:
                raise RuntimeError(
                    "Translation service is unavailable; the model "
                    "service package is not on the Python path."
                )
            self._translation = _get_translation_service()
        return self._translation

    def translate_user_input(self, text: str, session_id: str):
        """Translate the user-typed text from their UI language to English.

        Returns a ``TranslationResult`` with the original text and
        ``cached=True`` if no translation was needed.
        """
        return self._get_translation().translate_user_input(text, session_id)

    def translate_bot_reply(self, text: str, session_id: str):
        """Translate the LLM's English reply into the user's UI language."""
        return self._get_translation().translate_bot_reply(text, session_id)

    def set_language_preference(
        self,
        session_id: str,
        *,
        source: str | None = None,
        target: str | None = None,
    ):
        """Read or update a session's language preference."""
        return self._get_translation().set_preference(
            session_id, source=source, target=target
        )

    def get_language_preference(self, session_id: str):
        """Return the current preference (falls back to the V1 default)."""
        return self._get_translation().get_preference(session_id)

    def _translate_response(
        self, response: OrchestratorResponse
    ) -> OrchestratorResponse:
        """Translate the final ``message`` field back into the user's UI language.

        Other fields (plan, confirmation_required, tool_calls_made,
        metadata) are not translated — they're internal protocol
        surfaces, not user-facing strings. The confirmation popup's
        human-readable ``message`` *is* translated when present, since
        that's user-facing copy.
        """
        from dataclasses import replace as _dc_replace

        if not response.message:
            return response

        result = self.translate_bot_reply(response.message, response.session_id)
        translation_meta = {
            "source": result.source,
            "target": result.target,
            "cached": result.cached,
        }
        new_response = OrchestratorResponse(
            session_id=response.session_id,
            message=result.text,
            plan=response.plan,
            confirmation_required=response.confirmation_required,
            tool_calls_made=response.tool_calls_made,
            metadata={**response.metadata, "translation": translation_meta},
        )
        if response.confirmation_required is not None:
            original_msg = response.confirmation_required.message
            if original_msg:
                translated_msg = self.translate_bot_reply(
                    original_msg, response.session_id
                ).text
                new_response.confirmation_required = _dc_replace(
                    response.confirmation_required, message=translated_msg
                )
        return new_response

    def _field_approval_key(self, tool_name: str, arguments: dict) -> str | None:
        """Build a cache key for per-tool, per-field approval reuse.

        Only get_field_value is currently cached; other gated tools
        always prompt. Returns None for non-cacheable tools.
        """
        if tool_name != "get_field_value":
            return None
        document_id = str(arguments.get("document_id", ""))
        version = arguments.get("version", 1)
        field = str(arguments.get("field", arguments.get("field_name", "")))
        if not document_id or not field:
            return None
        user_segment = self._user_id or session_fallback_user_id(self._sessions)
        return f"{user_segment}|{tool_name}|{document_id}|{version}|{field}"

    def _field_approval_is_fresh(self, key: str) -> bool:
        """True iff a recent approval exists for this cache key."""
        expires_at = self._field_approvals.get(key)
        if expires_at is None:
            return False
        if expires_at < time.monotonic():
            del self._field_approvals[key]
            return False
        return True

    def _record_field_approval(self, key: str) -> None:
        """Cache a fresh approval for this key, expiring after TTL."""
        self._field_approvals[key] = time.monotonic() + self._field_approval_ttl_seconds
        logger.info(
            "Field approval cached: key=%s ttl=%ds",
            key, self._field_approval_ttl_seconds,
        )

    def handle_confirmation(
        self, session_id: str, request_id: str, approved: bool, correction: str | None = None
    ) -> OrchestratorResponse:
        """Handle user's confirmation response.

        Note: ``correction`` is expected to already be in English. The
        HTTP layer translates user input from the UI language before
        passing it in; the orchestrator itself never speaks anything but
        English internally.
        """
        response = self._handle_confirmation_english(
            session_id, request_id, approved, correction
        )
        return self._translate_response(response)

    def _handle_confirmation_english(
        self,
        session_id: str,
        request_id: str,
        approved: bool,
        correction: str | None = None,
    ) -> OrchestratorResponse:
        """Inner ``handle_confirmation`` that operates in English only."""
        # correction is the user-supplied corrected field name when they
        # deny a confirmation because the LLM picked the wrong field.
        self._confirmations.respond(request_id, approved, correction=correction)

        # Cache per-field approvals so the user is not re-prompted for the
        # same field within TTL (default 2 hours).
        if approved:
            confirmation = self._confirmations.get(request_id)
            if confirmation is not None:
                cache_key = self._field_approval_key(
                    confirmation.tool_name, confirmation.tool_args
                )
                if cache_key is not None:
                    self._record_field_approval(cache_key)

        session = self._sessions.get(session_id)
        if session is None:
            return OrchestratorResponse(
                session_id=session_id,
                message="Session not found.",
            )

        if not approved:
            # User denied. If they supplied a correction (e.g. they typed
            # the right field name in the chat), record it as a system note
            # so the LLM retries with the correct field.
            pending = self._confirmations.get(request_id)
            wrong_field = (pending.tool_args.get("field") if pending else None) or "(unknown)"
            if correction:
                session.messages.append(
                    ConversationMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            f"User denied the previous request for field '{wrong_field}' "
                            f"and indicated the correct field is '{correction}'. "
                            f"Re-call get_field_value with field='''{correction}''' so the user gets the right value."
                        ),
                    )
                )
            else:
                # Plain denial, no correction. Tell the LLM the user said no.
                session.messages.append(
                    ConversationMessage(
                        role=MessageRole.USER,
                        content="No, don't do that.",
                    )
                )
            self._sessions.save(session)
            return self._plan_and_execute(session)

        # Approved — find the step that was waiting and execute it
        return self._execute_confirmed_step(session, request_id)

    # ------------------------------------------------------------------
    # Auto-search: inject document search results before LLM sees the query
    # ------------------------------------------------------------------

    # Patterns that suggest the user is asking about a document they own
    _DOC_QUERY_PATTERNS = re.compile(
        r"\b(my|do i have|show me|get|find|what is|what are|tell me|download|read|open|view|list)\b"
        r".*(\b\w+\b)\b",
        re.IGNORECASE,
    )
    _DOC_KEYWORDS = re.compile(
        r"\b(aadhaar|adhaar|passport|pan|marksheet|mark\s*sheet|certificate|"
        r"degree|license|licence|insurance|policy|form|bill|receipt|invoice|"
        r"letter|agreement|contract|id|identity|document|paper|file)\b",
        re.IGNORECASE,
    )

    def _maybe_inject_search_results(self, session: Session) -> None:
        """If the user's last message is about a document, auto-search first.

        Injects search results as a system message so the LLM sees document_ids
        in context and doesn't need to call search_documents itself.
        """
        if not session.messages:
            return

        # Get the last user message
        last_user = None
        for msg in reversed(session.messages):
            if msg.role == MessageRole.USER:
                last_user = msg
                break
        if last_user is None:
            return

        text = last_user.content or ""
        if not text or len(text) < 4:
            return

        # Check if it looks like a document query
        if not self._DOC_KEYWORDS.search(text):
            return

        # Don't re-inject if we already did for this message
        for msg in reversed(session.messages):
            if msg.role == MessageRole.SYSTEM and "[auto-search]" in (msg.content or ""):
                break
            if msg.role == MessageRole.USER:
                if msg is last_user:
                    continue
                return  # different user message, skip

        # Run search against document service
        try:
            target = self._registry.get_service("search_documents")
            if target is None:
                return
            result = self._router.call_tool(
                target, "search_documents", {"query": text, "limit": 5}
            )
            if result.get("error"):
                return
            # Format search results for context
            results_list = result.get("results", [])
            if not results_list:
                # No semantic-text match. The document may still be in the
                # collection under a different filename or with empty OCR.
                # Tell the LLM to try list_documents so it doesn't conclude
                # "you haven't uploaded this" from a single empty search.
                injection = (
                    "[auto-search] The user's query mentions a document but "
                    "semantic search returned no matches. "
                    "Call list_documents() to see everything in the collection — "
                    "the document may be present under a different filename or "
                    "have no extracted text yet. If list_documents shows nothing, "
                    "ask the user to upload it. DO NOT tell the user the document "
                    "is missing until list_documents has been called and was empty."
                )
                session.messages.append(
                    ConversationMessage(role=MessageRole.SYSTEM, content=injection)
                )
                return
            docs_summary = []
            for r in results_list:
                doc_id = r.get("document_id", "?")
                ver = r.get("version", 1)
                desc = r.get("description", "no description")
                privacy = r.get("privacy", "?")
                docs_summary.append(
                    f"- document_id={doc_id}, version={ver}, description='{desc}', privacy={privacy}"
                )
            injection = (
                "[auto-search] The user's query is about documents. "
                "Here are matching documents from the database. "
                "Use the document_id and version from these results to call other tools. "
                "DO NOT ask the user for document IDs.\n\n"
                + "\n".join(docs_summary)
            )
            session.messages.append(
                ConversationMessage(role=MessageRole.SYSTEM, content=injection)
            )
        except Exception as e:
            logger.debug("Auto-search failed: %s", e)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _plan_and_execute(self, session: Session) -> OrchestratorResponse:
        """Plan → Execute → Respond loop."""
        tools = self._registry.get_all_tool_schemas()

        # Auto-search: if user asks about a document, pre-fetch search results
        # so the LLM sees document_ids in context without needing to call tools itself.
        self._maybe_inject_search_results(session)

        # Call LLM with conversation context + tools
        llm_response = self._llm.chat(
            messages=self._build_messages(session),
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

    def _build_messages(self, session) -> list:
        """Build the message list for the LLM, prepending the system prompt."""
        from orchestrator_service.domain.models import ConversationMessage, MessageRole
        system = build_system_prompt(user_name=self._user_name)
        return [
            ConversationMessage(role=MessageRole.SYSTEM, content=system),
            *session.messages,
        ]

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
                thought_signature=tc_data.get("thought_signature"),
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
                # Per-field approval cache: skip the gate if the user has
                # already approved this exact (tool, document, field)
                # within the TTL window. Lets "show me my aadhaar
                # number" work without re-prompting on the next ask.
                cache_key = self._field_approval_key(tool_call.tool_name, tool_call.arguments)
                if cache_key is not None and self._field_approval_is_fresh(cache_key):
                    logger.info(
                        "Field approval cache hit: key=%s; skipping confirmation gate",
                        cache_key,
                    )
                else:
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
                            thought_signature=tool_call.thought_signature,
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
                        target, tool_call.tool_name,
                        self._scoped_args(session, target, tool_call.arguments),
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
            messages=self._build_messages(session),
            tools=tools if (tools := self._registry.get_all_tool_schemas()) else None,
        )

        final_content = _humanize_reply(followup.content or "Done.")
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
            thought_signature=confirmation.thought_signature,
        )
        target = self._registry.get_service(tool_call.tool_name)

        if target:
            result = self._router.call_tool(
                target, tool_call.tool_name,
                self._scoped_args(session, target, tool_call.arguments),
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
            messages=self._build_messages(session),
            tools=tools if tools else None,
        )

        final_content = _humanize_reply(followup.content or "Done.")
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

    def _scoped_args(
        self, session: Session | None, target: Any, args: dict[str, Any],
        fallback_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Stamp the session's user_id onto document-service tool args.

        Only document tools are touched (other services would choke on
        unknown params); an explicit caller-supplied user_id always wins.
        """
        try:
            is_document = str(getattr(target, "value", target)) == ServiceTarget.DOCUMENT.value
        except Exception:
            is_document = False
        if not is_document or (args and args.get("user_id")):
            return args
        user_id = (getattr(session, "user_id", "") or "") or (self._user_id or "") or (fallback_user_id or "")
        if not user_id:
            return args
        return {**args, "user_id": user_id}

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
