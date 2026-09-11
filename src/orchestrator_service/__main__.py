"""Orchestrator service entry point.

Wires together:
- Session storage (file-based)
- Confirmation store (in-memory)
- Tool registry (static definitions)
- Service clients (HTTP to each microservice)
- LLM provider (Qwen3.8-27B via HF)
- Orchestration engine
- MCP + HTTP adapters
"""
from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

from orchestrator_service.application.orchestrator import OrchestrationEngine
from orchestrator_service.config import OrchestratorConfig
from orchestrator_service.domain.models import ServiceTarget
from orchestrator_service.infrastructure.file_confirmation_store import (
    FileConfirmationStore,
)
from orchestrator_service.infrastructure.llm_provider import QwenLLMProvider
from orchestrator_service.infrastructure.service_clients import (
    HttpServiceClient,
    ServiceClientRouter,
)
from orchestrator_service.infrastructure.session_store import FileSessionStore
from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry


def main() -> None:
    config = OrchestratorConfig.from_env()

    # Logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("orchestrator")
    logger.info("Starting orchestrator-service")

    # Session store
    sessions = FileSessionStore(base_dir=config.session_store_dir)

    # Confirmation store (persistent — survives restarts)
    confirmations = FileConfirmationStore(
        base_dir=config.session_store_dir.replace("sessions", "confirmations")
    )

    # Tool registry
    registry = DefaultToolRegistry()

    # Service clients
    clients = {
        ServiceTarget.DOCUMENT: HttpServiceClient(
            target=ServiceTarget.DOCUMENT,
            base_url=config.document_service_url,
        ),
        ServiceTarget.MODEL: HttpServiceClient(
            target=ServiceTarget.MODEL,
            base_url=config.model_service_url,
        ),
        ServiceTarget.VALIDATOR: HttpServiceClient(
            target=ServiceTarget.VALIDATOR,
            base_url=config.validator_service_url,
        ),
        ServiceTarget.WEB_SEARCH: HttpServiceClient(
            target=ServiceTarget.WEB_SEARCH,
            base_url=config.web_search_service_url,
        ),
    }
    router = ServiceClientRouter(clients)

    # LLM provider selection. Three options:
    #   1. GEMINI_API_KEY set -> Gemini as primary (free tier, generous RPD)
    #   2. Otherwise -> Qwen via HuggingFace
    # Either path can be wrapped in a Groq fallback if ORCHESTRATOR_FALLBACK_PROVIDER=groq
    # and GROQ_API_KEY is set.
    if config.gemini_api_key:
        from orchestrator_service.infrastructure.gemini_llm_provider import (
            GeminiLLMProvider,
        )
        primary_llm = GeminiLLMProvider(
            api_key=config.gemini_api_key,
            model_id=config.gemini_llm_model_id,
            timeout_seconds=config.llm_timeout_seconds,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        logger.info("LLM primary: Gemini (%s)", config.gemini_llm_model_id)
    else:
        primary_llm = QwenLLMProvider(
            token=config.hf_token or "",
            model_id=config.llm_model_id,
            timeout_seconds=config.llm_timeout_seconds,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )

    if config.fallback_provider == "groq" and config.groq_api_key:
        from orchestrator_service.infrastructure.fallback_llm_provider import (
            FallbackLLMProvider,
        )
        from orchestrator_service.infrastructure.groq_llm_provider import (
            GroqLLMProvider,
        )
        groq_llm = GroqLLMProvider(
            api_key=config.groq_api_key,
            model_id=config.groq_model_id,
            timeout_seconds=config.llm_timeout_seconds,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        if not config.hf_token and not config.gemini_api_key:
            # No primary LLM is configured — route every chat call straight
            # through Groq instead of wasting a 401 round-trip on the primary.
            llm = groq_llm
            logger.info(
                "LLM primary: Groq (%s) - no primary configured, "
                "using Groq as sole LLM",
                config.groq_model_id,
            )
        else:
            llm = FallbackLLMProvider(primary=primary_llm, fallback=groq_llm)
            logger.info("LLM fallback enabled: Groq (%s)", config.groq_model_id)
    else:
        llm = primary_llm

    # Translation service for V1 multilingual support. Constructed
    # eagerly so the first user message doesn't pay initialization cost.
    # When GEMINI_API_KEY is missing the provider raises at first use,
    # which is fine — the orchestrator should still boot for English-only
    # users.
    from model_service.application.translation_service import (
        get_translation_service,
    )
    translation = get_translation_service()

    # Pre-existing profile (if any) so restarts keep addressing the
    # user by name even before the UI syncs again.
    try:
        from orchestrator_service.onboarding import load_profile
        _existing = load_profile(config.profile_path or None)
    except Exception:
        _existing = None

    # Orchestration engine
    engine = OrchestrationEngine(
        llm=llm,
        sessions=sessions,
        confirmations=confirmations,
        tool_registry=registry,
        service_router=router,
        confirmation_threshold=config.confirmation_threshold,
        translation_service=translation,
    )
    if _existing is not None:
        engine.set_user_name(_existing.display_name)
        logger.info("Loaded profile for '%s'", _existing.display_name)

    # Start server
    mode = sys.argv[1] if len(sys.argv) > 1 else "http"

    if mode == "mcp":
        from orchestrator_service.adapters.mcp.server import MCPServer
        server = MCPServer(config=config, engine=engine)
        server.serve()
    else:
        # FastAPI + Uvicorn RESTful server
        from orchestrator_service.adapters.http.server import HTTPServer
        server = HTTPServer(config=config, engine=engine)
        server.serve()


if __name__ == "__main__":
    main()
