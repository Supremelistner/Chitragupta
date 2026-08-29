"""Orchestrator service configuration — all from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class OrchestratorConfig:
    app_name: str
    environment: str
    log_level: str
    http_host: str
    http_port: int

    # LLM
    llm_model_id: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: int

    # Service URLs
    document_service_url: str
    model_service_url: str
    validator_service_url: str
    web_search_service_url: str

    # Session storage
    session_store_dir: str

    # Confirmation
    confirmation_threshold: int

    # Local profile (single-user mode)
    profile_path: str
    require_profile: bool

    # HF token
    hf_token: str | None

    # Groq fallback
    groq_api_key: str | None
    groq_model_id: str
    fallback_provider: str | None

    @classmethod
    def from_env(cls) -> OrchestratorConfig:
        return cls(
            app_name=_env("ORCHESTRATOR_APP_NAME", "orchestrator-service") or "orchestrator-service",
            environment=_env("ORCHESTRATOR_ENV", "development") or "development",
            log_level=_env("ORCHESTRATOR_LOG_LEVEL", "INFO") or "INFO",
            http_host=_env("ORCHESTRATOR_HTTP_HOST", "127.0.0.1") or "127.0.0.1",
            http_port=_env_int("ORCHESTRATOR_HTTP_PORT", 8084),
            llm_model_id=_env("ORCHESTRATOR_MODEL", "Qwen/Qwen3.8-27B") or "Qwen/Qwen3.8-27B",
            llm_temperature=_env_float("ORCHESTRATOR_TEMPERATURE", 0.1),
            llm_max_tokens=_env_int("ORCHESTRATOR_MAX_TOKENS", 4096),
            llm_timeout_seconds=_env_int("ORCHESTRATOR_LLM_TIMEOUT", 60),
            document_service_url=_env("DOCUMENT_SERVICE_URL", "http://localhost:8080") or "http://localhost:8080",
            model_service_url=_env("MODEL_SERVICE_URL", "http://localhost:8081") or "http://localhost:8081",
            validator_service_url=_env("VALIDATOR_SERVICE_URL", "http://localhost:8083") or "http://localhost:8083",
            web_search_service_url=_env("WEB_SEARCH_SERVICE_URL", "http://localhost:8082") or "http://localhost:8082",
            session_store_dir=_env("ORCHESTRATOR_SESSION_DIR", "./data/sessions") or "./data/sessions",
            confirmation_threshold=_env_int("ORCHESTRATOR_CONFIRM_THRESHOLD", 3),
            profile_path=_env("CHITRAGUPTA_PROFILE_PATH", "") or "",
            require_profile=(_env("CHITRAGUPTA_REQUIRE_PROFILE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"},
            hf_token=_env("HF_TOKEN"),
            groq_api_key=_env("GROQ_API_KEY"),
            groq_model_id=_env("GROQ_LLM_MODEL", "qwen/qwen3.8-27b") or "qwen/qwen3.8-27b",
            fallback_provider=_env("ORCHESTRATOR_FALLBACK_PROVIDER"),
        )


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)
