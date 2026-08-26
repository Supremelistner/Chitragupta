"""Model Service configuration — all from environment variables.

Provider switching is a config change, not a code change.
"""

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
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelServiceConfig:
    app_name: str
    environment: str
    log_level: str
    http_host: str
    http_port: int
    mcp_server_name: str
    mcp_server_version: str

    # Provider selection
    active_provider: str  # "huggingface", "fireworks", "deepinfra", "openai"
    huggingface_token: str | None
    huggingface_model_id: str
    fireworks_api_key: str | None
    fireworks_model_id: str
    deepinfra_api_key: str | None
    deepinfra_model_id: str
    openai_api_key: str | None
    openai_model_id: str

    # Inference defaults
    max_image_size_bytes: int
    default_temperature: float
    default_max_tokens: int
    request_timeout_seconds: int

    # Shared filesystem with document service (for dev)
    file_storage_root: str | None

    @classmethod
    def from_env(cls, *, http_host: str | None = None, http_port: int | None = None) -> ModelServiceConfig:
        return cls(
            app_name=_env("MODEL_SERVICE_APP_NAME", "model-service") or "model-service",
            environment=_env("MODEL_SERVICE_ENV", "development") or "development",
            log_level=_env("MODEL_SERVICE_LOG_LEVEL", "INFO") or "INFO",
            http_host=http_host or _env("MODEL_SERVICE_HTTP_HOST", "127.0.0.1") or "127.0.0.1",
            http_port=http_port or _env_int("MODEL_SERVICE_HTTP_PORT", 8081),
            mcp_server_name=_env("MODEL_SERVICE_MCP_SERVER_NAME", "model-service") or "model-service",
            mcp_server_version=_env("MODEL_SERVICE_VERSION", "0.1.0") or "0.1.0",
            active_provider=_env("MODEL_SERVICE_PROVIDER", "huggingface") or "huggingface",
            huggingface_token=_env("HF_TOKEN"),
            huggingface_model_id=_env("MODEL_SERVICE_HF_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct") or "Qwen/Qwen2.5-VL-72B-Instruct",
            fireworks_api_key=_env("FIREWORKS_API_KEY"),
            fireworks_model_id=_env("MODEL_SERVICE_FIREWORKS_MODEL", "accounts/fireworks/models/qwen2p5-vl-3b-instruct") or "accounts/fireworks/models/qwen2p5-vl-3b-instruct",
            deepinfra_api_key=_env("DEEPINFRA_API_KEY"),
            deepinfra_model_id=_env("MODEL_SERVICE_DEEPINFRA_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct") or "Qwen/Qwen2.5-VL-3B-Instruct",
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model_id=_env("MODEL_SERVICE_OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            max_image_size_bytes=_env_int("MODEL_SERVICE_MAX_IMAGE_BYTES", 10 * 1024 * 1024),
            default_temperature=_env_float("MODEL_SERVICE_TEMPERATURE", 0.1),
            default_max_tokens=_env_int("MODEL_SERVICE_MAX_TOKENS", 2048),
            request_timeout_seconds=_env_int("MODEL_SERVICE_TIMEOUT_SECONDS", 30),
            file_storage_root=_env("MODEL_SERVICE_FILE_STORAGE_ROOT"),
        )


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)
