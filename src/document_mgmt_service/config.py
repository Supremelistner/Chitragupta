from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class AppConfig:
    app_name: str
    environment: str
    log_level: str
    http_host: str
    http_port: int
    mcp_server_name: str
    mcp_server_version: str
    postgres_dsn: str | None
    qdrant_url: str | None
    qdrant_collection_name: str
    semantic_embedding_dimension: int
    semantic_chunk_size: int
    semantic_chunk_overlap: int
    file_storage_root: Path
    sqlite_db_path: str | None
    encryption_master_key: str
    ocr_enabled: bool
    huggingface_token: str | None
    huggingface_model_id: str
    request_timeout_seconds: int

    @classmethod
    def from_env(
        cls,
        *,
        http_host: str | None = None,
        http_port: int | None = None,
    ) -> "AppConfig":
        file_storage_root = Path(
            _env("DOCUMENT_SERVICE_FILE_STORAGE_ROOT", "./data/files") or "./data/files"
        )
        return cls(
            app_name=_env("DOCUMENT_SERVICE_APP_NAME", "document-management-service")
            or "document-management-service",
            environment=_env("DOCUMENT_SERVICE_ENV", "development") or "development",
            log_level=_env("DOCUMENT_SERVICE_LOG_LEVEL", "INFO") or "INFO",
            http_host=http_host
            or _env("DOCUMENT_SERVICE_HTTP_HOST", "127.0.0.1")
            or "127.0.0.1",
            http_port=http_port or _env_int("DOCUMENT_SERVICE_HTTP_PORT", 8080),
            mcp_server_name=_env(
                "DOCUMENT_SERVICE_MCP_SERVER_NAME",
                "document-management-service",
            )
            or "document-management-service",
            mcp_server_version=_env("DOCUMENT_SERVICE_VERSION", "0.1.0") or "0.1.0",
            postgres_dsn=_env("DOCUMENT_SERVICE_POSTGRES_DSN"),
            qdrant_url=_env("DOCUMENT_SERVICE_QDRANT_URL"),
            qdrant_collection_name=_env("DOCUMENT_SERVICE_QDRANT_COLLECTION", "document_chunks")
            or "document_chunks",
            semantic_embedding_dimension=_env_int("DOCUMENT_SERVICE_SEMANTIC_EMBED_DIM", 256),
            semantic_chunk_size=_env_int("DOCUMENT_SERVICE_SEMANTIC_CHUNK_SIZE", 900),
            semantic_chunk_overlap=_env_int("DOCUMENT_SERVICE_SEMANTIC_CHUNK_OVERLAP", 120),
            file_storage_root=file_storage_root,
            sqlite_db_path=_env("DOCUMENT_SERVICE_SQLITE_DB_PATH"),
            encryption_master_key=_env("ENCRYPTION_MASTER_KEY", ""),
            ocr_enabled=_env_bool("DOCUMENT_SERVICE_OCR_ENABLED", False),
            huggingface_token=_env("HF_TOKEN"),
            huggingface_model_id=_env("MODEL_SERVICE_HF_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct") or "Qwen/Qwen2.5-VL-72B-Instruct",
            request_timeout_seconds=_env_int("MODEL_SERVICE_TIMEOUT_SECONDS", 30),
        )
