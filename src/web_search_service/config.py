"""Web Search Service configuration -- loads from central .env."""
from __future__ import annotations
from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _env(name, default=None):
    v = os.getenv(name)
    return default if v is None or v == "" else v

def _env_int(name, default):
    v = _env(name)
    return default if v is None else int(v)

@dataclass(frozen=True)
class WebSearchConfig:
    http_host: str
    http_port: int
    log_level: str
    firecrawl_api_key: str | None
    firecrawl_base_url: str
    fetch_timeout_ms: int

    @classmethod
    def from_env(cls, *, http_host=None, http_port=None):
        return cls(
            http_host=http_host or _env("WEB_SEARCH_SERVICE_HTTP_HOST", "127.0.0.1") or "127.0.0.1",
            http_port=http_port or _env_int("WEB_SEARCH_SERVICE_HTTP_PORT", 8082),
            log_level=_env("WEB_SEARCH_SERVICE_LOG_LEVEL", "INFO") or "INFO",
            firecrawl_api_key=_env("FIRECRAWL_API_KEY"),
            firecrawl_base_url=_env("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1") or "https://api.firecrawl.dev/v1",
            fetch_timeout_ms=_env_int("FETCH_TIMEOUT_MS", 15000),
        )
