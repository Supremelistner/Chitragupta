"""Domain models for the Web Search Service."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SearchProvider(str, Enum):
    FIRECRAWL = "firecrawl"
    FETCH = "fetch"


class ContentFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class SearchQuery:
    query: str
    limit: int = 10
    language: str | None = None
    country: str | None = None
    safe_search: bool = True
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float | None = None
    provider: SearchProvider | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    results: tuple[SearchResult, ...]
    provider: SearchProvider
    latency_ms: float | None = None
    request_id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ScrapeRequest:
    url: str
    format: ContentFormat = ContentFormat.MARKDOWN
    wait_for: str | None = None
    timeout_ms: int = 30000
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScrapeResult:
    url: str
    title: str | None
    content: str
    content_format: ContentFormat
    provider: SearchProvider
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    request_id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CrawlRequest:
    url: str
    limit: int = 10
    depth: int = 2
    include_external: bool = False
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class CrawlResult:
    url: str
    pages: tuple[ScrapeResult, ...]
    provider: SearchProvider
    latency_ms: float | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class FetchRequest:
    url: str
    format: ContentFormat = ContentFormat.MARKDOWN
    timeout_ms: int = 15000
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    content: str
    content_format: ContentFormat
    status_code: int | None = None
    content_type: str | None = None
    latency_ms: float | None = None
    request_id: str | None = None
    created_at: datetime | None = None
