"""Firecrawl adapter -- search, scrape, crawl via Firecrawl API."""
from __future__ import annotations
import json
import logging
import os
import time
from urllib.request import urlopen, Request
from web_search_service.domain.models import (
    CrawlRequest, CrawlResult, ContentFormat,
    ScrapeRequest, ScrapeResult,
    SearchProvider, SearchQuery, SearchResponse, SearchResult,
)

logger = logging.getLogger("web_search_service.firecrawl")



class FirecrawlAdapter:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self._base_url = base_url or os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")

    def ping(self) -> None:
        if not self._api_key:
            raise RuntimeError("FIRECRAWL_API_KEY is required")

    def search(self, query: SearchQuery) -> SearchResponse:
        start = time.monotonic()
        results = self._call_search(query)
        latency = (time.monotonic() - start) * 1000
        return SearchResponse(query=query.query, results=tuple(results),
            provider=SearchProvider.FIRECRAWL, latency_ms=latency, request_id=query.request_id)

    def scrape(self, request: ScrapeRequest) -> ScrapeResult:
        start = time.monotonic()
        data = self._call_scrape(request.url)
        latency = (time.monotonic() - start) * 1000
        return ScrapeResult(url=request.url, title=data.get("title"), content=data.get("content", ""),
            content_format=request.format, provider=SearchProvider.FIRECRAWL,
            metadata=data.get("metadata", {}), latency_ms=latency, request_id=request.request_id)

    def crawl(self, request: CrawlRequest) -> CrawlResult:
        start = time.monotonic()
        pages_data = self._call_crawl(request.url, request.limit, request.depth)
        latency = (time.monotonic() - start) * 1000
        pages = tuple(ScrapeResult(url=p.get("url", ""), title=p.get("title"),
            content=p.get("content", ""), content_format=ContentFormat.MARKDOWN,
            provider=SearchProvider.FIRECRAWL) for p in pages_data)
        return CrawlResult(url=request.url, pages=pages, provider=SearchProvider.FIRECRAWL,
            latency_ms=latency, request_id=request.request_id)

    def _api_call(self, endpoint, payload, timeout=30):
        data = json.dumps(payload).encode()
        req = Request(f"{self._base_url}/{endpoint}", data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _call_search(self, query):
        data = self._api_call("search", {"query": query.query, "limit": query.limit})
        return [SearchResult(url=r.get("url", ""), title=r.get("title", ""),
            snippet=r.get("description", ""), provider=SearchProvider.FIRECRAWL, metadata=r)
            for r in data.get("data", [])]

    def _call_scrape(self, url):
        data = self._api_call("scrape", {"url": url})
        result = data.get("data", {})
        return {"title": result.get("title"), "content": result.get("markdown") or result.get("content", ""),
                "metadata": result.get("metadata", {})}

    def _call_crawl(self, url, limit, depth):
        data = self._api_call("crawl", {"url": url, "limit": limit, "depth": depth}, timeout=60)
        return data.get("data", [])

    def close(self) -> None:
        pass
