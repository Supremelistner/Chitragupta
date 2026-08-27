"""Retrieval orchestration -- no model inference, just web data retrieval."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from web_search_service.domain.models import (
    CrawlRequest, CrawlResult, DownloadRequest, DownloadResult,
    FetchRequest, FetchResult,
    ScrapeRequest, ScrapeResult, SearchQuery, SearchResponse, SearchProvider,
)
from web_search_service.domain.ports import FileDownloader, UrlFetcher, WebScraperProvider, WebSearchProvider

logger = logging.getLogger("web_search_service.retrieval")

@dataclass(frozen=True, slots=True)
class RetrievalDependencies:
    search_provider: WebSearchProvider | None = None
    scraper_provider: WebScraperProvider | None = None
    fetcher: UrlFetcher | None = None
    downloader: FileDownloader | None = None

class RetrievalService:
    def __init__(self, dependencies):
        self._deps = dependencies

    def search(self, query):
        if self._deps.search_provider is None:
            raise RuntimeError("No search provider configured")
        return self._deps.search_provider.search(query)

    def scrape(self, request):
        if self._deps.scraper_provider is not None:
            try:
                return self._deps.scraper_provider.scrape(request)
            except Exception as exc:
                logger.warning("Scraper failed, falling back to fetch: %s", exc)
        if self._deps.fetcher is not None:
            fetch_req = FetchRequest(url=request.url, format=request.format, request_id=request.request_id)
            result = self._deps.fetcher.fetch(fetch_req)
            return ScrapeResult(url=result.url, title=None, content=result.content,
                content_format=result.content_format, provider=SearchProvider.FETCH,
                latency_ms=result.latency_ms, request_id=result.request_id)
        raise RuntimeError("No scraper or fetcher configured")

    def crawl(self, request):
        if self._deps.scraper_provider is None:
            raise RuntimeError("No crawl provider configured")
        return self._deps.scraper_provider.crawl(request)

    def fetch(self, request):
        if self._deps.fetcher is None:
            raise RuntimeError("No fetcher configured")
        return self._deps.fetcher.fetch(request)

    def download(self, request: DownloadRequest) -> DownloadResult:
        if self._deps.downloader is None:
            raise RuntimeError("No downloader configured")
        return self._deps.downloader.download(request)

    def health(self):
        providers = {}
        for name, provider in [("search", self._deps.search_provider), ("scraper", self._deps.scraper_provider), ("fetcher", self._deps.fetcher), ("downloader", self._deps.downloader)]:
            if provider is not None:
                try:
                    provider.ping()
                    providers[name] = {"status": "healthy"}
                except Exception as exc:
                    providers[name] = {"status": "degraded", "error": str(exc)}
        healthy = all(p.get("status") == "healthy" for p in providers.values()) if providers else False
        return {"status": "healthy" if healthy else "degraded", "providers": providers}
