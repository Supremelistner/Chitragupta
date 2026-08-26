"""Simple HTTP fetch adapter."""
from __future__ import annotations
import re
import time
import logging
from urllib.request import urlopen, Request
from web_search_service.domain.models import ContentFormat, FetchRequest, FetchResult

logger = logging.getLogger("web_search_service.fetch")


class SimpleFetchAdapter:
    def ping(self) -> None:
        pass

    def fetch(self, request: FetchRequest) -> FetchResult:
        start = time.monotonic()
        try:
            req = Request(request.url, headers={"User-Agent": "Chitragupta/1.0"})
            with urlopen(req, timeout=request.timeout_ms / 1000) as resp:
                raw = resp.read()
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
            content = self._decode(raw, request.format)
            latency = (time.monotonic() - start) * 1000
            return FetchResult(url=request.url, content=content, content_format=request.format,
                status_code=status, content_type=content_type, latency_ms=latency, request_id=request.request_id)
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return FetchResult(url=request.url, content=f"ERROR: {exc}", content_format=request.format,
                latency_ms=latency, request_id=request.request_id)

    def _decode(self, raw: bytes, fmt: ContentFormat) -> str:
        text = raw.decode("utf-8", errors="replace")
        if fmt in (ContentFormat.TEXT, ContentFormat.MARKDOWN):
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def close(self) -> None:
        pass
