"""Simple HTTP fetch adapter."""
from __future__ import annotations
import re
import time
import logging
from urllib.request import Request
from web_search_service.domain.models import ContentFormat, FetchRequest, FetchResult
from web_search_service.infrastructure.url_guard import (
    BlockedHostError,
    build_guarded_opener,
    validate_url,
)

logger = logging.getLogger("web_search_service.fetch")

# Hard cap on fetched bodies: fetch is for snippets/summaries, not bulk
# download (that's the downloader's job with its own limit).
MAX_FETCH_BYTES = 5 * 1024 * 1024


class SimpleFetchAdapter:
    def ping(self) -> None:
        pass

    def fetch(self, request: FetchRequest) -> FetchResult:
        start = time.monotonic()
        try:
            validate_url(request.url)
        except BlockedHostError as exc:
            latency = (time.monotonic() - start) * 1000
            return FetchResult(url=request.url, content=f"ERROR: blocked URL: {exc}",
                content_format=request.format, latency_ms=latency, request_id=request.request_id)
        try:
            req = Request(request.url, headers={"User-Agent": "Chitragupta/1.0"})
            opener = build_guarded_opener()
            with opener.open(req, timeout=request.timeout_ms / 1000) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                raw, truncated = self._read_capped(resp, MAX_FETCH_BYTES)
            content = self._decode(raw, request.format)
            if truncated:
                content += f"\n\n[truncated: response exceeded {MAX_FETCH_BYTES} bytes]"
            latency = (time.monotonic() - start) * 1000
            return FetchResult(url=request.url, content=content, content_format=request.format,
                status_code=status, content_type=content_type, latency_ms=latency, request_id=request.request_id)
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return FetchResult(url=request.url, content=f"ERROR: {exc}", content_format=request.format,
                latency_ms=latency, request_id=request.request_id)

    @staticmethod
    def _read_capped(resp, limit: int) -> tuple[bytes, bool]:
        """Read at most ``limit`` bytes. Returns (data, truncated)."""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = resp.read(min(65536, limit - total + 1))
            if not chunk:
                return b"".join(chunks), False
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                data = b"".join(chunks)[:limit]
                return data, True

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
