"""HTTP file downloader adapter.

Downloads binary files (PDFs, images, forms, etc.) from URLs to a local
directory. Returns metadata about the downloaded file so it can be fed
into the document management service's ingestion pipeline.
"""
from __future__ import annotations

import logging
import os
import re
import time
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from web_search_service.domain.models import (
    DownloadRequest, DownloadResult, DownloadStatus,
)

logger = logging.getLogger("web_search_service.downloader")

# Binary content types worth downloading
_DOWNLOADABLE_TYPES = {
    "application/pdf",
    "application/zip",
    "application/x-zip-compressed",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
    "image/gif",
}

# Extensions to infer content type from
_EXT_TO_TYPE = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".zip": "application/zip",
}


class HttpDownloader:
    """Downloads files from URLs to a local directory."""

    def __init__(self, download_dir: str | Path | None = None) -> None:
        if download_dir is None:
            download_dir = os.getenv("WEB_SEARCH_DOWNLOAD_DIR", "./data/downloads")
        self._dir = Path(download_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def ping(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def download(self, request: DownloadRequest) -> DownloadResult:
        start = time.monotonic()
        try:
            return self._do_download(request, start)
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.error("Download failed: %s — %s", request.url, exc)
            return DownloadResult(
                url=request.url,
                status=DownloadStatus.FAILED,
                error=str(exc),
                latency_ms=latency,
                request_id=request.request_id,
            )

    def _do_download(self, request: DownloadRequest, start: float) -> DownloadResult:
        req = Request(request.url, headers={
            "User-Agent": "Chitragupta/1.0 (document-companion)",
            "Accept": ", ".join(_DOWNLOADABLE_TYPES) + ", */*",
        })

        with urlopen(req, timeout=request.timeout_ms / 1000) as resp:
            status_code = resp.status
            content_type = resp.headers.get("Content-Type", "")
            content_length_str = resp.headers.get("Content-Length")

            # Check content length before downloading
            if content_length_str:
                content_length = int(content_length_str)
                if content_length > request.max_size_bytes:
                    latency = (time.monotonic() - start) * 1000
                    return DownloadResult(
                        url=request.url,
                        status=DownloadStatus.FAILED,
                        error=f"File too large: {content_length} bytes (max: {request.max_size_bytes})",
                        content_type=content_type,
                        content_length=content_length,
                        latency_ms=latency,
                        request_id=request.request_id,
                    )

            # Read in chunks to enforce size limit
            chunks = []
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > request.max_size_bytes:
                    latency = (time.monotonic() - start) * 1000
                    return DownloadResult(
                        url=request.url,
                        status=DownloadStatus.FAILED,
                        error=f"File exceeded size limit during download: >{request.max_size_bytes} bytes",
                        content_type=content_type,
                        latency_ms=latency,
                        request_id=request.request_id,
                    )

            raw = b"".join(chunks)

        # Determine filename
        filename = request.filename or self._guess_filename(request.url, content_type, raw)
        content_type = self._normalize_content_type(content_type, filename)

        # Check if downloadable
        if not self._is_downloadable(content_type, filename):
            latency = (time.monotonic() - start) * 1000
            return DownloadResult(
                url=request.url,
                status=DownloadStatus.UNSUPPORTED,
                error=f"Unsupported content type: {content_type}",
                content_type=content_type,
                content_length=len(raw),
                latency_ms=latency,
                request_id=request.request_id,
            )

        # Write to disk
        file_hash = hashlib.sha256(raw).hexdigest()[:12]
        safe_name = re.sub(r'[^\w.\-]', '_', filename)
        dest_name = f"{file_hash}_{safe_name}"
        dest_path = self._dir / dest_name
        dest_path.write_bytes(raw)

        latency = (time.monotonic() - start) * 1000
        logger.info("Downloaded %s → %s (%d bytes, %.0fms)", request.url, dest_path, len(raw), latency)

        return DownloadResult(
            url=request.url,
            status=DownloadStatus.SUCCESS,
            file_path=str(dest_path),
            filename=filename,
            content_type=content_type,
            content_length=len(raw),
            latency_ms=latency,
            request_id=request.request_id,
        )

    def _guess_filename(self, url: str, content_type: str, raw: bytes) -> str:
        # Try Content-Disposition header first (already extracted if present)
        # Try URL path
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path:
            basename = path.split("/")[-1]
            if "." in basename:
                return basename

        # Infer from content type
        ext = _TYPE_TO_EXT.get(content_type.split(";")[0].strip(), ".bin")
        return f"download{ext}"

    def _normalize_content_type(self, content_type: str, filename: str) -> str:
        ct = content_type.split(";")[0].strip()
        if ct in ("application/octet-stream", "text/html", "text/plain", ""):
            # Try to infer from extension
            ext = Path(filename).suffix.lower()
            if ext in _EXT_TO_TYPE:
                return _EXT_TO_TYPE[ext]
        return ct

    def _is_downloadable(self, content_type: str, filename: str) -> bool:
        if content_type in _DOWNLOADABLE_TYPES:
            return True
        # Check extension
        ext = Path(filename).suffix.lower()
        return ext in _EXT_TO_TYPE

    def close(self) -> None:
        pass


# Reverse mapping for content type → extension
_TYPE_TO_EXT = {v: k for k, v in _EXT_TO_TYPE.items()}
