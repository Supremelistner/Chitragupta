"""Infrastructure — in-memory document service client for V1.

Reads document metadata from the document management service via HTTP.
This allows the validator to look up document type without the caller
having to provide it.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger("validator_service.document_client")


class HTTPDocumentServiceClient:
    """Reads document metadata from the document management service."""

    def __init__(
        self,
        *,
        document_service_url: str = "http://localhost:8080",
        timeout_seconds: int = 10,
    ) -> None:
        self._url = document_service_url.rstrip("/")
        self._timeout = timeout_seconds

    def ping(self) -> None:
        try:
            req = urllib.request.Request(
                f"{self._url}/healthz",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Document service returned {resp.status}")
        except Exception as exc:
            raise RuntimeError(f"Document service unreachable: {exc}")

    def get_document_metadata(self, document_id: str, version: int) -> dict[str, Any]:
        """Fetch document metadata including model extraction fields."""
        try:
            req = urllib.request.Request(
                f"{self._url}/documents/{document_id}/versions/{version}/metadata",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise KeyError(f"Document {document_id} v{version} not found")
            raise RuntimeError(f"Document service error: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch document metadata: {exc}")

    def close(self) -> None:
        pass
