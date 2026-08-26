"""Model-backed OCR adapter — replaces NullOCRAdapter with Qwen inference.

This adapter delegates text extraction to the model service, treating it as
an OCR replacement.  When the model service is available, this adapter is
preferred over the NullOCRAdapter.

The model service handles:
- Text extraction from images (replaces OCR)
- Hindi/Devanagari + English bilingual support
- No separate OCR service needed
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger("document_mgmt_service.model_ocr")


class ModelInferenceClient(Protocol):
    """Protocol for calling the model service from the document service."""

    def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        """Extract text from an image using the model service."""
        ...


class ModelOCRAdapter:
    """Delegates text extraction to the model service.

    Implements the same interface as OCRService but uses the model service
    for inference instead of a traditional OCR engine.
    """

    def __init__(self, inference_client: ModelInferenceClient) -> None:
        self._client = inference_client

    def ping(self) -> None:
        return None

    def extract_text(self, file_path) -> str:
        """Read file and delegate to model service for text extraction."""
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            logger.warning("File not found for model OCR: %s", path)
            return ""

        image_bytes = path.read_bytes()
        # Detect MIME type from extension
        suffix = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".pdf": "application/pdf",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        try:
            return self._client.extract_text(image_bytes=image_bytes, mime_type=mime_type)
        except Exception as exc:
            logger.exception("Model OCR extraction failed: %s", exc)
            return ""

    def close(self) -> None:
        pass
