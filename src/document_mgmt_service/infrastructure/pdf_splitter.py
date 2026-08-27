"""PDF multi-page splitter — extracts individual pages as separate images.

When a multi-page PDF is uploaded, each page becomes its own document
with separate metadata and storage records. Single-page PDFs are
returned as-is.

Uses Pillow for PDF-to-image conversion (no external PDF libraries needed).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("document_mgmt_service.pdf_splitter")

# Maximum pages to split before treating as a single document
MAX_SPLIT_PAGES = 50


@dataclass(frozen=True)
class PDFPage:
    """A single extracted page from a PDF."""
    page_number: int  # 1-indexed
    image_bytes: bytes
    content_type: str  # image/png or image/jpeg
    width: int
    height: int


@dataclass(frozen=True)
class PDFSplitResult:
    """Result of splitting a PDF into pages."""
    total_pages: int
    pages: list[PDFPage]
    is_multi_page: bool  # True if >1 page extracted


def split_pdf(
    pdf_bytes: bytes,
    *,
    max_pages: int = MAX_SPLIT_PAGES,
    output_format: str = "png",
    dpi: int = 150,
) -> PDFSplitResult:
    """Split a PDF into individual page images.

    Args:
        pdf_bytes: Raw PDF file bytes.
        max_pages: Maximum number of pages to extract (safety limit).
        output_format: 'png' or 'jpeg' for extracted page images.
        dpi: Resolution for rendering (150 is good for VL models).

    Returns:
        PDFSplitResult with extracted page images.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not available, cannot split PDF")
        return PDFSplitResult(total_pages=0, pages=[], is_multi_page=False)

    pages: list[PDFPage] = []

    try:
        # Open PDF with Pillow — it can render PDF pages as images
        img = Image.open(io.BytesIO(pdf_bytes))

        # Check if PDF has multiple frames (pages)
        page_count = getattr(img, "n_frames", 1)
        is_multi = page_count > 1

        if not is_multi:
            # Single page — just encode it
            page_bytes = _encode_page(img, output_format)
            pages.append(PDFPage(
                page_number=1,
                image_bytes=page_bytes,
                content_type=f"image/{output_format}",
                width=img.width,
                height=img.height,
            ))
        else:
            # Multi-page — extract each page
            extract_count = min(page_count, max_pages)
            for i in range(extract_count):
                try:
                    img.seek(i)
                    # Convert to RGB for consistent output
                    if img.mode in ("RGBA", "P", "LA"):
                        img_rgb = img.convert("RGB")
                    else:
                        img_rgb = img

                    page_bytes = _encode_page(img_rgb, output_format)
                    pages.append(PDFPage(
                        page_number=i + 1,
                        image_bytes=page_bytes,
                        content_type=f"image/{output_format}",
                        width=img_rgb.width,
                        height=img_rgb.height,
                    ))
                except Exception as exc:
                    logger.warning("Failed to extract page %d: %s", i + 1, exc)
                    continue

            if page_count > max_pages:
                logger.warning(
                    "PDF has %d pages but only %d extracted (max_pages=%d)",
                    page_count, len(pages), max_pages,
                )

        return PDFSplitResult(
            total_pages=page_count,
            pages=pages,
            is_multi_page=is_multi,
        )

    except Exception as exc:
        logger.error("Failed to split PDF: %s", exc)
        return PDFSplitResult(total_pages=0, pages=[], is_multi_page=False)


def _encode_page(img: "Image.Image", fmt: str) -> bytes:
    """Encode a PIL Image to bytes."""
    buf = io.BytesIO()
    if fmt == "jpeg":
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def is_pdf(content_type: str | None, filename: str | None = None) -> bool:
    """Check if the file is a PDF based on content type or extension."""
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


def is_multi_page_pdf(pdf_bytes: bytes) -> bool:
    """Quick check if a PDF has multiple pages (without full extraction)."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(pdf_bytes))
        return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False
