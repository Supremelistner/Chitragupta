"""Image resize utility for ingestion pipeline.

Downscales images before sending to VL models to reduce token usage.
Configurable max dimensions and JPEG quality.
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

logger = logging.getLogger("document_mgmt_service.image_utils")

# Default: 1280px on longest side, JPEG quality 82
# VL models typically use ~170 tokens per 512x512 tile,
# so 1280px = ~12 tiles = ~2048 tokens vs ~8000+ tokens for a 3000px image.
DEFAULT_MAX_DIMENSION: int = 1024
DEFAULT_JPEG_QUALITY: int = 82


def resize_image(
    image_bytes: bytes,
    mime_type: str,
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Tuple[bytes, str]:
    """Resize image if it exceeds max_dimension on either side.

    Returns (resized_bytes, output_mime_type).
    If the image is already small enough, returns the original bytes unchanged.
    PNG images are converted to JPEG for smaller payload.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — skipping image resize")
        return image_bytes, mime_type

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        logger.warning("Could not open image for resize — returning original")
        return image_bytes, mime_type

    orig_w, orig_h = img.size
    if max(orig_w, orig_h) <= max_dimension:
        # Already small enough
        return image_bytes, mime_type

    # Calculate new dimensions preserving aspect ratio
    if orig_w >= orig_h:
        new_w = max_dimension
        new_h = int(orig_h * (max_dimension / orig_w))
    else:
        new_h = max_dimension
        new_w = int(orig_w * (max_dimension / orig_h))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to RGB if needed (RGBA, palette, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Always output JPEG for smaller payload
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    resized_bytes = buf.getvalue()

    reduction_pct = (1 - len(resized_bytes) / len(image_bytes)) * 100
    logger.info(
        "Image resized: %dx%d -> %dx%d (%.0f%% reduction, %dKB -> %dKB)",
        orig_w, orig_h, new_w, new_h,
        reduction_pct,
        len(image_bytes) // 1024,
        len(resized_bytes) // 1024,
    )

    return resized_bytes, "image/jpeg"


def estimate_token_count(image_bytes: bytes) -> int:
    """Rough estimate of image token count for VL models.

    VL models typically process images as tiles:
    - ~170 tokens per 512x512 tile
    - Images are resized to fit within the model's grid
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        # Approximate: each 512x512 tile ≈ 170 tokens
        tiles = ((w + 511) // 512) * ((h + 511) // 512)
        return tiles * 170
    except Exception:
        return 0
