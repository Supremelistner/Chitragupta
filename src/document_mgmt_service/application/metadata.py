from __future__ import annotations

import re
from pathlib import Path

from document_mgmt_service.domain.models import DocumentFileKind, DocumentPrivacyClassification


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
_LONG_NUMBER_RE = re.compile(r"\b\d{8,}\b")
_SECRET_RE = re.compile(
    r"\b(?:password|passcode|secret|token|api[_ -]?key|private key|confidential)\b",
    re.IGNORECASE,
)
_SENSITIVE_RE = re.compile(
    r"\b(?:ssn|social security|aadhaar|aadhar|passport|credit card|bank account|salary|medical|diagnosis|prescription)\b",
    re.IGNORECASE,
)
_PUBLIC_RE = re.compile(
    r"\b(?:public|press release|brochure|marketing|announcement|website)\b",
    re.IGNORECASE,
)
_PRIVATE_RE = re.compile(
    r"\b(?:invoice|statement|receipt|contract|nda|resume|cv|pay slip|payslip|internal|draft)\b",
    re.IGNORECASE,
)


def infer_file_kind(filename: str, content_type: str | None) -> DocumentFileKind:
    filename_lower = filename.lower()
    content_type_lower = (content_type or "").lower()
    if "pdf" in content_type_lower or filename_lower.endswith(".pdf"):
        return DocumentFileKind.PDF
    if content_type_lower.startswith("image/") or filename_lower.endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp")
    ):
        return DocumentFileKind.IMAGE
    return DocumentFileKind.UNKNOWN


def redact_sensitive_text(text: str, *, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())
    cleaned = _EMAIL_RE.sub("[redacted-email]", cleaned)
    cleaned = _PHONE_RE.sub("[redacted-phone]", cleaned)
    cleaned = _LONG_NUMBER_RE.sub("[redacted-number]", cleaned)
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 1].rstrip() + "…"
    return cleaned


def generate_safe_description(
    *,
    filename: str,
    file_kind: DocumentFileKind,
    extracted_text: str | None,
    metadata: dict[str, object],
    description_hint: str | None = None,
) -> str:
    if description_hint:
        return redact_sensitive_text(description_hint)

    parts = [f"{file_kind.value.lower()} document", f"named {filename!r}"]
    if metadata.get("page_count") is not None:
        parts.append(f"with {metadata['page_count']} page(s)")
    if extracted_text:
        snippet = redact_sensitive_text(extracted_text)
        if snippet:
            parts.append(f"containing {snippet}")
    description = ", ".join(parts)
    return redact_sensitive_text(description, max_length=300)


def classify_privacy(
    *,
    filename: str,
    content_type: str | None,
    extracted_text: str | None,
    description: str | None,
    privacy_hint: DocumentPrivacyClassification | None,
) -> DocumentPrivacyClassification:
    if privacy_hint is not None:
        return privacy_hint

    haystack = " ".join(
        part
        for part in (filename, content_type or "", extracted_text or "", description or "")
        if part
    )
    if _SECRET_RE.search(haystack) or _SENSITIVE_RE.search(haystack):
        return DocumentPrivacyClassification.SENSITIVE
    if _PRIVATE_RE.search(haystack):
        return DocumentPrivacyClassification.PRIVATE
    if _PUBLIC_RE.search(haystack):
        return DocumentPrivacyClassification.OPEN
    return DocumentPrivacyClassification.OPEN_NOT_PUBLIC

