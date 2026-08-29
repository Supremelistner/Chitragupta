"""Field value retrieval service.

This module is the read-side counterpart of the ingestion pipeline. When the
user (or the orchestrator) asks for a specific field value (e.g. "what is my
Aadhaar number?"), the service:

1. Loads the document version from Postgres (encrypted at rest).
2. Looks up the requested field in ``extracted_fields``.
3. Returns a structured response that names the source document but does NOT
   embed any URL \u2014 the chat UI shows "Source: <name> (<relation>)" and a
   "Press Retrieve original file to view the full document" suggestion.
4. The actual PII value is gated: callers must pass ``confirm: true`` to
   receive the value, and the access is audited via the existing policy
   layer in :mod:`document_mgmt_service.application.access`.

The split between "field_pointers in Qdrant" (just names) and "extracted_fields
in Postgres" (full values) is what lets the LLM answer "I have your Aadhaar
number cached" without ever surfacing the actual digits until the user
confirms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from document_mgmt_service.domain.models import DocumentVersionRecord
from document_mgmt_service.domain.ports import PostgreSQLDocumentRepository

logger = logging.getLogger("document_mgmt_service.fields")


@dataclass(frozen=True, slots=True)
class FieldValueSource:
    """Human-readable description of where a field value came from.

    Intentionally has no URL \u2014 the chat UI is the only place that surfaces this
    object, and URLs are stripped in the post-processor. If a future use case
    needs a link (e.g. an "open document" button), the value comes from a
    separate ``document_open_url`` field on this dataclass and is generated
    by the HTTP layer, not the application layer.
    """

    name: str                  # e.g. "Aadhaar card"
    document_id: str
    version: int
    relation: str | None      # "self" / "mother" / "spouse" / ...
    relation_name: str | None
    page_number: int | None
    field: str
    extracted_at: datetime | None


@dataclass(frozen=True, slots=True)
class FieldValueResult:
    """Result of a get_field_value call.

    Variants:

    * ``status == "not_found"`` \u2014 the document exists but doesn't carry the
      requested field. ``suggestion`` tells the user what is available.
    * ``status == "requires_confirmation"`` \u2014 the document carries the
      field; the caller must pass ``confirm=True`` to actually get the value.
    * ``status == "ok"`` \u2014 the value is returned. ``source`` describes the
      document in human terms. ``suggestion`` is the "press Retrieve to see
      original file" line.
    """

    status: str
    field: str
    value: Any | None = None
    source: FieldValueSource | None = None
    suggestion: str | None = None
    available_fields: list[str] | None = None


class FieldAccessError(RuntimeError):
    pass


def _describe_source(
    record: DocumentVersionRecord,
    field_name: str,
    page_number: int | None,
) -> FieldValueSource:
    """Build a human-readable source description for a field.

    The ``name`` field is the safe description (no PII). If that's missing,
    fall back to the original filename.
    """
    display_name = (
        record.description_safe
        or record.summary
        or record.original_filename
    )
    relation = record.owner_type
    relation_name = record.relation_name
    return FieldValueSource(
        name=display_name or "Document",
        document_id=record.document_id,
        version=record.version,
        relation=relation,
        relation_name=relation_name,
        page_number=page_number,
        field=field_name,
        extracted_at=record.completed_at or record.updated_at or record.created_at,
    )


def _suggestion_for(field_name: str) -> str:
    return (
        f"Source: {field_name} is available. "
        "Press \u2018Retrieve original file\u2019 to view the full document."
    )


def get_field_value(
    repository: PostgreSQLDocumentRepository,
    *,
    document_id: str,
    version: int,
    field_name: str,
    confirm: bool = False,
) -> FieldValueResult:
    """Look up a field value on a document version.

    Two-step protocol: the first call returns ``requires_confirmation`` and
    shows the source; only when the caller passes ``confirm=True`` is the
    actual value returned.
    """
    if not field_name:
        raise FieldAccessError("field_name is required")
    record = repository.get_version(document_id, version)
    if record is None:
        raise FieldAccessError(
            f"Document {document_id} v{version} not found"
        )
    extracted = record.extracted_fields or {}
    if field_name not in extracted:
        return FieldValueResult(
            status="not_found",
            field=field_name,
            suggestion=(
                f"The field '{field_name}' was not extracted from this "
                "document. Available fields: "
                + (", ".join(sorted(extracted.keys())) or "none")
            ),
            available_fields=sorted(extracted.keys()),
        )
    if not confirm:
        return FieldValueResult(
            status="requires_confirmation",
            field=field_name,
            source=_describe_source(record, field_name, page_number=None),
            suggestion=(
                f"Field '{field_name}' is cached for this document. "
                "Confirm to see the value."
            ),
            available_fields=sorted(extracted.keys()),
        )
    # Audit log \u2014 we deliberately don't log the value itself.
    logger.info(
        "Field value revealed: document=%s v=%d field=%s",
        document_id, version, field_name,
    )
    return FieldValueResult(
        status="ok",
        field=field_name,
        value=extracted[field_name],
        source=_describe_source(record, field_name, page_number=None),
        suggestion=_suggestion_for(field_name),
        available_fields=sorted(extracted.keys()),
    )
