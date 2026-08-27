"""Port interfaces for the Validator Service.

Providers implement these protocols.  Swapping providers is a config change.
"""

from __future__ import annotations

from typing import Protocol, Any

from validator_service.domain.models import (
    DocumentTemplate,
    ValidationResult,
    ValidationRequest,
    AlertPayload,
    AlertResult,
)


class TemplateRegistry(Protocol):
    """Registry of known document templates."""

    def ping(self) -> None: ...

    def get_templates(
        self, document_type: str | None = None, document_sub_type: str | None = None,
    ) -> list[DocumentTemplate]:
        """Return templates matching the given filters."""
        ...

    def get_template(self, template_id: str) -> DocumentTemplate | None:
        """Return a single template by ID."""
        ...

    def list_all(self) -> list[DocumentTemplate]:
        """Return every registered template."""
        ...

    def close(self) -> None: ...


class ModelValidator(Protocol):
    """Uses the model service to visually inspect a document for authenticity."""

    def ping(self) -> None: ...

    def validate_document(
        self,
        request: ValidationRequest,
        template: DocumentTemplate | None = None,
    ) -> ValidationResult:
        """Ask the model to check if the document matches the template."""
        ...

    def close(self) -> None: ...


class AlertSender(Protocol):
    """Sends alerts to the document service when validation fails."""

    def ping(self) -> None: ...

    def send_alert(self, alert: AlertPayload) -> AlertResult:
        """Send a validation-failure alert to the document service."""
        ...

    def close(self) -> None: ...


class DocumentServiceClient(Protocol):
    """Reads document metadata from the document management service."""

    def ping(self) -> None: ...

    def get_document_metadata(self, document_id: str, version: int) -> dict[str, Any]:
        """Fetch document metadata from the document service."""
        ...

    def close(self) -> None: ...
