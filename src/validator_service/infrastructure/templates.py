"""Infrastructure — in-memory template registry with pre-built document templates.

For V1, templates are defined in code.  Later, this can be backed by a database
or loaded from YAML/JSON files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from validator_service.domain.models import (
    DocumentTemplate,
    ExpiryStatus,
    FieldRule,
    FieldType,
    StructuralRule,
    TemporalRule,
    TemporalRuleType,
)

logger = logging.getLogger("validator_service.templates")


# ===========================================================================
# Pre-built templates for Indian documents
# ===========================================================================

_AADHAAR_CARD_TEMPLATE = DocumentTemplate(
    template_id="aadhaar_card_v1",
    document_type="identity_document",
    document_sub_type="aadhaar",
    variant_name="card_print",
    description="Aadhaar card — 2-sided printout (front: photo + number, back: address + govt text)",
    field_rules=(
        FieldRule(
            name="aadhaar_number",
            field_type=FieldType.GOVERNMENT_ID,
            required=True,
            description="12-digit Aadhaar number",
            regex=r"^\d{4}\s?\d{4}\s?\d{4}$",
            page=1,
        ),
        FieldRule(
            name="name",
            field_type=FieldType.PERSON_NAME,
            required=True,
            description="Cardholder's name (English)",
            min_length=2,
            bilingual=True,
            page=1,
        ),
        FieldRule(
            name="date_of_birth",
            field_type=FieldType.DATE,
            required=True,
            description="Date of birth in DD/MM/YYYY format",
            regex=r"^\d{2}/\d{2}/\d{4}$",
            page=1,
        ),
        FieldRule(
            name="gender",
            field_type=FieldType.FIXED_VALUE,
            required=True,
            description="Gender (MALE / FEMALE)",
            allowed_values=("MALE", "FEMALE", "महिला", "पुरुष"),
            page=1,
        ),
        FieldRule(
            name="mobile_number",
            field_type=FieldType.PHONE,
            required=False,
            description="Registered mobile number",
            regex=r"^\d{10}$",
            page=1,
        ),
        FieldRule(
            name="address",
            field_type=FieldType.ADDRESS,
            required=False,
            description="Registered address (back of card)",
            page=2,
        ),
    ),
    structural_rules=(
        StructuralRule(
            name="has_unique_id",
            description="Must contain exactly one 12-digit Aadhaar number",
            rule_type="format",
            parameters={"pattern": r"\d{4}\s?\d{4}\s?\d{4}", "count": 1},
        ),
    ),
    min_pages=1,
    max_pages=2,
    expected_content_types=("image/jpeg", "image/png", "application/pdf"),
    temporal_rules=(
        TemporalRule(
            rule_type=TemporalRuleType.NO_EXPIRY,
            description="Aadhaar is a government ID with no expiry — valid indefinitely",
        ),
    ),
)

_AADHAAR_LETTER_TEMPLATE = DocumentTemplate(
    template_id="aadhaar_letter_v1",
    document_type="identity_document",
    document_sub_type="aadhaar",
    variant_name="letter",
    description="Aadhaar letter — full A4 page with govt header, photo, details, and guidelines",
    field_rules=(
        FieldRule(
            name="aadhaar_number",
            field_type=FieldType.GOVERNMENT_ID,
            required=True,
            description="12-digit Aadhaar number",
            regex=r"^\d{4}\s?\d{4}\s?\d{4}$",
        ),
        FieldRule(
            name="name",
            field_type=FieldType.PERSON_NAME,
            required=True,
            description="Cardholder's name (English)",
            min_length=2,
            bilingual=True,
        ),
        FieldRule(
            name="date_of_birth",
            field_type=FieldType.DATE,
            required=True,
            description="Date of birth",
            regex=r"^\d{2}/\d{2}/\d{4}$",
        ),
        FieldRule(
            name="gender",
            field_type=FieldType.FIXED_VALUE,
            required=True,
            description="Gender",
            allowed_values=("MALE", "FEMALE", "महिला", "पुरुष"),
        ),
        FieldRule(
            name="mobile_number",
            field_type=FieldType.PHONE,
            required=False,
            description="Registered mobile number",
            regex=r"^\d{10}$",
        ),
        FieldRule(
            name="address",
            field_type=FieldType.ADDRESS,
            required=True,
            description="Full registered address",
        ),
    ),
    structural_rules=(
        StructuralRule(
            name="has_unique_id",
            description="Must contain exactly one 12-digit Aadhaar number",
            rule_type="format",
            parameters={"pattern": r"\d{4}\s?\d{4}\s?\d{4}", "count": 1},
        ),
        StructuralRule(
            name="govt_header",
            description="Must contain government branding text",
            rule_type="has_section",
            parameters={"text_hints": ["GOVERNMENT OF INDIA", "भारत सरकार", "UIDAI"]},
        ),
    ),
    min_pages=1,
    max_pages=2,
    expected_content_types=("image/jpeg", "image/png", "application/pdf"),
    temporal_rules=(
        TemporalRule(
            rule_type=TemporalRuleType.NO_EXPIRY,
            description="Aadhaar letter — no expiry, valid indefinitely",
        ),
    ),
)

_EAADHAAR_TEMPLATE = DocumentTemplate(
    template_id="eaadhaar_v1",
    document_type="identity_document",
    document_sub_type="aadhaar",
    variant_name="e_aadhaar",
    description="e-Aadhaar — downloaded PDF with UIDAI watermark and QR code",
    field_rules=(
        FieldRule(
            name="aadhaar_number",
            field_type=FieldType.GOVERNMENT_ID,
            required=True,
            description="12-digit Aadhaar number",
            regex=r"^\d{4}\s?\d{4}\s?\d{4}$",
        ),
        FieldRule(
            name="name",
            field_type=FieldType.PERSON_NAME,
            required=True,
            description="Cardholder's name",
            bilingual=True,
        ),
        FieldRule(
            name="date_of_birth",
            field_type=FieldType.DATE,
            required=True,
            regex=r"^\d{2}/\d{2}/\d{4}$",
        ),
        FieldRule(
            name="gender",
            field_type=FieldType.FIXED_VALUE,
            required=True,
            allowed_values=("MALE", "FEMALE", "महिला", "पुरुष"),
        ),
        FieldRule(
            name="address",
            field_type=FieldType.ADDRESS,
            required=True,
        ),
    ),
    structural_rules=(
        StructuralRule(
            name="has_unique_id",
            description="Must contain exactly one 12-digit Aadhaar number",
            rule_type="format",
            parameters={"pattern": r"\d{4}\s?\d{4}\s?\d{4}", "count": 1},
        ),
        StructuralRule(
            name="pdf_format",
            description="Must be a PDF document",
            rule_type="format",
            parameters={"content_type": "application/pdf"},
        ),
    ),
    min_pages=1,
    max_pages=3,
    expected_content_types=("application/pdf",),
    temporal_rules=(
        TemporalRule(
            rule_type=TemporalRuleType.NO_EXPIRY,
            description="e-Aadhaar — no expiry, valid indefinitely",
        ),
    ),
)

_PAN_CARD_TEMPLATE = DocumentTemplate(
    template_id="pan_card_v1",
    document_type="identity_document",
    document_sub_type="pan",
    variant_name="card",
    description="PAN card — plastic ID card with name, DOB, PAN number, photo",
    field_rules=(
        FieldRule(
            name="PAN_number",
            field_type=FieldType.GOVERNMENT_ID,
            required=True,
            description="10-character PAN number (5 letters, 4 digits, 1 letter)",
            regex=r"^[A-Z]{5}\d{4}[A-Z]$",
        ),
        FieldRule(
            name="name",
            field_type=FieldType.PERSON_NAME,
            required=True,
            min_length=2,
        ),
        FieldRule(
            name="date_of_birth",
            field_type=FieldType.DATE,
            required=True,
            regex=r"^\d{2}/\d{2}/\d{4}$",
        ),
        FieldRule(
            name="father_name",
            field_type=FieldType.PERSON_NAME,
            required=True,
        ),
    ),
    structural_rules=(
        StructuralRule(
            name="has_pan_format",
            description="Must contain a valid PAN format",
            rule_type="format",
            parameters={"pattern": r"[A-Z]{5}\d{4}[A-Z]"},
        ),
    ),
    min_pages=1,
    max_pages=1,
    expected_content_types=("image/jpeg", "image/png", "application/pdf"),
    temporal_rules=(
        TemporalRule(
            rule_type=TemporalRuleType.NO_EXPIRY,
            description="PAN card — no expiry, valid until surrender/cancellation",
        ),
    ),
)

_MARKSHEET_TEMPLATE = DocumentTemplate(
    template_id="marksheet_v1",
    document_type="academic_record",
    document_sub_type="marksheet",
    variant_name="standard",
    description="Academic marksheet — board/school examination results with subjects and grades",
    field_rules=(
        FieldRule(
            name="student_name",
            field_type=FieldType.PERSON_NAME,
            required=True,
            min_length=2,
        ),
        FieldRule(
            name="roll_number",
            field_type=FieldType.NUMBER,
            required=True,
            min_length=5,
        ),
        FieldRule(
            name="mother_name",
            field_type=FieldType.PERSON_NAME,
            required=False,
        ),
        FieldRule(
            name="father_name",
            field_type=FieldType.PERSON_NAME,
            required=False,
        ),
        FieldRule(
            name="date_of_birth",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldRule(
            name="school",
            field_type=FieldType.ORGANIZATION,
            required=True,
        ),
        FieldRule(
            name="result",
            field_type=FieldType.FIXED_VALUE,
            required=True,
            allowed_values=("PASS", "FAIL", "COMPARTMENTAL", "DISTINCTION", "FIRST CLASS", "SECOND CLASS"),
        ),
    ),
    structural_rules=(
        StructuralRule(
            name="has_subjects",
            description="Must contain at least 3 subject entries",
            rule_type="has_section",
            parameters={"min_subjects": 3},
        ),
    ),
    min_pages=1,
    max_pages=3,
    expected_content_types=("image/jpeg", "image/png", "application/pdf"),
    temporal_rules=(
        TemporalRule(
            rule_type=TemporalRuleType.ISSUE_BASED,
            description="Marksheet — valid from issue date, no fixed expiry, but consider freshness",
            issue_date_field="issue_date",
        ),
    ),
)

_PASSPORT_TEMPLATE = DocumentTemplate(
    template_id="passport_v1",
    document_type="identity_document",
    document_sub_type="passport",
    variant_name="bio_page",
    description="Indian passport — biographical data page",
    field_rules=(
        FieldRule(
            name="passport_number",
            field_type=FieldType.GOVERNMENT_ID,
            required=True,
            regex=r"^[A-Z]\d{8}$",
        ),
        FieldRule(
            name="name",
            field_type=FieldType.PERSON_NAME,
            required=True,
        ),
        FieldRule(
            name="date_of_birth",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldRule(
            name="nationality",
            field_type=FieldType.TEXT,
            required=True,
            allowed_values=("INDIAN", "भारतीय"),
        ),
        FieldRule(
            name="sex",
            field_type=FieldType.FIXED_VALUE,
            required=True,
            allowed_values=("M", "F", "MALE", "FEMALE"),
        ),
    ),
    structural_rules=(),
    min_pages=1,
    max_pages=1,
    expected_content_types=("image/jpeg", "image/png", "application/pdf"),
    temporal_rules=(
        TemporalRule(
            rule_type=TemporalRuleType.FIXED_LIFETIME,
            description="Indian passport — expires 10 years from date of issue",
            issue_date_field="issue_date",
            expiry_field="expiry_date",
            lifetime_days=3650,
            warning_days=180,
        ),
    ),
)


# All built-in templates
_BUILTIN_TEMPLATES: list[DocumentTemplate] = [
    _AADHAAR_CARD_TEMPLATE,
    _AADHAAR_LETTER_TEMPLATE,
    _EAADHAAR_TEMPLATE,
    _PAN_CARD_TEMPLATE,
    _MARKSHEET_TEMPLATE,
    _PASSPORT_TEMPLATE,
]


# ===========================================================================
# Registry implementation
# ===========================================================================

class InMemoryTemplateRegistry:
    """In-memory template registry for V1.

    Loads built-in templates at init.  Can be extended at runtime
    via add_template().  Later, this can load from a database or
    YAML files.
    """

    def __init__(self, extra_templates: list[DocumentTemplate] | None = None) -> None:
        self._templates: dict[str, DocumentTemplate] = {}
        for t in _BUILTIN_TEMPLATES:
            self._templates[t.template_id] = t
        if extra_templates:
            for t in extra_templates:
                self._templates[t.template_id] = t
        logger.info("Loaded %d document templates", len(self._templates))

    def ping(self) -> None:
        pass

    def get_templates(
        self, document_type: str | None = None, document_sub_type: str | None = None,
    ) -> list[DocumentTemplate]:
        results = list(self._templates.values())
        if document_type:
            results = [t for t in results if t.document_type == document_type]
        if document_sub_type:
            results = [t for t in results if t.document_sub_type == document_sub_type]
        return results

    def get_template(self, template_id: str) -> DocumentTemplate | None:
        return self._templates.get(template_id)

    def list_all(self) -> list[DocumentTemplate]:
        return list(self._templates.values())

    def add_template(self, template: DocumentTemplate) -> None:
        self._templates[template.template_id] = template
        logger.info("Added template: %s", template.template_id)

    def close(self) -> None:
        pass
