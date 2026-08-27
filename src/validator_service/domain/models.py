"""Domain models for the Validator Service.

Defines document templates (what a valid document looks like), validation rules,
validation results, and alert payloads.  No service logic — pure data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    """Result of a document validation."""
    VALID = "VALID"
    INVALID = "INVALID"
    SUSPICIOUS = "SUSPICIOUS"
    UNKNOWN_DOCUMENT = "UNKNOWN_DOCUMENT"
    ERROR = "ERROR"


class AlertSeverity(str, Enum):
    """How urgently the document service should react."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class FieldType(str, Enum):
    """Supported field types in template rules."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    GOVERNMENT_ID = "government_id"
    PERSON_NAME = "person_name"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    ORGANIZATION = "organization"
    GRADE = "grade"
    AMOUNT = "amount"
    FIXED_VALUE = "fixed_value"
    LIST = "list"


class ExpiryStatus(str, Enum):
    """Temporal validity status of a document."""
    VALID = "VALID"                      # Document is within its validity period
    EXPIRED = "EXPIRED"                  # Document has passed its expiry date
    EXPIRING_SOON = "EXPIRING_SOON"      # Document will expire within the warning window
    NO_EXPIRY = "NO_EXPIRY"              # Document type has no expiry (e.g., Aadhaar, PAN)
    UNKNOWN = "UNKNOWN"                  # Could not determine temporal validity


class TemporalRuleType(str, Enum):
    """Types of temporal rules."""
    NO_EXPIRY = "no_expiry"              # Document never expires
    FIXED_LIFETIME = "fixed_lifetime"    # Expires N days after issue date
    EXPIRY_FIELD = "expiry_field"        # Expiry date is in a specific field
    ISSUE_BASED = "issue_based"          # Valid from issue date, no fixed expiry
    DOB_BASED = "dob_based"              # Age-based validity (e.g., minor status)


# ---------------------------------------------------------------------------
# Template models — describe what a valid document looks like
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FieldRule:
    """A single field that must be present in the document, with constraints."""
    name: str
    field_type: FieldType
    required: bool = True
    description: str = ""
    regex: str | None = None          # Optional regex the value must match
    min_length: int | None = None
    max_length: int | None = None
    allowed_values: tuple[str, ...] = ()  # For FIXED_VALUE / LIST types
    page: int | None = None           # Expected page (None = any page)
    bilingual: bool = False           # True if field has Hindi + English variants


@dataclass(frozen=True, slots=True)
class StructuralRule:
    """A structural constraint on the document (layout, page count, sections)."""
    name: str
    description: str
    rule_type: str                    # "page_count" | "has_section" | "layout" | "format"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TemporalRule:
    """Time-based validity rule for a document.

    Defines when a document expires and how to check it.
    Examples:
    - Aadhaar: NO_EXPIRY (government ID, valid indefinitely)
    - Passport: FIXED_LIFETIME with lifetime_days=3650 (10 years)
    - Insurance: EXPIRY_FIELD reading from 'policy_expiry_date'
    - Marksheet: ISSUE_BASED (valid from issue, no fixed expiry)
    """
    rule_type: TemporalRuleType
    description: str
    expiry_field: str | None = None     # Field name containing expiry date (for EXPIRY_FIELD type)
    issue_date_field: str | None = None # Field name containing issue date
    lifetime_days: int | None = None    # Days after issue date until expiry (for FIXED_LIFETIME)
    warning_days: int | None = None     # Days before expiry to trigger EXPIRING_SOON
    dob_field: str | None = None        # Field name for DOB (for DOB_BASED)
    max_age_years: int | None = None    # Max age in years (for DOB_BASED)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentTemplate:
    """Template describing a valid document variant.

    A single document type (e.g., Aadhaar) can have multiple templates
    because the same document can appear in different physical formats:
    - Aadhaar card (2-sided printout)
    - Aadhaar letter (full A4 page with govt guidelines)
    - e-Aadhaar (downloaded PDF with watermark)
    """
    template_id: str
    document_type: str                 # e.g. "identity_document"
    document_sub_type: str             # e.g. "aadhaar"
    variant_name: str                  # e.g. "card_print", "letter", "e_aadhaar"
    description: str
    field_rules: tuple[FieldRule, ...] = ()
    structural_rules: tuple[StructuralRule, ...] = ()
    temporal_rules: tuple[TemporalRule, ...] = ()  # Time-based validity rules
    min_pages: int | None = None
    max_pages: int | None = None
    expected_content_types: tuple[str, ...] = ()  # e.g. ("image/jpeg", "application/pdf")
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation models — input and output of the validation pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ValidationRequest:
    """Request to validate a document against known templates."""
    document_id: str
    version: int
    document_type: str | None = None       # Pre-classified type (from model extraction)
    document_sub_type: str | None = None   # Pre-classified sub-type
    extracted_text: str | None = None      # OCR / model-extracted text
    metadata: dict[str, Any] = field(default_factory=dict)  # Model extraction output
    image_bytes: bytes | None = None       # For model-based visual validation
    image_mime_type: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class FieldValidation:
    """Validation result for a single field."""
    field_name: str
    expected_type: FieldType
    present: bool
    value: str | None = None
    matches_rules: bool = True
    violations: tuple[str, ...] = ()       # What went wrong
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ValidationStep:
    """Result of one validation step (template match or model check)."""
    step_name: str
    passed: bool
    details: str = ""
    field_results: tuple[FieldValidation, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class TemporalCheck:
    """Result of a temporal validity check."""
    rule_type: TemporalRuleType
    description: str
    status: ExpiryStatus
    expiry_date: datetime | None = None     # When the document expires (if known)
    issue_date: datetime | None = None      # When the document was issued
    days_until_expiry: int | None = None    # Days until expiry (negative = expired)
    details: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Complete validation result for a document."""
    document_id: str
    version: int
    status: ValidationStatus
    matched_template: str | None = None     # template_id if matched
    steps: tuple[ValidationStep, ...] = ()
    overall_confidence: float = 0.0
    is_authentic: bool = True               # False if fraud indicators detected
    risk_score: float = 0.0                 # 0.0 = safe, 1.0 = definitely fake
    alerts: tuple[AlertPayload, ...] = ()   # Alerts to send to document service
    notes: tuple[str, ...] = ()
    # Temporal validity
    temporal_status: ExpiryStatus = ExpiryStatus.UNKNOWN
    temporal_checks: tuple[TemporalCheck, ...] = ()
    temporal_tags: tuple[str, ...] = ()     # Human-readable tags: "EXPIRED", "VALID", etc.
    latency_ms: float | None = None
    request_id: str | None = None
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Alert models — sent to document service when validation fails
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AlertPayload:
    """Alert to notify the document service about a validation issue."""
    document_id: str
    version: int
    severity: AlertSeverity
    alert_type: str                       # "fake_document" | "structure_mismatch" | "template_unknown" | "high_risk"
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    request_id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AlertResult:
    """Result of sending an alert to the document service."""
    sent: bool
    alert_id: str | None = None
    error: str | None = None
    latency_ms: float | None = None
