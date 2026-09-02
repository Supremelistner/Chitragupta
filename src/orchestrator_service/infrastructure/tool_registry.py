"""Tool registry — maps tool names to services and confirmation rules.

The registry defines:
1. Which service owns each tool
2. Which tools require user confirmation before execution
3. Tool schemas for the LLM (OpenAI function calling format)
"""
from __future__ import annotations

import logging
from typing import Any

from orchestrator_service.domain.models import ConfirmationType, ServiceTarget

logger = logging.getLogger("orchestrator.tool_registry")


# ---------------------------------------------------------------------------
# Tool definitions: name → (service, description, schema, confirmation_type)
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: dict[str, tuple[ServiceTarget, str, dict[str, Any], ConfirmationType | None]] = {
    # ── Document Service ──────────────────────────────────────────────
    "upload_document": (
        ServiceTarget.DOCUMENT,
        "Ingest a document (PDF or image) into the system.",
        {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Original filename"},
                "content_base64": {"type": "string", "description": "Base64-encoded file content"},
                "content_type": {"type": "string", "description": "MIME type"},
                "description": {"type": "string", "description": "Optional description"},
                "privacy": {"type": "string", "description": "Privacy hint: SENSITIVE, PRIVATE, OPEN"},
            },
            "required": ["filename", "content_base64"],
        },
        None,  # No confirmation needed for upload
    ),
    "list_documents": (
        ServiceTarget.DOCUMENT,
        "List all known documents with summaries.",
        {"type": "object", "properties": {}},
        None,
    ),
    "search_documents": (
        ServiceTarget.DOCUMENT,
        "Search documents by semantic similarity. Returns document IDs and summaries. ALWAYS call this FIRST when the user mentions a document by name or description — use the returned document_id and version for all other tool calls.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What the user is looking for, e.g. 'Aadhaar card', 'marksheet', 'insurance policy'"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
        None,
    ),
    "search_document_content": (
        ServiceTarget.DOCUMENT,
        "Search within document content for specific information.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer"},
                "document_id": {"type": "string"},
                "version": {"type": "integer"},
            },
            "required": ["query"],
        },
        None,
    ),
    "get_document_metadata": (
        ServiceTarget.DOCUMENT,
        "Get processing status and metadata for a document version. You MUST first call search_documents or list_documents to find the document_id.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents or list_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents or list_documents results"},
            },
            "required": ["document_id", "version"],
        },
        None,
    ),
    "get_document_description": (
        ServiceTarget.DOCUMENT,
        "Get the safe (non-PII) description for a document version. You MUST first call search_documents or list_documents to find the document_id.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents or list_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents or list_documents results"},
            },
            "required": ["document_id", "version"],
        },
        None,
    ),
    "get_evidence": (
        ServiceTarget.DOCUMENT,
        "Retrieve supporting evidence (text chunks) for a document. You MUST first call search_documents to find the document_id. System will ask user to confirm before retrieving.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents results"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["document_id", "version", "query"],
        },
        ConfirmationType.SENSITIVE_ACCESS,  # Evidence may contain PII
    ),
    # ── Document: file retrieval (requires confirmation) ──────────────
    "get_document": (
        ServiceTarget.DOCUMENT,
        "Download and retrieve the actual document file. You MUST first call search_documents or list_documents to find the document_id, then call this tool. System will ask user to confirm before downloading.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents or list_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents or list_documents results"},
            },
            "required": ["document_id", "version"],
        },
        ConfirmationType.FILE_RETRIEVAL,
    ),
    "get_page": (
        ServiceTarget.DOCUMENT,
        "Retrieve a specific page from a document. You MUST first call search_documents or list_documents to find the document_id. Requires user confirmation.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents or list_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents or list_documents results"},
                "page_number": {"type": "integer"},
            },
            "required": ["document_id", "version", "page_number"],
        },
        ConfirmationType.FILE_RETRIEVAL,
    ),
    "request_sensitive_access": (
        ServiceTarget.DOCUMENT,
        "Request approval for accessing sensitive document data. You MUST first call search_documents to find the document_id. System will ask user to confirm — call this when user asks for ID numbers, personal info, etc.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents results"},
                "intent": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["document_id", "version", "intent"],
        },
        ConfirmationType.SENSITIVE_ACCESS,
    ),
    "get_field_value": (
        ServiceTarget.DOCUMENT,
        "Look up a specific extracted field value (e.g. aadhaar_number, pan_number, date_of_birth) on a document version. Two-step protocol (policy §5, failures #1): the FIRST call returns status=requires_confirmation with a popup shown in the UI; the orchestrator pauses for the user to approve. After approval, call get_field_value again with the same document_id, version, field, AND confirm=true — this returns the actual value. ALWAYS prefer this over get_evidence for single-field lookups; use get_evidence only for free-form question answering over OCR text. Field names are fuzzy-matched: case-insensitive, common-suffix stripping (_card/_number), and small typo corrections are applied automatically; the response includes resolved_field showing the actual key used. The response also includes available_fields listing every extracted field name; surface those to the user when status is not_found so they can pick the right one.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Get this from search_documents or list_documents results"},
                "version": {"type": "integer", "description": "Get this from search_documents or list_documents results"},
                "field": {"type": "string", "description": "Field name to look up, e.g. 'aadhaar_number', 'pan_number', 'date_of_birth'"},
                "confirm": {"type": "boolean", "description": "Pass true on the SECOND call after the user approves. Defaults to false (first call returns requires_confirmation).", "default": False},
            },
            "required": ["document_id", "version", "field"],
        },
        ConfirmationType.SENSITIVE_ACCESS,
    ),
    # ── Model Service ─────────────────────────────────────────────────
    "classify_document": (
        ServiceTarget.MODEL,
        "Classify a document type using vision model.",
        {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string"},
                "image_path": {"type": "string"},
                "image_mime_type": {"type": "string"},
            },
        },
        None,
    ),
    "verify_ocr": (
        ServiceTarget.MODEL,
        "Verify OCR output against original image.",
        {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string"},
                "ocr_text": {"type": "string"},
            },
        },
        None,
    ),
    "extract_metadata": (
        ServiceTarget.MODEL,
        "Extract structured metadata from a document image.",
        {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string"},
                "image_path": {"type": "string"},
                "ocr_text": {"type": "string"},
            },
        },
        None,
    ),
    "summarize_content": (
        ServiceTarget.MODEL,
        "Summarize text content.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
        None,
    ),
    # ── Validator Service ─────────────────────────────────────────────
    "validate_document": (
        ServiceTarget.VALIDATOR,
        "Validate a document against known templates for authenticity.",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "version": {"type": "integer"},
            },
            "required": ["document_id"],
        },
        None,
    ),
    "validate_document_image": (
        ServiceTarget.VALIDATOR,
        "Validate a document image directly for authenticity.",
        {
            "type": "object",
            "properties": {
                "content_base64": {"type": "string"},
                "document_type": {"type": "string"},
                "document_sub_type": {"type": "string"},
            },
            "required": ["content_base64"],
        },
        None,
    ),
    "list_templates": (
        ServiceTarget.VALIDATOR,
        "List all registered document templates and their required fields.",
        {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
            },
        },
        None,
    ),
    "get_template": (
        ServiceTarget.VALIDATOR,
        "Get detailed schema for a specific document template.",
        {
            "type": "object",
            "properties": {
                "template_id": {"type": "string"},
            },
            "required": ["template_id"],
        },
        None,
    ),
    "get_alerts": (
        ServiceTarget.VALIDATOR,
        "Get validation alerts (fake/suspicious document detections).",
        {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
            },
        },
        None,
    ),
    # ── Web Search Service ────────────────────────────────────────────
    "web_search": (
        ServiceTarget.WEB_SEARCH,
        "Search the web for information.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        None,
    ),
    "web_scrape": (
        ServiceTarget.WEB_SEARCH,
        "Scrape a URL and return its content as markdown.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
        ConfirmationType.EXTERNAL_ACTION,
    ),
    "fetch_url": (
        ServiceTarget.WEB_SEARCH,
        "Fetch content from a URL.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
        None,
    ),
    # ── File download (requires confirmation) ─────────────────────────
    "download_file": (
        ServiceTarget.WEB_SEARCH,
        "Download a file (PDF, form) from a URL to local storage. Requires confirmation.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "filename": {"type": "string"},
            },
            "required": ["url"],
        },
        ConfirmationType.FILE_RETRIEVAL,
    ),
    # ── Bulk operations ─────────────────────────────────────────────
    "bulk_download": (
        ServiceTarget.DOCUMENT,
        "Download multiple documents at once as a zip file. System will ask user to confirm.",
        {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "description": "List of {document_id, version} objects to download",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "version": {"type": "integer"},
                        },
                        "required": ["document_id"],
                    },
                },
            },
            "required": ["document_ids"],
        },
        ConfirmationType.FILE_RETRIEVAL,
    ),
    "bulk_metadata": (
        ServiceTarget.DOCUMENT,
        "Get metadata for multiple documents at once.",
        {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "description": "List of {document_id, version} objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "version": {"type": "integer"},
                        },
                        "required": ["document_id"],
                    },
                },
            },
            "required": ["document_ids"],
        },
        None,
    ),
    # ── Health checks ─────────────────────────────────────────────────
    "health_document": (
        ServiceTarget.DOCUMENT,
        "Check document service health.",
        {"type": "object", "properties": {}},
        None,
    ),
    "health_model": (
        ServiceTarget.MODEL,
        "Check model service health.",
        {"type": "object", "properties": {}},
        None,
    ),
    "health_validator": (
        ServiceTarget.VALIDATOR,
        "Check validator service health.",
        {"type": "object", "properties": {}},
        None,
    ),
    "health_web_search": (
        ServiceTarget.WEB_SEARCH,
        "Check web search service health.",
        {"type": "object", "properties": {}},
        None,
    ),
}

# Tools that are exposed to the LLM (subset of all tools)
# File retrieval tools are not exposed to the LLM; the orchestrator decides
# when to offer them after document search and confirmation.
LLM_VISIBLE_TOOLS = {
    "upload_document",
    "list_documents",
    "search_documents",
    "search_document_content",
    "get_document_metadata",
    "get_document_description",
    "get_field_value",
    "get_evidence",
    "classify_document",
    "verify_ocr",
    "extract_metadata",
    "summarize_content",
    "validate_document",
    "validate_document_image",
    "list_templates",
    "get_template",
    "get_alerts",
    "web_search",
    "web_scrape",
    "fetch_url",
    "bulk_download",
    "bulk_metadata",
    "health_document",
    "health_model",
    "health_validator",
    "health_web_search",
}


class DefaultToolRegistry:
    """Static tool registry with built-in tool definitions."""

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        defn = _TOOL_DEFINITIONS.get(tool_name)
        if defn is None:
            return None
        _service, description, schema, _confirm = defn
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": schema,
            },
        }

    def get_all_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            self.get_tool_schema(name)
            for name in LLM_VISIBLE_TOOLS
            if name in _TOOL_DEFINITIONS
        ]

    def get_service(self, tool_name: str) -> ServiceTarget | None:
        defn = _TOOL_DEFINITIONS.get(tool_name)
        return defn[0] if defn else None

    def requires_confirmation(
        self, tool_name: str, args: dict[str, Any]
    ) -> ConfirmationType | None:
        defn = _TOOL_DEFINITIONS.get(tool_name)
        if defn is None:
            return None
        return defn[3]  # confirmation_type
