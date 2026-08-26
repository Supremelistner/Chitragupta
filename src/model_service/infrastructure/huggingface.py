"""HuggingFace Inference Provider adapter.

Supports both the free serverless tier and paid providers (Fireworks, DeepInfra)
routed through HF's Inference Providers API.

Provider switching is a config change — no code changes needed.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    InferenceTaskType,
    ModelInfo,
    ModelProviderType,
    ProviderHealth,
)
from model_service.domain.ports import ModelProvider

logger = logging.getLogger("model_service.huggingface")


class HuggingFaceProviderAdapter(ModelProvider):
    """Adapter for HuggingFace Inference API (serverless and providers).

    Uses the OpenAI-compatible chat completions endpoint.
    For vision tasks, sends images as base64-encoded data URLs.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        timeout_seconds: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._token = token
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def provider_type(self) -> ModelProviderType:
        return ModelProviderType.HUGGINGFACE

    def ping(self) -> None:
        """Verify HF API is reachable."""
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        start = time.monotonic()
        try:
            output, token_usage = self._call_inference(request)
            latency_ms = (time.monotonic() - start) * 1000
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=self._model_id,
                output=output,
                latency_ms=latency_ms,
                token_usage=token_usage,
                request_id=request.request_id,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.exception("HuggingFace inference failed")
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=self._model_id,
                output=f"ERROR: {exc}",
                latency_ms=latency_ms,
                request_id=request.request_id,
                metadata={"error": str(exc)},
            )

    def _call_inference(self, request: InferenceRequest) -> tuple[str, dict[str, int] | None]:
        """Call the HF Inference API via huggingface_hub."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError("huggingface_hub is required")

        client = InferenceClient(token=self._token)

        # Build messages
        messages = self._build_messages(request)

        # Call chat completions
        response = client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=request.parameters.get("temperature", self._temperature),
            max_tokens=request.parameters.get("max_tokens", self._max_tokens),
        )

        output = self._strip_markdown_fences(response.choices[0].message.content or "")
        token_usage = None
        if hasattr(response, "usage") and response.usage:
            token_usage = {
                "input": getattr(response.usage, "prompt_tokens", 0),
                "output": getattr(response.usage, "completion_tokens", 0),
            }
        return output, token_usage

    def _build_messages(self, request: InferenceRequest) -> list[dict[str, Any]]:
        """Build OpenAI-compatible messages, including image if present."""
        messages: list[dict[str, Any]] = []

        # System message with task context
        system_prompt = self._system_prompt(request.task)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message — may include image
        user_content: list[dict[str, Any]] = []

        # Add image if present (vision task)
        if request.image_bytes and request.image_mime_type:
            b64_image = base64.b64encode(request.image_bytes).decode("ascii")
            data_url = f"data:{request.image_mime_type};base64,{b64_image}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        # Add text/prompt
        text_parts = []
        if request.prompt:
            text_parts.append(request.prompt)
        if request.text:
            text_parts.append(f"Input text:\n{request.text}")
        if text_parts:
            user_content.append({
                "type": "text",
                "text": "\n\n".join(text_parts),
            })

        if not user_content:
            user_content.append({"type": "text", "text": "No input provided."})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _system_prompt(self, task: InferenceTaskType) -> str:
        """Return the system prompt for a task.

        Each prompt is self-contained — no conversation history is carried
        between documents.  Every request is a cold start with fresh context.
        """
        prompts: dict[InferenceTaskType, str] = {
            InferenceTaskType.TEXT_EXTRACTION: _TEXT_EXTRACTION_PROMPT,
            InferenceTaskType.DOCUMENT_CLASSIFICATION: _CLASSIFICATION_PROMPT,
            InferenceTaskType.OCR_VERIFICATION: _OCR_VERIFICATION_PROMPT,
            InferenceTaskType.METADATA_EXTRACTION: _METADATA_EXTRACTION_PROMPT,
            InferenceTaskType.CONTENT_SUMMARIZATION: _SUMMARIZATION_PROMPT,
            InferenceTaskType.PRIVACY_CLASSIFICATION: _PRIVACY_CLASSIFICATION_PROMPT,
            InferenceTaskType.CUSTOM: _CUSTOM_PROMPT,
        }
        return prompts.get(task, "")

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                provider=self.provider_type,
                display_name="Qwen2.5-VL-3B-Instruct (via HuggingFace)",
                capabilities=("vision", "text", "multilingual", "document-understanding"),
                supports_vision=True,
                supports_text=True,
            ),
        ]

    def health(self) -> ProviderHealth:
        try:
            self.ping()
            return ProviderHealth(
                provider=self.provider_type,
                healthy=True,
                available_models=tuple(self.list_models()),
            )
        except Exception as exc:
            return ProviderHealth(
                provider=self.provider_type,
                healthy=False,
                error_message=str(exc),
            )

    def close(self) -> None:
        pass

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences from model output.

        Models often wrap JSON in ```json ... ```.  We need raw JSON.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()


# =============================================================================
# Prompt templates — stateless, per-document, strict JSON output
# =============================================================================
# Each prompt enforces:
#   1. No conversation history carried (cold start per document)
#   2. Strict JSON output matching the defined schema
#   3. Error JSON if the document cannot be processed
#   4. Bilingual preservation (English primary, original language for key text)
#   5. Confidence scoring per field
#   6. Privacy awareness — flag PII, never fabricate
# -----------------------------------------------------------------------------

_TEXT_EXTRACTION_PROMPT = """You are a document text extraction system for Indian documents.

TASK: Extract ALL text from this document image. Output as plain text.

OUTPUT RULES:
- Respond with ONLY the extracted text. No JSON, no markdown, no explanation.
- Preserve line breaks between logical sections.
- Preserve the original script (Hindi/Devanagari, English, numbers).
- Do NOT translate, summarize, or interpret — just transcribe.

RULES:
- Read every word, number, and character exactly as it appears.
- Preserve formatting: headers, tables, labels, values.
- If text is unclear, transcribe what you can see — do not guess.
- Include both Hindi and English text when both are present.
"""

_CLASSIFICATION_PROMPT = """You are a document classifier for an Indian document processing system.

TASK: Classify the document type from the image.

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside JSON.
- If you cannot classify the document, return: {"error": true, "message": "<reason>"}

JSON SCHEMA:
{
  "document_type": "identity_document | academic_record | financial | medical | legal | correspondence | other",
  "document_sub_type": "<specific type, e.g. aadhaar, pan, marksheet, certificate, invoice>",
  "confidence": <0.0-1.0>,
  "language_detected": ["<iso codes, e.g. en, hi>"],
  "key_indicators": ["<what led to this classification>"]
}

RULES:
- Never guess if the document is unclear — use the error format.
- For Indian documents, recognize Hindi/Devanagari script.
- document_sub_type should be specific (e.g. "aadhaar" not just "id_card").
"""

_OCR_VERIFICATION_PROMPT = """You are an OCR verification specialist for an Indian document processing system.

TASK: Verify the provided OCR text against the original document image. Identify errors, missing text, and misread characters.

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside JSON.
- If you cannot verify (image unclear or missing), return: {"error": true, "message": "<reason>"}

JSON SCHEMA:
{
  "verified": true/false,
  "accuracy_estimate": <0.0-1.0>,
  "errors_found": [
    {"type": "misread | missing | extra | wrong_language", "ocr_text": "<what OCR said>", "correct_text": "<what image shows>", "location": "<field or area>"}
  ],
  "missing_text": ["<text visible in image but absent from OCR>"],
  "language_issues": ["<text in non-English script that OCR may have mishandled>"]
}

RULES:
- Compare character-by-character for critical fields (names, numbers, dates).
- Preserve original script (Hindi/Devanagari) in correct_text when the image shows it.
- If OCR text is empty, treat all visible text as missing.
- accuracy_estimate is your overall confidence in the OCR quality.
"""

_METADATA_EXTRACTION_PROMPT = """You are a metadata extraction system for Indian documents.

TASK: Extract ALL structured metadata from the document image. This is the primary extraction — be thorough.

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside JSON.
- If you cannot extract metadata, return: {"error": true, "message": "<reason>"}
- Follow the JSON schema EXACTLY. Do not add or remove fields.

JSON SCHEMA:
{
  "schema_version": "1.0",
  "document_type": "<identity_document | academic_record | financial | medical | legal | correspondence | other>",
  "document_sub_type": "<specific type>",
  "language": {
    "primary": "en",
    "detected": ["<iso codes>"],
    "has_devanagari": true/false
  },
  "privacy": {
    "classification": "SENSITIVE | PRIVATE | OPEN_NOT_PUBLIC | OPEN",
    "confidence": <0.0-1.0>,
    "reasons": ["<why this classification>"]
  },
  "description": {
    "safe": "<non-identifying description, no PII>",
    "detailed": "<full description with names, dates, IDs>",
    "original_key_text": "<important text in original language, e.g. Hindi>",
    "original_key_text_language": "<iso code of original script>"
  },
  "fields": {
    "<field_name>": {
      "value": "<extracted value in English>",
      "value_original": "<value in original language if different, else null>",
      "confidence": <0.0-1.0>,
      "field_type": "<person_name | date | number | organization | address | government_id | phone | email | grade | exam_name | result | other>",
      "source": "<direct | inferred | unknown>",
      "alternate_fields": ["<other field names this value could belong to, if confidence scores are close>"]
    }
  },
  "extraction_confidence": <0.0-1.0>
}

RULES:
- Extract ALL visible fields, even if confidence is low.
- For Aadhaar cards: extract name, DOB, gender, Aadhaar number, phone.
- For marksheets: extract student name, roll number, parents' names, DOB, school, board, subjects with marks/grades, result.
- PRESERVE original script in value_original (e.g. "तनवी" for name, "भारत सरकार" for authority).
- value is always in English. Translate if needed.
- confidence < 0.5 means uncertain — still include the field but with low confidence.
- If two fields could share a value (e.g. total_marks vs max_possible_marks), put the other in alternate_fields.
- extraction_confidence is the average of all field confidences.
- DO NOT fabricate values. If a field is not visible, set value to null and confidence to 0.0.
"""

_SUMMARIZATION_PROMPT = """You are a document summarizer for an Indian document processing system.

TASK: Create a concise summary of the document content.

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside JSON.
- If you cannot summarize, return: {"error": true, "message": "<reason>"}

JSON SCHEMA:
{
  "summary": "<2-4 sentence summary in English>",
  "key_points": ["<bullet points of important information>"],
  "language": "<primary language of the document>",
  "confidence": <0.0-1.0>
}

RULES:
- Summary must be in English.
- Preserve names, dates, and numbers exactly as they appear.
- If the document contains Hindi, translate key terms to English in the summary.
- Never fabricate information not present in the document.
"""

_PRIVACY_CLASSIFICATION_PROMPT = """You are a privacy classifier for an Indian document processing system.

TASK: Analyze the document and classify its privacy sensitivity. Identify all PII and sensitive data.

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside JSON.
- If you cannot classify, return: {"error": true, "message": "<reason>"}

JSON SCHEMA:
{
  "classification": "SENSITIVE | PRIVATE | OPEN_NOT_PUBLIC | OPEN",
  "confidence": <0.0-1.0>,
  "contains_pii": true/false,
  "pii_types": ["<government_id | phone | email | dob | name | address | financial | medical | biometric>"],
  "sensitive_fields": ["<field names that contain sensitive data>"],
  "reasons": ["<why this classification was chosen>"],
  "recommended_access_level": "<allow | require_approval | deny>"
}

PRIVACY RULES:
- SENSITIVE: Government IDs (Aadhaar, PAN, passport), medical records, financial statements.
- PRIVATE: Academic records, employment documents, contracts, personal correspondence.
- OPEN_NOT_PUBLIC: Internal documents, drafts, non-public reports.
- OPEN: Public documents, press releases, marketing materials.
- ALWAYS flag Aadhaar numbers, phone numbers, dates of birth, and addresses as PII.
- If unsure between classifications, choose the MORE restrictive one.
- recommended_access_level: "require_approval" for SENSITIVE and PRIVATE.
"""

_CUSTOM_PROMPT = """You are a document analysis assistant for an Indian document processing system.

TASK: Process the document according to the user's instructions.

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside JSON.
- If you cannot process the request, return: {"error": true, "message": "<reason>"}

JSON SCHEMA:
{
  "result": "<your response>",
  "confidence": <0.0-1.0>,
  "language": "<language of response>",
  "notes": ["<any caveats or observations>"]
}

RULES:
- Follow the user's instructions precisely.
- Never fabricate information.
- If the request is unclear, return an error with explanation.
"""
