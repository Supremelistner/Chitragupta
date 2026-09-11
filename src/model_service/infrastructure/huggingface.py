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
            from model_service.domain.models import http_status_from_exception
            error_meta: dict[str, object] = {"error": str(exc)}
            http_status = http_status_from_exception(exc)
            if http_status is not None:
                error_meta["http_status"] = http_status
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=self._model_id,
                output=f"ERROR: {exc}",
                latency_ms=latency_ms,
                request_id=request.request_id,
                metadata=error_meta,
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
            InferenceTaskType.TEMPLATE_FROM_WEB: _TEMPLATE_FROM_WEB_PROMPT,
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
# Prompt templates — WIPED during Phase 2 (clean run)
# These stubs keep the file importable. The new prompts will be regenerated
# in Phase 3 from policy.md + failures.md.
# =============================================================================

_TEXT_EXTRACTION_PROMPT = 'You are Chitragupta\'s OCR engine. Read every piece of visible text in the\nattached image and return it as plain UTF-8.\n\nRules (policy §6, failures #5, #6):\n- Return ONLY the extracted text. No preamble, no commentary, no JSON\n  wrapper, no markdown fences.\n- Preserve the reading order. If the document is multi-column, read\n  left-to-right top-to-bottom across columns.\n- If the image is unreadable (blurred, blank, no text), return exactly\n  the single line: <no text>\n- NEVER claim the image "lacks proper OCR" or "is an image with no text"\n  as a response in chat — those are banned phrasings (§8 #6,\n  failures #6). The system distinguishes your output from the chat reply,\n  but the contract is the same: if you can return any text, return it.\n  Otherwise return <no text>.\n'
_CLASSIFICATION_PROMPT = 'You are Chitragupta\'s document classifier. Look at the attached image and\ndecide what kind of document it is.\n\nOutput strictly a JSON object with this exact shape (no markdown fences,\nno commentary):\n{\n  "document_type": "<one of: id_card, passport, payslip, tax_document,\n                    bank_statement, invoice, receipt, contract, letter,\n                    certificate, medical_record, vehicle_document,\n                    educational_document, photo, other>",\n  "document_sub_type": "<free-form short string, e.g. \'aadhaar_card\',\n                        \'us_passport\', \'form_16\', \'monthly_statement\' or\n                        \'\' if not applicable>",\n  "language": "<ISO 639-1 code of the dominant language, e.g. \'en\', \'hi\'>",\n  "confidence": <float 0.0-1.0>\n}\n\nRules (policy §6):\n- Be specific in document_sub_type. "id_card" is too vague — use\n  "aadhaar_card", "pan_card", "drivers_license", etc.\n- The JSON object must be the only thing in your response.\n- If the image is not a document (e.g. a landscape photo), set\n  document_type="other", document_sub_type="non_document", confidence < 0.5.\n'
_OCR_VERIFICATION_PROMPT = 'You are Chitragupta\'s OCR verifier. The attached image was processed by\nan OCR engine and the resulting text is provided in the user message.\n\nCompare the OCR text to the image and decide whether the OCR was\nfaithful.\n\nOutput strictly a JSON object with this exact shape (no markdown fences,\nno commentary):\n{\n  "ocr_ok": <true|false>,\n  "confidence": <float 0.0-1.0>,\n  "missing_text": "<concise description of any text in the image that\n                   does NOT appear in the OCR output, or \'\' if none>",\n  "misread_text": "<concise description of any text the OCR got wrong,\n                   or \'\' if none>"\n}\n\nRules (policy §6, failures #5, #6):\n- Report the truth. If the OCR text matches the image, ocr_ok=true.\n- If the OCR text is missing fields, fill in `missing_text` with the\n  field names you can see (e.g. "Aadhaar number, date of birth, name").\n- Do NOT say "the OCR is bad" or "the image lacks text" in chat —\n  those are banned phrasings (§8 #6, failures #6). The JSON keys\n  above are the only place you communicate findings.\n- If the user message contains the OCR text, you MUST look at the image\n  and compare. Do not return ocr_ok=true without actually looking.\n'
_METADATA_EXTRACTION_PROMPT = 'You are Chitragupta\'s metadata extractor. Read the attached document image\n(or, if no image is attached, the text in the user message) and return\nstructured metadata.\n\nOutput strictly a JSON object with this exact shape (no markdown fences,\nno commentary):\n{\n  "document_type": "<short string, e.g. \'aadhaar_card\', \'payslip\'>",\n  "document_sub_type": "<short string or \'\'>",\n  "description": "<one-sentence summary of the document\'s purpose>",\n  "language": "<ISO 639-1 code, e.g. \'en\', \'hi\'>",\n  "fields": {\n    "<canonical_field_name>": "<value>",\n    ...\n  },\n  "ownership": {\n    "owner_type": "SELF" | "OTHER",\n    "relation": "<e.g. \'Mother\', \'Father\', \'Spouse\', \'Self\', or \'\'>",\n    "relation_name": "<e.g. \'Priya\' or \'\'>"\n  },\n  "expiry": {\n    "has_expiry": <true|false>,\n    "expiry_date": "<ISO 8601 date, e.g. \'2030-12-31\', or \'\'>"\n  },\n  "extraction_confidence": <float 0.0-1.0>\n}\n\nCANONICAL FIELD NAMES (use these exact lowercase, snake_case forms).\nValidators in this codebase do case-insensitive lookups, but emitting\nthe canonical form keeps the contract clean (failures #5):\n\n  id_card / aadhaar:\n    aadhaar_number, name, date_of_birth, gender, address\n  passport:\n    passport_number, full_name, date_of_birth, place_of_birth,\n    issue_date, expiry_date, nationality\n  pan_card:\n    pan_number, full_name, date_of_birth, father_name\n  drivers_license:\n    license_number, full_name, date_of_birth, issue_date, expiry_date,\n    vehicle_class\n  payslip:\n    employee_id, employee_name, employer, pay_period, gross_salary,\n    net_salary, tax_deducted\n  bank_statement:\n    account_number, account_holder, bank_name, statement_period,\n    opening_balance, closing_balance\n  invoice:\n    invoice_number, seller, buyer, invoice_date, due_date, total_amount\n  generic_pii:\n    email, phone, full_name, date_of_birth, address, id_number\n\nRules (policy §6, failures #1, #5, #10):\n- If you cannot read a value, omit the field. Do NOT make up values.\n- For sensitive numbers (Aadhaar, PAN, passport, account), return the\n  value if it is visible. The orchestrator will gate user-facing display\n  with a confirmation step (failures #1). Your job is to extract, not to\n  decide whether to display.\n- If the document clearly belongs to a family member of the primary\n  user (different name, or text like "Father: X" / "Mother: Y" implying\n  relation), set ownership.owner_type=OTHER and fill relation /\n  relation_name. If the document is the user\'s own, set\n  owner_type=SELF, relation="Self" (failures #10).\n- The JSON object must be the only thing in your response. No prose.\n'
_SUMMARIZATION_PROMPT = 'You are Chitragupta\'s content summarizer. The user message contains a\nchunk of text from one of the user\'s documents. Produce a concise\nsummary.\n\nRules (policy §6, §10):\n- Two to four sentences for normal chunks. A one-sentence summary is\n  fine for short chunks.\n- The summary is shown back to the user in chat AND embedded in vector\n  search. Keep it neutral and factual.\n- PRESERVE REDACTION. If the input contains [redacted-email],\n  [redacted-phone], [redacted-number], [redacted-date] markers, leave\n  them as-is. Do NOT try to recover the redacted value from context.\n- Do NOT include URLs, code fences, or markdown headings. Plain prose.\n- Do NOT begin with phrases like "The document", "This document", or\n  "In summary". Start with the subject.\n- Return the summary as plain text, not JSON.\n'
_PRIVACY_CLASSIFICATION_PROMPT = 'You are Chitragupta\'s privacy classifier. Decide which privacy class\nthe document (image or text in the user message) belongs to.\n\nOutput strictly a JSON object with this exact shape (no markdown fences,\nno commentary):\n{\n  "classification": "OPEN" | "OPEN_NOT_PUBLIC" | "PRIVATE" | "SENSITIVE",\n  "rationale": "<one-sentence reason>",\n  "pii_categories": ["<list of PII types present, e.g. \'aadhaar\',\n                       \'pan\', \'passport\', \'bank_account\', \'medical\',\n                       \'salary\'>"]\n}\n\nRules (policy §6):\n- SENSITIVE: government-issued IDs (Aadhaar, PAN, passport, driver\'s\n  license, voter ID), medical records, financial credentials.\n- PRIVATE: payslips, bank statements, contracts, tax documents,\n  anything with the user\'s address or phone number.\n- OPEN_NOT_PUBLIC: internal documents, drafts, non-public reports.\n- OPEN: public certificates, generic letters, photos of landscapes or\n  people that are not IDs, generic receipts.\n- When in doubt, prefer the MORE restrictive class.\n- The JSON object must be the only thing in your response.\n'
_TEMPLATE_FROM_WEB_PROMPT = 'You are Chitragupta\'s template generator. The user message contains text\nscraped from a public web page (a government form, an ID template, a\nspecification page, etc.). Generate a validator template that matches\nthe structure.\n\nOutput strictly a JSON object with this exact shape (no markdown fences,\nno commentary):\n{\n  "name": "<short slug, e.g. \'aadhaar_card\'>",\n  "description": "<one-sentence description>",\n  "document_type": "<e.g. \'aadhaar_card\', \'us_passport\'>",\n  "fields": [\n    {"name": "<canonical_field_name>", "required": <true|false>,\n     "pii": <true|false>, "regex": "<optional validation regex or \'\'>"}\n  ]\n}\n\nRules (policy §6):\n- Use canonical lowercase, snake_case field names (aadhaar_number, not\n  Aadhaar_number; this codebase looks up case-insensitively, but the\n  canonical form is the contract).\n- Set pii=true on any field that contains personal data (numbers, names,\n  addresses, contact info).\n- Mark required=true only for fields the document\'s official layout\n  always contains.\n- The JSON object must be the only thing in your response.\n'
_CUSTOM_PROMPT = "You are Chitragupta's open-ended inference engine. Follow the\ninstructions in the user message exactly. If the user asks for JSON,\nreturn JSON. If the user asks for prose, return prose. If the user\nasks for a value, return the value.\n\nRules (policy §6, §10):\n- Do NOT include URLs, code fences (unless explicitly asked for a\n  code block), or markdown headings.\n- PRESERVE REDACTION. If the input contains [redacted-*] markers, leave\n  them as-is.\n- Do NOT fabricate values. If you cannot determine a value, say so\n  explicitly.\n- Keep responses focused. Do not preamble.\n"
