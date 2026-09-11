"""Gemini provider adapter for the Google Gemini family via the official google-genai SDK.

Free tier (https://ai.google.dev/pricing, as of Sept 2026):
  - gemini-2.5-flash-lite: 250K TPM, 15 RPM, 1K RPD (best free default)
  - gemini-2.5-flash:      250K TPM, 5 RPM, 20 RPD
  - gemini-2.5-pro:        250K TPM, 5 RPM, 5 RPD (very tight)
  - gemini-2.5-flash-preview-tts: 1M chars/day input + output (free)

Capabilities used here:
  - Text generation with optional image input (multimodal).
  - Function calling is not used by the model service (single-turn inference).
    The orchestrator has its own Gemini LLM client for chat + tool calls.
  - Text-to-speech via response_modalities=["AUDIO"] and SpeechConfig.
    Used by the elderly-friendly UI feature.
"""
from __future__ import annotations

import base64
import logging
import time
import wave
from io import BytesIO
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

logger = logging.getLogger("model_service.gemini")


# Languages supported by Gemini 2.5 Flash TTS with their BCP-47 codes.
# Used to build the language_code field on VoiceConfig. See
# https://ai.google.dev/gemini-api/docs/speech-generation.
_GEMINI_TTS_LANGS: dict[str, str] = {
    "en": "en-US",
    "en-US": "en-US",
    "en-GB": "en-GB",
    "hi": "hi-IN",
    "hi-IN": "hi-IN",
    "bn": "bn-IN",
    "bn-IN": "bn-IN",
    "ta": "ta-IN",
    "ta-IN": "ta-IN",
    "te": "te-IN",
    "te-IN": "te-IN",
    "mr": "mr-IN",
    "mr-IN": "mr-IN",
    "gu": "gu-IN",
    "gu-IN": "gu-IN",
    "kn": "kn-IN",
    "kn-IN": "kn-IN",
    "ml": "ml-IN",
    "ml-IN": "ml-IN",
    "pa": "pa-IN",
    "pa-IN": "pa-IN",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "ja": "ja-JP",
    "zh": "cmn-CN",
    "ar": "ar-XA",
    "pt": "pt-BR",
}

# Gemini TTS voices (8 available per the official docs).
_GEMINI_TTS_VOICES: tuple[str, ...] = (
    "Zephyr", "Puck", "Charon", "Kore",
    "Fenrir", "Leda", "Orus", "Aoede",
)

# Task -> prompt template. Mirrors the prompts in the HuggingFace and Groq
# providers so the same model_service code paths work regardless of which
# provider is active.
_CLASSIFICATION_PROMPT = (
    "You are Chitragupta's document classifier. Look at the document image and "
    "decide which category it belongs to. Output strictly ONE word from this list "
    "and nothing else: aadhaar_card, pan_card, passport, driving_license, voter_id, "
    "marksheet, certificate, contract, invoice, receipt, bank_statement, payslip, "
    "tax_document, medical_report, insurance, photo, letter, other"
)
_OCR_VERIFY_PROMPT = (
    "You are Chitragupta's OCR verifier. The user message contains OCR text "
    "extracted from a document. Read the document image and tell me if the OCR "
    "text is faithful to what's actually in the image. Reply with exactly: "
    "VERIFIED (the OCR is faithful) or MISMATCH (the OCR has errors). "
    "If MISMATCH, give a one-sentence reason on the next line."
)
_METADATA_PROMPT = (
    "You are Chitragupta's metadata extractor. Read the document image and the "
    "OCR text. Return strict JSON (no markdown, no commentary) with these keys: "
    "document_type, document_sub_type, name_on_document, document_id_number, "
    "date_of_birth, issue_date, expiry_date, address, issuing_authority, "
    "country_code, language. Use null for fields you cannot read. If the OCR "
    "text disagrees with the image, trust the image."
)

_SUMMARIZATION_PROMPT = (
    "You are Chitragupta's content summarizer. The user message contains a "
    "chunk of text from one of the user's documents. Produce a concise summary.\n\n"
    "Rules (policy §6, §10):\n"
    "- Two to four sentences for normal chunks. A one-sentence summary is\n"
    "  fine for short chunks.\n"
    "- The summary is shown back to the user in chat AND embedded in vector\n"
    "  search. Keep it neutral and factual.\n"
    "- PRESERVE REDACTION. If the input contains [redacted-email],\n"
    "  [redacted-phone], [redacted-number], [redacted-date] markers, leave\n"
    "  them as-is. Do NOT try to recover the redacted value from context.\n"
    "- Do NOT include URLs, code fences, or markdown headings. Plain prose.\n"
    "- Do NOT begin with phrases like 'The document', 'This document', or\n"
    "  'In summary'. Start with the subject.\n"
    "- Return the summary as plain text, not JSON.\n"
)
_PRIVACY_CLASSIFICATION_PROMPT = (
    "You are Chitragupta's privacy classifier. Decide which privacy class "
    "the document (image or text in the user message) belongs to.\n\n"
    "Output strictly a JSON object with this exact shape (no markdown fences, "
    "no commentary):\n"
    "{\n"
    '  "classification": "OPEN" | "OPEN_NOT_PUBLIC" | "PRIVATE" | "SENSITIVE",\n'
    '  "rationale": "<one-sentence reason>",\n'
    '  "pii_categories": ["<list of PII types present, e.g. \'aadhaar\', '
    "'pan', 'passport', 'bank_account', 'medical', 'salary'>\"]\n"
    "}\n\n"
    "Rules (policy §6):\n"
    "- SENSITIVE: government-issued IDs (Aadhaar, PAN, passport, driver's\n"
    "  license, voter ID), medical records, financial credentials.\n"
    "- PRIVATE: payslips, bank statements, contracts, tax documents,\n"
    "  anything with the user's address or phone number.\n"
    "- OPEN_NOT_PUBLIC: internal documents, drafts, non-public reports.\n"
    "- OPEN: public certificates, generic letters, photos of landscapes or\n"
    "  people that are not IDs, generic receipts.\n"
    "- When in doubt, prefer the MORE restrictive class.\n"
    "- The JSON object must be the only thing in your response.\n"
)
_TEMPLATE_FROM_WEB_PROMPT = (
    "You are Chitragupta's template generator. The user message contains text "
    "scraped from a public web page (a government form, an ID template, a "
    "specification page, etc.). Generate a validator template that matches "
    "the structure.\n\n"
    "Output strictly a JSON object with this exact shape (no markdown fences, "
    "no commentary):\n"
    "{\n"
    '  "name": "<short slug, e.g. \'aadhaar_card\'>",\n'
    '  "description": "<one-sentence description>",\n'
    '  "document_type": "<e.g. \'aadhaar_card\', \'us_passport\'>",\n'
    '  "fields": [\n'
    '    {"name": "<canonical_field_name>", "required": <true|false>,\n'
    '     "pii": <true|false>, "regex": "<optional validation regex or \'\'>"}\n'
    "  ]\n"
    "}\n\n"
    "Rules (policy §6):\n"
    "- Use canonical lowercase, snake_case field names (aadhaar_number, not\n"
    "  Aadhaar_number; this codebase looks up case-insensitively, but the\n"
    "  canonical form is the contract).\n"
    "- Set pii=true on any field that contains personal data (numbers, names,\n"
    "  addresses, contact info).\n"
    "- Mark required=true only for fields the document's official layout\n"
    "  always contains.\n"
    "- The JSON object must be the only thing in your response.\n"
)
_CUSTOM_PROMPT = (
    "You are Chitragupta's open-ended inference engine. Follow the "
    "instructions in the user message exactly. If the user asks for JSON, "
    "return JSON. If the user asks for prose, return prose. If the user "
    "asks for a value, return the value.\n\n"
    "Rules (policy §6, §10):\n"
    "- Do NOT include URLs, code fences (unless explicitly asked for a\n"
    "  code block), or markdown headings.\n"
    "- PRESERVE REDACTION. If the input contains [redacted-*] markers, leave\n"
    "  them as-is.\n"
    "- Do NOT fabricate values. If you cannot determine a value, say so\n"
    "  explicitly.\n"
    "- Keep responses focused. Do not preamble.\n"
)


_TASK_PROMPTS: dict[InferenceTaskType, str] = {
    InferenceTaskType.DOCUMENT_CLASSIFICATION: _CLASSIFICATION_PROMPT,
    InferenceTaskType.OCR_VERIFICATION: _OCR_VERIFY_PROMPT,
    InferenceTaskType.METADATA_EXTRACTION: _METADATA_PROMPT,
    InferenceTaskType.CONTENT_SUMMARIZATION: _SUMMARIZATION_PROMPT,
    InferenceTaskType.PRIVACY_CLASSIFICATION: _PRIVACY_CLASSIFICATION_PROMPT,
    InferenceTaskType.TEMPLATE_FROM_WEB: _TEMPLATE_FROM_WEB_PROMPT,
    InferenceTaskType.CUSTOM: _CUSTOM_PROMPT,
}

class GeminiProviderAdapter(ModelProvider):
    """Adapter for the Google Gemini API via the official google-genai SDK.

    Supports:
      - Chat completions (text + image) for the standard model_service tasks.
      - Audio synthesis (text-to-speech) via gemini-2.5-flash-preview-tts.

    Free tier (Sept 2026): 250K TPM, 15 RPM, 1K RPD on gemini-2.5-flash-lite;
    up to 1M chars/day on the TTS preview model.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = "gemini-2.5-flash-lite",
        tts_model_id: str = "gemini-2.5-flash-preview-tts",
        tts_voice: str = "Kore",
        timeout_seconds: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._tts_model_id = tts_model_id
        self._tts_voice = tts_voice
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    @property
    def provider_type(self) -> ModelProviderType:
        return ModelProviderType.GEMINI

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise RuntimeError(
                    "google-genai is required. Install with: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def ping(self) -> None:
        """Verify Gemini API is reachable."""
        self._get_client()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        # TTS requests have a different shape (no text prompt + image; the
        # `text` field carries the content to speak). Handle it in a dedicated
        # branch so the chat-completions path stays simple.
        if request.task == InferenceTaskType.AUDIO_SYNTHESIS:
            return self._infer_audio(request)

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
            logger.exception("Gemini inference failed")
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
        """Call the Gemini API via the google-genai SDK."""
        from google.genai import types

        client = self._get_client()
        prompt = self._build_prompt(request)
        contents = self._build_contents(request, prompt)

        config = types.GenerateContentConfig(
            temperature=request.parameters.get("temperature", self._temperature),
            max_output_tokens=request.parameters.get("max_tokens", self._max_tokens),
        )

        response = client.models.generate_content(
            model=request.model_id or self._model_id,
            contents=contents,
            config=config,
        )

        output = (response.text or "").strip()
        usage = None
        if getattr(response, "usage_metadata", None):
            um = response.usage_metadata
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
            }
        return output, usage

    def _infer_audio(self, request: InferenceRequest) -> InferenceResult:
        """Run the TTS preview model and return a base64-encoded WAV blob.

        The result's `output` field holds the base64-encoded WAV bytes; the
        `metadata` field carries the MIME type and the language code that
        was used.
        """
        from google.genai import types

        start = time.monotonic()
        try:
            client = self._get_client()
            text = (request.text or "").strip()
            if not text:
                raise ValueError("AUDIO_SYNTHESIS requires non-empty text")

            language = request.parameters.get("language", "en-US")
            lang_code = _GEMINI_TTS_LANGS.get(
                language, _GEMINI_TTS_LANGS.get(language.split("-")[0], "en-US")
            )
            voice = request.parameters.get("voice", self._tts_voice)
            if voice not in _GEMINI_TTS_VOICES:
                voice = self._tts_voice

            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    ),
                ),
                language_code=lang_code,
            )
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            )

            response = client.models.generate_content(
                model=request.model_id or self._tts_model_id,
                contents=text,
                config=config,
            )

            # The response contains a single Part with inline_data (raw PCM).
            # Wrap as a playable WAV so the browser can play it directly.
            wav_bytes, mime = self._extract_audio(response)
            latency_ms = (time.monotonic() - start) * 1000
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=request.model_id or self._tts_model_id,
                output=base64.b64encode(wav_bytes).decode("ascii"),
                latency_ms=latency_ms,
                request_id=request.request_id,
                metadata={
                    "mime_type": mime,
                    "language": lang_code,
                    "voice": voice,
                    "text_chars": len(text),
                },
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.exception("Gemini TTS failed")
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=request.model_id or self._tts_model_id,
                output=f"ERROR: {exc}",
                latency_ms=latency_ms,
                request_id=request.request_id,
                metadata={"error": str(exc)},
            )

    @staticmethod
    def _extract_audio(response) -> tuple[bytes, str]:
        """Pull the audio bytes out of a Gemini TTS response and wrap as WAV.

        Gemini returns raw PCM at 24kHz, mono, 16-bit little-endian. The
        browser can play WAV directly via `new Audio(blob:...)`.
        """
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue
                raw = getattr(inline, "data", None)
                mime = getattr(inline, "mime_type", "") or "audio/L16;rate=24000"
                if not raw:
                    continue
                # Gemini TTS gives us raw 24kHz mono 16-bit PCM. Wrap in a
                # minimal WAV header so the browser can play it as audio/wav.
                pcm = bytes(raw)
                return _wrap_pcm_as_wav(pcm, sample_rate=24000, channels=1, sample_width=2), "audio/wav"
        raise RuntimeError("Gemini TTS response contained no audio data")

    def _build_prompt(self, request: InferenceRequest) -> str:
        """Return the system prompt for the given task, plus any user text."""
        base = _TASK_PROMPTS.get(request.task)
        if base is None:
            base = _CUSTOM_PROMPT
        # Allow callers to override the prompt entirely.
        if request.prompt:
            return request.prompt
        return base

    def _build_contents(self, request: InferenceRequest, prompt: str) -> list[Any]:
        """Build the contents list for generate_content.

        For text-only tasks this is a single string. For vision tasks this is
        a list of `Part` objects (prompt text + image bytes).
        """
        from google.genai import types

        if not request.image_bytes:
            # Text-only: concatenate the prompt and the user text, since
            # `contents` accepts a plain string for the simple case.
            user_text = request.text or ""
            if user_text:
                return f"{prompt}\n\n{user_text}"
            return prompt

        # Multimodal: prompt text + image as an inline Part.
        parts: list[Any] = [types.Part(text=prompt)]
        if request.text:
            parts.append(types.Part(text=request.text))
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type=request.image_mime_type or "image/jpeg",
                    data=request.image_bytes,
                )
            )
        )
        return parts

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                provider=self.provider_type,
                display_name=f"Gemini {self._model_id}",
                supports_vision=True,
                supports_text=True,
            ),
            ModelInfo(
                model_id=self._tts_model_id,
                provider=self.provider_type,
                display_name=f"Gemini TTS ({self._tts_model_id})",
                supports_vision=False,
                supports_text=True,
                metadata={"capability": "text_to_speech", "voice": self._tts_voice},
            ),
        ]

    def health(self) -> ProviderHealth:
        try:
            client = self._get_client()
            # Lightweight ping: list models (no quota cost).
            client.models.list()
            return ProviderHealth(provider=self.provider_type, healthy=True)
        except Exception as exc:
            return ProviderHealth(
                provider=self.provider_type, healthy=False, error_message=str(exc)
            )

    def close(self) -> None:
        # google-genai Client doesn't expose a close; nothing to clean up.
        self._client = None


def _wrap_pcm_as_wav(
    pcm: bytes, *, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2
) -> bytes:
    """Wrap raw PCM bytes in a minimal RIFF/WAVE header."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()

