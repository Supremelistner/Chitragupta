"""MCP Server adapter for the Model Service.

Exposes model inference capabilities as MCP tools for agent consumption.
Follows the same MCP protocol as the Document Management service.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from typing import Any

from model_service.application.inference import InferenceService
from model_service.config import ModelServiceConfig
from model_service.domain.models import InferenceRequest, InferenceTaskType

logger = logging.getLogger("model_service.mcp")


class MCPServer:
    """MCP server that exposes model inference tools."""

    def __init__(
        self,
        *,
        config: ModelServiceConfig,
        inference_service: InferenceService,
        translation_service: Any | None = None,
    ) -> None:
        self._config = config
        self._inference = inference_service
        # Translation service is optional in __init__ so existing call
        # sites in __main__.py don't need to change. When not injected,
        # we build the module-level singleton lazily on first use.
        self._translation = translation_service
        self._logger = logger

    def serve(self) -> None:
        reader = sys.stdin.buffer
        writer = sys.stdout.buffer
        while True:
            message = self._read_message(reader)
            if message is None:
                return
            response = self._dispatch(message)
            if response is not None:
                self._write_message(writer, response)

    def _dispatch(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")

        try:
            if method == "initialize":
                return self._result(
                    request_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": self._config.mcp_server_name,
                            "version": self._config.mcp_server_version,
                        },
                        "capabilities": {"tools": {"listChanged": False}},
                    },
                )
            if method == "tools/list":
                return self._result(request_id, {"tools": self._tool_specs()})
            if method == "tools/call":
                return self._handle_tool_call(request_id, message.get("params", {}))
            if method == "ping":
                return self._result(request_id, {"pong": True})
            if request_id is not None:
                return self._error(request_id, -32601, f"Method not found: {method}")
            return None
        except Exception as exc:
            self._logger.exception("MCP request failed")
            if request_id is not None:
                return self._error(request_id, -32603, str(exc))
            return None

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name == "classify_document":
            return self._tool_result(request_id, self._classify_document(arguments))
        if name == "verify_ocr":
            return self._tool_result(request_id, self._verify_ocr(arguments))
        if name == "extract_metadata":
            return self._tool_result(request_id, self._extract_metadata(arguments))
        if name == "summarize_content":
            return self._tool_result(request_id, self._summarize_content(arguments))
        if name == "synthesize_speech":
            return self._tool_result(request_id, self._synthesize_speech(arguments))
        if name == "list_providers":
            return self._tool_result(request_id, {"providers": self._inference.list_providers()})
        if name == "get_active_provider":
            return self._tool_result(request_id, {"active_provider": self._inference.get_active_provider()})
        if name == "set_active_provider":
            provider = str(arguments.get("provider") or "")
            if not provider:
                raise ValueError("provider is required")
            return self._tool_result(request_id, self._inference.set_active_provider(provider))
        if name == "health":
            return self._tool_result(request_id, self._inference.health())

        if name == "translate_text":
            return self._tool_result(request_id, self._translate_text(arguments))
        if name == "set_language_preference":
            return self._tool_result(
                request_id, self._set_language_preference(arguments)
            )

        return self._error(request_id, -32602, f"Unknown tool: {name}")

    def _classify_document(self, args: dict[str, Any]) -> dict[str, Any]:
        image_bytes, mime = self._extract_image(args)
        request = InferenceRequest(
            task=InferenceTaskType.DOCUMENT_CLASSIFICATION,
            image_bytes=image_bytes,
            image_mime_type=mime,
            text=args.get("text"),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _verify_ocr(self, args: dict[str, Any]) -> dict[str, Any]:
        image_bytes, mime = self._extract_image(args)
        request = InferenceRequest(
            task=InferenceTaskType.OCR_VERIFICATION,
            image_bytes=image_bytes,
            image_mime_type=mime,
            text=args.get("ocr_text", ""),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _extract_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        image_bytes, mime = self._extract_image(args)
        request = InferenceRequest(
            task=InferenceTaskType.METADATA_EXTRACTION,
            image_bytes=image_bytes,
            image_mime_type=mime,
            text=args.get("ocr_text", ""),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _summarize_content(self, args: dict[str, Any]) -> dict[str, Any]:
        request = InferenceRequest(
            task=InferenceTaskType.CONTENT_SUMMARIZATION,
            text=args.get("text", ""),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _synthesize_speech(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP tool wrapper for the /infer/synthesize-audio endpoint."""
        text = (args.get("text") or "").strip()
        if not text:
            raise ValueError("text is required")
        parameters: dict[str, Any] = {}
        for key in ("language", "voice", "model_id"):
            if args.get(key):
                parameters[key] = args[key]
        request = InferenceRequest(
            task=InferenceTaskType.AUDIO_SYNTHESIS,
            text=text,
            parameters=parameters,
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _extract_image(self, args: dict[str, Any]) -> tuple[bytes | None, str | None]:
        """Extract image bytes and MIME type from arguments."""
        image_b64 = args.get("image_base64")
        mime = args.get("image_mime_type", "image/jpeg")
        if image_b64:
            return base64.b64decode(image_b64), mime

        image_path = args.get("image_path")
        if image_path:
            from pathlib import Path
            path = Path(image_path)
            if path.exists():
                return path.read_bytes(), mime

        return None, None

    def _result_payload(self, result: Any) -> dict[str, Any]:
        return {
            "task": result.task.value,
            "provider": result.provider.value,
            "model_id": result.model_id,
            "output": result.output,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "token_usage": result.token_usage,
            "request_id": result.request_id,
            "metadata": getattr(result, "metadata", None) or {},
        }

    # -- Tool specs --

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            self._tool_spec(
                "classify_document",
                "Classify a document using vision model. Send image bytes or path.",
                {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string", "description": "Base64-encoded image"},
                        "image_path": {"type": "string", "description": "Path to image file"},
                        "image_mime_type": {"type": "string", "default": "image/jpeg"},
                        "text": {"type": "string", "description": "Additional text context"},
                        "prompt": {"type": "string", "description": "Custom prompt override"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "verify_ocr",
                "Verify OCR output against the original image for accuracy.",
                {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string"},
                        "image_path": {"type": "string"},
                        "image_mime_type": {"type": "string", "default": "image/jpeg"},
                        "ocr_text": {"type": "string", "description": "OCR text to verify"},
                        "prompt": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "extract_metadata",
                "Extract structured metadata from a document image.",
                {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string"},
                        "image_path": {"type": "string"},
                        "image_mime_type": {"type": "string", "default": "image/jpeg"},
                        "ocr_text": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "summarize_content",
                "Summarize document text content.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "synthesize_speech",
                "Convert text to speech audio. Returns base64-encoded WAV. "
                "Supports multiple languages (e.g. en-US, hi-IN, ta-IN, te-IN, "
                "bn-IN, mr-IN, gu-IN, kn-IN, ml-IN, pa-IN). Used by the "
                "elderly-friendly UI voice output feature.",
                {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to speak",
                        },
                        "language": {
                            "type": "string",
                            "description": "BCP-47 language code, e.g. hi-IN. Default en-US.",
                            "default": "en-US",
                        },
                        "voice": {
                            "type": "string",
                            "description": "Voice name (Kore, Puck, Charon, Zephyr, "
                            "Fenrir, Leda, Orus, Aoede). Default Kore.",
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Override the TTS model id (e.g. gemini-2.5-pro-preview-tts).",
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "list_providers", "List available model providers and their status.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._tool_spec(
                "get_active_provider", "Get the currently active model provider.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._tool_spec(
                "set_active_provider",
                "Switch the active model provider.",
                {
                    "type": "object",
                    "properties": {"provider": {"type": "string"}},
                    "required": ["provider"],
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "health", "Health check for the model service.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._tool_spec(
                "translate_text",
                "Translate text between two BCP-47 language codes "
                "(e.g. en, hi-IN, ta-IN). Used by the orchestrator's LLM "
                "to translate the user's input into English for the "
                "tool loop, and to translate the LLM's English reply "
                "back into the user's preferred UI language. Returns "
                "the translated string in the ``text`` field; ``cached`` "
                "indicates whether the result came from the on-disk "
                "cache.",
                {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to translate.",
                        },
                        "source": {
                            "type": "string",
                            "description": "BCP-47 source language code "
                            "(e.g. 'hi', 'hi-IN'). Defaults to 'en'.",
                            "default": "en",
                        },
                        "target": {
                            "type": "string",
                            "description": "BCP-47 target language code "
                            "(e.g. 'hi', 'hi-IN'). Defaults to 'hi'.",
                            "default": "hi",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session id. When "
                            "provided and source/target are omitted, the "
                            "session's stored language preference is used.",
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "set_language_preference",
                "Set or read the language preference for a session. "
                "Pass at least one of source/target. Pass no language "
                "fields and only session_id to read the current value. "
                "Supported codes: en, en-US, en-GB, hi, hi-IN, ta, ta-IN, "
                "bn, bn-IN.",
                {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier.",
                        },
                        "source": {
                            "type": "string",
                            "description": "New source language (BCP-47).",
                        },
                        "target": {
                            "type": "string",
                            "description": "New target language (BCP-47).",
                        },
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            ),
        ]

    def _tool_spec(self, name: str, description: str, schema: dict) -> dict[str, Any]:
        return {"name": name, "description": description, "inputSchema": schema}

    # -- Translation tools (V1: en <-> hi, plus other supported codes) --

    def _get_translation(self) -> Any:
        """Resolve the translation service, falling back to the singleton."""
        if self._translation is None:
            from model_service.application.translation_service import (
                get_translation_service,
            )
            self._translation = get_translation_service()
        return self._translation

    def _translate_text(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP tool: translate a string between two BCP-47 codes.

        If ``session_id`` is provided and source/target are omitted,
        the session's stored preference drives the translation. If
        source == target, the input is echoed unchanged. Errors come
        back as ``isError`` from the surrounding MCP envelope.
        """
        text = (args.get("text") or "").strip()
        if not text:
            raise ValueError("text is required")
        source = args.get("source")
        target = args.get("target")
        session_id = args.get("session_id")

        service = self._get_translation()

        # Resolve direction. When session_id is given and either side is
        # missing, defer to the session's stored preference.
        if (source is None or target is None) and session_id:
            pref = service.get_preference(session_id)
            source = source or pref.source
            target = target or pref.target
        else:
            from model_service.infrastructure.language_preferences import (
                DEFAULT_PREFERENCE,
            )
            source = source or DEFAULT_PREFERENCE.source
            target = target or DEFAULT_PREFERENCE.target

        # Same-language short-circuit happens inside the service too,
        # but doing it here lets us return ``cached=True`` even for the
        # no-op path without paying any allocation.
        if source == target:
            return {
                "text": text,
                "source": source,
                "target": target,
                "cached": True,
                "latency_ms": 0,
            }

        from model_service.infrastructure.translation_provider import (
            TranslationRequest,
        )
        result = service._provider.translate(  # type: ignore[attr-defined]
            TranslationRequest(text=text, source=source, target=target)
        )
        return {
            "text": result.text,
            "source": result.source,
            "target": result.target,
            "cached": result.cached,
            "latency_ms": result.latency_ms,
        }

    def _set_language_preference(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP tool: read or update a session's language preference.

        With only ``session_id``: returns the current preference.
        With ``source`` / ``target``: updates and returns the new
        preference. Both ``source`` and ``target`` are optional but at
        least one must be provided when *setting*.
        """
        session_id = (args.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        source = args.get("source")
        target = args.get("target")
        service = self._get_translation()

        if source is None and target is None:
            pref = service.get_preference(session_id)
            return {"session_id": session_id, **pref.to_dict()}

        pref = service.set_preference(
            session_id, source=source, target=target
        )
        return {"session_id": session_id, **pref.to_dict()}

    def _tool_result(self, request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, default=str, separators=(",", ":"))}],
                "isError": False,
            },
        )

    def _result(self, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _read_message(self, reader: Any) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            line = reader.readline()
            if not line:
                return None
            line = line.decode("utf-8").rstrip("\r\n")
            if not line:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            return None
        payload = reader.read(content_length)
        return json.loads(payload.decode("utf-8"))

    def _write_message(self, writer: Any, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode()
        writer.write(header)
        writer.write(payload)
        writer.flush()
