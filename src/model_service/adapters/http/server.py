"""HTTP server adapter for the Model Service.

Provides REST endpoints for model inference, provider management, and health.
"""

from __future__ import annotations

import base64
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from model_service.application.inference import InferenceService
from model_service.domain.models import InferenceRequest, InferenceTaskType

logger = logging.getLogger("model_service.http")


class _ModelRequestHandler(BaseHTTPRequestHandler):
    inference_service: InferenceService

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path in {"/healthz", "/readyz", "/"}:
            self._send_json(HTTPStatus.OK, self._health())
            return

        if path == "/providers":
            self._send_json(HTTPStatus.OK, {"providers": self.inference_service.list_providers()})
            return

        if path == "/providers/active":
            self._send_json(HTTPStatus.OK, {"active_provider": self.inference_service.get_active_provider()})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json_body()

            if path == "/infer/classify":
                self._handle_infer(InferenceTaskType.DOCUMENT_CLASSIFICATION, payload)
            elif path == "/infer/verify-ocr":
                self._handle_infer(InferenceTaskType.OCR_VERIFICATION, payload)
            elif path == "/infer/extract-metadata":
                self._handle_infer(InferenceTaskType.METADATA_EXTRACTION, payload)
            elif path == "/infer/summarize":
                self._handle_infer(InferenceTaskType.CONTENT_SUMMARIZATION, payload)
            elif path == "/infer/synthesize-audio":
                self._handle_synthesize_audio(payload)
            elif path == "/providers/switch":
                provider = payload.get("provider", "")
                if not provider:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "provider is required"})
                    return
                result = self.inference_service.set_active_provider(provider)
                self._send_json(HTTPStatus.OK, result)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            logger.exception("Request failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_infer(self, task: InferenceTaskType, payload: dict[str, Any]) -> None:
        image_bytes = None
        mime_type = payload.get("image_mime_type", "image/jpeg")

        if payload.get("image_base64"):
            image_bytes = base64.b64decode(payload["image_base64"])
        elif payload.get("image_path"):
            from pathlib import Path
            path = Path(payload["image_path"])
            if path.exists():
                image_bytes = path.read_bytes()

        request = InferenceRequest(
            task=task,
            image_bytes=image_bytes,
            image_mime_type=mime_type if image_bytes else None,
            text=payload.get("text"),
            prompt=payload.get("prompt", ""),
            parameters={k: v for k, v in payload.items() if k.startswith("param_")},
            request_id=payload.get("request_id"),
        )
        result = self.inference_service.infer(request)
        self._send_json(HTTPStatus.OK, {
            "task": result.task.value,
            "provider": result.provider.value,
            "model_id": result.model_id,
            "output": result.output,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "token_usage": result.token_usage,
            "request_id": result.request_id,
        })

    def _handle_synthesize_audio(self, payload: dict[str, Any]) -> None:
        """POST /infer/synthesize-audio — text-to-speech via the active provider.

        Request body:
          {
            "text":   "<required, what to speak>",
            "language": "<BCP-47, e.g. hi-IN, en-US, default en-US>",
            "voice":  "<optional voice name, e.g. Kore>",
            "request_id": "<optional echo>"
          }

        Response: JSON envelope with `output` as base64-encoded WAV bytes
        and `metadata` carrying the MIME type / language / voice used.
        """
        text = (payload.get("text") or "").strip()
        if not text:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "text is required"})
            return

        # Build parameters dict from the well-known TTS fields.
        parameters: dict[str, Any] = {}
        if payload.get("language"):
            parameters["language"] = payload["language"]
        if payload.get("voice"):
            parameters["voice"] = payload["voice"]
        if payload.get("model_id"):
            parameters["model_id"] = payload["model_id"]
        # Any extra param_* keys flow through.
        for k, v in payload.items():
            if k.startswith("param_"):
                parameters[k[len("param_"):]] = v

        request = InferenceRequest(
            task=InferenceTaskType.AUDIO_SYNTHESIS,
            text=text,
            parameters=parameters,
            request_id=payload.get("request_id"),
        )
        result = self.inference_service.infer(request)
        if result.output.startswith("ERROR:"):
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": result.output[len("ERROR:"):].strip(),
                    "provider": result.provider.value,
                    "model_id": result.model_id,
                },
            )
            return
        self._send_json(HTTPStatus.OK, {
            "task": result.task.value,
            "provider": result.provider.value,
            "model_id": result.model_id,
            "output": result.output,  # base64-encoded WAV
            "metadata": result.metadata or {},
            "latency_ms": result.latency_ms,
            "request_id": result.request_id,
        })

    def _health(self) -> dict[str, Any]:
        return self.inference_service.health()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        logger.info(format, *args)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ModelServiceHTTPServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        inference_service: InferenceService,
    ) -> None:
        handler = type(
            "ModelRequestHandler",
            (_ModelRequestHandler,),
            {"inference_service": inference_service},
        )
        self._server = ThreadingHTTPServer((host, port), handler)

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.5)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
