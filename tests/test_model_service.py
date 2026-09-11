"""Tests for Model Service and Document Service OCR/image endpoint additions.

Covers:
- Model service domain models
- HuggingFace provider adapter (unit, no network)
- InferenceService orchestration
- MCP tool listing and dispatch
- Document service OCR endpoint
- Document service image endpoint
- MCP tools for OCR and image retrieval
"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    InferenceTaskType,
    ModelProviderType,
    ProviderHealth,
)
from model_service.domain.ports import ModelProvider
from model_service.application.inference import InferenceService
from model_service.infrastructure.huggingface import HuggingFaceProviderAdapter
from model_service.infrastructure.memory import InMemoryModelRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockProvider:
    """A mock model provider for testing without network."""

    def __init__(self, output: str = "mock output") -> None:
        self._output = output
        self._call_count = 0

    @property
    def provider_type(self) -> ModelProviderType:
        return ModelProviderType.LOCAL

    def ping(self) -> None:
        return None

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self._call_count += 1
        return InferenceResult(
            task=request.task,
            provider=self.provider_type,
            model_id="mock-model",
            output=self._output,
            request_id=request.request_id,
        )

    def list_models(self):
        return []

    def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.provider_type, healthy=True)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------

class TestDomainModels(unittest.TestCase):
    def test_inference_request_defaults(self) -> None:
        req = InferenceRequest(task=InferenceTaskType.CUSTOM)
        self.assertEqual(req.task, InferenceTaskType.CUSTOM)
        self.assertIsNone(req.image_bytes)
        self.assertEqual(req.prompt, "")

    def test_inference_result_fields(self) -> None:
        result = InferenceResult(
            task=InferenceTaskType.OCR_VERIFICATION,
            provider=ModelProviderType.HUGGINGFACE,
            model_id="test-model",
            output="verified",
            confidence=0.95,
        )
        self.assertEqual(result.output, "verified")
        self.assertEqual(result.confidence, 0.95)

    def test_task_types_covered(self) -> None:
        tasks = [
            InferenceTaskType.DOCUMENT_CLASSIFICATION,
            InferenceTaskType.OCR_VERIFICATION,
            InferenceTaskType.METADATA_EXTRACTION,
            InferenceTaskType.CONTENT_SUMMARIZATION,
            InferenceTaskType.PRIVACY_CLASSIFICATION,
            InferenceTaskType.CUSTOM,
        ]
        self.assertEqual(len(tasks), 6)


# ---------------------------------------------------------------------------
# InferenceService tests
# ---------------------------------------------------------------------------

class TestInferenceService(unittest.TestCase):
    def setUp(self) -> None:
        self.mock = _MockProvider("test output")
        self.repository = InMemoryModelRepository(initial_provider="mock")
        self.service = InferenceService(
            providers={"mock": self.mock},
            repository=self.repository,
        )

    def test_infer_uses_active_provider(self) -> None:
        result = self.service.infer(InferenceRequest(task=InferenceTaskType.CUSTOM))
        self.assertEqual(result.output, "test output")
        self.assertEqual(result.provider, ModelProviderType.LOCAL)
        self.assertEqual(self.mock._call_count, 1)

    def test_switch_provider(self) -> None:
        mock2 = _MockProvider("provider 2")
        self.service = InferenceService(
            providers={"mock": self.mock, "mock2": mock2},
            repository=self.repository,
        )
        self.service.set_active_provider("mock2")
        result = self.service.infer(InferenceRequest(task=InferenceTaskType.CUSTOM))
        self.assertEqual(result.output, "provider 2")
        self.assertEqual(self.mock._call_count, 0)

    def test_invalid_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.service.set_active_provider("nonexistent")

    def test_list_providers(self) -> None:
        providers = self.service.list_providers()
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["name"], "mock")
        self.assertTrue(providers[0]["healthy"])

    def test_health_check(self) -> None:
        health = self.service.health()
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["healthy"])


# ---------------------------------------------------------------------------
# HuggingFace adapter tests (unit, no network)
# ---------------------------------------------------------------------------

class TestHuggingFaceAdapter(unittest.TestCase):
    def test_provider_type(self) -> None:
        adapter = HuggingFaceProviderAdapter(model_id="test")
        self.assertEqual(adapter.provider_type, ModelProviderType.HUGGINGFACE)

    def test_list_models(self) -> None:
        adapter = HuggingFaceProviderAdapter(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
        models = adapter.list_models()
        self.assertEqual(len(models), 1)
        self.assertTrue(models[0].supports_vision)
        self.assertIn("vision", models[0].capabilities)

    def test_health_without_token(self) -> None:
        adapter = HuggingFaceProviderAdapter(model_id="test")
        # Should still report healthy (ping checks import, not auth)
        health = adapter.health()
        self.assertTrue(health.healthy)

    def test_system_prompts_not_empty(self) -> None:
        """Prompts should be populated with task-specific instructions.

        AUDIO_SYNTHESIS is skipped: TTS doesn't use a system prompt; the
        `text` field IS the content to speak.
        """
        adapter = HuggingFaceProviderAdapter(model_id="test")
        for task in InferenceTaskType:
            if task == InferenceTaskType.AUDIO_SYNTHESIS:
                continue
            prompt = adapter._system_prompt(task)
            self.assertGreater(len(prompt), 50, f"Prompt for {task.value} should be populated")


# ---------------------------------------------------------------------------
# MCP server tool listing tests
# ---------------------------------------------------------------------------

class TestModelMCPServerTools(unittest.TestCase):
    def setUp(self) -> None:
        self.config = type("Config", (), {
            "mcp_server_name": "test-model-service",
            "mcp_server_version": "0.1.0",
        })()
        self.mock = _MockProvider()
        self.repository = InMemoryModelRepository()
        self.inference = InferenceService(
            providers={"mock": self.mock},
            repository=self.repository,
        )

    def _build_server(self):
        from model_service.adapters.mcp.server import MCPServer
        return MCPServer(config=self.config, inference_service=self.inference)

    def test_tools_listed(self) -> None:
        server = self._build_server()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
        tool_names = [t["name"] for t in response["result"]["tools"]]
        self.assertIn("classify_document", tool_names)
        self.assertIn("verify_ocr", tool_names)
        self.assertIn("extract_metadata", tool_names)
        self.assertIn("summarize_content", tool_names)
        self.assertIn("list_providers", tool_names)
        self.assertIn("health", tool_names)

    def test_health_tool(self) -> None:
        server = self._build_server()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "health", "arguments": {}},
        })
        content = json.loads(response["result"]["content"][0]["text"])
        self.assertIn(content["status"], ("healthy", "degraded"))

    def test_list_providers_tool(self) -> None:
        server = self._build_server()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "list_providers", "arguments": {}},
        })
        content = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(len(content["providers"]), 1)


# ---------------------------------------------------------------------------
# Document service MCP policy-surface tests
# ---------------------------------------------------------------------------

class TestDocumentMCPPolicySurface(unittest.TestCase):
    """Test that document MCP exposes policy-managed document operations."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter
        from document_mgmt_service.application.search import (
            SemanticChunker, SemanticSearchService, TextEmbeddingService,
        )
        from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
        from document_mgmt_service.application.access import DocumentAccessService
        from document_mgmt_service.application.health import HealthService

        from tests._live_stack import build_postgres_repo, build_qdrant_store

        self.repo = build_postgres_repo(self)
        self.storage = LocalFileStorageAdapter(Path(self.tempdir.name))
        self.store = build_qdrant_store(self, dimension=32)
        self.search = SemanticSearchService(
            store=self.store,
            embedder=TextEmbeddingService(dimension=32),
            chunker=SemanticChunker(max_chunk_chars=60, chunk_overlap=10),
        )

        class _OCR:
            def ping(self): return None
            def extract_text(self, fp): return "extracted OCR text from document"
            def close(self): return None

        self.ingestion = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo, storage=self.storage,
                ocr=_OCR(), semantic_search=self.search,
            )
        )
        self.access = DocumentAccessService(
            repository=self.repo, storage=self.storage, search=self.search,
        )
        self.config = type("Config", (), {
            "mcp_server_name": "test", "mcp_server_version": "0.1.0",
        })()
        self.health = HealthService(self.config)

    def _build_mcp(self):
        from document_mgmt_service.adapters.mcp.server import MCPServer
        return MCPServer(
            config=self.config,
            health_service=self.health,
            ingestion_service=self.ingestion,
            access_service=self.access,
        )

    def _ingest(self, doc_id: str, filename: str = "test.pdf", content: bytes = b"test", privacy=None):
        from document_mgmt_service.domain.models import DocumentIngestionRequest
        return self.ingestion.ingest(DocumentIngestionRequest(
            original_filename=filename, content=content,
            content_type="application/pdf", document_id=doc_id,
            privacy_hint=privacy,
        ))

    def _rpc(self, server, name: str, args: dict | None = None) -> dict:
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args or {}},
        })
        return json.loads(response["result"]["content"][0]["text"])

    def test_mcp_tools_hide_raw_ocr_and_image(self) -> None:
        server = self._build_mcp()
        response = server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = [t["name"] for t in response["result"]["tools"]]
        self.assertIn("search_document_content", tool_names)
        self.assertIn("get_evidence", tool_names)
        self.assertIn("get_document", tool_names)
        self.assertNotIn("get_document_ocr", tool_names)
        self.assertNotIn("get_document_image", tool_names)

    def test_content_search_returns_policy_managed_text(self) -> None:
        self._ingest("doc-ocr-1")
        server = self._build_mcp()
        result = self._rpc(server, "search_document_content", {
            "query": "extracted text", "document_id": "doc-ocr-1", "version": 1,
        })
        self.assertEqual(result["results"][0]["document_id"], "doc-ocr-1")
        self.assertIn("provenance", result["results"][0])

    def test_document_metadata_includes_status_and_privacy(self) -> None:
        self._ingest("doc-ocr-2")
        server = self._build_mcp()
        result = self._rpc(server, "get_document_metadata", {
            "document_id": "doc-ocr-2", "version": 1,
        })
        self.assertIn("metadata", result)
        self.assertIn("description", result)
        self.assertIn("privacy", result)
        self.assertEqual(result["processing_status"], "INDEXED")

    def test_get_document_requires_approval(self) -> None:
        """Whole-document retrieval on non-public docs requires approval."""
        self._ingest("doc-img-1", content=b"fake-pdf-content")
        server = self._build_mcp()
        result = self._rpc(server, "get_document", {
            "document_id": "doc-img-1", "version": 1,
        })
        self.assertIn("access_action", result)
        self.assertEqual(result["access_action"], "REQUIRE_APPROVAL")
        self.assertIn("access_reason", result)

    def test_get_document_returns_content_for_open_documents(self) -> None:
        """Open whole-document retrieval uses original file storage."""
        from document_mgmt_service.domain.models import DocumentPrivacyClassification
        content = b"fake-pdf-content-for-image"
        self._ingest("doc-img-2", content=content, privacy=DocumentPrivacyClassification.OPEN)
        server = self._build_mcp()
        result = self._rpc(server, "get_document", {
            "document_id": "doc-img-2", "version": 1,
        })
        self.assertEqual(base64.b64decode(result["content_base64"]), content)

    def test_nonexistent_document_returns_error(self) -> None:
        server = self._build_mcp()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "get_document_metadata", "arguments": {
                "document_id": "nonexistent", "version": 1,
            }},
        })
        self.assertIn("error", response)


# ---------------------------------------------------------------------------
# Gemini provider tests
# ---------------------------------------------------------------------------

class _FakeGeminiBlob:
    def __init__(self, data: bytes, mime_type: str) -> None:
        self.data = data
        self.mime_type = mime_type


class _FakeGeminiPart:
    def __init__(self, data: bytes, mime_type: str) -> None:
        self.inline_data = _FakeGeminiBlob(data, mime_type)


class _FakeGeminiContent:
    def __init__(self, parts) -> None:
        self.parts = parts


class _FakeGeminiCandidate:
    def __init__(self, parts) -> None:
        self.content = _FakeGeminiContent(parts)


class _FakeGeminiUsage:
    def __init__(self, prompt: int = 5, candidates: int = 7) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates


class _FakeGeminiTTSResponse:
    """Fake response for `client.models.generate_content` on the TTS model."""

    def __init__(self, pcm: bytes | None = None) -> None:
        pcm = pcm if pcm is not None else b"\x00\x01" * 64
        self.candidates = [
            _FakeGeminiCandidate(
                [_FakeGeminiPart(pcm, "audio/L16;rate=24000")]
            )
        ]
        self.usage_metadata = _FakeGeminiUsage(prompt=2, candidates=0)


class _FakeGeminiTextResponse:
    def __init__(self, text: str = "OK") -> None:
        self.text = text
        self.usage_metadata = _FakeGeminiUsage()
        self.candidates = []


class _FakeGeminiModels:
    """Minimal stand-in for `client.models` for both chat and TTS."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if "tts" in (model or "").lower():
            return _FakeGeminiTTSResponse()
        return _FakeGeminiTextResponse("OK")

    def list(self):
        return []


class _FakeGeminiClient:
    def __init__(self, **kwargs) -> None:
        self.models = _FakeGeminiModels()
        self.init_kwargs = kwargs


# Shared google.genai.types fake (see tests/_genai_stub.py). This used to be
# a local stub incompatible with test_orchestrator_service.py's copy, which
# made one file's Gemini tests fail in full-suite runs while passing solo.
from _genai_stub import ensure_genai_stub as _ensure_genai_stub

_ensure_genai_stub()


class TestGeminiProviderAdapter(unittest.TestCase):
    """Tests for the Gemini provider. No network. The google.genai SDK is
    faked via a direct _client swap on the adapter."""

    def _make_adapter(self, **overrides):
        from model_service.infrastructure.gemini_provider import GeminiProviderAdapter
        defaults = dict(
            api_key="test-key",
            model_id="gemini-2.5-flash-lite",
            tts_model_id="gemini-2.5-flash-preview-tts",
            tts_voice="Kore",
        )
        defaults.update(overrides)
        adapter = GeminiProviderAdapter(**defaults)
        adapter._client = _FakeGeminiClient()
        return adapter

    def test_provider_type_is_gemini(self) -> None:
        adapter = self._make_adapter()
        self.assertEqual(adapter.provider_type.value, "gemini")

    def test_chat_completion_calls_generate_content(self) -> None:
        from model_service.domain.models import InferenceRequest, InferenceTaskType
        adapter = self._make_adapter()
        request = InferenceRequest(
            task=InferenceTaskType.CUSTOM,
            text="Hello",
            prompt="Reply with a single word.",
        )
        result = adapter.infer(request)
        self.assertFalse(result.output.startswith("ERROR:"))
        self.assertEqual(result.output, "OK")
        self.assertEqual(
            adapter._client.models.calls[0]["model"], "gemini-2.5-flash-lite"
        )
        self.assertIn(
            "Reply with a single word.",
            adapter._client.models.calls[0]["contents"],
        )

    def test_tts_returns_base64_wav(self) -> None:
        from model_service.domain.models import InferenceRequest, InferenceTaskType
        import base64
        adapter = self._make_adapter()
        request = InferenceRequest(
            task=InferenceTaskType.AUDIO_SYNTHESIS,
            text="hello",
            parameters={"language": "hi-IN", "voice": "Kore"},
        )
        result = adapter.infer(request)
        self.assertFalse(result.output.startswith("ERROR:"))
        wav = base64.b64decode(result.output)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")
        self.assertEqual(result.metadata.get("language"), "hi-IN")
        self.assertEqual(result.metadata.get("voice"), "Kore")
        self.assertEqual(result.metadata.get("mime_type"), "audio/wav")

    def test_tts_unknown_language_falls_back_to_en_us(self) -> None:
        from model_service.domain.models import InferenceRequest, InferenceTaskType
        adapter = self._make_adapter()
        request = InferenceRequest(
            task=InferenceTaskType.AUDIO_SYNTHESIS,
            text="hi",
            parameters={"language": "klingon-KL"},
        )
        result = adapter.infer(request)
        self.assertFalse(result.output.startswith("ERROR:"))
        self.assertEqual(result.metadata.get("language"), "en-US")

    def test_tts_short_code_resolved(self) -> None:
        from model_service.domain.models import InferenceRequest, InferenceTaskType
        adapter = self._make_adapter()
        request = InferenceRequest(
            task=InferenceTaskType.AUDIO_SYNTHESIS,
            text="hi",
            parameters={"language": "ta"},
        )
        result = adapter.infer(request)
        self.assertEqual(result.metadata.get("language"), "ta-IN")

    def test_tts_invalid_voice_falls_back_to_default(self) -> None:
        from model_service.domain.models import InferenceRequest, InferenceTaskType
        adapter = self._make_adapter(tts_voice="Kore")
        request = InferenceRequest(
            task=InferenceTaskType.AUDIO_SYNTHESIS,
            text="hi",
            parameters={"voice": "NotAGeminiVoice"},
        )
        result = adapter.infer(request)
        self.assertEqual(result.metadata.get("voice"), "Kore")

    def test_tts_empty_text_returns_error_envelope(self) -> None:
        from model_service.domain.models import InferenceRequest, InferenceTaskType
        adapter = self._make_adapter()
        request = InferenceRequest(task=InferenceTaskType.AUDIO_SYNTHESIS, text="")
        result = adapter.infer(request)
        self.assertTrue(result.output.startswith("ERROR:"))

    def test_list_models_reports_chat_and_tts(self) -> None:
        adapter = self._make_adapter()
        models = adapter.list_models()
        ids = {m.model_id for m in models}
        self.assertIn("gemini-2.5-flash-lite", ids)
        self.assertIn("gemini-2.5-flash-preview-tts", ids)
        tts = next(m for m in models if "tts" in m.model_id)
        self.assertTrue(tts.supports_text)
        self.assertFalse(tts.supports_vision)

    def test_health_pings_models_list(self) -> None:
        adapter = self._make_adapter()
        h = adapter.health()
        self.assertTrue(h.healthy)


class TestGeminiProviderInFactory(unittest.TestCase):
    """`_build_providers` must produce a Gemini adapter when configured."""

    def _config(self, **overrides):
        from model_service.config import ModelServiceConfig
        defaults = dict(
            app_name="t", environment="t", log_level="INFO",
            http_host="127.0.0.1", http_port=8081,
            mcp_server_name="t", mcp_server_version="0",
            active_provider="gemini",
            huggingface_token=None,
            huggingface_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
            fireworks_api_key=None, fireworks_model_id="",
            deepinfra_api_key=None, deepinfra_model_id="",
            openai_api_key=None, openai_model_id="gpt-4o-mini",
            gemini_api_key="test-key",
            gemini_model_id="gemini-2.5-flash-lite",
            gemini_tts_model_id="gemini-2.5-flash-preview-tts",
            gemini_tts_voice="Kore",
            max_image_size_bytes=10 * 1024 * 1024,
            default_temperature=0.1,
            default_max_tokens=2048,
            request_timeout_seconds=30,
            groq_api_key=None, groq_model_id="qwen/qwen3.6-27b",
            fallback_provider=None,
            file_storage_root=None,
        )
        defaults.update(overrides)
        return ModelServiceConfig(**defaults)

    def test_gemini_is_built_when_configured(self) -> None:
        from model_service.__main__ import _build_providers
        providers = _build_providers(self._config())
        self.assertIn("huggingface", providers)
        from model_service.infrastructure.gemini_provider import GeminiProviderAdapter
        self.assertIsInstance(providers["huggingface"], GeminiProviderAdapter)


if __name__ == "__main__":
    unittest.main()
