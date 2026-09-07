"""Model Service entrypoint."""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from model_service.adapters.http.server import ModelServiceHTTPServer
from model_service.adapters.mcp.server import MCPServer
from model_service.application.inference import InferenceService
from model_service.config import ModelServiceConfig
from model_service.infrastructure.huggingface import HuggingFaceProviderAdapter
from model_service.infrastructure.memory import InMemoryModelRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model Service")
    parser.add_argument("--http-host", default=None, help="HTTP server host")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP server port")
    parser.add_argument("--http-only", action="store_true", help="Run only the HTTP server")
    parser.add_argument("--mcp-only", action="store_true", help="Run only the MCP server over stdio")
    return parser


def _build_providers(config: ModelServiceConfig) -> dict[str, HuggingFaceProviderAdapter]:
    """Build provider adapters from config. Add new providers here."""
    providers = {}

    # Gemini is selected when MODEL_SERVICE_PROVIDER=gemini (and GEMINI_API_KEY
    # is set). It is the strictest switch: if you ask for Gemini, you get
    # Gemini. Falls back to Groq only if explicitly configured.
    if config.active_provider == "gemini" and config.gemini_api_key:
        from model_service.infrastructure.gemini_provider import GeminiProviderAdapter
        gemini = GeminiProviderAdapter(
            api_key=config.gemini_api_key,
            model_id=config.gemini_model_id,
            tts_model_id=config.gemini_tts_model_id,
            tts_voice=config.gemini_tts_voice,
            timeout_seconds=config.request_timeout_seconds,
            temperature=config.default_temperature,
            max_tokens=config.default_max_tokens,
        )
        providers["huggingface"] = gemini
        # Also register under the active-provider key so the
        # repository's lookup by `active_provider == "gemini"` resolves.
        # Without this, the health check and any subsequent infer call
        # report "Active provider not found" because the registry only
        # had the alias under "huggingface".
        providers["gemini"] = gemini
        print(f"  Primary: Gemini ({config.gemini_model_id}) + TTS ({config.gemini_tts_model_id})")

        # Optional: wrap with Groq fallback for Gemini failures.
        if config.fallback_provider == "groq" and config.groq_api_key:
            from model_service.infrastructure.groq_provider import GroqProviderAdapter
            from model_service.infrastructure.fallback_provider import FallbackProvider
            groq = GroqProviderAdapter(
                api_key=config.groq_api_key,
                model_id=config.groq_model_id,
                timeout_seconds=config.request_timeout_seconds,
                temperature=config.default_temperature,
                max_tokens=config.default_max_tokens,
            )
            primary = providers["huggingface"]
            providers["huggingface"] = FallbackProvider(primary=primary, fallback=groq)
            print(f"  Fallback: Groq ({config.groq_model_id}) -- activates on Gemini errors / timeouts / 4xx/5xx")
        return providers

    # If HF_TOKEN is empty AND Groq is configured as fallback, use Groq as the sole
    # primary. We alias the Groq adapter under the huggingface key so the repository
    # still tracks active_provider=huggingface (no caller change needed). When you
    # refill HF credits next month, set HF_TOKEN again and the system reverts to the
    # old wrap-with-fallback behavior automatically.
    if not config.huggingface_token and config.fallback_provider == "groq" and config.groq_api_key:
        from model_service.infrastructure.groq_provider import GroqProviderAdapter
        providers["huggingface"] = GroqProviderAdapter(
            api_key=config.groq_api_key,
            model_id=config.groq_model_id,
            timeout_seconds=config.request_timeout_seconds,
            temperature=config.default_temperature,
            max_tokens=config.default_max_tokens,
        )
        print(f"  Primary: Groq ({config.groq_model_id}) -- HF_TOKEN empty, serving as the sole model provider")
        return providers

    # HuggingFace (default -- works on free tier)
    providers["huggingface"] = HuggingFaceProviderAdapter(
        token=config.huggingface_token,
        model_id=config.huggingface_model_id,
        timeout_seconds=config.request_timeout_seconds,
        temperature=config.default_temperature,
        max_tokens=config.default_max_tokens,
    )

    # Groq fallback (when HF credits exhausted)
    if config.fallback_provider == "groq" and config.groq_api_key:
        from model_service.infrastructure.groq_provider import GroqProviderAdapter
        from model_service.infrastructure.fallback_provider import FallbackProvider
        groq = GroqProviderAdapter(
            api_key=config.groq_api_key,
            model_id=config.groq_model_id,
            timeout_seconds=config.request_timeout_seconds,
            temperature=config.default_temperature,
            max_tokens=config.default_max_tokens,
        )
        # Wrap the primary HF provider with fallback
        primary = providers["huggingface"]
        providers["huggingface"] = FallbackProvider(primary=primary, fallback=groq)
        print(f"  Fallback: Groq ({config.groq_model_id}) -- activates on HF errors / timeouts / 402/429 / 5xx")

    # Future providers can be added here:
    # if config.fireworks_api_key:
    #     providers["fireworks"] = FireworksProviderAdapter(...)
    # if config.deepinfra_api_key:
    #     providers["deepinfra"] = DeepInfraProviderAdapter(...)

    return providers


def main() -> None:
    args = build_parser().parse_args()
    config = ModelServiceConfig.from_env(
        http_host=args.http_host,
        http_port=args.http_port,
    )

    logging.getLogger("model_service").setLevel(
        getattr(logging, config.log_level.upper(), logging.INFO)
    )

    providers = _build_providers(config)
    repository = InMemoryModelRepository(initial_provider=config.active_provider)
    inference_service = InferenceService(providers=providers, repository=repository)

    http_server = None
    http_thread = None
    stop_event = threading.Event()

    def shutdown(*_: object) -> None:
        stop_event.set()
        if http_server is not None:
            http_server.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        if not args.mcp_only:
            http_server = ModelServiceHTTPServer(
                host=config.http_host,
                port=config.http_port,
                inference_service=inference_service,
            )
            http_thread = threading.Thread(
                target=http_server.serve_forever,
                name="model-http-server",
                daemon=True,
            )
            http_thread.start()
            print(f"Model Service HTTP listening on {config.http_host}:{config.http_port}")

        if not args.http_only:
            MCPServer(
                config=config,
                inference_service=inference_service,
            ).serve()
        else:
            stop_event.wait()
    finally:
        if http_server is not None:
            http_server.stop()
        if http_thread is not None:
            http_thread.join(timeout=5)
        for provider in providers.values():
            provider.close()
        repository.close()


if __name__ == "__main__":
    main()
