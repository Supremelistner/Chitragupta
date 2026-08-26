"""Model Service entrypoint."""

from __future__ import annotations

import argparse
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

    # HuggingFace (default — works on free tier)
    providers["huggingface"] = HuggingFaceProviderAdapter(
        token=config.huggingface_token,
        model_id=config.huggingface_model_id,
        timeout_seconds=config.request_timeout_seconds,
        temperature=config.default_temperature,
        max_tokens=config.default_max_tokens,
    )

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
