# Main entrypoint
from __future__ import annotations
import argparse, signal, threading
from web_search_service.adapters.http.server import WebSearchHTTPServer
from web_search_service.adapters.mcp.server import MCPServer
from web_search_service.application.retrieval import RetrievalDependencies, RetrievalService
from web_search_service.config import WebSearchConfig
from web_search_service.infrastructure.firecrawl import FirecrawlAdapter
from web_search_service.infrastructure.fetch import SimpleFetchAdapter

def main():
    parser = argparse.ArgumentParser(description="Web Search Service")
    parser.add_argument("--http-host", default=None)
    parser.add_argument("--http-port", type=int, default=None)
    parser.add_argument("--http-only", action="store_true")
    parser.add_argument("--mcp-only", action="store_true")
    args = parser.parse_args()
    config = WebSearchConfig.from_env(http_host=args.http_host, http_port=args.http_port)

    firecrawl = FirecrawlAdapter(api_key=config.firecrawl_api_key, base_url=config.firecrawl_base_url) if config.firecrawl_api_key else None
    fetcher = SimpleFetchAdapter()

    retrieval = RetrievalService(RetrievalDependencies(
        search_provider=firecrawl, scraper_provider=firecrawl, fetcher=fetcher))

    print(f"Web Search Service on {config.http_host}:{config.http_port}")
    print(f"  Firecrawl: {"enabled" if firecrawl else "disabled (no API key)"}")
    print(f"  Fetch: enabled")

    http_server = None; http_thread = None; stop = threading.Event()
    def shutdown(*_): stop.set(); http_server and http_server.stop()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        if not args.mcp_only:
            http_server = WebSearchHTTPServer(host=config.http_host, port=config.http_port, retrieval_service=retrieval)
            http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            http_thread.start()
        if not args.http_only:
            MCPServer(config=config, retrieval_service=retrieval).serve()
        else: stop.wait()
    finally:
        http_server and http_server.stop()
        http_thread and http_thread.join(timeout=5)
        fetcher.close()
        firecrawl and firecrawl.close()

if __name__ == "__main__": main()
