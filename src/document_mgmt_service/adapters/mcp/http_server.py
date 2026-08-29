"""HTTP transport adapter for the Document Management MCP server.

Exposes the same JSON-RPC protocol used by the stdio transport over HTTP so
the service can be consumed from inside a Docker container or any networked
agent. We deliberately reuse the existing :class:`MCPServer` core — the only
HTTP-specific work is parsing the request body, dispatching it through
``handle_message``, and shaping the response.

Endpoints
---------
``POST /mcp``
    Accepts a single JSON-RPC 2.0 message (or a JSON array of messages) and
    returns either a single response object or a JSON array, respectively.
    Responses for ``id``-less notifications are omitted.

``GET  /mcp/health``
    Returns MCP server metadata and a basic liveness signal. Useful for
    orchestrators (Docker / Kubernetes) that need a non-RPC health probe.

``GET  /mcp/sse``
    Server-Sent Events stream. The current implementation keeps the
    connection open and emits a ``ready`` event plus periodic heartbeats so
    clients can confirm reachability. Full streaming notification support
    is left as a future extension.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from document_mgmt_service.adapters.mcp.server import MCPServer

logger = logging.getLogger("document_mgmt_service.mcp.http")


def build_app(mcp_server: MCPServer) -> FastAPI:
    """Build the FastAPI app that fronts the MCP server over HTTP."""
    app = FastAPI(
        title="Document Management MCP (HTTP)",
        version=mcp_server._config.mcp_server_version,
        description=(
            "HTTP/JSON-RPC transport for the Document Management MCP server. "
            "Compatible with the Model Context Protocol 2024-11-05 spec."
        ),
    )

    @app.get("/mcp/health")
    async def mcp_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "transport": "http",
            "server": {
                "name": mcp_server._config.mcp_server_name,
                "version": mcp_server._config.mcp_server_version,
            },
            "protocolVersion": "2024-11-05",
        }

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        # Accept either a single message object or a batch.
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                },
            )
        if isinstance(payload, list):
            responses = []
            for item in payload:
                response = mcp_server.handle_message(item)
                if response is not None:
                    responses.append(response)
            return JSONResponse(responses)
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: expected object or array"},
                },
            )
        response = mcp_server.handle_message(payload)
        if response is None:
            # Notification: 202 Accepted, no body content per JSON-RPC.
            return Response(status_code=202)
        return JSONResponse(response)

    @app.get("/mcp/sse")
    async def mcp_sse(request: Request) -> Response:
        """Server-Sent Events endpoint (reachability / future streaming)."""

        async def event_stream() -> AsyncIterator[bytes]:
            ready = json.dumps({"transport": "http"})
            yield f"event: ready\ndata: {ready}\n\n".encode()
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    await asyncio.sleep(15)
                    yield b"event: heartbeat\ndata: {}\n\n"
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": mcp_server._config.app_name,
            "mcp": {
                "endpoint": "/mcp",
                "health": "/mcp/health",
                "sse": "/mcp/sse",
                "transport": "http",
            },
        }

    return app


def run(
    mcp_server: MCPServer,
    *,
    host: str,
    port: int,
) -> None:
    """Run the HTTP MCP server (blocking)."""
    import uvicorn

    app = build_app(mcp_server)
    logger.info("Starting MCP HTTP transport on %s:%d", host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=mcp_server._config.log_level.lower(),
    )
