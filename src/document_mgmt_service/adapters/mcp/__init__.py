"""MCP delivery adapter.

Two transports are supported:

* ``stdio`` - the default. The MCP server reads JSON-RPC frames from
  ``sys.stdin`` and writes them to ``sys.stdout``. Suitable for local
  agent/CLI usage.
* ``http``  - the MCP server exposes the same JSON-RPC protocol over
  ``POST /mcp`` via FastAPI/uvicorn. Required for Docker / network
  agents and any client that can't share the server's stdio.
"""
from document_mgmt_service.adapters.mcp.server import MCPServer

__all__ = ["MCPServer"]
