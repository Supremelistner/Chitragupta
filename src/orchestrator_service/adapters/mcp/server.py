"""MCP server adapter for the orchestrator service.

Exposes orchestrator tools via Model Context Protocol:
- chat: send a message to a session and get a response
- create_session / delete_session / list_sessions
- confirm: respond to a confirmation request
- health: service health check
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from orchestrator_service.application.orchestrator import OrchestrationEngine
from orchestrator_service.config import OrchestratorConfig

logger = logging.getLogger("orchestrator.mcp")


class MCPServer:
    """MCP server exposing orchestrator tools."""

    def __init__(
        self,
        config: OrchestratorConfig,
        engine: OrchestrationEngine,
    ) -> None:
        self._config = config
        self._engine = engine

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
                return self._result(request_id, {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "orchestrator-service", "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                })
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
            logger.exception("MCP request failed")
            if request_id is not None:
                return self._error(request_id, -32603, str(exc))
            return None

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}

        if name == "chat":
            session_id = args.get("session_id", "")
            message = args.get("message", "")
            if not session_id or not message:
                return self._error(request_id, -32602, "session_id and message required")
            response = self._engine.process_message(session_id, message)
            return self._tool_result(request_id, {
                "session_id": response.session_id,
                "message": response.message,
                "confirmation_required": (
                    {
                        "request_id": response.confirmation_required.request_id,
                        "type": response.confirmation_required.confirmation_type.value,
                        "message": response.confirmation_required.message,
                    }
                    if response.confirmation_required
                    else None
                ),
                "tool_calls_count": len(response.tool_calls_made),
            })

        if name == "confirm":
            session_id = args.get("session_id", "")
            request_id_conf = args.get("request_id", "")
            approved = args.get("approved", False)
            if not session_id or not request_id_conf:
                return self._error(request_id, -32602, "session_id and request_id required")
            response = self._engine.handle_confirmation(session_id, request_id_conf, approved)
            return self._tool_result(request_id, {
                "session_id": response.session_id,
                "message": response.message,
            })

        if name == "create_session":
            session = self._engine.create_session(
                user_id=args.get("user_id", ""),
                title=args.get("title", ""),
            )
            return self._tool_result(request_id, {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
            })

        if name == "list_sessions":
            sessions = self._engine.list_sessions(user_id=args.get("user_id"))
            return self._tool_result(request_id, {
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "title": s.title,
                        "status": s.status.value,
                        "created_at": s.created_at.isoformat(),
                        "message_count": len(s.messages),
                    }
                    for s in sessions
                ]
            })

        if name == "get_session":
            sid = args.get("session_id", "")
            session = self._engine.get_session(sid)
            if session is None:
                return self._error(request_id, -32602, f"Session {sid} not found")
            return self._tool_result(request_id, {
                "session_id": session.session_id,
                "title": session.title,
                "status": session.status.value,
                "message_count": len(session.messages),
            })

        if name == "delete_session":
            sid = args.get("session_id", "")
            ok = self._engine.delete_session(sid)
            return self._tool_result(request_id, {"deleted": ok})

        if name == "health":
            health = self._engine._router.health_all()
            return self._tool_result(request_id, health)

        return self._error(request_id, -32602, f"Unknown tool: {name}")

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "chat",
                "description": "Send a message to a session and get a response. The orchestrator plans and executes tool calls automatically.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "message": {"type": "string", "description": "User message"},
                    },
                    "required": ["session_id", "message"],
                },
            },
            {
                "name": "confirm",
                "description": "Respond to a pending confirmation request.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "request_id": {"type": "string"},
                        "approved": {"type": "boolean"},
                    },
                    "required": ["session_id", "request_id", "approved"],
                },
            },
            {
                "name": "create_session",
                "description": "Create a new conversation session with isolated context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "title": {"type": "string"},
                    },
                },
            },
            {
                "name": "list_sessions",
                "description": "List active sessions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                    },
                },
            },
            {
                "name": "get_session",
                "description": "Get session details.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "delete_session",
                "description": "Delete a session and all its context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "health",
                "description": "Check orchestrator and all downstream services.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    # -- Wire protocol --

    def _tool_result(self, request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._result(request_id, {
            "content": [{"type": "text", "text": json.dumps(payload, default=str, separators=(",", ":"))}],
            "isError": False,
        })

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
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
        writer.write(header)
        writer.write(payload)
        writer.flush()
