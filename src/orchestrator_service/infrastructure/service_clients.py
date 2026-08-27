"""HTTP clients for calling downstream microservices.

Each client wraps the MCP-compatible HTTP endpoints exposed by a service.
They share the same interface so the orchestrator can route tool calls uniformly.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from orchestrator_service.domain.models import ServiceTarget

logger = logging.getLogger("orchestrator.service_clients")


class HttpServiceClient:
    """Generic HTTP client that calls a microservice's tool endpoints."""

    def __init__(
        self,
        *,
        target: ServiceTarget,
        base_url: str,
        timeout_seconds: int = 30,
    ) -> None:
        self._target = target
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._session = requests.Session()

    @property
    def target(self) -> ServiceTarget:
        return self._target

    def health(self) -> dict[str, Any]:
        try:
            resp = self._session.get(
                f"{self._base_url}/healthz",
                timeout=self._timeout,
            )
            return resp.json()
        except Exception as e:
            logger.warning("Health check failed for %s: %s", self._target.value, e)
            return {"status": "unreachable", "error": str(e)}

    def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on this service via its HTTP endpoint."""
        start = time.monotonic()
        try:
            resp = self._session.post(
                f"{self._base_url}/tools/{tool_name}",
                json=arguments,
                timeout=self._timeout,
            )
            latency_ms = (time.monotonic() - start) * 1000

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                logger.error(
                    "Tool %s failed (%d): %s",
                    tool_name,
                    resp.status_code,
                    error_text,
                )
                return {
                    "error": True,
                    "status_code": resp.status_code,
                    "message": error_text,
                    "latency_ms": latency_ms,
                }

            result = resp.json()
            result["latency_ms"] = latency_ms
            return result

        except requests.exceptions.Timeout:
            return {
                "error": True,
                "message": f"Timeout calling {tool_name} after {self._timeout}s",
                "latency_ms": (time.monotonic() - start) * 1000,
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "error": True,
                "message": f"Cannot connect to {self._target.value}: {e}",
                "latency_ms": (time.monotonic() - start) * 1000,
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"Error calling {tool_name}: {e}",
                "latency_ms": (time.monotonic() - start) * 1000,
            }

    def close(self) -> None:
        self._session.close()


class ServiceClientRouter:
    """Routes tool calls to the correct service client based on tool name."""

    def __init__(self, clients: dict[ServiceTarget, HttpServiceClient]) -> None:
        self._clients = clients

    def call_tool(
        self,
        target: ServiceTarget,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        client = self._clients.get(target)
        if client is None:
            return {
                "error": True,
                "message": f"No client configured for {target.value}",
            }
        return client.call_tool(tool_name, arguments)

    def health_all(self) -> dict[str, Any]:
        results = {}
        for target, client in self._clients.items():
            results[target.value] = client.health()
        return results
