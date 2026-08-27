"""Unified health aggregator - pings all services and returns combined status."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class ServiceHealth:
    name: str
    url: str
    status: str
    latency_ms: float = 0.0
    version: str | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AggregatedHealth:
    overall_status: str
    services: tuple[ServiceHealth, ...]
    infrastructure: tuple[ServiceHealth, ...]
    timestamp: str
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "timestamp": self.timestamp,
            "total_latency_ms": self.total_latency_ms,
            "services": [{"name": s.name, "url": s.url, "status": s.status, "latency_ms": s.latency_ms, "version": s.version, "error": s.error, "details": s.details} for s in self.services],
            "infrastructure": [{"name": s.name, "url": s.url, "status": s.status, "latency_ms": s.latency_ms, "error": s.error, "details": s.details} for s in self.infrastructure],
        }


class HealthAggregator:
    def __init__(self, *, document_service_url: str = "http://localhost:8080", model_service_url: str = "http://localhost:8081", web_search_service_url: str = "http://localhost:8082", validator_service_url: str = "http://localhost:8083", postgres_dsn: str | None = None, qdrant_url: str | None = None, timeout_seconds: float = 5.0) -> None:
        self._services = [("document_service", document_service_url), ("model_service", model_service_url), ("web_search_service", web_search_service_url), ("validator_service", validator_service_url)]
        self._postgres_dsn = postgres_dsn
        self._qdrant_url = qdrant_url
        self._timeout = timeout_seconds

    def check_all(self) -> AggregatedHealth:
        start = time.monotonic()
        services = [self._check_service(name, url) for name, url in self._services]
        infrastructure = []
        if self._postgres_dsn:
            infrastructure.append(self._check_postgres())
        if self._qdrant_url:
            infrastructure.append(self._check_qdrant())
        total_latency = (time.monotonic() - start) * 1000
        all_statuses = [s.status for s in services + infrastructure]
        if all(s == "healthy" for s in all_statuses):
            overall = "healthy"
        elif any(s == "unreachable" for s in all_statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"
        return AggregatedHealth(overall_status=overall, services=tuple(services), infrastructure=tuple(infrastructure), timestamp=datetime.now(timezone.utc).isoformat(), total_latency_ms=total_latency)

    def _check_service(self, name: str, url: str) -> ServiceHealth:
        start = time.monotonic()
        try:
            req = urllib.request.Request(f"{url.rstrip('/')}/healthz", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                latency = (time.monotonic() - start) * 1000
                data = json.loads(resp.read().decode("utf-8"))
                return ServiceHealth(name=name, url=url, status=data.get("status", "unknown"), latency_ms=latency, version=data.get("version"), details=data)
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return ServiceHealth(name=name, url=url, status="unreachable", latency_ms=latency, error=str(exc))

    def _check_postgres(self) -> ServiceHealth:
        start = time.monotonic()
        try:
            import psycopg2
            conn = psycopg2.connect(self._postgres_dsn, connect_timeout=3)
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = %s", ("active",))
                active = cur.fetchone()[0]
            conn.close()
            latency = (time.monotonic() - start) * 1000
            return ServiceHealth(name="postgresql", url="postgres", status="healthy", latency_ms=latency, details={"version": version, "active_connections": active})
        except ImportError:
            latency = (time.monotonic() - start) * 1000
            return ServiceHealth(name="postgresql", url="postgres", status="degraded", latency_ms=latency, error="psycopg2 not installed")
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return ServiceHealth(name="postgresql", url="postgres", status="unreachable", latency_ms=latency, error=str(exc))

    def _check_qdrant(self) -> ServiceHealth:
        start = time.monotonic()
        try:
            url = self._qdrant_url.rstrip("/")
            req = urllib.request.Request(f"{url}/collections", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                latency = (time.monotonic() - start) * 1000
                data = json.loads(resp.read().decode("utf-8"))
                collections = data.get("result", {}).get("collections", [])
                return ServiceHealth(name="qdrant", url=url, status="healthy", latency_ms=latency, details={"collections": len(collections)})
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return ServiceHealth(name="qdrant", url=self._qdrant_url or "qdrant", status="unreachable", latency_ms=latency, error=str(exc))
