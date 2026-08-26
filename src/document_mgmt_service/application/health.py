from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable

from document_mgmt_service.config import AppConfig


@dataclass(frozen=True, slots=True)
class DependencyState:
    name: str
    healthy: bool
    details: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class HealthReport:
    status: str
    service: str
    version: str
    environment: str
    timestamp: str
    dependencies: tuple[DependencyState, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "service": self.service,
            "version": self.version,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "dependencies": [asdict(item) for item in self.dependencies],
        }


class HealthService:
    def __init__(self, config: AppConfig, dependencies: Iterable[DependencyState] = ()) -> None:
        self._config = config
        self._dependencies = tuple(dependencies)

    def report(self) -> HealthReport:
        status = "healthy"
        if any(not dep.healthy for dep in self._dependencies):
            status = "degraded"
        return HealthReport(
            status=status,
            service=self._config.app_name,
            version=self._config.mcp_server_version,
            environment=self._config.environment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dependencies=self._dependencies,
        )
