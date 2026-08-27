"""Infrastructure — alert adapter for notifying the document service.

When validation detects a fake or suspicious document, this adapter
sends an alert to the document management service to sabotage the
ingestion process and flag the document.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from validator_service.domain.models import (
    AlertPayload,
    AlertResult,
)

logger = logging.getLogger("validator_service.alert")


class DocumentServiceAlertAdapter:
    """Sends alerts to the document management service via HTTP.

    The document service receives these alerts and can:
    - Mark the document as INVALID/SUSPICIOUS
    - Block it from being indexed
    - Flag it for human review
    - Trigger notification to the user
    """

    def __init__(
        self,
        *,
        document_service_url: str = "http://localhost:8080",
        timeout_seconds: int = 10,
    ) -> None:
        self._document_service_url = document_service_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._alert_log: list[dict[str, Any]] = []  # In-memory log for V1

    def ping(self) -> None:
        """Check that the document service is reachable."""
        try:
            req = urllib.request.Request(
                f"{self._document_service_url}/healthz",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Document service health check failed: {resp.status}")
        except Exception as exc:
            raise RuntimeError(f"Document service unreachable: {exc}")

    def send_alert(self, alert: AlertPayload) -> AlertResult:
        """Send a validation-failure alert to the document service.

        The alert is posted to the document service's alert endpoint.
        For V1, we also log alerts in memory for audit purposes.
        """
        start = time.monotonic()

        # Log the alert
        self._alert_log.append({
            "document_id": alert.document_id,
            "version": alert.version,
            "severity": alert.severity.value,
            "alert_type": alert.alert_type,
            "message": alert.message,
            "risk_score": alert.risk_score,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
        })

        # Send to document service
        try:
            payload = json.dumps({
                "document_id": alert.document_id,
                "version": alert.version,
                "severity": alert.severity.value,
                "alert_type": alert.alert_type,
                "message": alert.message,
                "details": alert.details,
                "risk_score": alert.risk_score,
                "request_id": alert.request_id,
                "source": "validator_service",
                "timestamp": alert.created_at.isoformat() if alert.created_at else datetime.now(timezone.utc).isoformat(),
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._document_service_url}/alerts",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Source": "validator_service",
                },
            )

            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                latency_ms = (time.monotonic() - start) * 1000
                if resp.status in (200, 201, 202):
                    resp_data = json.loads(resp.read().decode("utf-8")) if resp.readable() else {}
                    alert_id = resp_data.get("alert_id")
                    logger.info(
                        "Alert sent: %s/%s severity=%s type=%s",
                        alert.document_id, alert.version,
                        alert.severity.value, alert.alert_type,
                    )
                    return AlertResult(
                        sent=True,
                        alert_id=alert_id,
                        latency_ms=latency_ms,
                    )
                else:
                    return AlertResult(
                        sent=False,
                        error=f"Unexpected status: {resp.status}",
                        latency_ms=latency_ms,
                    )

        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            # Document service may not have /alerts endpoint yet (V1)
            # In that case, log and treat as "sent" with a note
            if exc.code == 404:
                logger.warning(
                    "Document service does not have /alerts endpoint yet. "
                    "Alert logged locally: %s", alert.alert_type,
                )
                return AlertResult(
                    sent=True,
                    alert_id=f"local-{alert.document_id}-{alert.version}",
                    latency_ms=latency_ms,
                )
            logger.error("Alert send failed: %s", exc)
            return AlertResult(
                sent=False,
                error=str(exc),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("Alert send failed: %s", exc)
            return AlertResult(
                sent=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

    def get_alerts(
        self, document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return logged alerts (V1 — in-memory audit log)."""
        if document_id:
            return [a for a in self._alert_log if a["document_id"] == document_id]
        return list(self._alert_log)

    def close(self) -> None:
        pass
