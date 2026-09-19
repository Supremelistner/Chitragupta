"""Validator Service configuration — reads from central .env file."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Validator Service
# ---------------------------------------------------------------------------
VALIDATOR_SERVICE_HTTP_PORT = int(os.getenv("VALIDATOR_SERVICE_HTTP_PORT", "8083"))
VALIDATOR_SERVICE_MCP_SERVER_NAME = "validator-service"
VALIDATOR_SERVICE_MCP_SERVER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Model Service (used for visual validation)
# ---------------------------------------------------------------------------
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8081")

# ---------------------------------------------------------------------------
# Document Service (used for alerting and metadata lookup)
# ---------------------------------------------------------------------------
DOCUMENT_SERVICE_URL = os.getenv("DOCUMENT_SERVICE_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# Risk-score policy (single source of truth)
# ---------------------------------------------------------------------------
# One authoritative interpretation of the 0.0–1.0 risk score, used by status
# classification, alert dispatch, and alert severity so the same score can
# never produce disagreeing outcomes. Override any threshold via env.
#
#   risk >= RISK_INVALID_THRESHOLD      -> status INVALID   (+ CRITICAL alert)
#   risk >= RISK_SUSPICIOUS_THRESHOLD   -> status SUSPICIOUS (+ WARNING alert)
#   risk >= ALERT_RISK_THRESHOLD        -> an alert is dispatched
# (A structurally-failed doc is INVALID regardless of score.)
RISK_INVALID_THRESHOLD = float(os.getenv("RISK_INVALID_THRESHOLD", "0.7"))
RISK_SUSPICIOUS_THRESHOLD = float(os.getenv("RISK_SUSPICIOUS_THRESHOLD", "0.4"))

# ---------------------------------------------------------------------------
# Alert settings
# ---------------------------------------------------------------------------
ALERT_RISK_THRESHOLD = float(os.getenv("ALERT_RISK_THRESHOLD", "0.5"))
VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "30"))
