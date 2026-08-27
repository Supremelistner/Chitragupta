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
# Alert settings
# ---------------------------------------------------------------------------
ALERT_RISK_THRESHOLD = float(os.getenv("ALERT_RISK_THRESHOLD", "0.5"))
VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "30"))
