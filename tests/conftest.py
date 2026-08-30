"""Pytest conftest.

Adds ``src/`` to ``sys.path`` so the document-management service (and
related packages) can be imported as top-level modules. This is necessary
because the project uses a ``src/`` layout without an editable install in
the test environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
