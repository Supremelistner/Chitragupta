"""Shared fake for ``google.genai.types`` used by the test suite.

Two test modules (model-service Gemini adapter tests and orchestrator
Gemini LLM tests) each used to install their own incompatible stub into
``sys.modules`` with a "skip if present" guard. Whichever file pytest
imported first won, and the other file's tests failed in full-suite runs
while passing solo. This module provides ONE union stub with the full
attribute surface both suites need, so import order no longer matters.

The stub is only installed when the real ``google-genai`` package is
absent (detected via the ``_CHITRAGUPTA_FAKE_TYPES`` marker: real SDK
modules never carry it).
"""
from __future__ import annotations


class _FakeGenaiTypes:
    """Union of the model-service and orchestrator Gemini test needs.

    Every member stores ``**kwargs`` and also exposes them as attributes
    (the setattr-flavored superset both suites tolerate).
    """

    class Type:
        STRING = "STRING"
        NUMBER = "NUMBER"
        INTEGER = "INTEGER"
        BOOLEAN = "BOOLEAN"
        ARRAY = "ARRAY"
        OBJECT = "OBJECT"

    class Schema:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FunctionDeclaration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Tool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Part:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Blob:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Content:
        def __init__(self, *, role, parts):
            self.role = role
            self.parts = parts

    class FunctionCall:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FunctionResponse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class SpeechConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class VoiceConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PrebuiltVoiceConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)


def ensure_genai_stub() -> None:
    """Install the union fake into ``sys.modules`` if the real SDK is absent."""
    import sys
    import types as _types

    existing = sys.modules.get("google.genai")
    if existing is not None and getattr(existing, "_CHITRAGUPTA_FAKE_TYPES", False):
        return  # our stub (same union) already installed
    if existing is not None and hasattr(existing, "types"):
        return  # real google-genai SDK present; leave it alone
    google_pkg = sys.modules.get("google")
    if google_pkg is None:
        google_pkg = _types.ModuleType("google")
        sys.modules["google"] = google_pkg
    genai_pkg = _types.ModuleType("google.genai")
    genai_pkg.types = _FakeGenaiTypes
    genai_pkg._CHITRAGUPTA_FAKE_TYPES = True
    sys.modules["google.genai"] = genai_pkg
    google_pkg.genai = genai_pkg
