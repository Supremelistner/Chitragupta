"""Translation provider for Chitragupta V1.

Wraps the Google Gemini API (configured via the existing
``ModelServiceConfig.gemini_*`` fields) to translate user-facing strings
between the inner English pipeline and any supported UI language
(currently: Hindi, English variants; trivially extensible to Tamil,
Bengali, etc. via ``SUPPORTED_LANGUAGE_CODES``).

Why a separate module and not a method on ``GeminiProviderAdapter``?
* Keeps the translation hot path independent of vision/OCR/TTS code.
* Has its own cache (translation is hit way more often than
  classification; caching is the only way to stay under free-tier rate
  limits for repeated phrases like "Source: <name> (<relation>)").
* Has its own structured ``TranslationResult`` so callers don't need to
  know about Gemini SDK types.

Adding a new language:
1. Add the BCP-47 code to ``SUPPORTED_LANGUAGE_CODES`` in
   ``language_preferences.py``.
2. (Optional) extend the BCP-47 → display-name map here.
3. No code changes in the orchestrator, UI, or MCP server.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from model_service.config import ModelServiceConfig
from model_service.infrastructure.language_preferences import SUPPORTED_LANGUAGE_CODES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache layout: a small JSON file mapping
#   "<sha256(text|source|target)>" -> {"text": "...", "ts": <unix>}
# ---------------------------------------------------------------------------
DEFAULT_CACHE_PATH = Path(os.environ.get(
    "CHITRAGUPTA_TRANSLATION_CACHE_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "translation_cache.json"),
))


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """One translation job. Immutable so it can be a dict key."""

    text: str
    source: str
    target: str

    def cache_key(self) -> str:
        joined = f"{self.text}|{self.source}|{self.target}".encode()
        return hashlib.sha256(joined).hexdigest()


@dataclass(frozen=True, slots=True)
class TranslationResult:
    text: str
    source: str
    target: str
    cached: bool
    latency_ms: int


class TranslationProvider(Protocol):
    """Port interface for any translation backend.

    The default implementation is ``GeminiTranslationProvider``; tests
    substitute a fake. Adding a new backend (DeepL, local NLLB, etc.)
    means implementing this Protocol — no other code changes.
    """

    def translate(self, request: TranslationRequest) -> TranslationResult: ...

    def supported_pairs(self) -> list[tuple[str, str]]: ...

    def health(self) -> bool: ...

    def close(self) -> None: ...


class _FileCache:
    """Tiny JSON-backed LRU-ish cache.

    * Thread-safe via an RLock.
    * Atomic write via temp-file + ``os.replace`` so a crash mid-write
      can't corrupt the cache.
    * Bounded by ``max_entries`` to keep the file small.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        max_entries: int = 5000,
    ) -> None:
        self._path = Path(path)
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._store: dict[str, str] = {}
        self._loaded = False

    def _load_locked(self) -> None:
        if not self._path.exists():
            self._store = {}
            self._loaded = True
            return
        try:
            self._store = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read translation cache %s: %s; starting empty.",
                self._path,
                exc,
            )
            self._store = {}
        self._loaded = True

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".translation_cache.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(self._store, fh, ensure_ascii=False, indent=0)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, key: str) -> str | None:
        with self._lock:
            if not self._loaded:
                self._load_locked()
            return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if not self._loaded:
                self._load_locked()
            # Cheap eviction: when over capacity, drop ~10% of the oldest
            # entries by insertion order. dict preserves insertion order
            # in Python 3.7+, so the first ~max_entries//10 keys are the
            # oldest.
            if len(self._store) >= self._max_entries:
                excess = len(self._store) - self._max_entries + 1
                victims = list(self._store.keys())[: max(excess, self._max_entries // 10)]
                for victim in victims:
                    del self._store[victim]
            self._store[key] = value
            self._flush_locked()


# ---------------------------------------------------------------------------
# Gemini-backed provider
# ---------------------------------------------------------------------------
class GeminiTranslationProvider:
    """Translation via Google Gemini (default model: ``gemini-2.5-flash-lite``).

    The provider is intentionally small. There is no streaming, no
    function calling, no multimodal — translation is a text-in / text-out
    job. We send a tight system prompt instructing the model to translate
    literally and to never add commentary, because the orchestrator's LLM
    may then paraphrase what we hand back.
    """

    _SYSTEM_PROMPT = (
        "You are a translation engine. Translate the user's text from "
        "{source} to {target}. Output ONLY the translated text. No "
        "preamble, no commentary, no quotation marks, no markdown "
        "fences, no language tags. Preserve punctuation, numbers, "
        "proper nouns, and code identifiers exactly as written. If the "
        "input is already in {target}, return it unchanged."
    )

    def __init__(
        self,
        *,
        config: ModelServiceConfig | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self._config = config or ModelServiceConfig.from_env()
        self._cache = _FileCache(cache_path or DEFAULT_CACHE_PATH)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai is required for translation. "
                    "Install with: pip install google-genai"
                ) from exc
            if not self._config.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not configured; cannot translate."
                )
            self._client = genai.Client(api_key=self._config.gemini_api_key)
        return self._client

    def _validate(self, request: TranslationRequest) -> None:
        if request.source == request.target:
            # Translating to the same language is a no-op; we still want
            # to short-circuit it here so callers don't accidentally hit
            # the network.
            return
        if request.source not in SUPPORTED_LANGUAGE_CODES:
            raise ValueError(
                f"unsupported source {request.source!r}; "
                f"supported: {sorted(SUPPORTED_LANGUAGE_CODES)}"
            )
        if request.target not in SUPPORTED_LANGUAGE_CODES:
            raise ValueError(
                f"unsupported target {request.target!r}; "
                f"supported: {sorted(SUPPORTED_LANGUAGE_CODES)}"
            )

    def translate(self, request: TranslationRequest) -> TranslationResult:
        self._validate(request)
        started = time.monotonic()

        # Fast path: identical source/target -> echo input.
        if request.source == request.target:
            return TranslationResult(
                text=request.text,
                source=request.source,
                target=request.target,
                cached=True,
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        key = request.cache_key()
        cached = self._cache.get(key)
        if cached is not None:
            return TranslationResult(
                text=cached,
                source=request.source,
                target=request.target,
                cached=True,
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        client = self._get_client()
        system = self._SYSTEM_PROMPT.format(
            source=request.source, target=request.target
        )
        # ``google-genai`` exposes a chat-style ``generate_content`` that
        # accepts a single ``contents`` list. We pass the system prompt
        # via ``system_instruction`` (new SDK) when available, else as
        # the first content item.
        response = client.models.generate_content(
            model=self._config.gemini_model_id,
            contents=request.text,
            config={
                "system_instruction": system,
                "temperature": 0.0,  # deterministic for caching
                "max_output_tokens": max(64, len(request.text) * 4),
            },
        )
        translated = (response.text or "").strip()
        if not translated:
            raise RuntimeError(
                f"Gemini returned empty translation for {request.source}->{request.target}"
            )

        self._cache.put(key, translated)
        return TranslationResult(
            text=translated,
            source=request.source,
            target=request.target,
            cached=False,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def supported_pairs(self) -> list[tuple[str, str]]:
        codes = sorted(SUPPORTED_LANGUAGE_CODES)
        # Cartesian product minus self-pairs. V1 is en↔hi-centric; new
        # codes added to the frozenset flow through here automatically.
        return [
            (s, t) for s in codes for t in codes if s != t
        ]

    def health(self) -> bool:
        try:
            # Cheap probe: translate a single ASCII character.
            self.translate(TranslationRequest(text="a", source="en", target="hi"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Translation provider health probe failed: %s", exc)
            return False

    def close(self) -> None:
        self._client = None


# ---------------------------------------------------------------------------
# Module-level singleton for the MCP tool + HTTP route to share.
# ---------------------------------------------------------------------------
_provider: GeminiTranslationProvider | None = None
_provider_lock = threading.Lock()


def get_translation_provider() -> GeminiTranslationProvider:
    """Return the process-wide translation provider (lazy init)."""
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = GeminiTranslationProvider()
        return _provider


def reset_provider_for_tests() -> None:
    """Drop the singleton — tests inject their own."""
    global _provider
    with _provider_lock:
        _provider = None