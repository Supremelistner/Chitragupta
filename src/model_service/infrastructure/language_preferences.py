"""Per-session/per-user language preferences for Chitragupta V1.

V1 supports translating between English (the inner pipeline) and any user-
facing language. The user picks their preferred UI language via a switch
in the web UI; this module records that choice and exposes helpers to read
or update it.

The state is intentionally tiny — just a JSON file mapping
``session_id -> {source, target}``. We deliberately keep it inside the
model service (rather than in ``shared/``) because:

* The registry is *only* consulted by the translation provider and the
  orchestrator's tool loop, both of which already reach into the model
  service.
* Keeping language state with the translation code makes the future
  expansion to more languages a single-file change.
* The model service is the only service in the system that knows about
  *languages* in the first place — every other service works in English.

Adding a new language later (Tamil, Bengali, etc.) requires no change
here beyond updating the supported-pairs dict at startup; the registry
already stores arbitrary BCP-47 codes.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Default storage location: per-process state file. Kept inside the repo's
# data/ tree so dev runs are local-only and gitignored in production.
DEFAULT_STATE_PATH = Path(os.environ.get(
    "CHITRAGUPTA_LANGUAGE_STATE_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "translation_state.json"),
))


@dataclass(frozen=True, slots=True)
class LanguagePreference:
    """A single session's preferred source/target language pair.

    ``source`` is the language the user types in (the human-facing one).
    ``target`` is the language the user wants responses in. For V1 with
    one user-facing language toggle, ``target`` is the user-facing one
    and ``source`` is the inner English one. When the user flips the
    switch, ``source`` and ``target`` swap.

    Both fields hold BCP-47 codes (``en``, ``en-US``, ``hi``, ``hi-IN``,
    ``ta``, ``ta-IN``, ...). The translation provider accepts any pair.
    """

    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


# Supported language pairs are validated against this set so a typo in the
# UI never lands in storage. Adding a new pair here is the only thing
# required to support a new language end-to-end.
SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset(
    {"en", "en-US", "en-GB", "hi", "hi-IN", "ta", "ta-IN", "bn", "bn-IN"}
)

DEFAULT_PREFERENCE = LanguagePreference(source="en", target="hi")
"""V1 default: inner pipeline speaks English; user-facing language is Hindi.

Flip via ``set_preference()`` when the user toggles the UI switch.
"""


class LanguagePreferenceStore:
    """Thread-safe file-backed preference registry.

    Why a file instead of Postgres? State is tiny (one line per session),
    read on every chat turn, and we want it to survive service restarts
    without forcing a Postgres round-trip. This is the same pattern as
    the existing profile store in ``orchestrator_service.onboarding``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_STATE_PATH
        self._lock = threading.RLock()
        self._cache: dict[str, LanguagePreference] = {}
        self._loaded = False

    # ------------------------------------------------------------------ I/O
    def _load_locked(self) -> None:
        if not self._path.exists():
            self._cache = {}
            self._loaded = True
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read language preference file %s: %s; "
                "falling back to defaults.",
                self._path,
                exc,
            )
            self._cache = {}
            self._loaded = True
            return
        parsed: dict[str, LanguagePreference] = {}
        for session_id, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            source = payload.get("source")
            target = payload.get("target")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            if source not in SUPPORTED_LANGUAGE_CODES:
                continue
            if target not in SUPPORTED_LANGUAGE_CODES:
                continue
            parsed[session_id] = LanguagePreference(source=source, target=target)
        self._cache = parsed
        self._loaded = True

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a temp file in the same directory and rename.
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".translation_state.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {sid: pref.to_dict() for sid, pref in self._cache.items()},
                    fh,
                    indent=2,
                )
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ----------------------------------------------------------------- API
    def get(self, session_id: str) -> LanguagePreference:
        """Return the stored preference for ``session_id``, or the V1 default."""
        with self._lock:
            if not self._loaded:
                self._load_locked()
            return self._cache.get(session_id, DEFAULT_PREFERENCE)

    def set(
        self,
        session_id: str,
        *,
        source: str | None = None,
        target: str | None = None,
    ) -> LanguagePreference:
        """Update the preference for ``session_id``.

        Either ``source`` or ``target`` (or both) may be omitted; missing
        fields fall back to the existing value. At least one must be
        provided. Both must be in ``SUPPORTED_LANGUAGE_CODES``.
        """
        if source is None and target is None:
            raise ValueError("at least one of source/target must be provided")
        if source is not None and source not in SUPPORTED_LANGUAGE_CODES:
            raise ValueError(
                f"unsupported source language {source!r}; "
                f"supported: {sorted(SUPPORTED_LANGUAGE_CODES)}"
            )
        if target is not None and target not in SUPPORTED_LANGUAGE_CODES:
            raise ValueError(
                f"unsupported target language {target!r}; "
                f"supported: {sorted(SUPPORTED_LANGUAGE_CODES)}"
            )
        with self._lock:
            if not self._loaded:
                self._load_locked()
            current = self._cache.get(session_id, DEFAULT_PREFERENCE)
            new_pref = LanguagePreference(
                source=source if source is not None else current.source,
                target=target if target is not None else current.target,
            )
            self._cache[session_id] = new_pref
            self._flush_locked()
            return new_pref

    def clear(self, session_id: str) -> bool:
        """Forget a session's preference. Returns True if anything was cleared."""
        with self._lock:
            if not self._loaded:
                self._load_locked()
            if session_id in self._cache:
                del self._cache[session_id]
                self._flush_locked()
                return True
            return False

    def all_sessions(self) -> Iterable[tuple[str, LanguagePreference]]:
        with self._lock:
            if not self._loaded:
                self._load_locked()
            return list(self._cache.items())


# Module-level singleton, lazily constructed. Tests inject their own.
_store: LanguagePreferenceStore | None = None


def get_store() -> LanguagePreferenceStore:
    """Return the process-wide preference store."""
    global _store
    if _store is None:
        _store = LanguagePreferenceStore()
    return _store


def reset_store_for_tests() -> None:
    """Drop the singleton — tests tests inject their own."""
    global _store
    _store = None