"""Application-layer translation service.

Sits one level above ``GeminiTranslationProvider`` and ties it together
with the language-preference registry. Two entry points that the
orchestrator and the UI call:

* ``translate_for_session(text, session_id)`` — translate the *user's
  input* from their preferred UI language into English (so the LLM
  can reason about it in a single language). Returns the original text
  when the session is already in English mode (no translation, no cost).

* ``translate_from_english(text, session_id)`` — translate the *bot's
  English reply* into the user's preferred UI language. Returns the
  original text when the session is in English mode.

Both methods are no-ops when the preference for that session says the
user is on English, so the orchestrator can call them unconditionally
on every turn without paying for translations that aren't needed.

Adding a new language requires:
1. Adding the BCP-47 code to ``SUPPORTED_LANGUAGE_CODES`` (in
   ``language_preferences.py``).
2. Nothing else.
"""
from __future__ import annotations

import logging

from model_service.infrastructure.language_preferences import (
    DEFAULT_PREFERENCE,
    LanguagePreferenceStore,
)
from model_service.infrastructure.language_preferences import (
    get_store as get_preference_store,
)
from model_service.infrastructure.translation_provider import (
    TranslationProvider,
    TranslationRequest,
    TranslationResult,
    get_translation_provider,
)

logger = logging.getLogger(__name__)


class TranslationService:
    """Coordinate provider + preference store for the orchestrator's hot path.

    The orchestrator constructs one of these per process (or calls the
    module-level helpers). All methods are thread-safe.
    """

    def __init__(
        self,
        *,
        provider: TranslationProvider | None = None,
        preferences: LanguagePreferenceStore | None = None,
    ) -> None:
        self._provider = provider or get_translation_provider()
        self._preferences = preferences or get_preference_store()

    # ----------------------------------------------------------------- API
    def translate_user_input(
        self, text: str, session_id: str
    ) -> TranslationResult:
        """Translate user-typed text from their UI language into English.

        No-op when the session's preference has source == English (returns
        the original text with ``cached=True``).
        """
        pref = self._preferences.get(session_id)
        if pref.source == "en" or pref.source == DEFAULT_PREFERENCE.source:
            return TranslationResult(
                text=text,
                source=pref.source,
                target=pref.target,
                cached=True,
                latency_ms=0,
            )
        return self._provider.translate(
            TranslationRequest(text=text, source=pref.source, target="en")
        )

    def translate_bot_reply(
        self, text: str, session_id: str
    ) -> TranslationResult:
        """Translate the LLM's English reply into the user's UI language.

        No-op when target == English.
        """
        pref = self._preferences.get(session_id)
        if pref.target == "en":
            return TranslationResult(
                text=text,
                source="en",
                target=pref.target,
                cached=True,
                latency_ms=0,
            )
        return self._provider.translate(
            TranslationRequest(text=text, source="en", target=pref.target)
        )

    def get_preference(self, session_id: str):
        return self._preferences.get(session_id)

    def set_preference(
        self,
        session_id: str,
        *,
        source: str | None = None,
        target: str | None = None,
    ):
        return self._preferences.set(
            session_id, source=source, target=target
        )


# ---------------------------------------------------------------------------
# Module-level singleton.
# ---------------------------------------------------------------------------
_service: TranslationService | None = None


def get_translation_service() -> TranslationService:
    """Return the process-wide translation service (lazy init)."""
    global _service
    if _service is None:
        _service = TranslationService()
    return _service


def reset_service_for_tests() -> None:
    global _service
    _service = None