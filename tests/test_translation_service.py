"""Tests for the translation provider, preference registry, and service.

These tests are pure unit tests — no network calls, no live Postgres,
no live Gemini. They use a fake ``TranslationProvider`` to verify the
service's plumbing (when to translate, when to short-circuit, when to
fall back) and the preference store's read/write/validate logic.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

# Make the project src/ importable when running this file directly.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model_service.application.translation_service import (
    TranslationService,
    reset_service_for_tests,
)
from model_service.infrastructure.language_preferences import (
    DEFAULT_PREFERENCE,
    SUPPORTED_LANGUAGE_CODES,
    LanguagePreferenceStore,
    reset_store_for_tests,
)
from model_service.infrastructure.translation_provider import (
    TranslationRequest,
    TranslationResult,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeTranslationProvider:
    """TranslationProvider stub that records every call.

    Configurable to fail or succeed per-call. Used to verify the
    service's call-avoidance logic (cache hits, no-op short-circuits).
    """

    def __init__(self) -> None:
        self.calls: list[TranslationRequest] = []
        self.responses: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def translate(self, request: TranslationRequest) -> TranslationResult:
        with self.lock:
            self.calls.append(request)
            key = (request.source, request.target)
            translated = self.responses.get(
                key,
                f"[{request.source}->{request.target}] {request.text}",
            )
        return TranslationResult(
            text=translated,
            source=request.source,
            target=request.target,
            cached=False,
            latency_ms=1,
        )

    def supported_pairs(self) -> list[tuple[str, str]]:
        return [
            ("en", "hi"), ("hi", "en"),
            ("en", "ta"), ("ta", "en"),
            ("en", "bn"), ("bn", "en"),
        ]

    def health(self) -> bool:
        return True

    def close(self) -> None:
        pass


class _PreferenceStoreFixture:
    """Context manager yielding a fresh on-disk preference store."""

    def __enter__(self) -> LanguagePreferenceStore:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "prefs.json"
        self._store = LanguagePreferenceStore(self._path)
        return self._store

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmpdir.cleanup()


# ---------------------------------------------------------------------------
# LanguagePreferenceStore
# ---------------------------------------------------------------------------
class LanguagePreferenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_store_for_tests()

    def test_default_is_en_to_hi(self) -> None:
        with _PreferenceStoreFixture() as store:
            pref = store.get("unknown-session")
            self.assertEqual(pref, DEFAULT_PREFERENCE)
            self.assertEqual(pref.source, "en")
            self.assertEqual(pref.target, "hi")

    def test_set_persists_to_disk(self) -> None:
        with _PreferenceStoreFixture() as store:
            store.set("s1", target="ta")
            raw = json.loads(store._path.read_text(encoding="utf-8"))
            self.assertEqual(raw["s1"]["target"], "ta")

    def test_set_partial_keeps_existing(self) -> None:
        with _PreferenceStoreFixture() as store:
            store.set("s1", source="hi", target="hi")
            store.set("s1", source="ta")  # only source changes
            pref = store.get("s1")
            self.assertEqual(pref.source, "ta")
            self.assertEqual(pref.target, "hi")

    def test_set_rejects_unsupported_language(self) -> None:
        with _PreferenceStoreFixture() as store, self.assertRaises(ValueError):
            store.set("s1", target="klingon")

    def test_set_requires_at_least_one_arg(self) -> None:
        with _PreferenceStoreFixture() as store, self.assertRaises(ValueError):
            store.set("s1")

    def test_clear_removes_session(self) -> None:
        with _PreferenceStoreFixture() as store:
            store.set("s1", target="ta")
            self.assertTrue(store.clear("s1"))
            self.assertFalse(store.clear("s1"))  # idempotent
            # After clear, falls back to default
            self.assertEqual(store.get("s1"), DEFAULT_PREFERENCE)

    def test_atomic_write_does_not_leave_tmp(self) -> None:
        with _PreferenceStoreFixture() as store:
            store.set("s1", target="ta")
            tmp_files = list(Path(store._path.parent).glob(".translation_state.*.tmp"))
            self.assertEqual(tmp_files, [], "tmp file should be cleaned up")

    def test_corrupt_file_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            path.write_text("not-json", encoding="utf-8")
            store = LanguagePreferenceStore(path)
            # Should not raise; defaults used.
            self.assertEqual(store.get("any"), DEFAULT_PREFERENCE)

    def test_reload_after_set(self) -> None:
        """Reads see writes; a freshly-constructed store on the same path
        reads what the previous instance wrote."""
        with _PreferenceStoreFixture() as store:
            store.set("s1", target="ta")
            path = store._path
            fresh = LanguagePreferenceStore(path)
            self.assertEqual(fresh.get("s1").target, "ta")


# ---------------------------------------------------------------------------
# TranslationService
# ---------------------------------------------------------------------------
class TranslationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_service_for_tests()
        reset_store_for_tests()
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pref_path = Path(self._tmpdir.name) / "prefs.json"
        self._provider = _FakeTranslationProvider()
        self._store = LanguagePreferenceStore(self._pref_path)
        self._service = TranslationService(
            provider=self._provider,  # type: ignore[arg-type]
            preferences=self._store,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_no_translation_when_already_in_english(self) -> None:
        """Default preference has source=en, so the user's English text
        is passed through without hitting the provider."""
        result = self._service.translate_user_input("hello", "s1")
        self.assertEqual(result.text, "hello")
        self.assertTrue(result.cached)
        self.assertEqual(self._provider.calls, [])

    def test_translates_user_input_when_target_is_hindi(self) -> None:
        """Session is in Hindi mode -> user text is translated to English."""
        self._store.set("s1", source="hi", target="hi")
        self._provider.responses[("hi", "en")] = "hello"
        result = self._service.translate_user_input("नमस्ते", "s1")
        self.assertEqual(result.text, "hello")
        self.assertEqual(len(self._provider.calls), 1)
        self.assertEqual(self._provider.calls[0].source, "hi")
        self.assertEqual(self._provider.calls[0].target, "en")

    def test_translates_bot_reply_to_user_language(self) -> None:
        """Session is in Hindi mode -> bot's English reply is translated to Hindi."""
        self._store.set("s1", source="hi", target="hi")
        self._provider.responses[("en", "hi")] = "नमस्ते"
        result = self._service.translate_bot_reply("Hello there", "s1")
        self.assertEqual(result.text, "नमस्ते")
        self.assertEqual(len(self._provider.calls), 1)
        self.assertEqual(self._provider.calls[0].source, "en")
        self.assertEqual(self._provider.calls[0].target, "hi")

    def test_no_translation_when_user_english_and_target_english(self) -> None:
        """Both directions English -> everything passes through unchanged."""
        self._store.set("s1", source="en", target="en")
        user = self._service.translate_user_input("hello", "s1")
        bot = self._service.translate_bot_reply("hi back", "s1")
        self.assertEqual(user.text, "hello")
        self.assertEqual(bot.text, "hi back")
        self.assertEqual(self._provider.calls, [])

    def test_session_isolation(self) -> None:
        """One session in Hindi, another in English."""
        self._store.set("s1", source="hi", target="hi")
        self._store.set("s2", source="en", target="en")
        self._service.translate_user_input("नमस्त", "s1")
        self._service.translate_user_input("hello", "s2")
        # Only the Hindi session triggered a translation.
        self.assertEqual(len(self._provider.calls), 1)
        self.assertEqual(self._provider.calls[0].source, "hi")

    def test_get_and_set_preference_round_trip(self) -> None:
        pref = self._service.set_preference("s1", target="ta")
        self.assertEqual(pref.target, "ta")
        self.assertEqual(self._service.get_preference("s1").target, "ta")


# ---------------------------------------------------------------------------
# Supported-language contract (locks the public V1 surface)
# ---------------------------------------------------------------------------
class SupportedLanguagesTests(unittest.TestCase):
    def test_v1_includes_english_and_hindi(self) -> None:
        self.assertIn("en", SUPPORTED_LANGUAGE_CODES)
        self.assertIn("hi", SUPPORTED_LANGUAGE_CODES)

    def test_v1_includes_regional_codes(self) -> None:
        # Indian regional languages that may be added later; locking them
        # in now means the orchestrator's UI can mention them as "coming
       # soon" without code churn.
        self.assertIn("ta", SUPPORTED_LANGUAGE_CODES)
        self.assertIn("bn", SUPPORTED_LANGUAGE_CODES)

    def test_default_preference_is_en_hi(self) -> None:
        self.assertEqual(DEFAULT_PREFERENCE.source, "en")
        self.assertEqual(DEFAULT_PREFERENCE.target, "hi")


# ---------------------------------------------------------------------------
# TranslationRequest identity for caching
# ---------------------------------------------------------------------------
class TranslationRequestTests(unittest.TestCase):
    def test_same_text_same_pair_same_cache_key(self) -> None:
        a = TranslationRequest(text="hello", source="en", target="hi")
        b = TranslationRequest(text="hello", source="en", target="hi")
        self.assertEqual(a.cache_key(), b.cache_key())

    def test_different_text_different_cache_key(self) -> None:
        a = TranslationRequest(text="hello", source="en", target="hi")
        b = TranslationRequest(text="goodbye", source="en", target="hi")
        self.assertNotEqual(a.cache_key(), b.cache_key())

    def test_different_pair_different_cache_key(self) -> None:
        a = TranslationRequest(text="hello", source="en", target="hi")
        b = TranslationRequest(text="hello", source="en", target="ta")
        self.assertNotEqual(a.cache_key(), b.cache_key())


if __name__ == "__main__":
    unittest.main()