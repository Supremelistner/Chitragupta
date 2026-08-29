"""Tests for the local-user onboarding flow."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator_service import onboarding


class OnboardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "profile.json"

    def test_load_returns_none_when_no_profile(self) -> None:
        self.assertIsNone(onboarding.load_profile(self.path))

    def test_save_and_load_round_trip(self) -> None:
        profile = onboarding.save_profile("Manish", path=self.path)
        self.assertEqual(profile.display_name, "Manish")
        loaded = onboarding.load_profile(self.path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.display_name, "Manish")
        self.assertEqual(loaded.profile_path, self.path)

    def test_save_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            onboarding.save_profile("   ", path=self.path)

    def test_load_ignores_corrupt_file(self) -> None:
        self.path.write_text("not json", encoding="utf-8")
        self.assertIsNone(onboarding.load_profile(self.path))

    def test_load_ignores_missing_display_name(self) -> None:
        self.path.write_text(json.dumps({"created_at": "x"}), encoding="utf-8")
        self.assertIsNone(onboarding.load_profile(self.path))

    def test_save_creates_parent_directory(self) -> None:
        nested = Path(self.tempdir.name) / "deep" / "subdir" / "profile.json"
        onboarding.save_profile("A", path=nested)
        self.assertTrue(nested.exists())

    def test_profile_path_defaults_to_home(self) -> None:
        self.assertEqual(
            onboarding.DEFAULT_PROFILE_PATH,
            Path.home() / ".chitragupta" / "profile.json",
        )

    def test_to_dict_contains_human_fields(self) -> None:
        p = onboarding.save_profile("Manish", path=self.path)
        d = p.to_dict()
        self.assertEqual(d["display_name"], "Manish")
        self.assertIn("created_at", d)
        self.assertIn("profile_path", d)


if __name__ == "__main__":
    unittest.main()
