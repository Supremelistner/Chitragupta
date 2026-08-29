"""Tests for the persona + post-processor."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator_service.persona import build_system_prompt, humanize


class HumanizeTests(unittest.TestCase):
    def test_strips_markdown_fences(self) -> None:
        reply = "Here you go:\n```json\n{\"a\": 1}\n```\nDone."
        out = humanize(reply)
        self.assertNotIn("```", out)
        self.assertIn("a", out)

    def test_strips_urls(self) -> None:
        reply = "Check this out: http://example.com/x and https://foo.bar/baz"
        out = humanize(reply)
        self.assertNotIn("http://", out)
        self.assertNotIn("https://", out)

    def test_pure_json_replaced_with_one_liner(self) -> None:
        reply = '{"results": [{"x": 1}, {"x": 2}]}'
        out = humanize(reply)
        self.assertNotIn("results", out)
        self.assertIn("plain language", out)

    def test_passes_through_normal_text(self) -> None:
        reply = "I found your Aadhaar card. Source: Aadhaar card (self)."
        out = humanize(reply)
        self.assertEqual(out, reply)

    def test_collapses_blank_lines(self) -> None:
        reply = "Line 1\n\n\n\n\nLine 2"
        out = humanize(reply)
        self.assertIn("Line 1", out)
        self.assertIn("Line 2", out)
        self.assertNotIn("\n\n\n", out)

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(humanize(""), "")
        self.assertEqual(humanize(None or ""), "")


class SystemPromptTests(unittest.TestCase):
    def test_default_prompt_mentions_friend(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("friend", prompt.lower())

    def test_user_name_appears(self) -> None:
        prompt = build_system_prompt(user_name="Manish")
        self.assertIn("Manish", prompt)

    def test_prompt_includes_voice_rules(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        for marker in (
            "warm, direct",
            "raw JSON",
            "URLs in chat",
            "list_documents",
            "web_search",
            "get_field_value",
            "ALLOW, REDACT, REQUIRE_APPROVAL, DENY",
        ):
            self.assertIn(marker, prompt, msg=f"missing: {marker}")

    def test_prompt_has_no_urls(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        # The system prompt itself must not contain URLs.
        self.assertNotIn("http://", prompt)
        self.assertNotIn("https://", prompt)


if __name__ == "__main__":
    unittest.main()
