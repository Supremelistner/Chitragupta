"""Tests for the persona + post-processor.

The persona prompt is generated from `policy.md` + `failures.md`
(see project root).  These tests check the *behavior* of the prompt
generator and the post-processor, not the exact wording of the prompt
(which can be re-derived when either governance document changes).
"""
from __future__ import annotations

import re
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

    def test_salvages_message_key_from_json_blob(self) -> None:
        # The model wrapped its answer in JSON with a "message" key. We
        # should surface the prose, not ask the user to rephrase.
        reply = '{"message": "I found your Aadhaar card.", "confidence": 0.9}'
        out = humanize(reply)
        self.assertEqual(out, "I found your Aadhaar card.")
        self.assertNotIn("{", out)
        self.assertNotIn("confidence", out)

    def test_salvages_answer_key_from_json_blob(self) -> None:
        reply = '{"answer": "Your PAN card expires next month."}'
        out = humanize(reply)
        self.assertEqual(out, "Your PAN card expires next month.")

    def test_salvages_single_string_value(self) -> None:
        # No known prose key, but exactly one string value -> use it.
        reply = '{"foo": "Here is your document summary."}'
        out = humanize(reply)
        self.assertEqual(out, "Here is your document summary.")

    def test_salvages_list_of_strings(self) -> None:
        reply = '["First point.", "Second point."]'
        out = humanize(reply)
        self.assertIn("First point.", out)
        self.assertIn("Second point.", out)
        self.assertNotIn("[", out)

    def test_unrecoverable_json_falls_back_to_one_liner(self) -> None:
        # Numeric-only blob: nothing readable to salvage.
        reply = '{"code": 200, "count": 3}'
        out = humanize(reply)
        self.assertIn("plain language", out)

    def test_nested_prose_key_is_salvaged(self) -> None:
        reply = '{"data": {"reply": "All done saving your file."}}'
        out = humanize(reply)
        self.assertEqual(out, "All done saving your file.")

    def test_raw_provider_error_becomes_friendly_line(self) -> None:
        # The LLM adapters return content="Error: <status> ..." when every
        # provider is down/rate-limited. It must never reach the user raw.
        reply = (
            "Error: 503 UNAVAILABLE. {'error': {'code': 503, 'message': "
            "'This model is currently experiencing high demand.', "
            "'status': 'UNAVAILABLE'}}"
        )
        out = humanize(reply)
        self.assertNotIn("503", out)
        self.assertNotIn("UNAVAILABLE", out)
        self.assertNotIn("{", out)
        self.assertIn("try again", out.lower())

    def test_rate_limit_error_becomes_friendly_line(self) -> None:
        out = humanize("Error: 429 rate limit exceeded")
        self.assertNotIn("429", out)
        self.assertIn("try again", out.lower())

    def test_normal_reply_mentioning_error_word_is_untouched(self) -> None:
        # A legitimate reply that happens to contain "error" must NOT be
        # swallowed — only raw provider-error envelopes are mapped.
        reply = "I could not find an error in your document. It looks complete."
        out = humanize(reply)
        self.assertEqual(out, reply)


class SystemPromptTests(unittest.TestCase):
    """Behavioral checks on the system prompt.

    The prompt body is derived from `policy.md` and `failures.md` in the
    project root.  We assert that the prompt:

    * is non-empty,
    * injects the user's display name,
    * cites every policy clause (\u00a71-11) and every failures case (#1-15),
    * lists the four access_action values the orchestrator emits,
    * names the seven tools the LLM is allowed to call,
    * contains the banned-phrasing list from policy \u00a78,
    * itself contains no URLs (it is fed verbatim to the LLM),
    * uses different display names per call.
    """

    def test_prompt_is_non_empty(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        self.assertGreater(len(prompt), 200,
                           "prompt should be substantive (>=200 chars)")

    def test_user_name_appears(self) -> None:
        prompt = build_system_prompt(user_name="Manish")
        self.assertIn("Manish", prompt)

    def test_user_name_can_be_absent(self) -> None:
        # No user_name -> default is "friend" (voice fallback, policy \u00a711).
        prompt = build_system_prompt()
        self.assertIn("friend", prompt.lower())

    def test_user_name_replacement_is_per_call(self) -> None:
        # Two calls with different names should yield different first lines.
        a = build_system_prompt(user_name="Alice").splitlines()[0]
        b = build_system_prompt(user_name="Bob").splitlines()[0]
        self.assertIn("Alice", a)
        self.assertIn("Bob", b)

    def test_cites_every_policy_clause(self) -> None:
        # \u00a71-11 (regenerated prompt uses \u00a7 markers).
        prompt = build_system_prompt(user_name="Test")
        for n in range(1, 12):
            self.assertIn(f"\u00a7{n}", prompt,
                          msg=f"policy \u00a7{n} not cited in prompt")

    def test_cites_every_failure_case(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        for n in range(1, 16):
            self.assertIn(f"#{n}", prompt,
                          msg=f"failures #{n} not cited in prompt")

    def test_lists_all_four_access_action_values(self) -> None:
        # The orchestrator's tool router returns one of these four values;
        # the prompt must respect them all (policy \u00a74).
        prompt = build_system_prompt(user_name="Test")
        for value in ("ALLOW", "REDACT", "REQUIRE_APPROVAL", "DENY"):
            self.assertIn(value, prompt, msg=f"access_action {value} missing")

    def test_lists_all_seven_tools(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        for tool in (
            "list_documents",
            "search_documents",
            "get_field_value",
            "get_document",
            "get_page",
            "get_evidence",
            "web_search",
        ):
            self.assertIn(tool, prompt, msg=f"tool {tool} missing")

    def test_contains_banned_phrasings(self) -> None:
        # Policy \u00a78: 10 banned phrasings.  We assert a representative
        # subset; if any of these phrases appears in the banned-phrasings
        # section, the prompt has a banned-phrasing list.
        prompt = build_system_prompt(user_name="Test")
        for marker in (
            "I cannot retrieve your X",
            "Contact support",
            "Lacks proper OCR",
        ):
            self.assertIn(marker, prompt,
                          msg=f"banned-phrasing {marker!r} not listed")

    def test_contains_source_of_truth_footer(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        self.assertIn("policy.md", prompt)
        self.assertIn("failures.md", prompt)
        self.assertIn("policy.md wins", prompt)

    def test_prompt_itself_contains_no_urls(self) -> None:
        # The system prompt is sent to the LLM; URLs in it would be junk.
        prompt = build_system_prompt(user_name="Test")
        self.assertNotIn("http://", prompt)
        self.assertNotIn("https://", prompt)

    def test_mentions_owner_type_for_other_documents(self) -> None:
        # Failures #10: the prompt must instruct the model to say "Priya's
        # Aadhaar number" (not "your Aadhaar number") when the document
        # belongs to a relative.
        prompt = build_system_prompt(user_name="Test")
        self.assertIn("owner_type", prompt)
        self.assertIn("relation", prompt)
        self.assertTrue(
            re.search(r"Mother", prompt),
            "prompt should mention the Mother relation example (failures #10)",
        )


class TestElderlyAudienceRules(unittest.TestCase):
    """The ELDERLY AUDIENCE section must survive prompt edits."""

    def test_elderly_section_present(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        self.assertIn("ELDERLY AUDIENCE", prompt)

    def test_no_emoji_rule(self) -> None:
        prompt = build_system_prompt(user_name="Test").lower()
        self.assertIn("emojis", prompt)

    def test_speakable_numbers_rule(self) -> None:
        prompt = build_system_prompt(user_name="Test").lower()
        self.assertIn("digit", prompt)

    def test_one_ask_per_reply_rule(self) -> None:
        prompt = build_system_prompt(user_name="Test").lower()
        self.assertIn("one question or one action", prompt)

    def test_user_name_personalizes_prompt(self) -> None:
        prompt = build_system_prompt(user_name="Sharma")
        self.assertIn("Sharma", prompt)

    def test_expiring_tool_in_toolset(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        self.assertIn("list_expiring_documents", prompt)

    def test_scam_safety_rule_present(self) -> None:
        prompt = build_system_prompt(user_name="Test")
        self.assertIn("SCAM SAFETY", prompt)
        self.assertIn("OTP", prompt)

    def test_family_upload_guidance_present(self) -> None:
        prompt = build_system_prompt(user_name="Test").lower()
        self.assertIn("mother", prompt)


if __name__ == "__main__":
    unittest.main()
