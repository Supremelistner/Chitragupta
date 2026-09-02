"""Tests for the shared field-name resolver."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shared.field_resolver import resolve_field_name


AVAIL = ["aadhaar_number", "name", "date_of_birth", "gender", "address", "pan_number"]


class FieldResolverTests(unittest.TestCase):

    def test_exact(self):
        r = resolve_field_name("aadhaar_number", AVAIL)
        self.assertEqual(r.resolved, "aadhaar_number")
        self.assertEqual(r.match_type, "exact")
        self.assertEqual(r.confidence, 1.0)

    def test_case_insensitive(self):
        r = resolve_field_name("Aadhaar_Number", AVAIL)
        self.assertEqual(r.resolved, "aadhaar_number")
        self.assertEqual(r.match_type, "case_insensitive")
        self.assertEqual(r.confidence, 0.95)

    def test_substring(self):
        r = resolve_field_name("aadhaar", AVAIL)
        self.assertEqual(r.resolved, "aadhaar_number")
        self.assertEqual(r.match_type, "substring")

    def test_fuzzy_typo(self):
        r = resolve_field_name("aadhaar_nmbr", AVAIL)
        self.assertEqual(r.resolved, "aadhaar_number")
        self.assertEqual(r.match_type, "fuzzy")
        self.assertGreaterEqual(r.confidence, 0.75)

    def test_stem_fuzzy_adhar_card(self):
        r = resolve_field_name("adhar card", AVAIL)
        self.assertEqual(r.resolved, "aadhaar_number")
        self.assertEqual(r.match_type, "stem_fuzzy")
        self.assertGreaterEqual(r.confidence, 0.75)

    def test_stem_fuzzy_pan_number(self):
        r = resolve_field_name("PAN Number", AVAIL)
        self.assertEqual(r.resolved, "pan_number")

    def test_too_short_no_match(self):
        r = resolve_field_name("aad", AVAIL)
        self.assertIsNone(r.resolved)
        self.assertEqual(r.match_type, "none")

    def test_unrelated_no_match(self):
        r = resolve_field_name("phone", AVAIL)
        self.assertIsNone(r.resolved)
        self.assertEqual(r.match_type, "none")

    def test_empty_inputs(self):
        self.assertIsNone(resolve_field_name("", AVAIL).resolved)
        self.assertIsNone(resolve_field_name("aadhaar_number", []).resolved)

    def test_does_not_invent(self):
        # Resolver is conservative: a partial-but-noisy match must not
        # silently substitute. e.g. "PAN" alone (without _number)
        # should miss because the stem is too short for a safe match.
        r = resolve_field_name("PAN", AVAIL)
        self.assertIsNone(r.resolved)


if __name__ == "__main__":
    unittest.main()
