"""Unit tests for ``_split_fields`` accepting both prompt shapes.

The new METADATA_EXTRACTION prompt declares a *flat* fields shape:
    {"aadhaar_number": "1234 5678 9012", "name": "..."}

The legacy prompt declared a *nested* shape:
    {"aadhaar_number": {"value": "1234 5678 9012", "confidence": 0.9}}

``_split_fields`` must accept either, must normalize to ``{name: value}``,
and must skip nulls / empties.
"""
from __future__ import annotations

import unittest

from document_mgmt_service.application.ingestion import _split_fields


class SplitFieldsPureShapeTests(unittest.TestCase):
    """Pure shape behavior — no I/O, no fixtures."""

    def test_returns_empty_for_none(self) -> None:
        self.assertEqual(_split_fields(None), ({}, []))

    def test_returns_empty_for_error_marker(self) -> None:
        self.assertEqual(_split_fields({"error": True}), ({}, []))
        self.assertEqual(_split_fields({"error": "bad image"}), ({}, []))

    def test_returns_empty_when_fields_missing(self) -> None:
        self.assertEqual(_split_fields({}), ({}, []))

    def test_returns_empty_when_fields_not_a_dict(self) -> None:
        self.assertEqual(_split_fields({"fields": "oops"}), ({}, []))
        self.assertEqual(_split_fields({"fields": None}), ({}, []))
        self.assertEqual(_split_fields({"fields": [1, 2, 3]}), ({}, []))

    def test_flat_shape_passes_through(self) -> None:
        classification = {
            "fields": {
                "aadhaar_number": "1234 5678 9012",
                "name": "Manish",
            }
        }
        extracted, names = _split_fields(classification)
        self.assertEqual(
            extracted,
            {"aadhaar_number": "1234 5678 9012", "name": "Manish"},
        )
        self.assertEqual(set(names), {"aadhaar_number", "name"})

    def test_nested_shape_unwraps_value(self) -> None:
        classification = {
            "fields": {
                "aadhaar_number": {"value": "1234 5678 9012", "confidence": 0.95},
                "name": {"value": "Manish", "confidence": 0.9},
            }
        }
        extracted, _names = _split_fields(classification)
        self.assertEqual(
            extracted,
            {"aadhaar_number": "1234 5678 9012", "name": "Manish"},
        )

    def test_mixed_flat_and_nested_in_same_payload(self) -> None:
        """The realistic case: model flattens most fields but leaves one nested."""
        classification = {
            "fields": {
                "aadhaar_number": "1234 5678 9012",  # flat
                "name": {"value": "Manish", "confidence": 0.9},  # nested
                "date_of_birth": "1990-01-01",  # flat
                "address": {"value": "12 MG Road", "confidence": 0.7},  # nested
            }
        }
        extracted, names = _split_fields(classification)
        self.assertEqual(
            extracted,
            {
                "aadhaar_number": "1234 5678 9012",
                "name": "Manish",
                "date_of_birth": "1990-01-01",
                "address": "12 MG Road",
            },
        )
        self.assertEqual(
            set(names),
            {"aadhaar_number", "name", "date_of_birth", "address"},
        )

    def test_skips_null_and_empty_values(self) -> None:
        classification = {
            "fields": {
                "aadhaar_number": None,  # dropped
                "name": "",  # dropped
                "date_of_birth": "1990-01-01",
                "address": {"value": None, "confidence": 0.5},  # dropped
            }
        }
        extracted, names = _split_fields(classification)
        self.assertEqual(extracted, {"date_of_birth": "1990-01-01"})
        self.assertEqual(names, ["date_of_birth"])

    def test_non_dict_non_scalar_entry_is_skipped_safely(self) -> None:
        """A list value (not a dict, not a string) is treated as falsy-but-valid;
        the field is still extracted as the raw value. This locks the contract
        that ``_split_fields`` doesn't crash on unexpected JSON shapes."""
        classification = {"fields": {"some_list_field": [1, 2, 3]}}
        extracted, _ = _split_fields(classification)
        self.assertEqual(extracted, {"some_list_field": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()