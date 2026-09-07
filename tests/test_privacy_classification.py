"""Round-trip test: model-provider privacy prompts vs. _parse_privacy().

Each provider's _PRIVACY_CLASSIFICATION_PROMPT declares the set of strings the
model is allowed to emit for ``classification``. The downstream parser
(``_parse_privacy`` in document_mgmt_service.application.ingestion) must
recognize every string the prompt allows, AND must NOT silently downgrade
unrecognized values to a default.

Before this fix, the prompts emitted ``RESTRICTED`` for sensitive IDs but the
parser had no entry for it, so Aadhaar cards etc. were silently stored as
OPEN_NOT_PUBLIC — bypassing redaction. This test locks the contract.
"""
from __future__ import annotations

import importlib
import re
import unittest

from document_mgmt_service.application.ingestion import _parse_privacy
from document_mgmt_service.domain.models import DocumentPrivacyClassification


def _extract_allowed_classifications(prompt_text: str) -> set[str]:
    """Pull the four quoted classification values out of the prompt schema line.

    The prompts declare the allowed values inside a JSON example like:
        "classification": "OPEN" | "OPEN_NOT_PUBLIC" | "PRIVATE" | "SENSITIVE"
    """
    match = re.search(
        r'"classification"\s*:\s*"([^"]+)"\s*\|\s*"([^"]+)"(?:\s*\|\s*"([^"]+)")?(?:\s*\|\s*"([^"]+)")?',
        prompt_text,
    )
    if not match:
        return set()
    return {g for g in match.groups() if g}


class ProviderPromptContractTests(unittest.TestCase):
    """Every provider's privacy prompt vocabulary must be a subset of the enum."""

    PROVIDERS = [
        "model_service.infrastructure.huggingface",
        "model_service.infrastructure.groq_provider",
        "model_service.infrastructure.gemini_provider",
    ]

    def test_all_providers_emit_enum_values(self) -> None:
        allowed_enum = {e.name for e in DocumentPrivacyClassification}
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                mod = importlib.import_module(provider)
                prompt = mod._PRIVACY_CLASSIFICATION_PROMPT
                declared = _extract_allowed_classifications(prompt)
                self.assertTrue(
                    declared,
                    msg=f"{provider}: could not parse classification values from prompt",
                )
                unknown = declared - allowed_enum
                self.assertEqual(
                    unknown,
                    set(),
                    msg=(
                        f"{provider}: prompt declares values not in the "
                        f"DocumentPrivacyClassification enum: {sorted(unknown)}"
                    ),
                )

    def test_parser_handles_every_value_every_provider_emits(self) -> None:
        """For every classification value each prompt allows, _parse_privacy
        must NOT return the default OPEN_NOT_PUBLIC fallback."""
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                mod = importlib.import_module(provider)
                declared = _extract_allowed_classifications(mod._PRIVACY_CLASSIFICATION_PROMPT)
                for value in declared:
                    parsed = _parse_privacy(value)
                    self.assertEqual(
                        parsed.name,
                        value,
                        msg=(
                            f"{provider}: parser mapped {value!r} -> {parsed.name!r} "
                            f"(would silently misclassify)"
                        ),
                    )

    def test_parser_default_is_safe(self) -> None:
        """Unknown strings should default to OPEN_NOT_PUBLIC (conservative)."""
        self.assertEqual(
            _parse_privacy("RESTRICTED"),  # legacy stray value, would crash old code
            DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
        )
        self.assertEqual(
            _parse_privacy("garbage"),
            DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
        )
        self.assertEqual(
            _parse_privacy(None),
            DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
        )
        self.assertEqual(
            _parse_privacy(""),
            DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
        )


if __name__ == "__main__":
    unittest.main()