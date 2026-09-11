"""Unit tests for the v1 -> v2 Qdrant payload migration transform.

Covers ``scripts/reencrypt_qdrant_payloads._transform_payload`` as a pure
function (no Qdrant needed): already-v2 and unencrypted points pass
through untouched, legacy points are rewritten without ``_enc_desc``,
and corrupt points are skipped rather than mangled.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from document_mgmt_service.infrastructure import encryption as enc


def _load_transform():
    path = Path(__file__).resolve().parent.parent / "scripts" / "reencrypt_qdrant_payloads.py"
    spec = importlib.util.spec_from_file_location("_reencrypt_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._transform_payload


MASTER = "test-master-key"
DATE = "2026-09-11T00:00:00+00:00"


def _legacy_point(**overrides):
    """Build a v1-style payload: enc: fields + plaintext _enc_desc hint."""
    from cryptography.fernet import Fernet
    desc = "Aadhaar card for Test Person"
    key = enc.derive_encryption_key(desc, DATE, MASTER)
    fernet = Fernet(key)
    import json as _json

    def _e(value):
        return "enc:" + fernet.encrypt(_json.dumps(value).encode()).decode("ascii")

    payload = {
        "chunk_id": "c1", "document_id": "doc-9", "version": 3,
        "text": _e("sensitive text"), "description": _e(desc),
        "metadata": _e({"k": "v"}), "privacy": "SENSITIVE",
        "original_filename": "aadhaar.png",
        "_enc_desc": desc, "_enc_date": DATE,
    }
    payload.update(overrides)
    return payload


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transform = staticmethod(_load_transform())

    def test_v2_point_is_skipped(self):
        payload = _legacy_point()
        del payload["_enc_desc"]
        payload["_enc_v"] = 2
        self.assertIsNone(self.transform(payload, MASTER))

    def test_unencrypted_point_is_skipped(self):
        self.assertIsNone(self.transform({"chunk_id": "c", "text": "plain"}, MASTER))

    def test_legacy_point_rewritten_without_hint(self):
        out = self.transform(_legacy_point(), MASTER)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertNotIn("_enc_desc", out)
        self.assertEqual(out["_enc_v"], 2)
        # Round-trips through the v2 decrypt path with identical values.
        back = enc.decrypt_payload_with_date(dict(out), MASTER)
        self.assertEqual(back["text"], "sensitive text")
        self.assertEqual(back["description"], "Aadhaar card for Test Person")
        self.assertEqual(back["metadata"], {"k": "v"})

    def test_corrupt_legacy_point_is_skipped(self):
        payload = _legacy_point(text="enc:corrupted!!")
        self.assertIsNone(self.transform(payload, MASTER))

    def test_missing_document_id_is_skipped(self):
        payload = _legacy_point()
        del payload["document_id"]
        self.assertIsNone(self.transform(payload, MASTER))


if __name__ == "__main__":
    unittest.main()
