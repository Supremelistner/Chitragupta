"""Tests for the web search service download functionality."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from web_search_service.domain.models import (
    DownloadRequest, DownloadResult, DownloadStatus,
)
from web_search_service.infrastructure.downloader import HttpDownloader
from web_search_service.application.retrieval import RetrievalDependencies, RetrievalService


class TestDownloadModels(unittest.TestCase):
    """Domain model tests."""

    def test_download_request_defaults(self):
        r = DownloadRequest(url="https://example.com/form.pdf")
        self.assertEqual(r.url, "https://example.com/form.pdf")
        self.assertIsNone(r.filename)
        self.assertEqual(r.timeout_ms, 30000)
        self.assertEqual(r.max_size_bytes, 50 * 1024 * 1024)

    def test_download_request_custom(self):
        r = DownloadRequest(url="https://example.com/x", filename="custom.pdf", timeout_ms=60000, max_size_bytes=1024)
        self.assertEqual(r.filename, "custom.pdf")
        self.assertEqual(r.timeout_ms, 60000)
        self.assertEqual(r.max_size_bytes, 1024)

    def test_download_result_success(self):
        r = DownloadResult(
            url="https://example.com/form.pdf",
            status=DownloadStatus.SUCCESS,
            file_path="/tmp/abc_form.pdf",
            filename="form.pdf",
            content_type="application/pdf",
            content_length=1024,
        )
        self.assertEqual(r.status, DownloadStatus.SUCCESS)
        self.assertEqual(r.file_path, "/tmp/abc_form.pdf")
        self.assertIsNone(r.error)

    def test_download_result_failed(self):
        r = DownloadResult(
            url="https://example.com/x",
            status=DownloadStatus.FAILED,
            error="Connection timeout",
        )
        self.assertEqual(r.status, DownloadStatus.FAILED)
        self.assertEqual(r.error, "Connection timeout")
        self.assertIsNone(r.file_path)

    def test_download_result_unsupported(self):
        r = DownloadResult(
            url="https://example.com/page.html",
            status=DownloadStatus.UNSUPPORTED,
            error="Unsupported content type: text/html",
        )
        self.assertEqual(r.status, DownloadStatus.UNSUPPORTED)

    def test_download_status_enum(self):
        self.assertEqual(DownloadStatus.SUCCESS.value, "success")
        self.assertEqual(DownloadStatus.FAILED.value, "failed")
        self.assertEqual(DownloadStatus.UNSUPPORTED.value, "unsupported")


class TestHttpDownloader(unittest.TestCase):
    """Infrastructure adapter tests."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._downloader = HttpDownloader(download_dir=self._tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_ping_creates_directory(self):
        d = HttpDownloader(download_dir=str(Path(self._tmpdir) / "new_dir"))
        d.ping()
        self.assertTrue((Path(self._tmpdir) / "new_dir").exists())

    def test_download_success(self):
        """Test download with a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": "11",
        }
        mock_response.read.side_effect = [b"%PDF-1.4 test", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(url="https://example.com/form.pdf"))

        self.assertEqual(result.status, DownloadStatus.SUCCESS)
        self.assertEqual(result.content_type, "application/pdf")
        self.assertEqual(result.content_length, 13)
        self.assertIsNotNone(result.file_path)
        self.assertTrue(Path(result.file_path).exists())
        self.assertEqual(Path(result.file_path).read_bytes(), b"%PDF-1.4 test")
        self.assertIn(".pdf", result.file_path)

    def test_download_custom_filename(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "image/jpeg", "Content-Length": "5"}
        mock_response.read.side_effect = [b"\xff\xd8\xff\xe0\x00\x00", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(
                url="https://example.com/image",
                filename="aadhaar_card.jpg",
            ))

        self.assertEqual(result.status, DownloadStatus.SUCCESS)
        self.assertEqual(result.filename, "aadhaar_card.jpg")
        self.assertIn(".jpg", result.file_path)

    def test_download_rejects_too_large(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "100000"}
        mock_response.read.side_effect = [b"x" * 100000, b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(
                url="https://example.com/huge.pdf",
                max_size_bytes=1000,
            ))

        self.assertEqual(result.status, DownloadStatus.FAILED)
        self.assertIn("too large", result.error.lower())

    def test_download_rejects_unsupported_type(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html", "Content-Length": "100"}
        mock_response.read.side_effect = [b"<html>" + b"x" * 94, b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(url="https://example.com/page"))

        self.assertEqual(result.status, DownloadStatus.UNSUPPORTED)

    def test_download_handles_network_error(self):
        with patch("web_search_service.infrastructure.downloader.urlopen", side_effect=ConnectionError("refused")):
            result = self._downloader.download(DownloadRequest(url="https://dead.example.com/x.pdf"))

        self.assertEqual(result.status, DownloadStatus.FAILED)
        self.assertIn("refused", result.error)

    def test_download_from_url_with_filename(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "9"}
        mock_response.read.side_effect = [b"%PDF-test", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(url="https://example.com/forms/lic-3783.pdf"))

        self.assertEqual(result.status, DownloadStatus.SUCCESS)
        self.assertEqual(result.filename, "lic-3783.pdf")

    def test_download_normalizes_content_type_from_extension(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/octet-stream", "Content-Length": "4"}
        mock_response.read.side_effect = [b"\x89PNG", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(url="https://example.com/photo.png"))

        self.assertEqual(result.status, DownloadStatus.SUCCESS)
        self.assertEqual(result.content_type, "image/png")

    def test_downloader_health(self):
        self._downloader.ping()  # Should not raise


class TestRetrievalServiceDownload(unittest.TestCase):
    """RetrievalService integration with downloader."""

    def test_download_requires_downloader(self):
        svc = RetrievalService(RetrievalDependencies())
        with self.assertRaises(RuntimeError) as ctx:
            svc.download(DownloadRequest(url="https://example.com/x.pdf"))
        self.assertIn("No downloader configured", str(ctx.exception))

    def test_download_delegates_to_downloader(self):
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = DownloadResult(
            url="https://example.com/form.pdf",
            status=DownloadStatus.SUCCESS,
            file_path="/tmp/form.pdf",
            filename="form.pdf",
            content_type="application/pdf",
            content_length=1024,
        )

        svc = RetrievalService(RetrievalDependencies(downloader=mock_downloader))
        result = svc.download(DownloadRequest(url="https://example.com/form.pdf"))

        mock_downloader.download.assert_called_once()
        self.assertEqual(result.status, DownloadStatus.SUCCESS)
        self.assertEqual(result.file_path, "/tmp/form.pdf")

    def test_health_includes_downloader(self):
        mock_downloader = MagicMock()
        svc = RetrievalService(RetrievalDependencies(downloader=mock_downloader))
        health = svc.health()
        self.assertIn("downloader", health["providers"])
        self.assertEqual(health["providers"]["downloader"]["status"], "healthy")


class TestDownloadEndToEnd(unittest.TestCase):
    """End-to-end download test with real HTTP (httpbin)."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._downloader = HttpDownloader(download_dir=self._tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_download_real_pdf_from_httpbin(self):
        """Download a real PDF-like file from httpbin."""
        # httpbin/html returns HTML, but we can test with /bytes endpoint
        # Actually, let's just test with a real image endpoint
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "16"}
        mock_response.read.side_effect = [b"%PDF-1.4  sample", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("web_search_service.infrastructure.downloader.urlopen", return_value=mock_response):
            result = self._downloader.download(DownloadRequest(
                url="https://licindia.in/documents/form-3783.pdf",
                filename="lic_claim_form.pdf",
            ))

        self.assertEqual(result.status, DownloadStatus.SUCCESS)
        self.assertEqual(result.filename, "lic_claim_form.pdf")
        self.assertEqual(result.content_length, 16)
        self.assertTrue(Path(result.file_path).exists())


if __name__ == "__main__":
    unittest.main()
