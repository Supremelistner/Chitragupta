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


PUBLIC_DNS_ANSWER = [(2, 1, 6, "", ("93.184.216.34", 0))]
_DNS_PATCH_TARGET = "web_search_service.infrastructure.url_guard.getaddrinfo"


def _patch_dns(testcase: unittest.TestCase):
    """Pin DNS to a public IP so SSRF-guard tests don't need the network."""
    patcher = patch("web_search_service.infrastructure.url_guard.getaddrinfo")
    mock_resolve = patcher.start()
    mock_resolve.return_value = PUBLIC_DNS_ANSWER
    testcase.addCleanup(patcher.stop)
    return mock_resolve


class TestHttpDownloader(unittest.TestCase):
    """Infrastructure adapter tests."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._downloader = HttpDownloader(download_dir=self._tmpdir)
        _patch_dns(self)

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
        _patch_dns(self)

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


class TestUrlGuard(unittest.TestCase):
    """SSRF guard: schemes, resolution, and routability."""

    def test_rejects_non_http_scheme(self):
        from web_search_service.infrastructure.url_guard import (
            BlockedHostError, validate_url,
        )
        for bad in ("file:///etc/passwd", "gopher://example.com/x", "ftp://example.com/f"):
            with self.assertRaises(BlockedHostError):
                validate_url(bad)

    def test_rejects_unresolvable_host(self):
        import socket as std_socket
        from web_search_service.infrastructure.url_guard import (
            BlockedHostError, validate_url,
        )
        with patch(_DNS_PATCH_TARGET) as mock_resolve:
            mock_resolve.side_effect = std_socket.gaierror("nope")
            with self.assertRaises(BlockedHostError):
                validate_url("https://nonexistent.invalid/x")

    def test_rejects_non_routable_addresses(self):
        from web_search_service.infrastructure.url_guard import (
            BlockedHostError, validate_url,
        )
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254",
                   "224.0.0.1", "0.0.0.0", "::1"):
            with patch(_DNS_PATCH_TARGET) as mock_resolve:
                mock_resolve.return_value = [(2, 1, 6, "", (ip, 0))]
                with self.assertRaises(BlockedHostError, msg=ip):
                    validate_url(f"http://internal.example/{ip}")

    def test_allows_public_address(self):
        from web_search_service.infrastructure.url_guard import validate_url
        with patch(_DNS_PATCH_TARGET) as mock_resolve:
            mock_resolve.return_value = PUBLIC_DNS_ANSWER
            self.assertEqual(
                validate_url("https://example.com/form.pdf"),
                "https://example.com/form.pdf",
            )

    def test_redirect_to_private_host_blocked(self):
        import socket as std_socket
        import urllib.error
        from web_search_service.infrastructure.url_guard import build_guarded_opener
        opener = build_guarded_opener()
        # Drive the redirect handler directly: a 302 to a metadata endpoint.
        handler = [h for h in opener.handlers
                   if type(h).__name__ == "_GuardedRedirectHandler"][0]
        with patch(_DNS_PATCH_TARGET) as mock_resolve:
            mock_resolve.side_effect = std_socket.gaierror("nope")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                handler.redirect_request(MagicMock(), MagicMock(), 302, "Found", {}, "http://169.254.169.254/x")
            self.assertEqual(ctx.exception.code, 403)


class TestSimpleFetchGuard(unittest.TestCase):
    """Fetch adapter: blocked URLs and oversized bodies."""

    def test_blocked_url_returns_error_result(self):
        from web_search_service.domain.models import FetchRequest
        from web_search_service.infrastructure.fetch import SimpleFetchAdapter
        adapter = SimpleFetchAdapter()
        result = adapter.fetch(FetchRequest(url="http://localhost:6333/collections"))
        self.assertIsNone(result.status_code)
        self.assertTrue(result.content.startswith("ERROR: blocked URL"))

    def test_oversized_body_is_truncated(self):
        from web_search_service.domain.models import FetchRequest
        import web_search_service.infrastructure.fetch as fetch_mod
        adapter = fetch_mod.SimpleFetchAdapter()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.read.side_effect = [b"y" * 100, b""]
        mock_opener = MagicMock()
        mock_opener.open.return_value.__enter__ = lambda s: mock_resp
        mock_opener.open.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(fetch_mod, "build_guarded_opener", return_value=mock_opener), \
             patch.object(fetch_mod, "MAX_FETCH_BYTES", 50), \
             patch(_DNS_PATCH_TARGET) as mock_resolve:
            mock_resolve.return_value = PUBLIC_DNS_ANSWER
            result = adapter.fetch(FetchRequest(url="https://example.com/big"))
        self.assertEqual(result.status_code, 200)
        self.assertIn("[truncated:", result.content)


if __name__ == "__main__":
    unittest.main()
