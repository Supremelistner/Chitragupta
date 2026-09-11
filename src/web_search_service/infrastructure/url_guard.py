"""SSRF guard for outbound fetches/downloads.

Both :mod:`fetch` and :mod:`downloader` retrieve LLM- or user-influenced
URLs. Without validation an attacker (or a misled model) can make the
service probe localhost, cloud metadata endpoints (169.254.169.254), or
other link-local targets, and can follow redirects to them.

What this enforces (fail-closed):
  * scheme must be http or https (no file://, gopher://, etc.)
  * hostname must resolve, and EVERY resolved address must be globally
    routable — private, loopback, link-local, multicast, reserved, and
    unspecified ranges are all rejected
  * every redirect hop is re-validated (urllib follows redirects silently
    by default, which would otherwise bypass the initial check)

Known limitation (documented, not fixed here): DNS is resolved
ahead-of-connect, so a DNS-rebinding attack (record flips between check
and connect) is not covered. Full protection would require pinning the
validated address at connect time.
"""
from __future__ import annotations

import ipaddress
import logging
import urllib.error
import urllib.request
from socket import gaierror, getaddrinfo
from urllib.parse import urlparse

logger = logging.getLogger("web_search_service.url_guard")


class BlockedHostError(ValueError):
    """Raised when a URL fails SSRF validation."""


def validate_url(url: str) -> str:
    """Validate ``url`` for server-side fetching. Returns it unchanged.

    Raises :class:`BlockedHostError` for disallowed schemes, unresolvable
    hosts, or any resolved address that is not globally routable.
    DNS failures fail closed: an unresolvable host is indistinguishable
    from a malicious one at this layer.
    """
    try:
        parts = urlparse(url)
    except Exception as exc:
        raise BlockedHostError(f"Malformed URL: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise BlockedHostError(
            f"Blocked URL scheme {parts.scheme!r}: only http/https are fetchable"
        )
    host = parts.hostname or ""
    if not host:
        raise BlockedHostError("URL has no hostname")
    try:
        infos = getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except gaierror as exc:
        raise BlockedHostError(f"Host {host!r} does not resolve: {exc}") from exc
    for info in infos:
        raw_ip = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise BlockedHostError(f"Unparseable address {raw_ip!r} for {host!r}") from exc
        # NOTE: `is_global` alone is insufficient — IPv4 multicast
        # (224.0.0.0/4) reports is_global True, so it needs its own check.
        if ip.is_multicast or not ip.is_global:
            raise BlockedHostError(
                f"Blocked non-routable address {ip} for host {host!r}"
            )
    return url


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that re-validates every hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN202
        try:
            validate_url(newurl)
        except BlockedHostError as exc:
            raise urllib.error.HTTPError(
                newurl, 403, f"Blocked redirect target: {exc}", headers, fp
            ) from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_guarded_opener() -> urllib.request.OpenerDirector:
    """urllib opener with redirect-hop validation."""
    return urllib.request.build_opener(_GuardedRedirectHandler)
