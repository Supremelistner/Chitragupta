"""First-launch onboarding for Chitragupta.

A local, single-user deployment needs to know who the user is before it
starts handling documents. This module:

* loads a profile from ``~/.chitragupta/profile.json`` if it exists
* on first launch, prompts (CLI) or returns 428 (UI) so the user can submit
  their name
* persists the profile so subsequent runs are silent
* exposes :func:`require_profile` for the service entry-point to refuse to
  start without one when ``CHITRAGUPTA_REQUIRE_PROFILE=1``

The profile is intentionally minimal: just a display name. Future fields
(preferred language, locale, etc.) can be added without breaking the file
format.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.onboarding")


DEFAULT_PROFILE_PATH = Path.home() / ".chitragupta" / "profile.json"
REQUIRE_PROFILE_ENV = "CHITRAGUPTA_REQUIRE_PROFILE"
USER_NAME_ENV = "CHITRAGUPTA_USER_NAME"


@dataclass(frozen=True, slots=True)
class Profile:
    display_name: str
    created_at: str
    profile_path: Path

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["profile_path"] = str(self.profile_path)
        return d


def _resolve_path(path: str | os.PathLike[str] | None) -> Path:
    if path is None:
        return DEFAULT_PROFILE_PATH
    p = Path(path).expanduser()
    return p


def load_profile(path: str | os.PathLike[str] | None = None) -> Profile | None:
    """Load an existing profile, or return ``None`` if none exists.

    The file is treated as the single source of truth. If it does not exist
    or cannot be read, :func:`load_profile` returns ``None`` and the caller
    should trigger onboarding.
    """
    p = _resolve_path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        display_name = (data.get("display_name") or "").strip()
        if not display_name:
            logger.warning("Profile at %s is missing display_name", p)
            return None
        created_at = data.get("created_at") or datetime.now(timezone.utc).isoformat()
        return Profile(
            display_name=display_name,
            created_at=created_at,
            profile_path=p,
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read profile at %s: %s", p, exc)
        return None


def save_profile(
    display_name: str,
    *,
    path: str | os.PathLike[str] | None = None,
) -> Profile:
    """Persist a new profile and return the loaded :class:`Profile`."""
    cleaned = (display_name or "").strip()
    if not cleaned:
        raise ValueError("display_name must not be empty")
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    profile = Profile(
        display_name=cleaned,
        created_at=datetime.now(timezone.utc).isoformat(),
        profile_path=p,
    )
    p.write_text(
        json.dumps({"display_name": profile.display_name, "created_at": profile.created_at}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved profile for '%s' at %s", cleaned, p)
    return profile


def prompt_for_name_cli() -> str:
    """Interactively ask the user for their display name (CLI mode).

    Falls back to the ``CHITRAGUPTA_USER_NAME`` env var if stdin is not a
    TTY (e.g. when running under ``nohup`` or in a container with no
    interactive shell).
    """
    env_name = os.environ.get(USER_NAME_ENV, "").strip()
    if env_name:
        return env_name
    try:
        is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        is_tty = False
    if not is_tty:
        raise RuntimeError(
            f"Cannot prompt interactively: stdin/stdout is not a TTY. "
            f"Set {USER_NAME_ENV} or call POST /api/profile first."
        )
    print("Welcome to Chitragupta — your personal document agent.")
    while True:
        name = input("What's your name? ").strip()
        if name:
            return name
        print("Please enter a name (it can be anything you'll recognize).")


def require_profile(
    path: str | os.PathLike[str] | None = None,
) -> Profile:
    """Return an existing profile, or create one by prompting.

    Used by the service entry-point when ``CHITRAGUPTA_REQUIRE_PROFILE=1``.
    Raises :class:`RuntimeError` if the user cannot be prompted.
    """
    existing = load_profile(path)
    if existing is not None:
        return existing
    name = prompt_for_name_cli()
    return save_profile(name, path=path)


def is_required() -> bool:
    """Whether the service should refuse to start without a profile."""
    val = os.environ.get(REQUIRE_PROFILE_ENV, "").strip().lower()
    return val in {"1", "true", "yes", "on"}
