"""Fuzzy field-name resolver.

Matches a user-supplied field name (which may have typos, wrong case, or
wrong format) against the canonical field names stored on a document.

Used by both the document service (for actual lookups) and the orchestrator
(for cache-key normalization and approval-message construction), so that
both sides agree on what field was meant.

Match ladder (highest confidence first):

* "exact" (case-sensitive equality)              -- 1.00
* "case_insensitive" (casefold equality)         -- 0.95
* "substring" (case-insensitive contains)        -- 0.85
  only if requested name has length >= 4
* "stem_exact" (after stripping _card/_number/   -- 0.90
  _no/_id suffixes from both sides)
* "stem_substring" (after stem-stripping)         -- 0.80
* "stem_fuzzy" (after stem-stripping)            -- ratio
* "fuzzy" (raw SequenceMatcher ratio >= 0.75)    -- ratio

Anything below 0.75 is a miss; the resolver returns resolved=None so the
caller can show a not_found with available_fields.

The match is intentionally non-throwing: a miss is a normal outcome, not
an error. The denial-with-correction flow in the orchestrator is the
safety net for when the resolver picks the wrong field.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

# Minimum length for substring matching. Shorter requested names are too
# ambiguous to risk auto-matching (e.g. "aad" would falsely match
# "aadhaar_number").
_MIN_SUBSTRING_LENGTH = 4
# Minimum SequenceMatcher ratio to accept a fuzzy match.
_MIN_FUZZY_RATIO = 0.75
# Suffixes to strip from both sides before re-trying substring/fuzzy.
# This lets "adhar card" match "aadhaar_number" (both strip to
# "adhar" / "aadhaar", which then fuzzy-match).
_STEM_SUFFIXES = ("_card", "_number", "_no",
                  " card", " number", " no",
                  " card", " id", "_id")


@dataclass(frozen=True, slots=True)
class ResolverResult:
    """Outcome of a single field-name lookup.

    Attributes:
        requested: The name the caller asked for, verbatim.
        resolved: The canonical name we matched against. None when no
            candidate exceeded the thresholds.
        match_type: One of "exact", "case_insensitive", "substring", "stem_exact", "stem_substring", "stem_fuzzy", "fuzzy", "none".
        confidence: 0.0 - 1.0; higher = more confident. 0.0 when no match.
    """
    requested: str
    resolved: str | None
    match_type: str
    confidence: float


def resolve_field_name(requested: str, available: list[str]) -> ResolverResult:
    """Resolve a caller-supplied field name against a list of canonical names.

    The match ladder runs in order; the first hit wins. This is deliberate:
    we trust exact matches over fuzzy ones, and substring matches over
    fuzzy ones, so that a perfectly-typed field name never gets
    accidentally rerouted.
    """
    if not requested or not available:
        return ResolverResult(requested=requested or "", resolved=None, match_type="none", confidence=0.0)

    # 1. Exact (case-sensitive).
    if requested in available:
        return ResolverResult(requested=requested, resolved=requested, match_type="exact", confidence=1.0)

    # 2. Case-insensitive.
    req_cf = requested.casefold()
    for name in available:
        if name.casefold() == req_cf:
            return ResolverResult(requested=requested, resolved=name, match_type="case_insensitive", confidence=0.95)

    # 3. Substring (only if requested name is long enough to be safe).
    if len(requested) >= _MIN_SUBSTRING_LENGTH:
        for name in available:
            if req_cf in name.casefold() or name.casefold() in req_cf:
                return ResolverResult(requested=requested, resolved=name, match_type="substring", confidence=0.85)

    # 4. Stem stripping. Strip common suffixes (_card, _number, _no, etc.)
    #    from BOTH sides and retry the substring / fuzzy ladders. This
    #    handles "adhar card" -> "aadhaar_number" (both -> "aadhaar" / "aadhaar").
    def _strip_stem(s: str) -> str:
        out = s.strip()
        changed = True
        while changed:
            changed = False
            for suf in _STEM_SUFFIXES:
                if out.casefold().endswith(suf):
                    out = out[: -len(suf)]
                    changed = True
                    break
        return out.strip()
    req_stem = _strip_stem(requested)
    if req_stem and req_stem.casefold() != req_cf and len(req_stem) >= _MIN_SUBSTRING_LENGTH:
        for name in available:
            name_stem = _strip_stem(name)
            if name_stem.casefold() == req_stem.casefold():
                return ResolverResult(requested=requested, resolved=name, match_type="stem_exact", confidence=0.9)
        # Stem + substring
        for name in available:
            name_stem = _strip_stem(name)
            name_stem_cf = name_stem.casefold()
            if req_stem.casefold() in name_stem_cf or name_stem_cf in req_stem.casefold():
                return ResolverResult(requested=requested, resolved=name, match_type="stem_substring", confidence=0.8)
        # Stem + fuzzy
        best_name_stem: str | None = None
        best_ratio_stem = 0.0
        for name in available:
            name_stem = _strip_stem(name)
            ratio = SequenceMatcher(None, req_stem.casefold(), name_stem.casefold()).ratio()
            if ratio > best_ratio_stem:
                best_ratio_stem = ratio
                best_name_stem = name
        if best_name_stem is not None and best_ratio_stem >= _MIN_FUZZY_RATIO:
            return ResolverResult(requested=requested, resolved=best_name_stem, match_type="stem_fuzzy", confidence=best_ratio_stem)

    # 5. Fuzzy on the original (unstemmed) names. Last resort before
    #    declaring a miss.
    best_name: str | None = None
    best_ratio = 0.0
    for name in available:
        ratio = SequenceMatcher(None, req_cf, name.casefold()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name
    if best_name is not None and best_ratio >= _MIN_FUZZY_RATIO:
        return ResolverResult(requested=requested, resolved=best_name, match_type="fuzzy", confidence=best_ratio)

    return ResolverResult(requested=requested, resolved=None, match_type="none", confidence=0.0)
