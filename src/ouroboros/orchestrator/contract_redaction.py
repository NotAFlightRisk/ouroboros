"""Shared redaction for harness-owned verifier values."""

from __future__ import annotations

from collections.abc import Iterable
import html
import json
import re
import shlex


def hidden_contract_variants(values: Iterable[str | None]) -> tuple[str, ...]:
    """Return longest-first raw, quoted, and escaped hidden values."""

    variants: set[str] = set()
    for hidden in values:
        if not hidden:
            continue
        json_quoted = json.dumps(hidden, ensure_ascii=False)
        variants.update(
            {
                hidden,
                repr(hidden),
                shlex.quote(hidden),
                json_quoted,
                json_quoted[1:-1],
            }
        )
    return tuple(
        sorted((value for value in variants if value), key=lambda value: (-len(value), value))
    )


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LINE_PREFIX_RE = re.compile(r"(?m)^\s*(?:[EIWF]\s+|[+>~-]\s?)")


def _normalized_contract_text(text: str) -> str:
    """Normalize routine verifier-output transformations for leak detection."""
    unescaped = html.unescape(text)
    without_ansi = _ANSI_ESCAPE_RE.sub("", unescaped)
    without_prefixes = _LINE_PREFIX_RE.sub("", without_ansi)
    return "".join(char.casefold() for char in without_prefixes if char.isalnum())


def contains_transformed_hidden_contract_value(
    text: str,
    values: Iterable[str | None],
) -> bool:
    """Return whether normalized output still carries a readable hidden value."""
    normalized_text = _normalized_contract_text(text)
    for hidden in values:
        if not hidden:
            continue
        normalized_hidden = _normalized_contract_text(hidden)
        if normalized_hidden and normalized_hidden in normalized_text:
            return True
    return False


def redact_hidden_contract_values(
    text: str,
    values: Iterable[str | None],
    *,
    replacement: str = "[REDACTED CONTRACT VALUE]",
) -> str:
    """Remove every supported encoding of harness-owned values from text."""

    redacted = text
    for hidden in hidden_contract_variants(values):
        redacted = redacted.replace(hidden, replacement)
    return redacted
