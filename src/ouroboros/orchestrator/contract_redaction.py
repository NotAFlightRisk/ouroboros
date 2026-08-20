"""Shared redaction for harness-owned verifier values."""

from __future__ import annotations

from collections.abc import Iterable
import html
import json
import re
import shlex
import unicodedata


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
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_LINE_PREFIX_RE = re.compile(r"(?m)^\s*(?:[EIWF]\s+|[+>~-]\s?)")
_UNSUPPORTED_TERMINAL_CONTROL_RE = re.compile(
    r"(?:\x1b(?:P|_|\^|X)|[\x90\x98\x9e\x9f]).*?(?:\x1b\\|\x9c)",
    re.DOTALL,
)


def _normalized_contract_text(text: str, *, preserve_punctuation: bool) -> str:
    """Normalize routine verifier-output transformations for leak detection."""
    unescaped = text
    for _ in range(12):
        decoded = html.unescape(unescaped)
        if decoded == unescaped:
            break
        unescaped = decoded
    unescaped = unicodedata.normalize("NFKC", unescaped)
    without_ansi = _ANSI_ESCAPE_RE.sub("", unescaped)
    without_ansi = _OSC_ESCAPE_RE.sub("", without_ansi)
    without_ansi = "".join(
        char
        for char in without_ansi
        if (ord(char) >= 32 or char in "\n\r\t") and unicodedata.category(char) != "Cf"
    )
    without_prefixes = _LINE_PREFIX_RE.sub("", without_ansi)
    if preserve_punctuation:
        return "".join(char.casefold() for char in without_prefixes if not char.isspace())
    return "".join(char.casefold() for char in without_prefixes if char.isalnum())


def contains_unsupported_terminal_control(text: str) -> bool:
    """Return whether output carries controls outside normalized CSI/OSC."""
    without_known = _ANSI_ESCAPE_RE.sub("", _OSC_ESCAPE_RE.sub("", text))
    if _UNSUPPORTED_TERMINAL_CONTROL_RE.search(without_known):
        return True
    return any(
        char == "\x1b" or 0x80 <= ord(char) <= 0x9F or (ord(char) < 32 and char not in "\n\r\t")
        for char in without_known
    )


def contains_transformed_hidden_contract_value(
    text: str,
    values: Iterable[str | None],
) -> bool:
    """Return whether a non-exact normalized copy carries a hidden value."""
    for hidden in values:
        if not hidden:
            continue
        remaining = text
        for variant in hidden_contract_variants((hidden,)):
            remaining = remaining.replace(variant, "")
        normalized_remaining = _normalized_contract_text(
            remaining,
            preserve_punctuation=False,
        )
        normalized_hidden = _normalized_contract_text(hidden, preserve_punctuation=False)
        if normalized_hidden:
            if normalized_hidden in normalized_remaining:
                return True
            continue
        compact_remaining = _normalized_contract_text(remaining, preserve_punctuation=True)
        compact_hidden = _normalized_contract_text(hidden, preserve_punctuation=True)
        if compact_hidden and compact_hidden in compact_remaining:
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
