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


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;:]*m")
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ESCAPED_TERMINAL_CONTROL_RE = re.compile(
    r"(?ix)(?:"
    r"\\(?:x(?:1b|08|9[08bef])|u(?:001b|0008|009[08bef])|U(?:0000001b|00000008|0000009[08bef])|0(?:33|10)|e|b)"
    r"|\\?\^\["
    r")"
)
_ESCAPED_WHITESPACE_RE = re.compile(
    r"(?ix)\\(?:n|r|t|x0[9ad]|u000[9ad]|U0000000[9ad]|0(?:11|12|15))"
)
_ESCAPED_UNICODE_RE = re.compile(
    r"\\(?:x(?P<byte>[0-9a-fA-F]{2})|u(?P<short>[0-9a-fA-F]{4})|U(?P<long>[0-9a-fA-F]{8}))"
)
_NUMERIC_ENTITY_RE = re.compile(r"&#(?:(?P<decimal>\d+)|[xX](?P<hex>[0-9a-fA-F]+));?")
_LINE_PREFIX_RE = re.compile(r"(?m)^\s*(?:[EIWF]\s+|[+>~-]\s?)")
_UNSUPPORTED_TERMINAL_CONTROL_RE = re.compile(
    r"(?:\x1b(?:P|_|\^|X)|[\x90\x98\x9e\x9f]).*?(?:\x1b\\|\x9c)",
    re.DOTALL,
)


_MAX_HTML_ENTITY_DECODE_PASSES = 64


def _decode_html_entities(text: str) -> str | None:
    def decode_numeric(match: re.Match[str]) -> str:
        encoded = match.group("decimal") or match.group("hex")
        assert encoded is not None
        base = 10 if match.group("decimal") is not None else 16
        try:
            return chr(int(encoded, base))
        except (ValueError, OverflowError):
            return match.group(0)

    decoded_text = text
    for _ in range(_MAX_HTML_ENTITY_DECODE_PASSES):
        numeric_decoded = _NUMERIC_ENTITY_RE.sub(decode_numeric, decoded_text)
        decoded = html.unescape(numeric_decoded)
        if decoded == decoded_text:
            return decoded_text
        decoded_text = decoded
    return None


def _decode_escaped_unicode(text: str) -> str | None:
    invalid = False

    def replace(match: re.Match[str]) -> str:
        nonlocal invalid
        encoded = match.group("byte") or match.group("short") or match.group("long")
        assert encoded is not None
        try:
            return chr(int(encoded, 16))
        except (ValueError, OverflowError):
            invalid = True
            return ""

    decoded = _ESCAPED_UNICODE_RE.sub(replace, text)
    return None if invalid else decoded


def _decode_contract_encodings(text: str) -> str | None:
    decoded_text = text
    for _ in range(_MAX_HTML_ENTITY_DECODE_PASSES):
        html_decoded = _decode_html_entities(decoded_text)
        if html_decoded is None:
            return None
        unicode_decoded = _decode_escaped_unicode(html_decoded)
        if unicode_decoded is None:
            return None
        decoded = _ESCAPED_WHITESPACE_RE.sub(" ", unicode_decoded)
        if decoded == decoded_text:
            return decoded_text
        decoded_text = decoded
    return None


def _normalized_contract_text(text: str, *, preserve_punctuation: bool) -> str | None:
    """Normalize routine verifier-output transformations for leak detection."""
    unescaped = _decode_contract_encodings(text)
    if unescaped is None:
        return None
    unescaped = unicodedata.normalize("NFKC", unescaped)
    without_ansi = _ANSI_ESCAPE_RE.sub("", unescaped)
    without_ansi = _OSC_ESCAPE_RE.sub("", without_ansi)
    without_ansi = "".join(
        char
        for char in without_ansi
        if (ord(char) >= 32 or char in "\n\r\t")
        and unicodedata.category(char) not in {"Cf", "Mn", "Me"}
    )
    without_prefixes = _LINE_PREFIX_RE.sub("", without_ansi)
    folded = "".join(char.casefold() for char in without_prefixes)
    if preserve_punctuation:
        return "".join(
            char
            for char in folded
            if not char.isspace() and unicodedata.category(char) not in {"Cf", "Mn", "Me"}
        )
    return "".join(char for char in folded if char.isalnum())


def contains_unsupported_terminal_control(text: str) -> bool:
    """Return whether output carries controls outside normalized CSI/OSC."""
    decoded = _decode_contract_encodings(text)
    if decoded is None:
        return True
    if _ESCAPED_TERMINAL_CONTROL_RE.search(decoded):
        return True
    without_known = _ANSI_ESCAPE_RE.sub("", _OSC_ESCAPE_RE.sub("", decoded))
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
        if normalized_remaining is None or normalized_hidden is None:
            return True
        if normalized_hidden:
            if normalized_hidden in normalized_remaining:
                return True
            continue
        compact_remaining = _normalized_contract_text(remaining, preserve_punctuation=True)
        compact_hidden = _normalized_contract_text(hidden, preserve_punctuation=True)
        if compact_remaining is None or compact_hidden is None:
            return True
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
