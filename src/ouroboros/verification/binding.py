"""Deterministic binding between verifier evidence and an acceptance criterion."""

from __future__ import annotations

import re

# Words that describe the verification shape rather than the thing being
# verified. They cannot bind evidence to a criterion: finding an unrelated
# ``class Foo`` does not prove a request for ``CameraProvider`` merely because
# both the request and the match can be described as a class.
_NON_TARGET_WORDS = frozenset(
    {
        "a",
        "accept",
        "accepts",
        "an",
        "and",
        "be",
        "class",
        "constant",
        "contain",
        "contains",
        "define",
        "defines",
        "directory",
        "exist",
        "exists",
        "file",
        "flag",
        "for",
        "function",
        "has",
        "have",
        "in",
        "interface",
        "is",
        "it",
        "match",
        "maximum",
        "must",
        "of",
        "or",
        "remain",
        "required",
        "requires",
        "set",
        "should",
        "struct",
        "test",
        "the",
        "to",
        "trait",
        "value",
        "whether",
        "with",
    }
)

_TARGET_TOKEN = re.compile(r"[\w][\w./:-]*", re.UNICODE)


def literal_spans(text: str, literal: str) -> tuple[tuple[int, int], ...]:
    """Return case-insensitive whole-literal spans without prefix matches."""
    literal = literal.strip()
    if not literal:
        return ()
    left = r"(?<![A-Za-z0-9])" if literal[0].isalnum() or literal[0] == "_" else ""
    right = r"(?![A-Za-z0-9])" if literal[-1].isalnum() or literal[-1] == "_" else ""
    expression = re.compile(left + re.escape(literal) + right, re.IGNORECASE)
    return tuple((match.start(), match.end()) for match in expression.finditer(text))


def literal_is_bound(text: str, literal: str) -> bool:
    """Whether ``literal`` is present as a complete value in trusted text."""
    return bool(literal_spans(text, literal))


def acceptance_targets(
    ac_text: str,
    expected_value: str = "",
    *,
    prefer_expected: bool = False,
) -> tuple[str, ...]:
    """Return criterion-derived literals that may bind verification evidence.

    ``ac_text`` is supplied by the caller, not by the extraction model. An
    expected value is usable only when that caller-authored text contains it.
    Structural assertions prefer that exact expected name. Constant assertions
    instead bind the matched declaration/key and verify the expected scalar in
    a separate comparison.
    """
    expected = expected_value.strip()
    if expected and not literal_is_bound(ac_text, expected):
        return ()
    if prefer_expected and expected:
        return (expected,)

    expected_parts = {part.casefold() for part in _TARGET_TOKEN.findall(expected)}
    targets: list[str] = []
    seen: set[str] = set()
    for token in _TARGET_TOKEN.findall(ac_text):
        folded = token.casefold()
        if (
            folded in seen
            or folded in _NON_TARGET_WORDS
            or folded in expected_parts
            or token.isdecimal()
            or len(token) < 2
        ):
            continue
        seen.add(folded)
        targets.append(token)
    return tuple(targets)
