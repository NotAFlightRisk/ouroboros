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
        "create",
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
        "project",
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
_STRUCTURAL_WORDS = frozenset(
    {"class", "constant", "directory", "file", "flag", "function", "interface", "struct", "trait"}
)


def literal_spans(text: str, literal: str) -> tuple[tuple[int, int], ...]:
    """Return case-insensitive whole-literal spans without prefix matches."""
    literal = literal.strip()
    if not literal:
        return ()
    left = r"(?<!\w)" if literal[0].isalnum() or literal[0] == "_" else ""
    right = r"(?!\w)" if literal[-1].isalnum() or literal[-1] == "_" else ""
    expression = re.compile(left + re.escape(literal) + right, re.IGNORECASE)
    return tuple((match.start(), match.end()) for match in expression.finditer(text))


def literal_is_bound(text: str, literal: str) -> bool:
    """Whether ``literal`` is present as a complete value in trusted text."""
    return bool(literal_spans(text, literal))


def identifier_component_spans(text: str, literal: str) -> tuple[tuple[int, int], ...]:
    """Return spans where a literal is one component of a snake-case identifier.

    This compatibility is T1-only: prose commonly says ``Warmup frames`` while
    source spells the key ``WARMUP_FRAMES``. Structural names do not receive
    this relaxation, so ``CameraProvider`` cannot bind ``CameraProvider_fake``.
    """
    literal = literal.strip()
    if not literal:
        return ()
    expression = re.compile(
        r"(?<![^\W_])" + re.escape(literal) + r"(?![^\W_])",
        re.IGNORECASE,
    )
    return tuple((match.start(), match.end()) for match in expression.finditer(text))


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
    expected_parts = (
        set() if prefer_expected else {part.casefold() for part in _TARGET_TOKEN.findall(expected)}
    )
    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for match in _TARGET_TOKEN.finditer(ac_text):
        token = match.group(0)
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
        candidates.append((token, match.start(), match.end()))

    if prefer_expected:
        selected: list[str] = []
        for word in _TARGET_TOKEN.finditer(ac_text):
            if word.group(0).casefold() not in _STRUCTURAL_WORDS:
                continue
            ranked = sorted(
                candidates,
                key=lambda candidate: min(
                    abs(candidate[2] - word.start()),
                    abs(candidate[1] - word.end()),
                ),
            )
            if ranked and ranked[0][0].casefold() not in {target.casefold() for target in selected}:
                selected.append(ranked[0][0])
        if selected:
            return tuple(selected)
        if len(candidates) == 1:
            return (candidates[0][0],)
        return ()

    if expected:
        value_spans = literal_spans(ac_text, expected)
        if not value_spans:
            return ()
        value_start, value_end = value_spans[0]
        before = [candidate for candidate in candidates if candidate[2] <= value_start]
        if before:
            return (before[-1][0],)
        after = [candidate for candidate in candidates if candidate[1] >= value_end]
        return (after[0][0],) if after else ()
    return tuple(token for token, _start, _end in candidates)
