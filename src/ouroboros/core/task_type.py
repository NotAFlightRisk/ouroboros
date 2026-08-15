"""Task-type contracts shared by Seed authoring and repair."""

from __future__ import annotations

import re

SUPPORTED_TASK_TYPES = frozenset(
    {
        "analysis",
        "artifact",
        "code",
        "document",
        "documentation",
        "presentation",
        "research",
    }
)
_TASK_TYPE_PATTERN = (
    r"(?P<task_type>code|research|analysis|artifact|document|documentation|presentation)"
)
_TASK_TYPE_CONTRACT_PATTERNS = (
    re.compile(
        rf"\btask[_\s-]*type\b\s*(?:=|:)\s*{_TASK_TYPE_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:set\s+(?:the\s+)?)?task[_\s-]*type\b\s+(?:is|equals?|to)\s+"
        rf"{_TASK_TYPE_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\btask[_\s-]*type\b.{{0,80}}?\b(?:must|should|needs?\s+to)\s+"
        rf"(?:be|remain|use|equal)\s+{_TASK_TYPE_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bkeep\s+(?:the\s+)?task[_\s-]*type\b\s+(?:as\s+)?{_TASK_TYPE_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\buse\s+{_TASK_TYPE_PATTERN}\s+as\s+(?:the\s+)?task[_\s-]*type\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<![A-Za-z0-9_])task[_\s-]*type\s*(?:은|는|이|가)?\s*"
        rf"{_TASK_TYPE_PATTERN}\s*(?:이어야|여야|로\s*(?:설정|지정))",
        re.IGNORECASE,
    ),
)

_NON_BINDING_CONTRACT_PATTERN = re.compile(
    r"\b(?:ignore|discard|superseded|obsolete|example|literal|discussed|phrase|proposal)\b"
    r"|\b(?:do\s+not|don't|never|avoid(?:ed|ing)?|cannot|can't|can\s+not|without)\b"
    r"|\b(?:am|is|are|was|were)\s+not\b"
    r"|\bnot\s+true\b"
    r"|\b(?:must|should|may)\s+not\b"
    r"|\b(?:will|would)\s+not\b"
    r"|\b(?:won't|wouldn't|shouldn't|mustn't|isn't|aren't|wasn't|weren't)\b"
    r"|\b(?:won’t|wouldn’t|shouldn’t|mustn’t|isn’t|aren’t|wasn’t|weren’t)\b"
    r"|\b(?:am|is|are|was|were)\s+no\s+longer\b"
    r"|\bnot\s+allowed\b",
    re.IGNORECASE,
)
_HISTORICAL_GOVERNOR_PATTERN = re.compile(
    r"(?:\b(?:the\s+)?(?:previous|prior|historical)\b"
    r"|\b(?:rejected|superseded|obsolete)\b[^\n.!?]{0,50}\b(?:proposal|contract|reference|request)\b)"
    r"[^\n.!?]{0,120}?\b(?:but|and|while|although|though|because|despite)\b"
    r"[^\n.!?]{0,80}$",
    re.IGNORECASE,
)
_HISTORICAL_PREFIX_PATTERN = re.compile(
    r"\b(?:in|from|under)\s+(?:the\s+)?(?:previous|prior|historical)\s+"
    r"(?:proposal|contract|reference|request)\b[^\n.!?]{0,120}$",
    re.IGNORECASE,
)
_NEGATIVE_GOVERNOR_PATTERN = re.compile(
    r"\b(?:instead\s+of|rather\s+than|it\s+is\s+false\s+that|false\s+that|no\s+longer)\b"
    r"[^\n.!?]{0,120}$",
    re.IGNORECASE,
)
_POST_MATCH_REJECTION_PATTERN = re.compile(
    r"^(?:the\s+)?(?:requirement|contract|proposal)?\s*"
    r"(?:(?:was|is|has\s+been)?\s*(?:rejected|superseded|obsolete|discarded)"
    r"|(?:will|would)\s+not\s+(?:be\s+)?(?:used|required|adopted|applied)"
    r"|(?:am|is|are|was|were)\s+no\s+longer\s+(?:used|required|adopted|applied))\b",
    re.IGNORECASE,
)
_CANDIDATE_BOUNDARY_PATTERN = re.compile(
    r"[,!?;.\n]|\b(?:and|but|instead|without|although|though|because|while|despite)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_CONTRACT_PATTERN = re.compile(
    r"\b(?:if|unless|whether|either|otherwise|depending|choose\s+between)\b"
    r"|\bor\b",
    re.IGNORECASE,
)


def _candidate_segment(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the punctuation-or-conjunction-bounded candidate clause."""
    segment_start = 0
    for boundary in _CANDIDATE_BOUNDARY_PATTERN.finditer(text, 0, start):
        # A leading ``without``/causal clause is itself often the negation
        # governing the contract (``without using task_type``).  Keep it in
        # the candidate so the non-binding guard can reject it; these words
        # still delimit a positive contract when they occur after the match.
        if boundary.group().strip().casefold() in {"without", "despite"}:
            continue
        segment_start = boundary.end()
    next_boundary = _CANDIDATE_BOUNDARY_PATTERN.search(text, end)
    segment_end = next_boundary.start() if next_boundary is not None else len(text)
    return segment_start, segment_end


def _governor_scope_start(text: str, start: int) -> int:
    """Return the nearest hard clause boundary before a contract."""
    boundary = max(text.rfind(token, 0, start) for token in (";", ".", "!", "?", "\n"))
    return boundary + 1


def explicit_task_type_from_goal(goal: str) -> str | None:
    """Return the task type only when the goal states a binding contract."""
    normalized = goal
    matches: list[tuple[int, str]] = []
    for pattern in _TASK_TYPE_CONTRACT_PATTERNS:
        for match in pattern.finditer(normalized):
            line_start = normalized.rfind("\n", 0, match.start()) + 1
            line_end = normalized.find("\n", match.end())
            if line_end < 0:
                line_end = len(normalized)
            line = normalized[line_start:line_end]
            segment_start, segment_end = _candidate_segment(normalized, match.start(), match.end())
            segment = normalized[segment_start:segment_end]
            segment_terminator = normalized[segment_end : segment_end + 1]
            if re.match(r"\s*(?:q|question|interviewer)\s*:", line, re.IGNORECASE):
                continue
            if "?" in segment or segment_terminator == "?":
                continue
            if re.match(r"\s*\?", normalized[match.end() :]):
                continue
            if (
                is_non_binding_contract_segment(segment)
                or is_quoted_contract(normalized, match.start(), match.end())
                or has_historical_governor(
                    normalized, match.start(), _governor_scope_start(normalized, match.start())
                )
                or has_negative_governor(
                    normalized, match.start(), _governor_scope_start(normalized, match.start())
                )
                or has_post_match_rejection(normalized, match.end())
                or is_ambiguous_contract_segment(segment)
            ):
                continue
            matches.append((match.start(), match.group("task_type").casefold()))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def is_non_binding_contract_segment(segment: str) -> bool:
    """Return whether a clause describes negated, historical, or quoted text."""
    return _NON_BINDING_CONTRACT_PATTERN.search(segment) is not None


def is_quoted_contract(text: str, start: int, end: int) -> bool:
    """Return whether the matched contract itself is enclosed in quotes."""
    before = text[:start]
    after = text[end:]
    for quote in ('"', "`"):
        if before.count(quote) % 2 == 1 and after.count(quote) > 0:
            return True
    for opening, closing in (("“", "”"), ("‘", "’")):
        if before.count(opening) > before.count(closing) and after.count(closing) > 0:
            return True
    return False


def is_ambiguous_contract_segment(segment: str) -> bool:
    """Reject conditional or multi-option language at an authority boundary."""
    return _AMBIGUOUS_CONTRACT_PATTERN.search(segment) is not None


def has_historical_governor(text: str, start: int, scope_start: int | None = None) -> bool:
    """Return whether a preceding clause marks this contract as historical."""
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[max(line_start, scope_start or line_start) : start]
    return (
        _HISTORICAL_GOVERNOR_PATTERN.search(prefix) is not None
        or _HISTORICAL_PREFIX_PATTERN.search(prefix) is not None
    )


def has_negative_governor(text: str, start: int, scope_start: int | None = None) -> bool:
    """Return whether a preceding phrase explicitly negates this contract."""
    line_start = text.rfind("\n", 0, start) + 1
    return (
        _NEGATIVE_GOVERNOR_PATTERN.search(text[max(line_start, scope_start or line_start) : start])
        is not None
    )


def has_post_match_rejection(text: str, end: int) -> bool:
    """Return whether the current sentence immediately rejects the contract."""
    tail = text[end:]
    sentence_end = re.search(r"[.!?\n]", tail)
    suffix = tail[: sentence_end.start()] if sentence_end is not None else tail
    return _POST_MATCH_REJECTION_PATTERN.search(suffix.lstrip(" \t,;:")) is not None


def normalize_task_type(value: object) -> str:
    """Return a supported task type or raise a precise extraction error."""
    task_type = str(value).strip().casefold()
    if task_type not in SUPPORTED_TASK_TYPES:
        valid = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(f"Invalid TASK_TYPE: {task_type!r}. Expected one of: {valid}")
    return task_type
