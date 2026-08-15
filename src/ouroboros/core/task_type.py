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
        rf"\btask[_\s-]*type\b.{{0,80}}?\b(?:must|should|needs?\s+to)\s+"
        rf"(?:be|use|equal)\s+{_TASK_TYPE_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<![A-Za-z0-9_])task[_\s-]*type\s*(?:은|는|이|가)?\s*"
        rf"{_TASK_TYPE_PATTERN}\s*(?:이어야|여야|로\s*(?:설정|지정))",
        re.IGNORECASE,
    ),
)

_NON_BINDING_CONTRACT_PATTERN = re.compile(
    r"\b(?:ignore|discard|superseded|obsolete|example|literal)\b"
    r"|\b(?:do\s+not|don't|never|avoid|cannot|can't|can\s+not|without)\b"
    r"|\b(?:must|should|may)\s+not\b"
    r"|\bnot\s+allowed\b",
    re.IGNORECASE,
)


def _candidate_segment(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the comma-or-sentence-bounded segment containing a candidate."""
    segment_start = max(text.rfind(separator, 0, start) for separator in ",.!?;\n") + 1
    segment_end_candidates = [
        index for separator in ",.!?;\n" if (index := text.find(separator, end)) >= 0
    ]
    segment_end = min(segment_end_candidates) if segment_end_candidates else len(text)
    return segment_start, segment_end


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
            if _NON_BINDING_CONTRACT_PATTERN.search(segment):
                continue
            matches.append((match.start(), match.group("task_type").casefold()))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def normalize_task_type(value: object) -> str:
    """Return a supported task type or raise a precise extraction error."""
    task_type = str(value).strip().casefold()
    if task_type not in SUPPORTED_TASK_TYPES:
        valid = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(f"Invalid TASK_TYPE: {task_type!r}. Expected one of: {valid}")
    return task_type
