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


def explicit_task_type_from_goal(goal: str) -> str | None:
    """Return the task type only when the goal states a binding contract."""
    normalized = goal.replace("`", "").replace("'", "").replace('"', "")
    matches: list[tuple[int, str]] = []
    for pattern in _TASK_TYPE_CONTRACT_PATTERNS:
        for match in pattern.finditer(normalized):
            line_start = normalized.rfind("\n", 0, match.start()) + 1
            line_end = normalized.find("\n", match.end())
            if line_end < 0:
                line_end = len(normalized)
            line = normalized[line_start:line_end]
            clause_start = max(
                normalized.rfind(separator, line_start, match.start()) for separator in ".!?;"
            )
            clause_start = max(line_start, clause_start + 1)
            prefix = normalized[clause_start : match.start()]
            if re.match(r"\s*(?:q|question|interviewer)\s*:", line, re.IGNORECASE):
                continue
            if "?" in normalized[clause_start : match.end()]:
                continue
            if re.search(
                r"\b(?:ignore|discard|superseded|obsolete)\b",
                prefix,
                re.IGNORECASE,
            ):
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
