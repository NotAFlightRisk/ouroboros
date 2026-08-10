"""Bind structural evidence to caller-authored declaration kinds."""

from __future__ import annotations

import re

from ouroboros.verification.binding import (
    acceptance_declaration_kind,
    literal_spans,
)
from ouroboros.verification.models import SpecAssertion, VerificationTier

_DECLARATION_PATTERNS = {
    "class": (r"\bclass\s+(?P<target>{target})\b",),
    "interface": (
        r"\binterface\s+(?P<target>{target})\b",
        r"\btype\s+(?P<target>{target})\s+interface\b",
    ),
    "struct": (
        r"\bstruct\s+(?P<target>{target})\b",
        r"\btype\s+(?P<target>{target})\s+struct\b",
    ),
    "trait": (r"\btrait\s+(?P<target>{target})\b",),
    "function": (r"\b(?:def|fn|func|function)\s+(?P<target>{target})\b",),
}


def _literal_occurrences(
    subject: str,
    literal: str,
    start: int,
    end: int,
    flags: int,
) -> tuple[tuple[int, int], ...]:
    start = max(0, start)
    end = min(len(subject), end)
    if start >= end or not literal:
        return ()
    literal_flags = re.IGNORECASE if flags & re.IGNORECASE else 0
    return tuple(
        (start + match.start(), start + match.end())
        for match in re.finditer(re.escape(literal), subject[start:end], literal_flags)
    )


def _source_span_has_declaration_kind(
    source: str,
    target_span: tuple[int, int],
    target: str,
    kind: str,
) -> bool:
    escaped = re.escape(target)
    return any(
        match.span("target") == target_span
        for template in _DECLARATION_PATTERNS.get(kind, ())
        for match in re.finditer(template.format(target=escaped), source)
    )


def match_has_bound_declaration_kind(
    pattern: re.Pattern,
    match: re.Match,
    searched_text: str,
    assertion: SpecAssertion,
    evidence_target: str,
    finite_witnesses: tuple[tuple[int, int, int], ...],
) -> bool:
    """Whether the exact matched occurrence has the caller-requested kind."""
    if not assertion.input_binding_required or assertion.tier is not VerificationTier.T2_STRUCTURAL:
        return True
    kind_required, kind = acceptance_declaration_kind(assertion.ac_text, evidence_target)
    if not kind_required:
        return True
    if kind is None:
        return False

    trusted_spans = literal_spans(searched_text, evidence_target)
    candidate_spans: list[tuple[int, int]] = []
    if match.start() != match.end():
        occurrences = _literal_occurrences(
            searched_text,
            evidence_target,
            match.start(),
            match.end(),
            pattern.flags,
        )
        if len(occurrences) == 1 and occurrences[0] in trusted_spans:
            candidate_spans.append(occurrences[0])
    else:
        for direction, offset, maximum in finite_witnesses:
            assertion_position = match.start() + offset
            start = assertion_position if direction == 1 else assertion_position - maximum
            end = assertion_position + maximum if direction == 1 else assertion_position
            occurrences = _literal_occurrences(
                searched_text,
                evidence_target,
                start,
                end,
                pattern.flags,
            )
            if len(occurrences) == 1 and occurrences[0] in trusted_spans:
                candidate_spans.append(occurrences[0])

    return any(
        _source_span_has_declaration_kind(searched_text, span, evidence_target, kind)
        for span in candidate_spans
    )
