"""Bind structural evidence to caller-authored declaration kinds."""

from __future__ import annotations

import os
import re

from ouroboros.verification.binding import (
    acceptance_declaration_kind,
    literal_spans,
)
from ouroboros.verification.models import SpecAssertion, VerificationTier

_CLASS_SUFFIXES = frozenset(
    {
        ".cc",
        ".cpp",
        ".cs",
        ".h",
        ".hpp",
        ".hs",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".mm",
        ".py",
        ".pyi",
        ".rb",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_INTERFACE_SUFFIXES = frozenset({".cs", ".java", ".kt", ".kts", ".ts", ".tsx"})
_STRUCT_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cs", ".h", ".hpp", ".mm", ".rs", ".swift"})
_C_LIKE_FUNCTION_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cs", ".h", ".hpp", ".java", ".mm"})
_FUNCTION_KEYWORDS = {
    ".go": "func",
    ".js": "function",
    ".jsx": "function",
    ".kt": "fun",
    ".kts": "fun",
    ".lua": "function",
    ".pl": "sub",
    ".py": "(?:async\\s+)?def",
    ".pyi": "(?:async\\s+)?def",
    ".rb": "def",
    ".rs": "fn",
    ".swift": "func",
    ".ts": "function",
    ".tsx": "function",
}
_SHELL_SUFFIXES = frozenset({".bash", ".sh", ".zsh"})
_C_LIKE_FUNCTION = (
    r"(?m)^[ \t]*(?!(?:return|if|for|while|switch|catch|throw|new|sizeof)\b)"
    r"(?:[A-Za-z_]\w*\s+){{1,8}}[*&\s]*(?P<target>{target})[ \t]*"
    r"\([^;{{}}\r\n]*\)[ \t]*(?:throws\s+[^;{{\r\n]+)?(?:\{{|;)"
)


def _declaration_patterns(file_path: str, kind: str) -> tuple[str, ...]:
    """Return only declaration grammars admitted for the file's language."""
    suffix = os.path.splitext(file_path)[1].casefold()
    if kind == "class" and suffix in _CLASS_SUFFIXES:
        return (r"\bclass\s+(?P<target>{target})\b",)
    if kind == "interface":
        if suffix in _INTERFACE_SUFFIXES:
            return (r"\binterface\s+(?P<target>{target})\b",)
        if suffix == ".go":
            return (r"\btype\s+(?P<target>{target})\s+interface\b",)
    if kind == "struct":
        if suffix in _STRUCT_SUFFIXES:
            return (r"\bstruct\s+(?P<target>{target})\b",)
        if suffix == ".go":
            return (r"\btype\s+(?P<target>{target})\s+struct\b",)
    if kind == "trait" and suffix == ".rs":
        return (r"\btrait\s+(?P<target>{target})\b",)
    if kind == "function":
        if keyword := _FUNCTION_KEYWORDS.get(suffix):
            return (rf"\b{keyword}\s+(?P<target>{{target}})\b",)
        if suffix in _C_LIKE_FUNCTION_SUFFIXES:
            return (_C_LIKE_FUNCTION,)
        if suffix == ".r":
            return (r"(?P<target>{target})\s*(?:<-|=)\s*function\s*\(",)
        if suffix in _SHELL_SUFFIXES:
            return (r"(?m)^[ \t]*(?:function\s+)?(?P<target>{target})\s*\(\s*\)\s*\{",)
    return ()


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
    file_path: str,
) -> bool:
    escaped = re.escape(target)
    for template in _declaration_patterns(file_path, kind):
        for match in re.finditer(template.format(target=escaped), source):
            if match.span("target") != target_span:
                continue
            declaration_prefix = source[max(0, match.start() - 256) : match.start()]
            if kind in {"class", "struct"} and re.search(
                r"\benum(?:\s|\[\[[^\]]*\]\])*$",
                declaration_prefix,
            ):
                continue
            target_start = match.start("target")
            declaration_boundary = max(source.rfind(marker, 0, target_start) for marker in ";{}")
            target_prefix = source[declaration_boundary + 1 : target_start]
            if kind == "function" and re.search(
                r"\b(?:delegate|typedef|using)\b",
                target_prefix,
            ):
                continue
            return True
    return False


def match_has_bound_declaration_kind(
    pattern: re.Pattern,
    match: re.Match,
    searched_text: str,
    assertion: SpecAssertion,
    evidence_target: str,
    finite_witnesses: tuple[tuple[int, int, int], ...],
    file_path: str,
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
        _source_span_has_declaration_kind(searched_text, span, evidence_target, kind, file_path)
        for span in candidate_spans
    )
