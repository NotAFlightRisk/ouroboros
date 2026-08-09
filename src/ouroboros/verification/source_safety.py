"""Source-aware masking for positive regex evidence.

The verifier searches generated regexes over source text. A textual match
inside a comment or string is not executable evidence, so supported source
syntaxes are masked without changing offsets before regex evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
import io
import os
import tokenize

_C_STYLE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".m",
        ".mm",
        ".rs",
        ".scss",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_HASH_STYLE_SUFFIXES = frozenset(
    {
        ".bash",
        ".cfg",
        ".conf",
        ".ini",
        ".pl",
        ".r",
        ".rb",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
_PLAIN_DATA_SUFFIXES = frozenset({".csv", ".json", ".lock", ".txt"})
_KNOWN_COMMENT_MARKERS = ("//", "/*", "--", "#", "<!--", "{-")


def _mask_ranges(text: str, ranges: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in ranges:
        for index in range(max(0, start), min(len(chars), end)):
            if chars[index] not in "\r\n":
                chars[index] = " "
    return "".join(chars)


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _python_noncode_ranges(text: str) -> tuple[tuple[int, int], ...] | None:
    """Return Python comments and string literals."""
    offsets = _line_offsets(text)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (IndentationError, tokenize.TokenError):
        return None

    ranges: list[tuple[int, int]] = []
    for token in tokens:
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            continue
        start_line, start_column = token.start
        end_line, end_column = token.end
        if start_line > len(offsets) or end_line > len(offsets):
            return None
        start = offsets[start_line - 1] + start_column
        end = offsets[end_line - 1] + end_column
        if token.type == tokenize.STRING:
            quote_offsets = [
                offset for offset in (token.string.find("'"), token.string.find('"')) if offset >= 0
            ]
            if not quote_offsets:
                return None
            start += min(quote_offsets) + 1
            end -= 1
        ranges.append((start, end))
    return tuple(ranges)


def _delimited_noncode_ranges(
    text: str,
    *,
    line_markers: tuple[str, ...],
    block_markers: tuple[tuple[str, str], ...],
) -> tuple[tuple[int, int], ...]:
    """Scan comments and strings without confusing their delimiters."""
    ranges: list[tuple[int, int]] = []
    index = 0
    quote = ""
    quote_start = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                ranges.append((quote_start + 1, index))
                quote = ""
            index += 1
            continue
        if char in {"'", '"'} or char == chr(96):
            quote = char
            quote_start = index
            index += 1
            continue
        block = next(
            ((start, end) for start, end in block_markers if text.startswith(start, index)),
            None,
        )
        if block is not None:
            start_marker, end_marker = block
            end = text.find(end_marker, index + len(start_marker))
            end = len(text) if end < 0 else end + len(end_marker)
            ranges.append((index, end))
            index = end
            continue
        marker = next((item for item in line_markers if text.startswith(item, index)), None)
        if marker is not None:
            end = text.find("\n", index + len(marker))
            end = len(text) if end < 0 else end
            ranges.append((index, end))
            index = end
            continue
        index += 1
    if quote:
        ranges.append((quote_start + 1, len(text)))
    return tuple(ranges)


def mask_non_executable_source(text: str, file_path: str) -> str | None:
    """Return offset-identical text with comments and strings masked.

    Unknown source syntaxes fail closed when they contain a recognizable
    comment marker. Plain data formats are explicitly comment-free.
    """
    suffix = os.path.splitext(file_path)[1].casefold()
    if suffix in {".py", ".pyi"}:
        ranges = _python_noncode_ranges(text)
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix in _C_STYLE_SUFFIXES:
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=("//",),
            block_markers=(("/*", "*/"),),
        )
        return _mask_ranges(text, ranges)
    if suffix in _HASH_STYLE_SUFFIXES:
        ranges = _delimited_noncode_ranges(text, line_markers=("#",), block_markers=())
        return _mask_ranges(text, ranges)
    if suffix in {".lua", ".sql"}:
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=("--",),
            block_markers=(("--[[", "]]"), ("/*", "*/")),
        )
        return _mask_ranges(text, ranges)
    if suffix == ".hs":
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=("--",),
            block_markers=(("{-", "-}"),),
        )
        return _mask_ranges(text, ranges)
    if suffix in {".htm", ".html", ".md", ".svg", ".xml"}:
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=(),
            block_markers=(("<!--", "-->"),),
        )
        return _mask_ranges(text, ranges)
    if suffix in _PLAIN_DATA_SUFFIXES:
        return text
    return None if any(marker in text for marker in _KNOWN_COMMENT_MARKERS) else text
