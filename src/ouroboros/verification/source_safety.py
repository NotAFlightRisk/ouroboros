"""Source-aware masking for positive regex evidence.

The verifier searches generated regexes over source text. A textual match
inside a comment or string is not executable evidence, so supported source
syntaxes are masked without changing offsets before regex evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
import io
import os
import re
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
        ".kt",
        ".kts",
        ".m",
        ".mm",
        ".scss",
        ".swift",
    }
)
_JAVASCRIPT_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx"})
_HASH_STYLE_SUFFIXES = frozenset(
    {
        ".bash",
        ".pl",
        ".r",
        ".rb",
        ".sh",
        ".zsh",
    }
)
_CONFIG_SUFFIXES = frozenset({".cfg", ".conf", ".ini", ".toml", ".yaml", ".yml"})
_NON_SOURCE_SUFFIXES = frozenset({".csv", ".json", ".lock", ".md"})
_MARKUP_SUFFIXES = frozenset({".htm", ".html", ".svg", ".xml"})


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
    nested_blocks: bool = False,
) -> tuple[tuple[int, int], ...] | None:
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
            cursor = index + len(start_marker)
            depth = 1
            while depth:
                next_end = text.find(end_marker, cursor)
                if next_end < 0:
                    cursor = len(text)
                    break
                next_start = text.find(start_marker, cursor)
                if next_start >= 0 and next_start < next_end:
                    if not nested_blocks:
                        return None
                    depth += 1
                    cursor = next_start + len(start_marker)
                    continue
                depth -= 1
                cursor = next_end + len(end_marker)
            end = cursor
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


def _lua_long_delimiter(text: str, index: int) -> tuple[int, str] | None:
    """Return a Lua long-bracket opener length and its exact closer."""
    if index >= len(text) or text[index] != "[":
        return None
    cursor = index + 1
    while cursor < len(text) and text[cursor] == "=":
        cursor += 1
    if cursor >= len(text) or text[cursor] != "[":
        return None
    equals = text[index + 1 : cursor]
    return cursor - index + 1, f"]{equals}]"


def _lua_noncode_ranges(text: str) -> tuple[tuple[int, int], ...] | None:
    """Scan Lua quoted strings, long strings, and extended long comments."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text.startswith("--", index):
            long_delimiter = _lua_long_delimiter(text, index + 2)
            if long_delimiter is None:
                end = text.find("\n", index + 2)
                end = len(text) if end < 0 else end
            else:
                opener_length, closer = long_delimiter
                end = text.find(closer, index + 2 + opener_length)
                end = len(text) if end < 0 else end + len(closer)
            ranges.append((index, end))
            index = end
            continue
        long_delimiter = _lua_long_delimiter(text, index)
        if long_delimiter is not None:
            opener_length, closer = long_delimiter
            end = text.find(closer, index + opener_length)
            end = len(text) if end < 0 else end + len(closer)
            ranges.append((index, end))
            index = end
            continue
        if text[index] in {"'", '"'}:
            quote = text[index]
            cursor = index + 1
            while cursor < len(text):
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == quote:
                    cursor += 1
                    break
                cursor += 1
            ranges.append((index, min(cursor, len(text))))
            index = cursor
            continue
        index += 1
    return tuple(ranges)


def _can_start_javascript_regex(text: str, index: int) -> bool:
    prefix = text[:index].rstrip()
    if not prefix or prefix[-1] in "=(:,[!&|?{};":
        return True
    word = re.search(r"([A-Za-z_$][\w$]*)$", prefix)
    return bool(
        word
        and word.group(1)
        in {
            "await",
            "case",
            "delete",
            "in",
            "instanceof",
            "of",
            "return",
            "throw",
            "typeof",
            "void",
            "yield",
        }
    )


def _javascript_noncode_ranges(text: str) -> tuple[tuple[int, int], ...] | None:
    """Mask JavaScript comments, strings, templates, and regex literals."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            end = len(text) if end < 0 else end
            ranges.append((index, end))
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            nested = text.find("/*", index + 2)
            if nested >= 0 and (end < 0 or nested < end):
                return None
            end = len(text) if end < 0 else end + 2
            ranges.append((index, end))
            index = end
            continue
        if text[index] in {"'", '"', "`"}:
            quote = text[index]
            cursor = index + 1
            while cursor < len(text):
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == quote:
                    cursor += 1
                    break
                cursor += 1
            ranges.append((index, min(cursor, len(text))))
            index = cursor
            continue
        if text[index] == "/" and _can_start_javascript_regex(text, index):
            cursor = index + 1
            in_class = False
            while cursor < len(text):
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == "[":
                    in_class = True
                elif text[cursor] == "]":
                    in_class = False
                elif text[cursor] == "/" and not in_class:
                    cursor += 1
                    while cursor < len(text) and text[cursor].isalpha():
                        cursor += 1
                    break
                elif text[cursor] in "\r\n":
                    return None
                cursor += 1
            else:
                return None
            ranges.append((index, cursor))
            index = cursor
            continue
        index += 1
    return tuple(ranges)


def _markup_noncode_ranges(text: str) -> tuple[tuple[int, int], ...] | None:
    """Keep markup syntax while masking comments, text nodes, and values."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        start = text.find("<", index)
        if start < 0:
            ranges.append((index, len(text)))
            break
        ranges.append((index, start))
        if text.startswith("<!--", start):
            end = text.find("-->", start + 4)
            end = len(text) if end < 0 else end + 3
            ranges.append((start, end))
            index = end
            continue
        if text.startswith("<![CDATA[", start):
            end = text.find("]]>", start + 9)
            end = len(text) if end < 0 else end + 3
            ranges.append((start, end))
            index = end
            continue
        cursor = start + 1
        quote = ""
        quote_start = 0
        while cursor < len(text):
            char = text[cursor]
            if quote:
                if char == quote:
                    ranges.append((quote_start, cursor + 1))
                    quote = ""
            elif char in {"'", '"'}:
                quote = char
                quote_start = cursor
            elif char == ">":
                cursor += 1
                break
            cursor += 1
        if cursor > len(text) or quote or (cursor == len(text) and text[-1:] != ">"):
            return None
        index = cursor
    return tuple(ranges)


def mask_non_executable_source(
    text: str,
    file_path: str,
    *,
    allow_configuration: bool = False,
    allow_plain_text: bool = False,
) -> str | None:
    """Return offset-identical text with comments and strings masked.

    Unknown and non-source syntaxes fail closed. Configuration syntax is
    available only to the T1 constant path; it cannot prove a T2 declaration.
    Raw ``.txt`` output is searchable only when the verifier has established
    an explicit observable-output contract; arbitrary prose is not source.
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
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix in _JAVASCRIPT_SUFFIXES:
        ranges = _javascript_noncode_ranges(text)
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix == ".rs":
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=("//",),
            block_markers=(("/*", "*/"),),
            nested_blocks=True,
        )
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix in _HASH_STYLE_SUFFIXES:
        ranges = _delimited_noncode_ranges(text, line_markers=("#",), block_markers=())
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix in _CONFIG_SUFFIXES:
        if not allow_configuration:
            return None
        ranges = _delimited_noncode_ranges(text, line_markers=("#",), block_markers=())
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix == ".lua":
        ranges = _lua_noncode_ranges(text)
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix == ".sql":
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=("--",),
            block_markers=(("/*", "*/"),),
            nested_blocks=True,
        )
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix == ".hs":
        ranges = _delimited_noncode_ranges(
            text,
            line_markers=("--",),
            block_markers=(("{-", "-}"),),
            nested_blocks=True,
        )
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix in _MARKUP_SUFFIXES:
        ranges = _markup_noncode_ranges(text)
        return None if ranges is None else _mask_ranges(text, ranges)
    if suffix in _NON_SOURCE_SUFFIXES:
        return None
    if suffix == ".txt":
        return text if allow_plain_text else None
    return None
