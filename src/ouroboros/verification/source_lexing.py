"""Conservative source lexical classification for formal evidence binding.

This is deliberately not a language parser.  It recognizes only source forms
that must never mint positive evidence (comments, strings, regex literals,
Perl/Ruby quote-like operators, and heredoc bodies).  Ambiguous constructs are
classified as non-code so unsupported syntax fails closed.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import re

_LOGICAL_LINE_BOUNDARY = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")
_PAIRED_DELIMITERS = {"(": ")", "[": "]", "{": "}", "<": ">"}
_SINGLE_QUOTE_LIKE = ("qr", "qq", "qx", "qw", "%r", "%q", "%Q", "m", "q")
_DOUBLE_QUOTE_LIKE = ("tr", "s", "y")
_CPP_RAW_PREFIXES = ('u8R"', 'uR"', 'UR"', 'LR"', 'R"')
_REGEX_PREFIX_KEYWORDS = frozenset(
    {
        "await",
        "break",
        "case",
        "const",
        "continue",
        "debugger",
        "delete",
        "default",
        "do",
        "else",
        "export",
        "extends",
        "in",
        "instanceof",
        "let",
        "new",
        "of",
        "return",
        "throw",
        "typeof",
        "var",
        "void",
        "yield",
    }
)
_CONTROL_PAREN_KEYWORDS = frozenset({"catch", "for", "if", "switch", "while", "with"})


@dataclass(frozen=True, slots=True)
class SourceLexicalMap:
    """Precomputed non-code and logical-line bounds for one source text."""

    noncode_regions: tuple[tuple[int, int], ...]
    noncode_starts: tuple[int, ...]
    line_starts: tuple[int, ...]
    line_ends: tuple[int, ...]

    def is_noncode(self, position: int) -> bool:
        """Return whether ``position`` lies in a classified non-code region."""
        index = bisect_right(self.noncode_starts, position) - 1
        return index >= 0 and position < self.noncode_regions[index][1]

    def line_bounds(self, position: int) -> tuple[int, int]:
        """Return logical line bounds containing ``position``."""
        index = max(0, bisect_right(self.line_starts, position) - 1)
        return self.line_starts[index], self.line_ends[index]


@dataclass(frozen=True, slots=True)
class _HeredocBounds:
    body_start: int
    body_end: int
    scan_end: int


@dataclass(frozen=True, slots=True)
class _HeredocOpener:
    delimiter: str | None
    indent_mode: str
    end: int


def _identifier_start(character: str) -> bool:
    return character == "_" or character == "$" or character.isalpha()


def _identifier_continue(character: str) -> bool:
    return _identifier_start(character) or character.isdigit()


def _line_layout(text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    starts = [0]
    ends: list[int] = []
    for boundary in _LOGICAL_LINE_BOUNDARY.finditer(text):
        ends.append(boundary.start())
        starts.append(boundary.end())
    ends.append(len(text))
    return tuple(starts), tuple(ends)


def _quoted_end(text: str, index: int, quote: str) -> int:
    cursor = index + len(quote)
    while cursor < len(text):
        if text.startswith(quote, cursor):
            return cursor + len(quote)
        if text[cursor] == "\\":
            cursor += min(2, len(text) - cursor)
        else:
            cursor += 1
    return len(text)


def _nested_comment_end(text: str, index: int) -> int:
    """Return the end of a possibly nested ``/* ... */`` comment."""
    depth = 1
    cursor = index + 2
    while cursor < len(text):
        if text.startswith("/*", cursor):
            depth += 1
            cursor += 2
        elif text.startswith("*/", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
        else:
            cursor += 1
    return len(text)


def _cpp_raw_string_end(text: str, index: int) -> int | None:
    """Return a C++ raw-string end, including custom delimiters."""
    if index and _identifier_continue(text[index - 1]):
        return None
    prefix = next((item for item in _CPP_RAW_PREFIXES if text.startswith(item, index)), None)
    if prefix is None:
        return None
    delimiter_start = index + len(prefix)
    opening = text.find("(", delimiter_start, min(len(text), delimiter_start + 17))
    if opening < 0:
        return None
    delimiter = text[delimiter_start:opening]
    if any(character.isspace() or character in "()\\" for character in delimiter):
        return None
    terminator = ")" + delimiter + '"'
    closing = text.find(terminator, opening + 1)
    return len(text) if closing < 0 else closing + len(terminator)


def _delimited_end(text: str, opening_index: int) -> int:
    opening = text[opening_index]
    closing = _PAIRED_DELIMITERS.get(opening, opening)
    paired = opening in _PAIRED_DELIMITERS
    depth = 1
    cursor = opening_index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += min(2, len(text) - cursor)
            continue
        if paired and text[cursor] == opening:
            depth += 1
        elif text[cursor] == closing:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return len(text)


def _unopened_delimited_end(text: str, index: int, delimiter: str) -> int:
    cursor = index
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += min(2, len(text) - cursor)
            continue
        if text.startswith(delimiter, cursor):
            return cursor + len(delimiter)
        cursor += 1
    return len(text)


def _quote_like_end(text: str, index: int) -> int | None:
    """Return the end of a Perl/Ruby quote-like operator starting at ``index``."""
    if index and _identifier_continue(text[index - 1]):
        return None
    operator = next(
        (
            candidate
            for candidate in (*_SINGLE_QUOTE_LIKE, *_DOUBLE_QUOTE_LIKE)
            if text.startswith(candidate, index)
        ),
        None,
    )
    if operator is None:
        return None
    cursor = index + len(operator)
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    if cursor >= len(text):
        return None
    newline = _LOGICAL_LINE_BOUNDARY.match(text, cursor)
    if newline is not None:
        delimiter = newline.group()
        first_end = _unopened_delimited_end(text, newline.end(), delimiter)
    else:
        delimiter = text[cursor]
        if delimiter == "\\" or delimiter.isspace() or _identifier_continue(delimiter):
            return None
        first_end = _delimited_end(text, cursor)

    if not delimiter:
        return None
    if operator not in _DOUBLE_QUOTE_LIKE:
        end = first_end
    elif first_end >= len(text):
        return len(text)
    elif len(delimiter) == 1 and delimiter in _PAIRED_DELIMITERS:
        second_start = first_end
        while second_start < len(text) and text[second_start] in " \t":
            second_start += 1
        if second_start >= len(text) or text[second_start].isspace():
            return len(text)
        end = _delimited_end(text, second_start)
    else:
        end = _unopened_delimited_end(text, first_end, delimiter)

    while end < len(text) and text[end].isalpha():
        end += 1
    return end


def _heredoc_opener(text: str, index: int, line_end: int) -> _HeredocOpener | None:
    """Parse one shell/Ruby-style heredoc opener conservatively.

    Once ``<<`` is seen, unsupported delimiter syntax is still returned as an
    opener with no delimiter.  Its body therefore fails closed through EOF
    instead of being mistaken for executable source.
    """
    if not text.startswith("<<", index):
        return None
    if (index and text[index - 1] == "<") or text.startswith("<<<", index):
        return None

    cursor = index + 2
    indent_mode = ""
    if cursor < line_end and text[cursor] in "-~":
        indent_mode = text[cursor]
        cursor += 1
    while cursor < line_end and text[cursor] in " \t":
        cursor += 1
    if cursor >= line_end:
        return _HeredocOpener(None, indent_mode, cursor)

    if text[cursor] in "'\"`":
        quote = text[cursor]
        delimiter_start = cursor + 1
        closing = text.find(quote, delimiter_start, line_end)
        if closing < 0 or closing == delimiter_start:
            return _HeredocOpener(None, indent_mode, line_end)
        return _HeredocOpener(text[delimiter_start:closing], indent_mode, closing + 1)

    escaped = text[cursor] == "\\"
    if escaped:
        cursor += 1
    delimiter_start = cursor
    while cursor < line_end and not text[cursor].isspace():
        if text[cursor] in ";|&()<>\"'`\\":
            break
        cursor += 1
    if cursor == delimiter_start:
        return _HeredocOpener(None, indent_mode, max(cursor, delimiter_start))

    delimiter = text[delimiter_start:cursor]
    if cursor < line_end and not text[cursor].isspace() and text[cursor] not in ";|&":
        return _HeredocOpener(None, indent_mode, cursor)
    return _HeredocOpener(delimiter, indent_mode, cursor)


def _block_comment_end(
    text: str,
    index: int,
    line_starts: tuple[int, ...],
    line_ends: tuple[int, ...],
) -> int | None:
    """Return the end of a Perl POD or Ruby embedded-document block."""
    line_index = bisect_right(line_starts, index) - 1
    if line_index < 0 or line_starts[line_index] != index:
        return None
    line = text[index : line_ends[line_index]]
    if line == "=begin":
        terminator = "=end"
    elif line == "=pod" or line.startswith(("=pod ", "=pod\t")):
        terminator = "=cut"
    else:
        return None

    for candidate in range(line_index + 1, len(line_starts)):
        candidate_line = text[line_starts[candidate] : line_ends[candidate]]
        if candidate_line == terminator:
            if candidate + 1 < len(line_starts):
                return line_starts[candidate + 1]
            return line_ends[candidate]
    return len(text)


def _regex_end(text: str, index: int) -> int:
    cursor = index + 1
    in_class = False
    while cursor < len(text):
        if _LOGICAL_LINE_BOUNDARY.match(text, cursor):
            return len(text)
        if text[cursor] == "\\":
            cursor += min(2, len(text) - cursor)
            continue
        if text[cursor] == "[":
            in_class = True
        elif text[cursor] == "]":
            in_class = False
        elif text[cursor] == "/" and not in_class:
            cursor += 1
            while cursor < len(text) and text[cursor].isalpha():
                cursor += 1
            return cursor
        cursor += 1
    return len(text)


def _heredoc_bounds(
    text: str,
    index: int,
    line_starts: tuple[int, ...],
    line_ends: tuple[int, ...],
) -> _HeredocBounds | None:
    line_index = max(0, bisect_right(line_starts, index) - 1)
    opener = _heredoc_opener(text, index, line_ends[line_index])
    if opener is None:
        return None
    opener_line = max(0, bisect_right(line_starts, opener.end) - 1)
    if opener_line + 1 >= len(line_starts):
        return _HeredocBounds(opener.end, len(text), len(text))
    body_start = line_starts[opener_line + 1]
    if opener.delimiter is None:
        return _HeredocBounds(body_start, len(text), len(text))
    if text.find("<<", opener.end, line_ends[opener_line]) >= 0:
        # Multiple queued heredocs require shell parsing to pair bodies with
        # delimiters.  Until that grammar is supported, no later body is code.
        return _HeredocBounds(body_start, len(text), len(text))
    delimiter = opener.delimiter
    indent_mode = opener.indent_mode
    for line_index in range(opener_line + 1, len(line_starts)):
        line_start = line_starts[line_index]
        line_end = line_ends[line_index]
        line = text[line_start:line_end]
        if indent_mode == "-":
            terminates = line.lstrip("\t") == delimiter
        elif indent_mode == "~":
            terminates = line.lstrip(" \t") == delimiter
        else:
            terminates = line == delimiter
        if terminates:
            scan_end = (
                line_starts[line_index + 1] if line_index + 1 < len(line_starts) else line_end
            )
            return _HeredocBounds(body_start, line_start, scan_end)
    return _HeredocBounds(body_start, len(text), len(text))


def classify_source(text: str) -> SourceLexicalMap:
    """Classify source in one bounded linear pass.

    Slash tokens use JavaScript's expression-ending distinction: after an
    expression they are division, while after a statement/operator boundary
    they start a regex.  Ambiguous block boundaries prefer regex/non-code.
    """
    line_starts, line_ends = _line_layout(text)
    regions: list[tuple[int, int]] = []
    can_end_expression: bool | None = False
    pending_control_paren = False
    paren_controls: list[bool] = []
    index = 0

    def add_region(start: int, end: int) -> None:
        if end <= start:
            return
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))

    while index < len(text):
        character = text[index]
        if character.isspace():
            index += 1
            continue
        block_comment_end = _block_comment_end(text, index, line_starts, line_ends)
        if block_comment_end is not None:
            add_region(index, block_comment_end)
            index = block_comment_end
            can_end_expression = False
            pending_control_paren = False
            continue
        if text.startswith("/*", index):
            end = _nested_comment_end(text, index)
            add_region(index, end)
            index = end
            continue
        if text.startswith("<!--", index):
            closing = text.find("-->", index + 4)
            end = len(text) if closing < 0 else closing + 3
            add_region(index, end)
            index = end
            continue
        if character == "#" or text.startswith("//", index):
            boundary = _LOGICAL_LINE_BOUNDARY.search(text, index)
            end = len(text) if boundary is None else boundary.start()
            add_region(index, end)
            index = len(text) if boundary is None else boundary.end()
            continue

        heredoc = _heredoc_bounds(text, index, line_starts, line_ends)
        if heredoc is not None:
            add_region(heredoc.body_start, heredoc.body_end)
            index = heredoc.scan_end
            can_end_expression = False
            pending_control_paren = False
            continue

        cpp_raw_end = _cpp_raw_string_end(text, index)
        if cpp_raw_end is not None:
            add_region(index, cpp_raw_end)
            index = cpp_raw_end
            can_end_expression = True
            pending_control_paren = False
            continue

        quote_like_end = _quote_like_end(text, index)
        if quote_like_end is not None:
            add_region(index, quote_like_end)
            index = quote_like_end
            can_end_expression = True
            pending_control_paren = False
            continue

        if text.startswith(('"""', "'''"), index):
            end = _quoted_end(text, index, text[index : index + 3])
            add_region(index, end)
            index = end
            can_end_expression = True
            pending_control_paren = False
            continue
        if character in {'"', "'", "`"}:
            end = _quoted_end(text, index, character)
            add_region(index, end)
            index = end
            can_end_expression = True
            pending_control_paren = False
            continue

        if character == "/":
            if can_end_expression is True:
                index += 1
                can_end_expression = False
            elif can_end_expression is None:
                line_index = max(0, bisect_right(line_starts, index) - 1)
                end = line_ends[line_index]
                add_region(index, end)
                index = end
                can_end_expression = False
            else:
                end = _regex_end(text, index)
                add_region(index, end)
                index = end
                can_end_expression = True
            pending_control_paren = False
            continue

        if _identifier_start(character):
            end = index + 1
            while end < len(text) and _identifier_continue(text[end]):
                end += 1
            word = text[index:end]
            pending_control_paren = word in _CONTROL_PAREN_KEYWORDS
            can_end_expression = word not in _REGEX_PREFIX_KEYWORDS and not pending_control_paren
            index = end
            continue
        if character.isdigit():
            end = index + 1
            while end < len(text) and (text[end].isalnum() or text[end] in "._"):
                end += 1
            index = end
            can_end_expression = True
            pending_control_paren = False
            continue

        if text.startswith(("++", "--"), index):
            index += 2
            can_end_expression = True
            pending_control_paren = False
            continue

        if character == "(":
            paren_controls.append(pending_control_paren)
            can_end_expression = False
        elif character == ")":
            control = paren_controls.pop() if paren_controls else None
            can_end_expression = None if control is None else not control
        elif character == "]":
            can_end_expression = True
        elif character == "}":
            can_end_expression = None
        else:
            can_end_expression = False
        pending_control_paren = False
        index += 1

    immutable_regions = tuple(regions)
    return SourceLexicalMap(
        noncode_regions=immutable_regions,
        noncode_starts=tuple(start for start, _end in immutable_regions),
        line_starts=line_starts,
        line_ends=line_ends,
    )
