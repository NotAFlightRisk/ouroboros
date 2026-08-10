"""Bounded whole-compilation-unit checks for parser-free declarations."""

from __future__ import annotations

import re

_C_STYLE_COMMENT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".mm",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_C_FAMILY_TYPE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".h", ".hpp", ".mm"})
_OPTIONAL_EMPTY_STATEMENT_SUFFIXES = frozenset(
    {".go", ".js", ".jsx", ".kt", ".kts", ".swift", ".ts", ".tsx"}
)


def source_without_comments(source: str, suffix: str) -> str:
    """Replace proven comment text with spaces while preserving line boundaries."""
    if suffix not in _C_STYLE_COMMENT_SUFFIXES:
        return source
    characters = list(source)
    cursor = 0
    while cursor < len(source) - 1:
        marker = source[cursor : cursor + 2]
        if marker == "//":
            end = source.find("\n", cursor + 2)
            if end < 0:
                end = len(source)
            for position in range(cursor, end):
                if characters[position] not in "\r\n":
                    characters[position] = " "
            cursor = end
            continue
        if marker == "/*":
            depth = 1
            end = cursor + 2
            while end < len(source) and depth:
                if suffix == ".rs" and source[end : end + 2] == "/*":
                    depth += 1
                    end += 2
                elif source[end : end + 2] == "*/":
                    depth -= 1
                    end += 2
                else:
                    end += 1
            if depth:
                return source
            for position in range(cursor, end):
                if characters[position] not in "\r\n":
                    characters[position] = " "
            cursor = end
            continue
        cursor += 1
    return "".join(characters)


def compilation_unit_prefix_is_valid(
    prefix: str,
    suffix: str,
    original_prefix: str | None,
    *,
    java_keyword: str,
    csharp_keyword: str,
    cpp_suffixes: frozenset[str],
) -> bool:
    """Admit only a bounded compilation-unit prefix before a declaration."""
    prefix = (
        source_without_comments(original_prefix, suffix) if original_prefix is not None else prefix
    )
    if suffix == ".java":
        identifier = rf"(?!(?:{java_keyword})\b)[A-Za-z_]\w*"
        qualified_name = rf"{identifier}(?:\.{identifier})*"
        grammar = (
            rf"\s*(?:package\s+{qualified_name}\s*;\s*)?"
            rf"(?:import\s+(?:static\s+)?{qualified_name}(?:\.\*)?\s*;\s*)*"
        )
        return re.fullmatch(grammar, prefix) is not None
    if suffix == ".cs":
        identifier = rf"(?!(?:{csharp_keyword})\b)[A-Za-z_]\w*"
        qualified_name = rf"{identifier}(?:\.{identifier})*"
        extern_alias = rf"extern\s+alias\s+{identifier}\s*;\s*"
        global_using = rf"global\s+using\s+(?:static\s+)?{qualified_name}\s*;\s*"
        local_using = rf"using\s+(?:static\s+)?{qualified_name}\s*;\s*"
        outer_directives = rf"(?:{extern_alias})*(?:{global_using})*(?:{local_using})*"
        inner_directives = rf"(?:{extern_alias})*(?:{local_using})*"
        file_namespace = rf"namespace\s+{qualified_name}\s*;\s*{inner_directives}"
        return re.fullmatch(rf"\s*{outer_directives}(?:{file_namespace})?", prefix) is not None
    if suffix in cpp_suffixes and re.fullmatch(
        r"\s*template\s*<[^(){};=]+>\s*",
        prefix,
    ):
        return True
    return not prefix.strip()


def go_compilation_unit_prefix_is_valid(
    prefix: str,
    original_prefix: str,
    *,
    go_keyword: str,
) -> bool:
    """Require one Go package clause and conservative prior declarations."""
    identifier = rf"(?!(?:{go_keyword})\b)[A-Za-z_]\w*"
    package_clause = rf"\s*package[ \t]+{identifier}[ \t]*(?:;[ \t]*|\r?\n)"
    string_literal = r'(?:`[^`]*`|"(?:\\.|[^"\\\r\n])*")'
    rune_literal = r"'(?:\\.|[^'\\\r\n])'"
    scalar_literal = r"(?:[-+]?\d+(?:\.\d+)?|false|nil|true)"
    declaration = (
        rf"(?:var|const)[ \t]+(?P<name>{identifier})[ \t]*=[ \t]*"
        rf"(?:{string_literal}|{rune_literal}|{scalar_literal})[ \t]*;?[ \t]*(?:\r?\n|$)"
    )
    grammar_match = re.fullmatch(rf"{package_clause}\s*(?:{declaration}\s*)*", original_prefix)
    names = tuple(match.group("name") for match in re.finditer(declaration, original_prefix))
    masked_grammar = rf"{package_clause}\s*(?:(?:var|const)[^\r\n]*(?:\r?\n|$)\s*)*"
    return (
        len(prefix) == len(original_prefix)
        and grammar_match is not None
        and len(set(names)) == len(names)
        and re.fullmatch(masked_grammar, prefix) is not None
    )


def region_is_ignorable(source: str, suffix: str, *, allow_semicolon: bool = False) -> bool:
    """Whether a source region contains only whitespace, comments, and an optional semicolon."""
    normalized = source_without_comments(source, suffix)
    grammar = r"\s*;?\s*" if allow_semicolon else r"\s*"
    return re.fullmatch(grammar, normalized) is not None


def _matching_brace(source: str, opening: int) -> int | None:
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return position
            if depth < 0:
                return None
    return None


def top_level_braced_declaration_is_final(
    source: str,
    original_source: str,
    header_start: int,
    suffix: str,
    declaration_kind: str,
    header_limit: int,
) -> bool:
    """Require a top-level braced declaration to own the remaining compilation unit."""
    body_start = source.find("{", header_start, min(len(source), header_start + header_limit))
    if body_start < 0:
        return False
    body_end = _matching_brace(source, body_start)
    if body_end is None:
        return False
    tail = original_source[body_end + 1 :]
    if declaration_kind != "function" and suffix in _C_FAMILY_TYPE_SUFFIXES:
        return re.fullmatch(r"\s*;\s*", source_without_comments(tail, suffix)) is not None
    return region_is_ignorable(
        tail,
        suffix,
        allow_semicolon=suffix in _OPTIONAL_EMPTY_STATEMENT_SUFFIXES,
    )


def nested_method_owns_type_body(
    source: str,
    original_source: str,
    enclosing_start: int,
    method_start: int,
    method_header_start: int,
    suffix: str,
    enclosing_kind: str,
    header_limit: int,
    *,
    java_keyword: str,
) -> bool:
    """Require a nested method to account for its complete enclosing type body."""
    enclosing_end = _matching_brace(source, enclosing_start)
    body_start = source.find(
        "{",
        method_header_start,
        min(len(source), method_header_start + header_limit),
    )
    method_end = None if body_start < 0 else _matching_brace(source, body_start)
    if enclosing_end is None or method_end is None or method_end > enclosing_end:
        return False
    leading = source_without_comments(
        original_source[enclosing_start + 1 : method_start],
        suffix,
    )
    if suffix == ".java" and enclosing_kind == "enum":
        identifier = rf"(?!(?:{java_keyword})\b)[A-Za-z_]\w*"
        if re.fullmatch(rf"\s*(?:{identifier}(?:\s*,\s*{identifier})*)?\s*;\s*", leading) is None:
            return False
    elif leading.strip():
        return False
    return region_is_ignorable(
        original_source[method_end + 1 : enclosing_end],
        suffix,
    ) and region_is_ignorable(original_source[enclosing_end + 1 :], suffix)
