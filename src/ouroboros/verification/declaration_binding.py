"""Bind structural evidence to caller-authored declaration kinds."""

from __future__ import annotations

import ast
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
_INTERFACE_SUFFIXES = frozenset({".cs", ".java", ".kt", ".kts"})
_STRUCT_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cs", ".h", ".hpp", ".mm", ".rs", ".swift"})
_C_LIKE_FUNCTION_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cs", ".h", ".hpp", ".java", ".mm"})
_CPP_SUFFIXES = frozenset({".cc", ".cpp", ".h", ".hpp", ".mm"})
_BRACED_TYPE_SUFFIXES = {
    "class": frozenset(
        {
            ".cc",
            ".cpp",
            ".cs",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".jsx",
            ".kt",
            ".kts",
            ".mm",
            ".swift",
            ".ts",
            ".tsx",
        }
    ),
    "interface": frozenset({".cs", ".go", ".java", ".kt", ".kts"}),
    "struct": frozenset({".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".mm", ".rs", ".swift"}),
    "trait": frozenset({".rs"}),
}
_BRACED_FUNCTION_SUFFIXES = frozenset(
    {
        ".bash",
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
        ".pl",
        ".rs",
        ".sh",
        ".swift",
        ".ts",
        ".tsx",
        ".zsh",
    }
)
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
    r"\([^;{{}}\r\n]*\)[ \t]*(?:throws\s+[^;{{\r\n]+)?\{{"
)

_CPP_BUILTIN_PARAMETER_TYPES = frozenset(
    {
        "bool",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "double",
        "float",
        "int",
        "long",
        "short",
        "signed",
        "unsigned",
        "void",
        "wchar_t",
    }
)
_CPP_EXPRESSION_WORDS = frozenset(
    {
        "and",
        "and_eq",
        "bitand",
        "bitor",
        "compl",
        "not",
        "not_eq",
        "or",
        "or_eq",
        "xor",
        "xor_eq",
    }
)
_C_LIKE_DECLARATION_MODIFIERS = frozenset(
    {
        "abstract",
        "async",
        "explicit",
        "extern",
        "final",
        "inline",
        "internal",
        "override",
        "private",
        "protected",
        "public",
        "sealed",
        "static",
        "virtual",
    }
)

_DECLARATION_HEADER_LIMIT = 4096
_NESTED_DECLARATION = re.compile(
    r"\b(?:class|interface|struct|trait)\s+[A-Za-z_]\w*"
    r"|\b(?:def|fn|func|function|sub)\s+[A-Za-z_]\w*"
    r"|(?m:^[ \t]*(?!(?:extends|implements|where)\b)"
    r"(?:[A-Za-z_]\w*[ \t]+){0,8}[A-Za-z_]\w*[ \t]*\()"
)


def _matching_delimiter(
    source: str,
    opening_position: int,
    opening: str,
    closing: str,
) -> int | None:
    """Return the matching delimiter in already-masked source, if complete."""
    if opening_position < 0 or source[opening_position : opening_position + 1] != opening:
        return None
    depth = 0
    for position in range(opening_position, len(source)):
        character = source[position]
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return position
            if depth < 0:
                return None
    return None


def _type_body_header_is_valid(header: str, kind: str, suffix: str) -> bool:
    """Recognize a conservative executable subset of braced type headers."""
    if suffix == ".go":
        return re.fullmatch(rf"\s+{re.escape(kind)}\s*", header) is not None
    if not header.strip():
        return True
    if any(token in header for token in ("=", ";", "{", "}", "@", "<", ">")):
        return False
    if suffix in _CPP_SUFFIXES:
        return (
            re.fullmatch(
                r"\s*(?:final\s*)?(?::\s*(?:[A-Za-z_]\w*\s+)*"
                r"[A-Za-z_]\w*(?:::\w+)*(?:\s*<[^(){};=]+>)?"
                r"(?:\s*,\s*(?:[A-Za-z_]\w*\s+)*[A-Za-z_]\w*"
                r"(?:::\w+)*(?:\s*<[^(){};=]+>)?)*)?\s*",
                header,
            )
            is not None
        )
    if suffix == ".java":
        return (
            re.fullmatch(
                r"\s*(?:<[^(){};=@]+>\s*)?"
                r"(?:(?:extends|implements|permits)\s+"
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\s*<[^(){};=@]+>)?"
                r"(?:\s*,\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
                r"(?:\s*<[^(){};=@]+>)?)*\s*)*",
                header,
            )
            is not None
        )
    if suffix in {".js", ".jsx"}:
        return (
            re.fullmatch(
                r"\s*(?:extends\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)?\s*",
                header,
            )
            is not None
        )
    if suffix in {".ts", ".tsx"}:
        return (
            re.fullmatch(
                r"\s*(?:<[^(){};=@]+>\s*)?"
                r"(?:(?:extends|implements)\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*"
                r"(?:\s*<[^(){};=@]+>)?(?:\s*,\s*[A-Za-z_$][\w$]*"
                r"(?:\.[A-Za-z_$][\w$]*)*(?:\s*<[^(){};=@]+>)?)*\s*)*",
                header,
            )
            is not None
        )
    if suffix == ".cs":
        return (
            re.fullmatch(
                r"\s*(?:<[^(){};=@]+>\s*)?(?::\s*[A-Za-z_]\w*"
                r"(?:\.[A-Za-z_]\w*)*(?:\s*<[^(){};=@]+>)?"
                r"(?:\s*,\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
                r"(?:\s*<[^(){};=@]+>)?)*)?\s*",
                header,
            )
            is not None
        )
    if suffix == ".swift":
        return (
            re.fullmatch(
                r"\s*(?:<[^(){};=@]+>\s*)?(?::\s*[A-Za-z_]\w*"
                r"(?:\.[A-Za-z_]\w*)*(?:\s*,\s*[A-Za-z_]\w*"
                r"(?:\.[A-Za-z_]\w*)*)*)?(?:\s+where\s+[^(){};=@]+)?\s*",
                header,
            )
            is not None
        )
    if suffix in {".kt", ".kts"}:
        return (
            re.fullmatch(
                r"\s*(?:<[^{};=@]+>\s*)?(?:\([^{};=@]*\)\s*)?"
                r"(?::\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
                r"(?:\s*<[^{};=@]+>)?(?:\([^{};=@]*\))?"
                r"(?:\s*,\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
                r"(?:\s*<[^{};=@]+>)?(?:\([^{};=@]*\))?)*)?"
                r"(?:\s+where\s+[^{};=@]+)?\s*",
                header,
            )
            is not None
        )
    if suffix == ".rs":
        return (
            re.fullmatch(
                r"\s*(?:<[^(){};=@]+>\s*)?(?:where\s+[^(){};=@]+)?\s*",
                header,
            )
            is not None
        )
    return False


def _function_body_header_is_valid(header: str, suffix: str) -> bool:
    """Recognize a conservative executable subset between parameters and body."""
    if any(token in header for token in ("=", ";", "{", "}", "@")):
        return False
    if suffix in {".bash", ".js", ".jsx", ".pl", ".r", ".sh", ".zsh"}:
        return not header.strip()
    if suffix in {".ts", ".tsx"}:
        return re.fullmatch(r"\s*(?::\s*[^=;{}@]+)?\s*", header) is not None
    if suffix == ".go":
        return re.fullmatch(r"[\sA-Za-z0-9_.*\[\],()<>{}|~:/+-]*", header) is not None
    if suffix == ".rs":
        return (
            re.fullmatch(
                r"\s*(?:->\s*[^=;{}@]+)?(?:\s+where\s+[^=;{}@]+)?\s*",
                header,
            )
            is not None
        )
    if suffix == ".swift":
        return (
            re.fullmatch(
                r"\s*(?:(?:async|rethrows|throws)\s+)*(?:->\s*[^=;{}@]+)?"
                r"(?:\s+where\s+[^=;{}@]+)?\s*",
                header,
            )
            is not None
        )
    if suffix in {".kt", ".kts"}:
        return (
            re.fullmatch(
                r"\s*(?::\s*[^=;{}@]+)?(?:\s+where\s+[^=;{}@]+)?\s*",
                header,
            )
            is not None
        )
    if suffix == ".java":
        return (
            re.fullmatch(
                r"\s*(?:throws\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
                r"(?:\s*,\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)*)?\s*",
                header,
            )
            is not None
        )
    if suffix == ".cs":
        return re.fullmatch(r"\s*(?:where\s+[^=;{}@]+)?\s*", header) is not None
    if suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".mm"}:
        return (
            re.fullmatch(
                r"\s*(?:(?:const|constexpr|final|noexcept|override|volatile)\s*)*"
                r"(?:->\s*[^=;{}@]+)?(?:requires\s+[^=;{}@]+)?\s*",
                header,
            )
            is not None
        )
    return False


def _declaration_body_is_valid(body: str, declaration_kind: str, suffix: str) -> bool:
    """Admit only body syntax the verifier can prove without a language parser."""
    stripped = body.strip()
    if not stripped:
        return True
    if declaration_kind != "function":
        return False
    if suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".mm"}:
        return (
            re.fullmatch(
                r"return\s+(?:[A-Za-z_]\w*|[-+]?\d+|false|nullptr|true)\s*;",
                stripped,
            )
            is not None
        )
    return False


def _complete_braced_body(
    source: str,
    header_start: int,
    declaration_kind: str,
    suffix: str,
) -> bool:
    """Require one bounded declaration header and its complete braced body."""
    search_end = min(len(source), header_start + _DECLARATION_HEADER_LIMIT)
    body_start = source.find("{", header_start, search_end)
    if body_start < 0:
        return False
    header = source[header_start:body_start]
    if ";" in header or "}" in header or _NESTED_DECLARATION.search(header):
        return False
    if declaration_kind == "function":
        if not _function_body_header_is_valid(header, suffix):
            return False
    elif not _type_body_header_is_valid(header, declaration_kind, suffix):
        return False
    body_end = _matching_delimiter(source, body_start, "{", "}")
    if body_end is None:
        return False
    if not _declaration_body_is_valid(
        source[body_start + 1 : body_end],
        declaration_kind,
        suffix,
    ):
        return False
    return all(
        _all_delimiters_balanced(source, opening, closing)
        for opening, closing in (("{", "}"), ("(", ")"), ("[", "]"))
    )


def _all_delimiters_balanced(source: str, opening: str, closing: str) -> bool:
    """Fail a masked source closed when a basic delimiter is unbalanced."""
    depth = 0
    for character in source:
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _complete_ruby_block(source: str, declaration_start: int) -> bool:
    """Conservatively balance a Ruby class or method block through its ``end``."""
    body_start = source.find("\n", declaration_start)
    if body_start < 0:
        return False
    depth = 0
    for token in re.finditer(
        r"\b(?:begin|case|class|def|do|for|if|module|unless|until|while|end)\b",
        source[declaration_start:],
    ):
        if token.group() == "end":
            depth -= 1
            if depth == 0:
                token_start = declaration_start + token.start()
                return not source[body_start + 1 : token_start].strip()
            if depth < 0:
                return False
        else:
            depth += 1
    return False


def _complete_lua_function(
    source: str,
    declaration_start: int,
    body_start: int,
) -> bool:
    """Conservatively balance a Lua function and nested block terminators."""
    stack: list[str] = []
    for token in re.finditer(
        r"\b(?:do|function|if|repeat|end|until)\b",
        source[declaration_start:],
    ):
        word = token.group()
        if word == "repeat":
            stack.append("until")
        elif word in {"do", "function", "if"}:
            stack.append("end")
        elif not stack or stack[-1] != word:
            return False
        else:
            stack.pop()
            if not stack:
                token_start = declaration_start + token.start()
                return not source[body_start + 1 : token_start].strip()
    return False


def _complete_haskell_class(source: str, target_end: int) -> bool:
    """Require the bounded core of a Haskell typeclass declaration."""
    tail = source[target_end : target_end + _DECLARATION_HEADER_LIMIT]
    header = re.match(
        r"[ \t]+[a-z_]\w*(?:[ \t]+[a-z_]\w*)*[ \t]+where[ \t]*(?:\r?\n|$)",
        tail,
    )
    return header is not None and not source[target_end + header.end() :].strip()


def _complete_parameter_list(source: str, target_end: int) -> int | None:
    """Return the closing parenthesis for a bounded function header."""
    search_end = min(len(source), target_end + _DECLARATION_HEADER_LIMIT)
    parameters_start = source.find("(", target_end, search_end)
    if parameters_start < 0:
        return None
    header = source[target_end:parameters_start]
    if any(marker in header for marker in ";{}") or _NESTED_DECLARATION.search(header):
        return None
    return _matching_delimiter(source, parameters_start, "(", ")")


def _function_parameters_are_valid(
    source: str,
    original_source: str,
    target_end: int,
    parameters_end: int,
    suffix: str,
) -> bool:
    """Admit only parameter syntax provable without a language parser."""
    parameters_start = source.find("(", target_end, parameters_end + 1)
    if parameters_start < 0:
        return False
    masked = source[parameters_start + 1 : parameters_end]
    original = original_source[parameters_start + 1 : parameters_end]
    if suffix in _CPP_SUFFIXES:
        return _cpp_parameters_are_declarations(masked, original)
    if suffix == ".c":
        return masked.strip() in {"", "void"} and original.strip() in {"", "void"}
    return not masked.strip() and not original.strip()


def _complete_type_definition(
    source: str,
    original_source: str,
    match: re.Match[str],
    kind: str,
    suffix: str,
) -> bool:
    """Require a complete definition for every admitted structural type grammar."""
    if kind == "class" and suffix == ".py":
        try:
            ast.parse(original_source)
        except (SyntaxError, ValueError):
            return False
        return True
    if kind == "class" and suffix == ".rb":
        line_end = source.find("\n", match.end("target"))
        if line_end < 0:
            return False
        if (
            re.fullmatch(
                r"\s*(?:<\s*[A-Z]\w*(?:::[A-Z]\w*)*)?\s*",
                source[match.end("target") : line_end],
            )
            is None
        ):
            return False
        return _complete_ruby_block(source, match.start())
    if kind == "class" and suffix == ".hs":
        return _complete_haskell_class(source, match.end("target"))
    if kind in {"class", "interface"} and suffix in {".kt", ".kts"}:
        line_end = source.find("\n", match.end("target"))
        if line_end < 0:
            line_end = len(source)
        if not source[match.end("target") : line_end].strip():
            return (
                all(
                    _all_delimiters_balanced(source, opening, closing)
                    for opening, closing in (("{", "}"), ("(", ")"), ("[", "]"))
                )
                and not source[line_end:].strip()
            )
    if suffix in _BRACED_TYPE_SUFFIXES.get(kind, ()):
        return _complete_braced_body(
            source,
            match.end("target"),
            kind,
            suffix,
        )
    return False


def _complete_ruby_function(
    source: str,
    original_source: str,
    match: re.Match[str],
) -> bool:
    line_end = source.find("\n", match.end("target"))
    if line_end < 0:
        line_end = len(source)
    parameters_start = source.find("(", match.end("target"), line_end)
    if parameters_start >= 0:
        parameters_end = _matching_delimiter(source, parameters_start, "(", ")")
        if parameters_end is None or parameters_end > line_end:
            return False
        if (
            source[parameters_start + 1 : parameters_end].strip()
            or original_source[parameters_start + 1 : parameters_end].strip()
        ):
            return False
        if (
            source[match.end("target") : parameters_start].strip()
            or source[parameters_end + 1 : line_end].strip()
        ):
            return False
    elif source[match.end("target") : line_end].strip():
        return False
    return _complete_ruby_block(source, match.start())


def _complete_expression_function(source: str, match: re.Match[str]) -> bool:
    """Validate an R function expression with either a body block or expression."""
    parameters_end = _complete_parameter_list(source, match.end("target"))
    if parameters_end is None or not _function_parameters_are_valid(
        source,
        source,
        match.end("target"),
        parameters_end,
        ".r",
    ):
        return False
    line_end = source.find("\n", parameters_end)
    if line_end < 0:
        line_end = len(source)
    implementation = source[parameters_end + 1 : line_end]
    body_offset = implementation.find("{")
    if body_offset >= 0:
        return _complete_braced_body(
            source,
            parameters_end + 1,
            "function",
            ".r",
        )
    return (
        re.fullmatch(
            r"(?:[A-Za-z_.]\w*|[-+]?\d+(?:\.\d+)?|FALSE|NULL|TRUE)",
            implementation.strip(),
        )
        is not None
    )


def _complete_function_definition(
    source: str,
    original_source: str,
    match: re.Match[str],
    suffix: str,
) -> bool:
    """Require a complete implementation for every admitted function grammar."""
    if suffix == ".py":
        try:
            ast.parse(original_source)
        except (SyntaxError, ValueError):
            return False
        return True
    if suffix == ".pyi":
        return False
    if suffix == ".rb":
        return _complete_ruby_function(source, original_source, match)
    if suffix == ".lua":
        parameters_end = _complete_parameter_list(source, match.end("target"))
        return (
            parameters_end is not None
            and _function_parameters_are_valid(
                source,
                original_source,
                match.end("target"),
                parameters_end,
                suffix,
            )
            and _complete_lua_function(source, match.start(), parameters_end)
        )
    if suffix == ".r":
        return _complete_expression_function(source, match)
    if suffix in {".kt", ".kts"}:
        parameters_end = _complete_parameter_list(source, match.end("target"))
        if parameters_end is None or not _function_parameters_are_valid(
            source,
            original_source,
            match.end("target"),
            parameters_end,
            suffix,
        ):
            return False
        line_end = source.find("\n", parameters_end)
        if line_end < 0:
            line_end = len(source)
        implementation = source[parameters_end + 1 : line_end]
        body_offset = implementation.find("{")
        if body_offset >= 0:
            return _complete_braced_body(
                source,
                parameters_end + 1,
                "function",
                suffix,
            )
        expression_offset = implementation.find("=")
        if expression_offset < 0:
            return False
        return (
            re.fullmatch(
                r"(?:[A-Za-z_]\w*|[-+]?\d+(?:\.\d+)?|false|null|true|Unit)",
                implementation[expression_offset + 1 :].strip(),
            )
            is not None
        )
    if suffix in _BRACED_FUNCTION_SUFFIXES:
        parameters_end = _complete_parameter_list(source, match.end("target"))
        if parameters_end is None:
            if suffix != ".pl":
                return False
            header_start = match.end("target")
        else:
            if not _function_parameters_are_valid(
                source,
                original_source,
                match.end("target"),
                parameters_end,
                suffix,
            ):
                return False
            header_start = parameters_end + 1
        return _complete_braced_body(
            source,
            header_start,
            "function",
            suffix,
        )
    return False


def _inside_cpp_template_parameter_list(source: str, position: int) -> bool:
    """Whether ``position`` is inside a preceding ``template <...>`` list.

    C++ angle-bracket parsing is context-sensitive. This deliberately uses a
    conservative balance check over already-masked source: if a template list
    cannot be shown to close before the target, its ``class`` token cannot mint
    declaration evidence.
    """
    for introducer in re.finditer(r"\btemplate\s*<", source[:position]):
        depth = 1
        cursor = introducer.end()
        while cursor < position and depth:
            if source[cursor] == "<":
                depth += 1
            elif source[cursor] == ">":
                depth -= 1
            cursor += 1
        if depth:
            return True
    return False


def _split_cpp_parameters(parameters: str) -> tuple[str, ...] | None:
    """Split a flat C++ parameter list, failing closed on nested expressions."""
    if any(character in parameters for character in "(){}[]"):
        return None
    return tuple(part.strip() for part in parameters.split(","))


def _cpp_parameters_are_declarations(masked: str, original: str) -> bool:
    """Reject C++ initializer expressions that resemble function parameters."""
    if not masked.strip():
        return not original.strip()
    if '"' in original or "'" in original:
        return False
    parameters = _split_cpp_parameters(masked)
    if parameters is None or any(not parameter for parameter in parameters):
        return False
    for parameter in parameters:
        declaration = parameter.split("=", 1)[0].strip()
        if declaration == "...":
            continue
        if re.search(r"\b(?:false|nullptr|true)\b|(?:^|\W)(?:0[xX][0-9A-Fa-f]+|\d)", declaration):
            return False
        if re.search(r"[+/%!?|^~]", declaration) or "&&" in declaration:
            return False
        identifiers = re.findall(r"\b[A-Za-z_]\w*\b", declaration)
        if any(identifier in _CPP_EXPRESSION_WORDS for identifier in identifiers):
            return False
        if any(identifier in _CPP_BUILTIN_PARAMETER_TYPES for identifier in identifiers):
            continue
        if re.search(r"(?:->|::)|[.*&<>:-]", declaration):
            return False
        if len(identifiers) >= 2 and re.fullmatch(
            r"[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)+",
            declaration,
        ):
            continue
        return False
    return True


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
    original_source: str,
    target_span: tuple[int, int],
    target: str,
    kind: str,
    file_path: str,
) -> bool:
    escaped = re.escape(target)
    suffix = os.path.splitext(file_path)[1].casefold()
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
            if (
                kind == "class"
                and suffix in _CPP_SUFFIXES
                and _inside_cpp_template_parameter_list(source, match.start())
            ):
                continue
            target_start = match.start("target")
            declaration_boundary = max(source.rfind(marker, 0, target_start) for marker in ";{}")
            target_prefix = source[declaration_boundary + 1 : target_start]
            if kind == "function" and re.search(
                r"\b(?:delegate|operator|typedef|using)\b",
                target_prefix,
            ):
                continue
            if kind == "function" and suffix in _C_LIKE_FUNCTION_SUFFIXES:
                prefix_words = re.findall(r"\b[A-Za-z_]\w*\b", target_prefix)
                if prefix_words and all(
                    word in _C_LIKE_DECLARATION_MODIFIERS for word in prefix_words
                ):
                    continue
            if kind == "function" and suffix in _CPP_SUFFIXES:
                parameters_start = source.find("(", match.end("target"), match.end())
                parameters_end = source.rfind(")", parameters_start, match.end())
                if parameters_start < 0 or parameters_end < parameters_start:
                    continue
                if not _cpp_parameters_are_declarations(
                    source[parameters_start + 1 : parameters_end],
                    original_source[parameters_start + 1 : parameters_end],
                ):
                    continue
            if kind == "function" and suffix in {".kt", ".kts", ".swift"}:
                line_start = source.rfind("\n", 0, match.start()) + 1
                declaration_prefix = source[line_start : match.start()]
                if suffix in {".kt", ".kts"} and re.search(
                    r"\b(?:abstract|expect|external)\b",
                    declaration_prefix,
                ):
                    continue
            if kind in {"class", "interface"} and suffix in {".kt", ".kts"}:
                line_start = source.rfind("\n", 0, match.start()) + 1
                if re.search(
                    r"\b(?:expect|external)\b",
                    source[line_start : match.start()],
                ):
                    continue
            if kind in {"class", "interface", "struct", "trait"} and not _complete_type_definition(
                source,
                original_source,
                match,
                kind,
                suffix,
            ):
                continue
            if kind == "function" and not _complete_function_definition(
                source,
                original_source,
                match,
                suffix,
            ):
                continue
            return True
    return False


def match_has_bound_declaration_kind(
    pattern: re.Pattern,
    match: re.Match,
    source_texts: tuple[str, str],
    assertion: SpecAssertion,
    evidence_target: str,
    finite_witnesses: tuple[tuple[int, int, int], ...],
    file_path: str,
) -> bool:
    """Whether the exact matched occurrence has the caller-requested kind."""
    searched_text, original_text = source_texts
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
        _source_span_has_declaration_kind(
            searched_text,
            original_text,
            span,
            evidence_target,
            kind,
            file_path,
        )
        for span in candidate_spans
    )
