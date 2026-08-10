"""Conservative C++ parameter declarations for parser-free evidence."""

from __future__ import annotations

import re

CPP_BUILTIN_PARAMETER_TYPES = frozenset(
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
CPP_BUILTIN_TYPE_SEQUENCES = frozenset(
    {
        ("bool",),
        ("char",),
        ("char8_t",),
        ("char16_t",),
        ("char32_t",),
        ("double",),
        ("float",),
        ("int",),
        ("long",),
        ("long", "double"),
        ("long", "int"),
        ("long", "long"),
        ("long", "long", "int"),
        ("short",),
        ("short", "int"),
        ("signed",),
        ("signed", "char"),
        ("signed", "int"),
        ("signed", "long"),
        ("signed", "long", "int"),
        ("signed", "long", "long"),
        ("signed", "long", "long", "int"),
        ("signed", "short"),
        ("signed", "short", "int"),
        ("unsigned",),
        ("unsigned", "char"),
        ("unsigned", "int"),
        ("unsigned", "long"),
        ("unsigned", "long", "int"),
        ("unsigned", "long", "long"),
        ("unsigned", "long", "long", "int"),
        ("unsigned", "short"),
        ("unsigned", "short", "int"),
        ("void",),
        ("wchar_t",),
    }
)
_CPP_EXPRESSION_WORDS = frozenset(
    {"and", "and_eq", "bitand", "bitor", "compl", "not", "not_eq", "or", "or_eq", "xor", "xor_eq"}
)
_CPP_PARAMETER_KEYWORDS = frozenset(
    [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "import",
        "inline",
        "int",
        "long",
        "module",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ]
)
_CPP_TERMINAL_TYPE_WORDS = CPP_BUILTIN_PARAMETER_TYPES | frozenset({"const", "volatile"})


def _split_parameters(parameters: str) -> tuple[str, ...] | None:
    """Split a flat parameter list, failing closed on nested expressions."""
    if any(character in parameters for character in "(){}[]"):
        return None
    return tuple(part.strip() for part in parameters.split(","))


def _parameter_name(declaration: str) -> str | None:
    """Return an unambiguous name from an admitted flat declaration."""
    identifiers = re.findall(r"\b[A-Za-z_]\w*\b", declaration)
    if len(identifiers) < 2:
        return None
    candidate = identifiers[-1]
    if candidate in _CPP_PARAMETER_KEYWORDS:
        return None
    prefix = identifiers[:-1]
    type_words = tuple(word for word in prefix if word not in {"const", "volatile"})
    if not type_words or all(
        word in {"class", "enum", "struct", "typename"} for word in type_words
    ):
        return None
    return candidate


def cpp_parameters_are_declarations(masked: str, original: str) -> bool:
    """Reject expressions and invalid names that resemble C++ parameters."""
    if not masked.strip():
        return not original.strip()
    if '"' in original or "'" in original:
        return False
    parameters = _split_parameters(masked)
    if parameters is None or any(not parameter for parameter in parameters):
        return False
    names: set[str] = set()
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
        if identifiers and identifiers[-1] in _CPP_PARAMETER_KEYWORDS - _CPP_TERMINAL_TYPE_WORDS:
            return False
        name = _parameter_name(declaration)
        if any(identifier in CPP_BUILTIN_PARAMETER_TYPES for identifier in identifiers):
            type_words = tuple(
                identifier
                for identifier in identifiers[: -1 if name is not None else None]
                if identifier not in {"const", "volatile"}
            )
            if type_words not in CPP_BUILTIN_TYPE_SEQUENCES or (
                type_words == ("void",) and name is not None
            ):
                return False
        else:
            if re.search(r"(?:->|::)|[.*&<>:-]", declaration):
                return False
            if (
                len(identifiers) < 2
                or re.fullmatch(
                    r"[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)+",
                    declaration,
                )
                is None
            ):
                return False
        if name is not None:
            if name in names:
                return False
            names.add(name)
    return True


def cpp_parameter_bindings(
    parameters: str,
) -> tuple[tuple[str, tuple[str, ...], bool], ...]:
    """Return name, normalized type words, and pointer shape for parameters."""
    parts = _split_parameters(parameters)
    if parts is None:
        return ()
    bindings: list[tuple[str, tuple[str, ...], bool]] = []
    for parameter in parts:
        declaration = parameter.split("=", 1)[0].strip()
        name = _parameter_name(declaration)
        if name is None:
            continue
        identifiers = re.findall(r"\b[A-Za-z_]\w*\b", declaration)
        type_words = tuple(
            identifier
            for identifier in identifiers[:-1]
            if identifier not in {"class", "const", "enum", "struct", "typename", "volatile"}
        )
        bindings.append((name, type_words, "*" in declaration))
    return tuple(bindings)
