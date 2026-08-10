"""Bind structural evidence to caller-authored declaration kinds."""

from __future__ import annotations

import ast
import os
import re

from ouroboros.verification.binding import (
    acceptance_declaration_kind,
    literal_is_bound,
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

_TYPE_PREFIX_MODIFIERS = {
    ".cc": frozenset(),
    ".cpp": frozenset(),
    ".cs": frozenset(
        {
            "abstract",
            "file",
            "internal",
            "partial",
            "public",
            "sealed",
            "static",
        }
    ),
    ".h": frozenset(),
    ".hpp": frozenset(),
    ".hs": frozenset(),
    ".java": frozenset(
        {
            "abstract",
            "final",
            "non-sealed",
            "public",
            "sealed",
            "strictfp",
        }
    ),
    ".js": frozenset({"default", "export"}),
    ".jsx": frozenset({"default", "export"}),
    ".kt": frozenset(
        {
            "abstract",
            "actual",
            "annotation",
            "data",
            "enum",
            "final",
            "inner",
            "internal",
            "open",
            "private",
            "protected",
            "public",
            "sealed",
            "value",
        }
    ),
    ".kts": frozenset(
        {
            "abstract",
            "actual",
            "annotation",
            "data",
            "enum",
            "final",
            "inner",
            "internal",
            "open",
            "private",
            "protected",
            "public",
            "sealed",
            "value",
        }
    ),
    ".mm": frozenset(),
    ".py": frozenset(),
    ".pyi": frozenset(),
    ".rb": frozenset(),
    ".rs": frozenset({"pub", "unsafe"}),
    ".swift": frozenset({"fileprivate", "final", "internal", "open", "private", "public"}),
    ".ts": frozenset({"abstract", "default", "declare", "export"}),
    ".tsx": frozenset({"abstract", "default", "declare", "export"}),
}
_TYPE_KIND_PREFIX_MODIFIERS = {
    (".cs", "enum"): frozenset({"file", "internal", "public"}),
    (".cs", "interface"): frozenset({"file", "internal", "partial", "public"}),
    (".cs", "record"): frozenset({"abstract", "file", "internal", "partial", "public", "sealed"}),
    (".cs", "struct"): frozenset({"file", "internal", "partial", "public"}),
    (".java", "enum"): frozenset({"public", "strictfp"}),
    (".java", "interface"): frozenset({"abstract", "non-sealed", "public", "sealed", "strictfp"}),
    (".java", "record"): frozenset({"public", "strictfp"}),
    (".kt", "class"): frozenset(
        {
            "abstract",
            "actual",
            "annotation",
            "enum",
            "final",
            "internal",
            "open",
            "private",
            "public",
            "sealed",
        }
    ),
    (".kt", "interface"): frozenset(
        {"abstract", "actual", "internal", "private", "public", "sealed"}
    ),
    (".kts", "class"): frozenset(
        {
            "abstract",
            "actual",
            "annotation",
            "enum",
            "final",
            "internal",
            "open",
            "private",
            "public",
            "sealed",
        }
    ),
    (".kts", "interface"): frozenset(
        {"abstract", "actual", "internal", "private", "public", "sealed"}
    ),
    (".rs", "struct"): frozenset({"pub"}),
    (".rs", "trait"): frozenset({"pub", "unsafe"}),
    (".swift", "struct"): frozenset({"fileprivate", "internal", "private", "public"}),
}
_FUNCTION_PREFIX_MODIFIERS = {
    ".go": frozenset(),
    ".js": frozenset({"async", "default", "export"}),
    ".jsx": frozenset({"async", "default", "export"}),
    ".kt": frozenset({"actual", "internal", "private", "public"}),
    ".kts": frozenset({"actual", "internal", "private", "public"}),
    ".lua": frozenset({"local"}),
    ".pl": frozenset(),
    ".py": frozenset({"async"}),
    ".pyi": frozenset({"async"}),
    ".rb": frozenset(),
    ".rs": frozenset({"async", "const", "extern", "pub", "unsafe"}),
    ".swift": frozenset(
        {
            "fileprivate",
            "internal",
            "private",
            "public",
        }
    ),
    ".ts": frozenset({"async", "default", "export"}),
    ".tsx": frozenset({"async", "default", "export"}),
}
_ACCESS_MODIFIERS = frozenset(
    {"file", "fileprivate", "internal", "open", "private", "protected", "public"}
)
_PREFIX_TOKEN = r"[A-Za-z_][\w-]*(?:\([^()\r\n]*\))?"

_DECLARATION_HEADER_LIMIT = 4096
_NESTED_DECLARATION = re.compile(
    r"\b(?:class|interface|struct|trait)\s+[A-Za-z_]\w*"
    r"|\b(?:def|fn|func|function|sub)\s+[A-Za-z_]\w*"
    r"|(?m:^[ \t]*(?!(?:extends|implements|where)\b)"
    r"(?:[A-Za-z_]\w*[ \t]+){0,8}[A-Za-z_]\w*[ \t]*\()"
)


def _modifier_base(token: str) -> str:
    return token.split("(", 1)[0]


def _type_modifier_combination_is_valid(
    bases: tuple[str, ...],
    suffix: str,
    kind: str,
) -> bool:
    """Reject language-invalid combinations among individually valid modifiers."""
    modifiers = frozenset(bases)
    if suffix == ".cs":
        # A C# static class is sealed by definition, but neither keyword may be
        # written alongside ``static``; abstract and sealed are also exclusive.
        return not any(
            conflict.issubset(modifiers)
            for conflict in (
                frozenset({"abstract", "sealed"}),
                frozenset({"abstract", "static"}),
                frozenset({"sealed", "static"}),
            )
        )
    if suffix in {".kt", ".kts"}:
        # Kotlin permits at most one inheritance/modality modifier. Treating
        # these as independent allowlisted tokens admits non-compilable forms
        # such as ``sealed open class``.
        modality = frozenset({"abstract", "final", "open", "sealed"})
        special_class_kind = frozenset({"annotation", "enum"})
        return (
            len(modality.intersection(modifiers)) <= 1
            and len(special_class_kind.intersection(modifiers)) <= 1
            and not (
                kind == "class"
                and special_class_kind.intersection(modifiers)
                and modality.intersection(modifiers)
            )
        )
    return True


def _declaration_prefix_is_valid(match: re.Match[str], suffix: str, kind: str) -> bool:
    """Validate the complete same-line prefix captured before a declaration."""
    prefix = match.groupdict().get("prefix", "")
    tokens = re.findall(_PREFIX_TOKEN, prefix)
    bases = tuple(_modifier_base(token) for token in tokens)
    allowed = (
        _FUNCTION_PREFIX_MODIFIERS.get(suffix, frozenset())
        if kind == "function"
        else _TYPE_KIND_PREFIX_MODIFIERS.get(
            (suffix, kind),
            _TYPE_PREFIX_MODIFIERS.get(suffix, frozenset()),
        )
    )
    if any(base not in allowed for base in bases) or len(set(bases)) != len(bases):
        return False
    if len(_ACCESS_MODIFIERS.intersection(bases)) > 1:
        return False
    conflicts = (
        frozenset({"abstract", "final"}),
        frozenset({"final", "open"}),
        frozenset({"final", "sealed"}),
        frozenset({"non-sealed", "sealed"}),
    )
    if any(conflict.issubset(bases) for conflict in conflicts):
        return False
    if kind != "function" and not _type_modifier_combination_is_valid(bases, suffix, kind):
        return False
    if suffix == ".rs":
        if kind == "function":
            order = {"pub": 0, "const": 1, "async": 1, "unsafe": 2, "extern": 3}
            return not {"async", "const"}.issubset(bases) and bases == tuple(
                sorted(bases, key=order.__getitem__)
            )
        if kind == "trait":
            return bases in {(), ("pub",), ("unsafe",), ("pub", "unsafe")}
        if kind == "struct":
            return bases in {(), ("pub",)}
    if suffix in {".js", ".jsx"} and kind == "class":
        return bases in {(), ("export",), ("export", "default")}
    if suffix in {".js", ".jsx"} and kind == "function":
        return bases in {
            (),
            ("async",),
            ("export",),
            ("export", "async"),
            ("export", "default"),
            ("export", "default", "async"),
        }
    if suffix in {".ts", ".tsx"} and kind == "class":
        return bases in {
            (),
            ("abstract",),
            ("declare",),
            ("export",),
            ("export", "abstract"),
            ("export", "declare"),
            ("export", "default"),
            ("export", "default", "abstract"),
        }
    if suffix in {".ts", ".tsx"} and kind == "function":
        return bases in {
            (),
            ("async",),
            ("declare",),
            ("export",),
            ("export", "async"),
            ("export", "declare"),
            ("export", "default"),
            ("export", "default", "async"),
        }
    return "default" not in bases or "export" in bases


def _c_like_function_prefix_is_valid(
    source: str,
    match: re.Match[str],
    suffix: str,
) -> bool:
    """Validate canonical modifiers before a C-like function return type."""
    words = re.findall(
        r"\b[A-Za-z_]\w*\b",
        source[match.start() : match.start("target")],
    )
    if not words:
        return False
    modifiers = words[:-1]
    if suffix == ".java":
        allowed = frozenset(
            {
                "abstract",
                "final",
                "native",
                "private",
                "protected",
                "public",
                "static",
                "strictfp",
                "synchronized",
            }
        )
    elif suffix == ".cs":
        allowed = frozenset(
            {
                "abstract",
                "async",
                "extern",
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
    elif suffix == ".c":
        allowed = frozenset({"extern", "inline", "register", "static"})
    else:
        allowed = frozenset({"extern", "inline", "static"}).union(_CPP_BUILTIN_PARAMETER_TYPES)
    if any(word not in allowed for word in modifiers) or len(set(modifiers)) != len(modifiers):
        return False
    return len(_ACCESS_MODIFIERS.intersection(modifiers)) <= 1


def _java_declaration_context_is_valid(
    source: str,
    match: re.Match[str],
    target: str,
    kind: str,
    file_path: str,
) -> bool:
    """Enforce Java rules that depend on both a declaration and its context."""
    if kind == "function":
        words = re.findall(
            r"\b[A-Za-z_]\w*\b",
            source[match.start() : match.start("target")],
        )
        return (
            bool(words)
            and words[-1] == "void"
            and not {
                "abstract",
                "native",
            }.intersection(words[:-1])
        )

    prefix = match.groupdict().get("prefix", "")
    modifiers = {_modifier_base(token) for token in re.findall(_PREFIX_TOKEN, prefix)}
    search_end = min(len(source), match.end("target") + _DECLARATION_HEADER_LIMIT)
    body_start = source.find("{", match.end("target"), search_end)
    if body_start < 0:
        return False
    header = source[match.end("target") : body_start]
    if {"non-sealed", "sealed"}.intersection(modifiers):
        return False
    if re.search(r"\bpermits\b", header) and "sealed" not in modifiers:
        return False
    filename_stem = os.path.splitext(os.path.basename(file_path))[0]
    return "public" not in modifiers or target == filename_stem


def _csharp_function_context_is_valid(source: str, match: re.Match[str]) -> bool:
    """Reject C# method bodies incompatible with their return type or modifiers."""
    words = re.findall(
        r"\b[A-Za-z_]\w*\b",
        source[match.start() : match.start("target")],
    )
    if not words or words[-1] != "void":
        return False
    modifiers = frozenset(words[:-1])
    if {"abstract", "extern", "override", "sealed"}.intersection(modifiers):
        return False
    if "static" in modifiers and "virtual" in modifiers:
        return False
    if "private" in modifiers and "virtual" in modifiers:
        return False
    braces = _enclosing_braces(source, match.start())
    if braces is None or len(braces) != 1:
        return False
    enclosing = _enclosing_type_context(source, braces[0], ".cs")
    if enclosing is None:
        return False
    enclosing_kind, enclosing_modifiers = enclosing
    if "static" in enclosing_modifiers and "static" not in modifiers:
        return False
    return "virtual" not in modifiers or (
        enclosing_kind != "struct" and not {"sealed", "static"}.intersection(enclosing_modifiers)
    )


def _enclosing_braces(source: str, position: int) -> tuple[int, ...] | None:
    """Return unmatched opening braces before ``position`` in masked source."""
    stack: list[int] = []
    for cursor, character in enumerate(source[:position]):
        if character == "{":
            stack.append(cursor)
        elif character == "}":
            if not stack:
                return None
            stack.pop()
    return tuple(stack)


def _compilation_unit_prefix_is_valid(prefix: str, suffix: str) -> bool:
    """Admit only a provable top-level context before a declaration."""
    qualified_name = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    if suffix == ".java":
        return (
            re.fullmatch(
                rf"\s*(?:package\s+{qualified_name}\s*;\s*)?"
                rf"(?:import\s+(?:static\s+)?{qualified_name}(?:\.\*)?\s*;\s*)*",
                prefix,
            )
            is not None
        )
    if suffix == ".cs":
        directive = (
            rf"extern\s+alias\s+[A-Za-z_]\w*"
            rf"|(?:global\s+)?using\s+(?:static\s+)?{qualified_name}"
            rf"|namespace\s+{qualified_name}"
        )
        return re.fullmatch(rf"\s*(?:(?:{directive})\s*;\s*)*", prefix) is not None

    boundary = max(prefix.rfind(";"), prefix.rfind("{"), prefix.rfind("}"))
    context = prefix[boundary + 1 :]
    if suffix in _CPP_SUFFIXES and re.fullmatch(
        r"\s*template\s*<[^(){};=]+>\s*",
        context,
    ):
        return True
    return not context.strip()


def _type_placement_is_valid(source: str, match: re.Match[str], suffix: str) -> bool:
    """Require a top-level type in a canonical compilation-unit context."""
    if suffix == ".py":
        return True
    braces = _enclosing_braces(source, match.start())
    if braces is None or braces:
        return False
    if not _compilation_unit_prefix_is_valid(source[: match.start()], suffix):
        return False
    if suffix not in {".cs", ".java"}:
        return True
    search_end = min(len(source), match.end("target") + _DECLARATION_HEADER_LIMIT)
    body_start = source.find("{", match.end("target"), search_end)
    body_end = _matching_delimiter(source, body_start, "{", "}")
    return body_end is not None and not source[body_end + 1 :].strip()


def _enclosing_type_context(
    source: str,
    opening_brace: int,
    suffix: str,
) -> tuple[str, frozenset[str]] | None:
    """Classify one complete canonical type header owning an opening brace."""
    boundary = max(
        source.rfind(";", 0, opening_brace),
        source.rfind("{", 0, opening_brace),
        source.rfind("}", 0, opening_brace),
    )
    header = source[boundary + 1 : opening_brace]
    declaration = re.fullmatch(
        rf"\s*(?P<prefix>(?:{_PREFIX_TOKEN}\s+)*)"
        r"(?P<kind>class|enum|interface|record|struct)\s+[A-Za-z_]\w*"
        r"(?P<header>[^{};\r\n]*)",
        header,
    )
    if declaration is None:
        return None
    kind = declaration.group("kind")
    type_header = declaration.group("header")
    if suffix == ".java" and kind == "record":
        if re.fullmatch(r"\s*\(\s*\)\s*", type_header) is None:
            return None
    elif type_header.strip():
        return None
    if not _declaration_prefix_is_valid(declaration, suffix, kind):
        return None
    if not _compilation_unit_prefix_is_valid(source[: boundary + 1], suffix):
        return None
    body_end = _matching_delimiter(source, opening_brace, "{", "}")
    if body_end is None or source[body_end + 1 :].strip():
        return None
    if suffix == ".java" and kind == "enum":
        enum_body = source[opening_brace + 1 : body_end]
        if re.match(r"\s*(?:[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)?\s*;", enum_body) is None:
            return None
    prefix = declaration.group("prefix")
    modifiers = frozenset(_modifier_base(token) for token in re.findall(_PREFIX_TOKEN, prefix))
    return kind, modifiers


def _enclosing_type_kind(source: str, opening_brace: int, suffix: str) -> str | None:
    """Return the validated kind of a canonical enclosing type."""
    context = _enclosing_type_context(source, opening_brace, suffix)
    return None if context is None else context[0]


def _function_placement_is_valid(source: str, match: re.Match[str], suffix: str) -> bool:
    """Require a canonical legal placement for parser-free function evidence."""
    if suffix in {".py", ".pyi", ".rb", ".lua", ".r"}:
        return True
    braces = _enclosing_braces(source, match.start())
    if braces is None:
        return False
    if suffix == ".java":
        return len(braces) == 1 and _enclosing_type_kind(source, braces[0], suffix) in {
            "class",
            "enum",
            "record",
        }
    if suffix == ".cs":
        return len(braces) == 1 and _enclosing_type_kind(source, braces[0], suffix) in {
            "class",
            "record",
            "struct",
        }
    return not braces


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


def _split_flat_generic_type_list(value: str) -> tuple[str, ...]:
    """Split a comma list while preserving the verifier's flat generic subset."""
    parts: list[str] = []
    start = 0
    angle_depth = 0
    for position, character in enumerate(value):
        if character == "<":
            angle_depth += 1
        elif character == ">":
            angle_depth -= 1
        elif character == "," and angle_depth == 0:
            parts.append(value[start:position].strip())
            start = position + 1
    parts.append(value[start:].strip())
    return tuple(parts)


def _relationship_type_names(
    value: str,
    declaration_name: str,
) -> tuple[str, ...] | None:
    """Normalize flat relationship types and reject intrinsic duplicates/self-use."""
    names = tuple(
        re.sub(r"\s*<.*>\s*$", "", re.sub(r"\(\s*\)\s*$", "", item.strip()))
        for item in _split_flat_generic_type_list(value)
    )
    if len(set(names)) != len(names):
        return None
    if any(name.rsplit(".", 1)[-1] == declaration_name for name in names):
        return None
    return names


def _type_body_header_is_valid(
    header: str,
    kind: str,
    suffix: str,
    declaration_name: str = "",
    declaration_modifiers: frozenset[str] = frozenset(),
) -> bool:
    """Recognize a conservative executable subset of braced type headers."""
    if suffix == ".go":
        return re.fullmatch(rf"\s+{re.escape(kind)}\s*", header) is not None
    if not header.strip():
        return True
    if any(token in header for token in ("=", ";", "{", "}", "@")):
        return False
    generic = r"<\s*[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*\s*>"
    if suffix in {".cs", ".java", ".kt", ".kts", ".rs", ".swift", ".ts", ".tsx"}:
        parameters = re.match(
            r"\s*<\s*(?P<parameters>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*>",
            header,
        )
        if parameters is not None:
            names = tuple(name.strip() for name in parameters.group("parameters").split(","))
            if len(set(names)) != len(names):
                return False
    qualified = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    qualified_generic = rf"{qualified}(?:\s*{generic})?"
    if suffix in _CPP_SUFFIXES:
        stripped = header.strip()
        if stripped.startswith("final"):
            stripped = stripped.removeprefix("final").strip()
        if not stripped:
            return True
        cpp_type = r"(?:[A-Za-z_]\w*::)*[A-Z]\w*"
        base = rf"(?:(?:private|protected|public|virtual)\s+)*{cpp_type}"
        if re.fullmatch(rf":\s*{base}(?:\s*,\s*{base})*", stripped) is None:
            return False
        bases: list[str] = []
        for base_specifier in stripped[1:].split(","):
            type_match = re.search(rf"(?P<type>{cpp_type})\s*$", base_specifier)
            if type_match is None:
                return False
            modifiers = re.findall(r"\b(?:private|protected|public|virtual)\b", base_specifier)
            if len(set(modifiers)) != len(modifiers):
                return False
            if len({"private", "protected", "public"}.intersection(modifiers)) > 1:
                return False
            base_name = type_match.group("type")
            if base_name.split("::")[-1] == declaration_name:
                return False
            bases.append(base_name)
        return len(set(bases)) == len(bases)
    if suffix == ".java":
        java_non_type = (
            r"abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|"
            r"default|do|double|else|enum|extends|false|final|finally|float|for|goto|"
            r"if|implements|import|instanceof|int|interface|long|native|new|null|package|"
            r"private|protected|public|return|short|static|strictfp|super|switch|"
            r"synchronized|this|throw|throws|transient|true|try|var|void|volatile|while"
        )
        java_identifier = rf"(?!(?:{java_non_type})\b)[A-Za-z_]\w*"
        java_generic = r"<\s*[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*\s*>"
        java_qualified = rf"{java_identifier}(?:\.{java_identifier})*"
        java_type = rf"{java_qualified}(?:\s*{java_generic})?"
        if kind == "class":
            clauses = (
                rf"(?:extends\s+{java_type}\s*)?"
                rf"(?:implements\s+{java_type}"
                rf"(?:\s*,\s*{java_type})*\s*)?"
                rf"(?:permits\s+{java_type}"
                rf"(?:\s*,\s*{java_type})*\s*)?"
            )
        elif kind == "interface":
            clauses = (
                rf"(?:extends\s+{java_type}"
                rf"(?:\s*,\s*{java_type})*\s*)?"
                rf"(?:permits\s+{java_type}"
                rf"(?:\s*,\s*{java_type})*\s*)?"
            )
        else:
            return False
        if re.fullmatch(rf"\s*(?:{java_generic}\s*)?{clauses}", header) is None:
            return False
        relationships: dict[str, tuple[str, ...]] = {}
        for relationship in re.finditer(
            r"\b(?P<kind>extends|implements|permits)\s+"
            r"(?P<types>.*?)(?=\b(?:implements|permits)\b|$)",
            header,
        ):
            type_names = _relationship_type_names(
                relationship.group("types"),
                declaration_name,
            )
            if type_names is None:
                return False
            relationships[relationship.group("kind")] = type_names
        direct_supertypes = relationships.get("extends", ()) + relationships.get("implements", ())
        return len(set(direct_supertypes)) == len(direct_supertypes)
    if suffix in {".js", ".jsx"}:
        match = re.fullmatch(
            r"\s*(?:extends\s+(?P<base>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*))?\s*",
            header,
        )
        return match is not None and (
            match.group("base") is None
            or match.group("base").rsplit(".", 1)[-1] != declaration_name
        )
    if suffix in {".ts", ".tsx"}:
        if kind != "class":
            return False
        ts_name = r"(?:[A-Za-z_$][\w$]*\.)*[A-Z_$][\w$]*"
        ts_type = rf"{ts_name}(?:\s*{generic})?"
        if (
            re.fullmatch(
                rf"\s*(?:{generic}\s*)?"
                rf"(?:extends\s+{ts_type}\s*)?"
                rf"(?:implements\s+{ts_type}(?:\s*,\s*{ts_type})*\s*)?",
                header,
            )
            is None
        ):
            return False
        typescript_relationships: list[str] = []
        for relationship in re.finditer(
            r"\b(?:extends|implements)\s+(?P<types>.*?)(?=\bimplements\b|$)",
            header,
        ):
            names = _relationship_type_names(relationship.group("types"), declaration_name)
            if names is None:
                return False
            typescript_relationships.extend(names)
        return len(set(typescript_relationships)) == len(typescript_relationships)
    if suffix == ".cs":
        cs_name = r"(?:[A-Za-z_]\w*\.)*[A-Z]\w*"
        cs_type = rf"{cs_name}(?:\s*{generic})?"
        if (
            re.fullmatch(
                rf"\s*(?:{generic}\s*)?(?::\s*{cs_type}"
                rf"(?:\s*,\s*{cs_type})*)?\s*",
                header,
            )
            is None
        ):
            return False
        _, separator, base_list = header.partition(":")
        if not separator:
            return True
        bases = _relationship_type_names(base_list, declaration_name)
        return (
            bases is not None
            and "static" not in declaration_modifiers
            and not {
                "Boolean",
                "Byte",
                "Char",
                "Decimal",
                "Double",
                "Int16",
                "Int32",
                "Int64",
                "Object",
                "SByte",
                "Single",
                "String",
                "UInt16",
                "UInt32",
                "UInt64",
            }.intersection(name.rsplit(".", 1)[-1] for name in bases)
        )
    if suffix == ".swift":
        if (
            re.fullmatch(
                rf"\s*(?:{generic}\s*)?(?::\s*{qualified}"
                rf"(?:\s*,\s*{qualified})*)?\s*",
                header,
            )
            is None
        ):
            return False
        _, separator, base_list = header.partition(":")
        if not separator:
            return True
        bases = _relationship_type_names(base_list, declaration_name)
        swift_final_types = frozenset(
            {
                "Array",
                "Bool",
                "Character",
                "Dictionary",
                "Double",
                "Float",
                "Float16",
                "Float80",
                "Int",
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Never",
                "Optional",
                "Result",
                "Set",
                "String",
                "UInt",
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
            }
        )
        return bases is not None and not swift_final_types.intersection(
            name.rsplit(".", 1)[-1] for name in bases
        )
    if suffix in {".kt", ".kts"}:
        kotlin_parent = rf"{qualified_generic}(?:\(\s*\))?"
        if (
            re.fullmatch(
                rf"\s*(?:{generic}\s*)?(?:\(\s*\)\s*)?"
                rf"(?::\s*{kotlin_parent}(?:\s*,\s*{kotlin_parent})*)?\s*",
                header,
            )
            is None
        ):
            return False
        _, separator, base_list = header.partition(":")
        if not separator:
            return True
        bases = _relationship_type_names(base_list, declaration_name)
        kotlin_final_types = frozenset(
            {
                "Boolean",
                "BooleanArray",
                "Byte",
                "ByteArray",
                "Char",
                "CharArray",
                "Double",
                "DoubleArray",
                "Float",
                "FloatArray",
                "Int",
                "IntArray",
                "Long",
                "LongArray",
                "Nothing",
                "Short",
                "ShortArray",
                "String",
                "UByte",
                "UByteArray",
                "UInt",
                "UIntArray",
                "ULong",
                "ULongArray",
                "UShort",
                "UShortArray",
                "Unit",
            }
        )
        return bases is not None and not kotlin_final_types.intersection(
            name.rsplit(".", 1)[-1] for name in bases
        )
    if suffix == ".rs":
        return re.fullmatch(rf"\s*(?:{generic}\s*)?", header) is not None
    return False


def _function_body_header_is_valid(header: str, suffix: str) -> bool:
    """Recognize a conservative executable subset between parameters and body."""
    if any(token in header for token in ("=", ";", "{", "}", "@")):
        return False
    if suffix in {".bash", ".js", ".jsx", ".pl", ".r", ".sh", ".zsh"}:
        return not header.strip()
    if suffix in {".ts", ".tsx"}:
        return re.fullmatch(r"\s*(?::\s*void)?\s*", header) is not None
    if suffix == ".go":
        return not header.strip()
    if suffix == ".rs":
        return re.fullmatch(r"\s*(?:->\s*\(\s*\))?\s*", header) is not None
    if suffix == ".swift":
        return (
            re.fullmatch(
                r"\s*(?:async\s+)?(?:(?:rethrows|throws)\s+)?"
                r"(?:->\s*(?:Void|\(\s*\)))?\s*",
                header,
            )
            is not None
        )
    if suffix in {".kt", ".kts"}:
        return re.fullmatch(r"\s*(?::\s*Unit)?\s*", header) is not None
    if suffix == ".java":
        return not header.strip()
    if suffix == ".cs":
        return not header.strip()
    if suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".mm"}:
        return re.fullmatch(r"\s*(?:->\s*void)?\s*", header) is not None
    return False


def _declaration_body_is_valid(body: str, declaration_kind: str, suffix: str) -> bool:
    """Admit only body syntax the verifier can prove without a language parser."""
    stripped = body.strip()
    if not stripped:
        return True
    if declaration_kind != "function":
        if suffix == ".java" and declaration_kind == "class":
            return (
                re.fullmatch(
                    r"(?:boolean|byte|char|double|float|int|long|short)(?:\[\])?\s+"
                    r"[A-Za-z_]\w*\s*;",
                    stripped,
                )
                is not None
            )
        if suffix == ".rs" and declaration_kind == "struct":
            return (
                re.fullmatch(
                    r"[A-Za-z_]\w*\s*:\s*[A-Za-z_]\w*"
                    r"(?:::[A-Za-z_]\w*)*\s*,",
                    stripped,
                )
                is not None
            )
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
    declaration_name: str = "",
    declaration_modifiers: frozenset[str] = frozenset(),
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
    elif not _type_body_header_is_valid(
        header,
        declaration_kind,
        suffix,
        declaration_name,
        declaration_modifiers,
    ):
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
    if masked != original:
        return False
    if suffix != ".java":
        return not masked.strip()
    java_keyword = (
        r"abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|"
        r"default|do|double|else|enum|exports|extends|false|final|finally|float|for|"
        r"goto|if|implements|import|instanceof|int|interface|long|module|native|new|"
        r"non-sealed|null|open|opens|package|permits|private|protected|provides|public|"
        r"record|requires|return|sealed|short|static|strictfp|super|switch|synchronized|"
        r"this|throw|throws|to|transient|transitive|true|try|uses|var|void|volatile|"
        r"while|with|yield"
    )
    identifier = rf"(?!(?:{java_keyword})\b)[A-Za-z_]\w*"
    primitive = r"(?:boolean|byte|char|double|float|int|long|short)"
    parameter = rf"{primitive}(?:\[\])?\s+{identifier}"
    if re.fullmatch(rf"\s*(?:{parameter}(?:\s*,\s*{parameter})*)?\s*", masked) is None:
        return False
    names = tuple(
        match.group("name")
        for match in re.finditer(
            rf"{primitive}(?:\[\])?\s+(?P<name>{identifier})",
            masked,
        )
    )
    return len(set(names)) == len(names)


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
        prefix = match.groupdict().get("prefix", "")
        modifiers = frozenset(_modifier_base(token) for token in re.findall(_PREFIX_TOKEN, prefix))
        return _complete_braced_body(
            source,
            match.end("target"),
            kind,
            suffix,
            match.group("target"),
            modifiers,
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
    prefix = rf"(?m)^(?P<prefix>[ \t]*(?:{_PREFIX_TOKEN}[ \t]+)*)"
    if kind == "class" and suffix in _CLASS_SUFFIXES:
        return (prefix + r"class\s+(?P<target>{target})\b",)
    if kind == "interface":
        if suffix in _INTERFACE_SUFFIXES:
            return (prefix + r"interface\s+(?P<target>{target})\b",)
        if suffix == ".go":
            return (prefix + r"type\s+(?P<target>{target})\s+interface\b",)
    if kind == "struct":
        if suffix in _STRUCT_SUFFIXES:
            return (prefix + r"struct\s+(?P<target>{target})\b",)
        if suffix == ".go":
            return (prefix + r"type\s+(?P<target>{target})\s+struct\b",)
    if kind == "trait" and suffix == ".rs":
        return (prefix + r"trait\s+(?P<target>{target})\b",)
    if kind == "function":
        if keyword := _FUNCTION_KEYWORDS.get(suffix):
            return (prefix + rf"{keyword}\s+(?P<target>{{target}})\b",)
        if suffix in _C_LIKE_FUNCTION_SUFFIXES:
            return (_C_LIKE_FUNCTION,)
        if suffix == ".r":
            return (prefix + r"(?P<target>{target})\s*(?:<-|=)\s*function\s*\(",)
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
            if not _declaration_prefix_is_valid(match, suffix, kind):
                continue
            if suffix == ".java" and not _java_declaration_context_is_valid(
                source,
                match,
                target,
                kind,
                file_path,
            ):
                continue
            if (
                kind == "function"
                and suffix == ".cs"
                and not _csharp_function_context_is_valid(source, match)
            ):
                continue
            if kind in {"class", "interface", "struct", "trait"} and not _type_placement_is_valid(
                source,
                match,
                suffix,
            ):
                continue
            if kind == "function" and not _function_placement_is_valid(
                source,
                match,
                suffix,
            ):
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
                if not _c_like_function_prefix_is_valid(source, match, suffix):
                    continue
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


def source_has_declaration_kind(
    source: str,
    original_source: str,
    target: str,
    kind: str,
    file_path: str,
) -> bool:
    """Whether trusted source contains the requested complete declaration kind."""
    return any(
        _source_span_has_declaration_kind(
            source,
            original_source,
            span,
            target,
            kind,
            file_path,
        )
        for span in literal_spans(source, target)
    )


def _source_has_declaration_shape(
    source: str,
    target: str,
    kind: str,
    file_path: str,
) -> bool:
    """Whether source contains the requested language-specific declaration header."""
    target_spans = literal_spans(source, target)
    escaped = re.escape(target)
    if any(
        match.span("target") in target_spans
        for template in _declaration_patterns(file_path, kind)
        for match in re.finditer(template.format(target=escaped), source)
    ):
        return True
    suffix = os.path.splitext(file_path)[1].casefold()
    return (
        kind == "function"
        and suffix in _C_LIKE_FUNCTION_SUFFIXES
        and bool(
            re.search(
                rf"\b(?:[A-Za-z_]\w*\s+){{1,8}}[*&\s]*{escaped}\s*\(",
                source,
            )
        )
    )


def matches_criterion(
    source: str,
    original_source: str,
    target: str,
    assertion: SpecAssertion,
    file_path: str,
) -> bool:
    """Apply the caller-requested declaration kind to a trusted inventory scan."""
    kind_required, kind = acceptance_declaration_kind(assertion.ac_text, target)
    if not kind_required:
        return literal_is_bound(source, target)
    return kind is None or (
        source_has_declaration_kind(source, original_source, target, kind, file_path)
        or _source_has_declaration_shape(source, target, kind, file_path)
    )


def matches_any(
    source: str,
    original_source: str,
    targets: tuple[str, ...],
    assertion: SpecAssertion,
    file_path: str,
) -> bool:
    """Whether any criterion target has a requested declaration shape."""
    return any(
        matches_criterion(source, original_source, target, assertion, file_path)
        for target in targets
    )


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
