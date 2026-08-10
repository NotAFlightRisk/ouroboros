"""Deterministic binding between verifier evidence and an acceptance criterion."""

from __future__ import annotations

import os
import re

# Read model-supplied patterns without executing them.  This is the same
# CPython parser used by the verifier's bounded structural checks and is
# available on every supported interpreter (3.12--3.14).
from re import _parser as regex_parser

from ouroboros.verification.models import EvidencePolarity

_TARGET_TOKEN = re.compile(r"--[\w][\w-]*|[\w][\w./:-]*", re.UNICODE)
_QUOTED_TOKEN = re.compile(
    r"(?P<quote>[`'\"])(?P<token>--[\w][\w-]*|[\w][\w./:-]*)(?P=quote)",
    re.UNICODE,
)
_STRUCTURAL_WORDS = frozenset(
    {"class", "constant", "directory", "file", "flag", "function", "interface", "struct", "trait"}
)
_DECLARATION_KIND_WORD = re.compile(
    r"\b(?P<kind>class|function|interface|struct|trait)\b",
    re.IGNORECASE,
)
_DECLARATION_KIND_BEFORE_TARGET = re.compile(
    r"\b(?P<kind>class|function|interface|struct|trait)\b"
    r"(?:\s+(?:must|shall|should)\s+be)?(?:\s+(?:called|named))?\s*$",
    re.IGNORECASE,
)
_DECLARATION_KIND_AFTER_TARGET = re.compile(
    r"^\s+(?:(?:implementation|service)\s+)?"
    r"(?P<kind>class|function|interface|struct|trait)\b",
    re.IGNORECASE,
)
_CONSTANT_BINDING_SUFFIX = re.compile(
    r"(?:=|:|\bto\b|\bof\b|\bis\b|\bshould\s+be\b|\bmust\s+be\b)\s*$",
    re.IGNORECASE,
)
_TOKEN_PUNCTUATION = frozenset("._/:-+@#?&=%~")
_MAX_IDENTITY_PATTERN_LENGTH = 200
_SENTENCE_BOUNDARY = r"(?:[!?]+|\.+(?=\s|$|[A-Z])|[\r\n]+)"
_CLAUSE_BOUNDARY = re.compile(
    r"\b(?:in\s+order\s+to|because|so|and|but|while|whereas)\b|[;,]|" + _SENTENCE_BOUNDARY,
    re.IGNORECASE,
)
_FOLLOWING_SENTENCE_CONTENT = re.compile(
    r"(?:[!?！？]+|…+|[\r\n\u0085\u2028\u2029]+)\s*(?=[^\s!?！？.。．｡…])"
    r"|[.。．｡]+(?:\s+(?=\S)|(?=[^\s\d!?！？.。．｡…]))"
)
_UNRECOGNIZED_CLAUSE_SEPARATOR = re.compile(
    # Commas, semicolons, sentence marks, and the explicit causal joiners in
    # ``_CLAUSE_BOUNDARY`` are recognized elsewhere.  Dashes, brackets, and
    # prose slashes/pipes are not grammar: content after one can be a hidden
    # second predicate and therefore makes polarity unknowable.
    r"(?:"
    r"(?:(?<=\s)-{1,3}(?=\s|\w)|(?<=\w)-{2,3}(?=\w))"
    r"|[\u058a\u05be\u1400\u1806\u2010-\u2015\u2e17\u2e1a\u2e3a-\u2e3b\u2e40\u301c\u3030\ufe31-\ufe32\ufe58\ufe63\uff0d]"
    r"|[()\[\]{}<>（）［］｛｝〈〉《》]"
    r"|(?<=\s)[/|\\](?=\s|\w)"
    r"|(?<=\S)[:：]\s*(?=\S)"
    r"|(?:=>|->|[→⇒⟶])"
    r"|[·•‧∙⋅]"
    r"|[\"'`“”‘’«»‹›]\s*(?=\w)"
    r")\s*(?=\S)"
)
_FOLLOWING_AMBIGUOUS_CLAUSE = re.compile(
    r"(?:[,;，；]|\b(?:but|nor|or|while|whereas|yet)\b)\s*(?=\S)",
    re.IGNORECASE,
)
_FOLLOWING_CONJUNCTIVE_PREDICATE = re.compile(
    r"\band\b[\s\S]*\b"
    r"(?:must|shall|should|may|can|is|are|was|were|be|becomes?|do|does|did|not|never|"
    r"without|avoid|prevent|omit|remove|delete|forbid|prohibit|exclude|exists?|remains?)\b",
    re.IGNORECASE,
)
_FORBIDDEN_COMMAND_PREFIX = re.compile(
    r"(?:\b(?:must|shall|should|may|can)\s+(?:not|never)\b|\b(?:do|does|did)\s+not\b)",
    re.IGNORECASE,
)
_FORBIDDEN_TARGET_PREFIX = re.compile(
    r"\b(?:"
    r"(?:without|no)\s+(?:(?:an?|the)\s+)?(?:\w+\s+){0,2}"
    r"|(?:avoid|prevent|omit|forbid|prohibit|disallow|exclude)\w*\s+"
    r"(?:(?:defin|creat|set|includ|us)\w*\s+)?(?:(?:an?|the)\s+)?(?:\w+\s+){0,2}"
    r")$",
    re.IGNORECASE,
)
_FORBIDDEN_TARGET_SUFFIX = re.compile(
    r"^\s*(?:(?:\w+\s+){0,2}(?:class|interface|struct|trait|function|file|directory|"
    r"flag|constant|value|setting|assignment|definition|declaration)\s+)?(?:"
    r"(?:is|are|must\s+be|should\s+be|shall\s+be)\s+"
    r"(?:forbidden|prohibited|disallowed|absent|excluded)\b"
    r"|(?:must|shall|should)\s+(?:not|never)\s+"
    r"(?:exist|appear|remain|be\s+(?:present|defined|created|included|used))\b"
    r"|(?:do|does)\s+not\s+exist\b"
    r")",
    re.IGNORECASE,
)
_TARGET_RELATIVE_PREDICATE = re.compile(
    r"(?:^|[;,])\s*(?:(?:\w+\s+){0,2}(?:class|interface|struct|trait|function|file|"
    r"directory|flag|constant|value|setting|assignment|definition|declaration)\s+)?"
    r"(?:must|shall|should|may|can|is|are|was|were|be|becomes?)\b",
    re.IGNORECASE,
)
_EXPLICIT_REQUIRED_TARGET_PREFIX = re.compile(
    r"^\s*(?:"
    r"(?:must|shall|should)\s+"
    r"(?:add|create|declare|define|expose|implement|include|provide|retain|set|use)\s+"
    r"(?:(?:an?|the)\s+)?(?:(?:class|interface|struct|trait|function|file|directory|"
    r"flag|constant|value|setting|assignment|definition|declaration)\s+)?(?:named\s+)?"
    r"|(?:an?|the)\s+(?:class|interface|struct|trait|function|file|directory|flag|"
    r"constant|value|setting|assignment|definition|declaration)\s+"
    r"(?:must|shall|should)\s+be\s+"
    r")[`'\"]?$",
    re.IGNORECASE,
)
_EXPLICIT_REQUIRED_HAS_TARGET_PREFIX = re.compile(
    r"^\s*has\s+(?:(?:an?|the)\s+)?(?:class|interface|struct|trait|function|file|"
    r"directory|flag|constant|value|setting|assignment|definition|declaration)\s+"
    r"(?:named\s+)?$",
    re.IGNORECASE,
)
_EXPLICIT_REQUIRED_CLI_FLAG_PREFIX = re.compile(
    r"^\s*(?:the\s+)?cli\s+accepts\s+$",
    re.IGNORECASE,
)
_EXPLICIT_REQUIRED_CLI_FLAG_SUFFIX = re.compile(
    r"^\s+flag\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_REQUIRED_STRUCTURAL_SUFFIX = re.compile(
    r"^(?:(?:[\w-]+\s+){0,2})?(?:class|interface|struct|trait|function|file|directory|"
    r"flag|constant|value|setting|assignment|definition|declaration)\b",
    re.IGNORECASE,
)
_DISTINCT_CAUSAL_EFFECT = (
    r"(?:outages?|(?:decorator\s+)?boilerplate|(?:runtime\s+)?errors?|"
    r"(?:(?:transient|stale-state)\s+)?failures?|(?:accidental\s+)?failure\s+loops?|"
    r"(?:redundant\s+)?error\s+retries|retry/bounce\s+chains?)"
)
_DISTINCT_CAUSAL_PREDICATE = (
    r"(?:that\s+)?(?:must|shall|should)\s+(?:not|never)\s+"
    r"(?:remain|recur|persist|occur|appear|exist)"
)
_EXPLICIT_REQUIRED_CAUSAL_SUFFIX = re.compile(
    rf"^(?:"
    rf"to\s+(?:avoid|omit|prevent)\s+{_DISTINCT_CAUSAL_EFFECT}"
    rf"|(?:to\s+ensure|in\s+order\s+to\s+ensure|because|for|so)\s+"
    rf"{_DISTINCT_CAUSAL_EFFECT}(?:\s+{_DISTINCT_CAUSAL_PREDICATE})?"
    rf")$",
    re.IGNORECASE,
)
_LOCATION_BODY = (
    r"(?:(?:an?|the)\s+)?(?:[\w-]+\s+){0,2}"
    r"(?:project|config(?:uration)?|codebase|repository|module|package|source)"
)
_EXPLICIT_REQUIRED_LOCATION_SUFFIX = re.compile(
    rf"^in\s+{_LOCATION_BODY}\s*$",
    re.IGNORECASE,
)
_VALUE_OPERATOR = r"(?:(?:=|:)|(?:to|of|is)\b|(?:must|shall|should)\s+be\b)"
_VALUE_SCALAR = r"(?:`[^`\r\n]+`|'[^'\r\n]+'|\"[^\"\r\n]+\"|[^\s,;()\[\]{}<>]+)"
_COMPOUND_CONSTANT_KEY = r"(?:--[\w][\w-]*|(?-i:[A-Z][A-Z0-9_]*(?:[./:-][A-Z0-9_]+)*))"
_REQUIRED_VALUE_TAIL = re.compile(
    rf"^\s*{_VALUE_OPERATOR}\s*{_VALUE_SCALAR}"
    rf"(?:\s+in\s+{_LOCATION_BODY})?"
    rf"(?:\s+and\s+{_COMPOUND_CONSTANT_KEY}\s*{_VALUE_OPERATOR}\s*{_VALUE_SCALAR}"
    rf"(?:\s+in\s+{_LOCATION_BODY})?)*\s*$",
    re.IGNORECASE,
)
_CONSTANT_ASSIGNMENT = re.compile(
    rf"(?<![\w./:-])(?P<key>{_COMPOUND_CONSTANT_KEY})\s*{_VALUE_OPERATOR}\s*"
    rf"(?P<scalar>{_VALUE_SCALAR})",
    re.IGNORECASE,
)
_EXPLICIT_REQUIRED_VALUE_SUFFIX = re.compile(
    r"^\s*(?:(?:=|:)|\b(?:to|of|is)\b\s+(?!not\b|never\b)"
    r"|(?:must|shall|should)\s+be\s+(?!not\b|never\b))",
    re.IGNORECASE,
)


def _strip_terminal_target_marks(target_suffix: str) -> str:
    """Remove only closing quotes and terminal sentence marks from a suffix."""
    suffix = target_suffix.rstrip()
    while suffix and suffix[-1] in "`'\".!?！？。．｡…":
        suffix = suffix[:-1].rstrip()
    return suffix


def _required_target_suffix_is_bounded(_target: str, target_suffix: str) -> bool:
    """Accept only known structural, qualifier, or distinct-cause suffixes.

    The positive command before a target is not authority over arbitrary prose
    after it.  Unknown trailing words can introduce a second predicate without
    punctuation (``plus it must not exist``), so positive fallback is allowed
    only when the entire suffix belongs to this bounded grammar.
    """
    suffix = _strip_terminal_target_marks(target_suffix).strip()
    if not suffix:
        return True

    structural = _EXPLICIT_REQUIRED_STRUCTURAL_SUFFIX.match(suffix)
    if structural is not None:
        suffix = suffix[structural.end() :].strip()
        if not suffix:
            return True

    if _EXPLICIT_REQUIRED_CAUSAL_SUFFIX.fullmatch(suffix):
        return True

    return _EXPLICIT_REQUIRED_LOCATION_SUFFIX.fullmatch(suffix) is not None


def _required_value_tail_is_bounded(target_suffix: str) -> bool:
    """Require the complete scalar tail to match one bounded value grammar."""
    suffix = _strip_terminal_target_marks(target_suffix)
    return _REQUIRED_VALUE_TAIL.fullmatch(suffix) is not None


def _has_duplicate_constant_keys(ac_text: str) -> bool:
    """Whether one AC assigns more than one scalar clause to the same key."""
    keys = [match.group("key") for match in _CONSTANT_ASSIGNMENT.finditer(ac_text)]
    return len(keys) != len(set(keys))


def _has_explicit_required_target(
    ac_prefix: str,
    target: str,
    target_suffix: str,
) -> bool:
    """Whether the AC starts with a bounded command that requires this target."""
    if _EXPLICIT_REQUIRED_TARGET_PREFIX.search(ac_prefix):
        return _required_target_suffix_is_bounded(target, target_suffix)
    if _EXPLICIT_REQUIRED_HAS_TARGET_PREFIX.fullmatch(ac_prefix):
        return not target_suffix.strip()
    return bool(
        target.startswith("--")
        and _EXPLICIT_REQUIRED_CLI_FLAG_PREFIX.fullmatch(ac_prefix)
        and _EXPLICIT_REQUIRED_CLI_FLAG_SUFFIX.fullmatch(target_suffix)
    )


def _has_explicit_required_value(target: str, target_suffix: str) -> bool:
    """Whether assignment syntax positively binds a scalar target."""
    match = _EXPLICIT_REQUIRED_VALUE_SUFFIX.search(target_suffix)
    if match is None:
        return False
    operator = match.group(0).strip()
    return operator[:1] in {"=", ":"} or (target.isascii() and target.isupper())


def _is_identifier_continue(character: str) -> bool:
    r"""Return whether a character can continue a Unicode identifier.

    Python's regex ``\w`` omits valid XID continuations such as combining
    marks, ZWNJ, and the middle dot. Prefixing a known identifier starter asks
    Python's Unicode identifier table directly and keeps literal boundaries
    conservative for every such continuation.
    """
    return bool(character) and (
        character in {"$", "\u200c", "\u200d"} or ("A" + character).isidentifier()
    )


def literal_spans(text: str, literal: str) -> tuple[tuple[int, int], ...]:
    """Return exact-case whole-literal spans with Unicode XID boundaries."""
    literal = literal.strip()
    if not literal:
        return ()
    spans: list[tuple[int, int]] = []
    is_flag = literal.startswith("--")
    boundary_punctuation = (
        _TOKEN_PUNCTUATION
        if any(character in _TOKEN_PUNCTUATION for character in literal)
        else frozenset(".")
        if literal.isdecimal()
        else frozenset()
    )
    for match in re.finditer(re.escape(literal), text):
        if match.start() > 0:
            previous = text[match.start() - 1]
            if (
                (literal[0].isalnum() or literal[0] == "_" or is_flag)
                and _is_identifier_continue(previous)
            ) or previous in boundary_punctuation:
                continue
        if match.end() < len(text):
            following = text[match.end()]
            if (
                (literal[-1].isalnum() or literal[-1] == "_") and _is_identifier_continue(following)
            ) or following in boundary_punctuation:
                continue
        spans.append((match.start(), match.end()))
    return tuple(spans)


def literal_is_bound(text: str, literal: str) -> bool:
    """Whether ``literal`` is present as a complete value in trusted text."""
    return bool(literal_spans(text, literal))


def _java_enclosing_field_type(source: str, line_start: int) -> str | None:
    """Return a simple canonical Java type enclosing a field declaration."""
    braces: list[int] = []
    for position, character in enumerate(source[:line_start]):
        if character == "{":
            braces.append(position)
        elif character == "}":
            if not braces:
                return None
            braces.pop()
    if len(braces) != 1:
        return None
    opening = braces[0]
    boundary = max(source.rfind(marker, 0, opening) for marker in ";{}")
    header = source[boundary + 1 : opening]
    declaration = re.fullmatch(
        r"\s*(?P<modifiers>(?:(?:abstract|final|non-sealed|public|sealed|strictfp)\s+)*)"
        r"(?P<kind>class|enum|interface|record)\s+[A-Za-z_]\w*\s*",
        header,
    )
    if declaration is None:
        return None
    modifiers = tuple(
        re.findall(
            r"\b(?:abstract|final|non-sealed|public|sealed|strictfp)\b",
            declaration.group("modifiers"),
        )
    )
    if len(set(modifiers)) != len(modifiers):
        return None
    if any(
        conflict.issubset(modifiers)
        for conflict in (
            frozenset({"abstract", "final"}),
            frozenset({"final", "sealed"}),
            frozenset({"non-sealed", "sealed"}),
        )
    ):
        return None
    kind = declaration.group("kind")
    if kind in {"enum", "record"} and {"abstract", "final", "non-sealed", "sealed"}.intersection(
        modifiers
    ):
        return None
    if kind == "interface" and "final" in modifiers:
        return None
    return kind


def _java_field_assignment_is_valid(
    source: str,
    prefix: str,
    tail: str,
    line_start: int,
) -> bool:
    """Validate a conservative Java field assignment and its enclosing type."""
    modifier = r"final|private|protected|public|static|transient|volatile"
    field_type = r"(?:boolean|byte|char|double|float|int|long|short|String)(?:\[\])?"
    declaration = re.fullmatch(
        rf"\s*(?P<modifiers>(?:(?:{modifier})\s+)*){field_type}\s+",
        prefix,
    )
    if declaration is None or re.match(r"\s*=(?!=)", tail) is None:
        return False
    modifiers = tuple(re.findall(rf"\b(?:{modifier})\b", declaration.group("modifiers")))
    if len(set(modifiers)) != len(modifiers):
        return False
    if len({"private", "protected", "public"}.intersection(modifiers)) > 1:
        return False
    if {"final", "volatile"}.issubset(modifiers):
        return False
    enclosing_kind = _java_enclosing_field_type(source, line_start)
    if enclosing_kind == "interface":
        return set(modifiers).issubset({"final", "public", "static"})
    return enclosing_kind in {"class", "enum", "record"}


def match_has_assignment_semantics(
    source: str,
    match: re.Match[str],
    target: str,
    file_path: str,
) -> bool:
    """Whether the concrete bound target is a canonical trusted assignment key."""
    for start, end in literal_spans(source, target):
        associated = match.start() <= start and end <= match.end()
        if match.start() == match.end():
            associated = start == match.start()
        if not associated:
            continue
        line_start = source.rfind("\n", 0, start) + 1
        line_end = source.find("\n", end)
        if line_end < 0:
            line_end = len(source)
        suffix = os.path.splitext(file_path)[1].casefold()
        prefix = source[line_start:start]
        tail = source[end:line_end]
        if suffix == ".py":
            valid = not prefix.strip() and re.match(
                r"\s*(?::\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?\s*=(?!=)",
                tail,
            )
        elif suffix == ".java":
            valid = _java_field_assignment_is_valid(source, prefix, tail, line_start)
        elif suffix in {".js", ".jsx"}:
            valid = re.fullmatch(r"\s*(?:const|let|var)\s+", prefix) and re.match(
                r"\s*=(?!=)", tail
            )
        elif suffix in {".ts", ".tsx"}:
            valid = re.fullmatch(r"\s*(?:const|let|var)\s+", prefix) and re.match(
                r"\s*(?::\s*[A-Za-z_$][\w$]*)?\s*=(?!=)", tail
            )
        elif suffix in {".yaml", ".yml"}:
            valid = not prefix.strip() and re.match(r"\s*:", tail)
        else:
            valid = not prefix.strip() and re.match(r"\s*=(?!=)", tail)
        if valid:
            return True
    return False


def acceptance_declaration_kind(
    ac_text: str,
    target: str,
) -> tuple[bool, str | None]:
    """Return whether the AC requires a declaration kind and its exact kind.

    The extraction model may select a regex, but it cannot reinterpret a
    caller-authored ``class`` as a function (or vice versa). Only a small,
    explicit grammar is authoritative. If a declaration kind is mentioned but
    cannot be bound uniquely to every occurrence of ``target``, callers must
    fail closed.
    """
    if not target.isidentifier() or _DECLARATION_KIND_WORD.search(ac_text) is None:
        return False, None

    kinds: set[str] = set()
    spans = literal_spans(ac_text, target)
    if not spans:
        return True, None
    for start, end in spans:
        before = _DECLARATION_KIND_BEFORE_TARGET.search(ac_text[:start])
        after = _DECLARATION_KIND_AFTER_TARGET.match(ac_text[end:])
        bound = after or before
        if bound is None:
            return True, None
        kinds.add(bound.group("kind").casefold())
    return (True, next(iter(kinds))) if len(kinds) == 1 else (True, None)


def _looks_code_like(token: str) -> bool:
    """Recognize conservative code literals without interpreting prose nouns."""
    if token.startswith("--"):
        return True
    if any(separator in token for separator in ("_", "/", ".", ":", "-")):
        return True
    if token.isascii() and token.isupper() and token.isalpha() and len(token) <= 3:
        return True
    return bool(re.search(r"[a-z][A-Z]", token))


def _looks_constant_literal(token: str) -> bool:
    """Recognize exact constant keys without admitting arbitrary prose."""
    return _looks_code_like(token) or (token.isascii() and token.isupper() and token.isalpha())


def _token_candidates(ac_text: str) -> tuple[tuple[str, int, int, bool], ...]:
    """Return unquoted and explicitly quoted literals with source spans."""
    quoted = tuple(
        (match.group("token"), match.start(), match.end(), True)
        for match in _QUOTED_TOKEN.finditer(ac_text)
    )
    quoted_inner_spans = tuple(
        (match.start("token"), match.end("token")) for match in _QUOTED_TOKEN.finditer(ac_text)
    )
    plain = tuple(
        (match.group(0), match.start(), match.end(), False)
        for match in _TARGET_TOKEN.finditer(ac_text)
        if not any(
            start <= match.start() and match.end() <= end for start, end in quoted_inner_spans
        )
    )
    return tuple(sorted((*quoted, *plain), key=lambda candidate: candidate[1]))


def _structural_targets(ac_text: str) -> tuple[str, ...]:
    """Derive structural names under conservative, input-only grammar."""
    candidates = _token_candidates(ac_text)
    words = [candidate for candidate in candidates if not candidate[3]]
    selected: list[str] = []

    for word, word_start, word_end, _quoted in words:
        folded = word.casefold().rstrip(".")
        if folded not in _STRUCTURAL_WORDS:
            continue
        previous = next((item for item in reversed(words) if item[2] <= word_start), None)
        if previous is not None and not ac_text[previous[2] : word_start].strip():
            token = previous[0]
            if _looks_code_like(token):
                selected.append(token)
                continue
        following = next((item for item in words if item[1] >= word_end), None)
        if following is not None and not ac_text[word_end : following[1]].strip():
            token = following[0]
            if _looks_code_like(token) or (
                token.isascii() and token[:1].isupper() and token[1:].islower()
            ):
                selected.append(token)

    if selected:
        return tuple(dict.fromkeys(selected))

    explicit = [token for token, _start, _end, quoted in candidates if quoted]
    code_literals = [
        token for token, _start, _end, quoted in candidates if quoted or _looks_code_like(token)
    ]
    unique = tuple(dict.fromkeys(explicit or code_literals))
    return unique if len(unique) == 1 else ()


def _constant_targets(ac_text: str, expected: str) -> tuple[str, ...]:
    """Return every explicit code literal bound to an equal scalar occurrence."""
    value_spans = literal_spans(ac_text, expected)
    if not value_spans:
        return ()
    candidates = [
        candidate
        for candidate in _token_candidates(ac_text)
        if candidate[3] or _looks_constant_literal(candidate[0])
    ]
    selected: list[str] = []
    for value_start, _value_end in value_spans:
        for token, _start, end, _quoted in reversed(candidates):
            if end > value_start:
                continue
            if _CONSTANT_BINDING_SUFFIX.fullmatch(ac_text[end:value_start].strip()):
                selected.append(token)
                break
    return tuple(dict.fromkeys(selected))


def _regex_literal_runs(sequence: object) -> tuple[str, ...]:
    """Return fixed literal runs from a parsed regex without matching it.

    The result is deliberately incomplete.  Character classes, backreferences,
    flags, and other computed constructs cannot establish caller-clause identity;
    an honest repeated-scalar pattern must name the key as fixed syntax.  A
    conservative miss rejects evidence, while executing the model pattern here
    would permit catastrophic backtracking before verifier checks run.
    """
    runs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            runs.append("".join(current))
            current.clear()

    for opcode, argument in sequence:  # type: ignore[attr-defined]
        name = getattr(opcode, "name", str(opcode))
        if name == "LITERAL":
            current.append(chr(argument))
            continue
        flush()
        if name == "SUBPATTERN":
            runs.extend(_regex_literal_runs(argument[-1]))
        elif name == "BRANCH":
            for branch in argument[1]:
                runs.extend(_regex_literal_runs(branch))
        elif name in {
            "ASSERT",
            "ASSERT_NOT",
            "MAX_REPEAT",
            "MIN_REPEAT",
            "POSSESSIVE_REPEAT",
        }:
            runs.extend(_regex_literal_runs(argument[-1]))
        elif name == "ATOMIC_GROUP":
            runs.extend(_regex_literal_runs(argument))
    flush()
    return tuple(runs)


def _pattern_literal_targets(pattern: str, targets: tuple[str, ...]) -> tuple[str, ...]:
    """Select caller targets named by fixed regex literals, without execution."""
    if not pattern or len(pattern) > _MAX_IDENTITY_PATTERN_LENGTH:
        return ()
    try:
        parsed = regex_parser.parse(pattern, 0)
    except (re.error, OverflowError, RecursionError):
        return ()
    literal_runs = _regex_literal_runs(parsed)
    return tuple(
        target for target in targets if any(literal_is_bound(run, target) for run in literal_runs)
    )


def _constant_target(ac_text: str, expected: str, pattern: str) -> tuple[str, ...]:
    """Bind a repeated scalar to the unique caller key selected by its regex.

    The model controls the regex, so it cannot create a key. It may only select
    one of the explicit caller-authored clauses by consuming that clause's key.
    A generic regex that selects more than one equal-valued clause fails closed.
    """
    targets = _constant_targets(ac_text, expected)
    if len(targets) <= 1:
        return targets
    selected = _pattern_literal_targets(pattern, targets)
    return selected if len(selected) == 1 else ()


def acceptance_polarity(
    ac_text: str,
    targets: tuple[str, ...],
) -> EvidencePolarity | None:
    """Derive one clause-level polarity for trusted criterion targets.

    Mixed-polarity multi-target assertions are ambiguous and return ``None``;
    callers must split them or fail closed.
    """
    if _has_duplicate_constant_keys(ac_text):
        return None
    polarities: list[EvidencePolarity] = []
    boundaries = tuple(_CLAUSE_BOUNDARY.finditer(ac_text))
    for target in targets:
        spans = literal_spans(ac_text, target)
        if not spans:
            return None
        target_polarities: set[EvidencePolarity] = set()
        for start, end in spans:
            clause_start = max(
                (match.end() for match in boundaries if match.end() <= start),
                default=0,
            )
            clause_end = min(
                (match.start() for match in boundaries if match.start() >= end),
                default=len(ac_text),
            )
            target_prefix = ac_text[clause_start:start]
            target_suffix = ac_text[end:clause_end]
            complete_target_suffix = ac_text[end:]
            if _FOLLOWING_SENTENCE_CONTENT.search(ac_text[end:]):
                return None
            if _UNRECOGNIZED_CLAUSE_SEPARATOR.search(ac_text[end:]):
                return None
            if _FOLLOWING_AMBIGUOUS_CLAUSE.search(ac_text[end:]):
                return None
            if _FOLLOWING_CONJUNCTIVE_PREDICATE.search(ac_text[end:]):
                return None
            required_value = _has_explicit_required_value(target, target_suffix)
            forbidden_command = _FORBIDDEN_COMMAND_PREFIX.search(target_prefix)
            if required_value and not _required_value_tail_is_bounded(complete_target_suffix):
                return None
            if (
                forbidden_command
                and not required_value
                and not _required_target_suffix_is_bounded(target, complete_target_suffix)
            ):
                return None
            if (
                forbidden_command
                or _FORBIDDEN_TARGET_PREFIX.search(target_prefix)
                or _FORBIDDEN_TARGET_SUFFIX.search(target_suffix)
            ):
                target_polarities.add(EvidencePolarity.FORBIDDEN)
            elif required_value:
                target_polarities.add(EvidencePolarity.REQUIRED)
            elif _TARGET_RELATIVE_PREDICATE.search(ac_text[end:]):
                return None
            elif _has_explicit_required_target(ac_text[:start], target, complete_target_suffix):
                target_polarities.add(EvidencePolarity.REQUIRED)
            else:
                # Only an explicit, bounded positive grammar can mint REQUIRED
                # evidence. Unknown commands and implicit prose fail closed;
                # expanding this allowlist requires end-to-end proof.
                return None
        if len(target_polarities) != 1:
            return None
        polarities.append(next(iter(target_polarities)))
    unique = set(polarities)
    return next(iter(unique)) if len(unique) == 1 else None


def acceptance_targets(
    ac_text: str,
    expected_value: str = "",
    *,
    prefer_expected: bool = False,
    pattern: str = "",
) -> tuple[str, ...]:
    """Return exact code literals derived only from caller-authored criterion text.

    Structural targets are selected independently of model ``expected_value``;
    the extractor subsequently requires the model value to equal one of them.
    Constant targets require an explicit code-shaped literal bound to the
    expected scalar's clause. Ambiguous prose has no target and fails closed.
    """
    expected = expected_value.strip()
    if expected and not literal_is_bound(ac_text, expected):
        return ()
    if prefer_expected:
        return _structural_targets(ac_text)
    if expected:
        return _constant_target(ac_text, expected, pattern)
    return ()
