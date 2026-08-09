"""Static execution-safety proof for model-supplied stdlib regexes.

The verifier uses Python's backtracking engine only for the regular subset
proved here. Unknown syntax, ambiguous repeat partitions, variable-width
backreferences, and unbounded whole-subject assertion retries fail closed.
"""

from __future__ import annotations

import re
from re import _parser as regex_parser

MAX_PATTERN_LENGTH = 200
MAX_SUBJECT_LENGTH = 50 * 1024
_MAX_PARSE_DEPTH = 40
_REPEATS = frozenset({"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"})


def _contains_repeat(sequence: object, depth: int = 0) -> bool:
    if depth > _MAX_PARSE_DEPTH:
        return True
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return True
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name in _REPEATS:
            return True
        if name == "SUBPATTERN" and _contains_repeat(argument[-1], depth + 1):
            return True
        if name == "ATOMIC_GROUP" and _contains_repeat(argument, depth + 1):
            return True
        if name in {"ASSERT", "ASSERT_NOT"} and _contains_repeat(argument[1], depth + 1):
            return True
        if name == "BRANCH" and any(_contains_repeat(branch, depth + 1) for branch in argument[1]):
            return True
        if name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if _contains_repeat(yes_arm, depth + 1) or (
                no_arm is not None and _contains_repeat(no_arm, depth + 1)
            ):
                return True
    return False


def _contains_unbounded_repeat(sequence: object, depth: int = 0) -> bool:
    if depth > _MAX_PARSE_DEPTH:
        return True
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return True
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name in _REPEATS:
            if argument[1] == regex_parser.MAXREPEAT:
                return True
            if _contains_unbounded_repeat(argument[2], depth + 1):
                return True
        elif name in {"SUBPATTERN", "ATOMIC_GROUP", "ASSERT", "ASSERT_NOT"}:
            child = (
                argument[-1]
                if name == "SUBPATTERN"
                else argument[1]
                if name in {"ASSERT", "ASSERT_NOT"}
                else argument
            )
            if _contains_unbounded_repeat(child, depth + 1):
                return True
        elif name == "BRANCH" and any(
            _contains_unbounded_repeat(branch, depth + 1) for branch in argument[1]
        ):
            return True
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if _contains_unbounded_repeat(yes_arm, depth + 1) or (
                no_arm is not None and _contains_unbounded_repeat(no_arm, depth + 1)
            ):
                return True
    return False


def _contains_unbounded_lazy_repeat(sequence: object, depth: int = 0) -> bool:
    """Whether a lazy repeat can retry across an unbounded subject suffix."""
    if depth > _MAX_PARSE_DEPTH:
        return True
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return True
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name in _REPEATS:
            if name == "MIN_REPEAT" and argument[1] == regex_parser.MAXREPEAT:
                return True
            if _contains_unbounded_lazy_repeat(argument[2], depth + 1):
                return True
        elif name in {"SUBPATTERN", "ATOMIC_GROUP", "ASSERT", "ASSERT_NOT"}:
            child = (
                argument[-1]
                if name == "SUBPATTERN"
                else argument[1]
                if name in {"ASSERT", "ASSERT_NOT"}
                else argument
            )
            if _contains_unbounded_lazy_repeat(child, depth + 1):
                return True
        elif name == "BRANCH" and any(
            _contains_unbounded_lazy_repeat(branch, depth + 1) for branch in argument[1]
        ):
            return True
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if _contains_unbounded_lazy_repeat(yes_arm, depth + 1) or (
                no_arm is not None and _contains_unbounded_lazy_repeat(no_arm, depth + 1)
            ):
                return True
    return False


def _is_start_anchored(sequence: object, flags: int, depth: int = 0) -> bool:
    if depth > _MAX_PARSE_DEPTH:
        return False
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return False
    if not items:
        return False
    opcode, argument = items[0]
    name = getattr(opcode, "name", str(opcode))
    if name == "AT":
        anchor = getattr(argument, "name", str(argument))
        return anchor == "AT_BEGINNING_STRING" or (
            anchor == "AT_BEGINNING" and not flags & re.MULTILINE
        )
    if name == "SUBPATTERN":
        _number, add_flags, del_flags, body = argument
        return _is_start_anchored(body, (flags | add_flags) & ~del_flags, depth + 1)
    if name == "BRANCH":
        return bool(argument[1]) and all(
            _is_start_anchored(branch, flags, depth + 1) for branch in argument[1]
        )
    return False


def _contains_scoped_flags(sequence: object, depth: int = 0) -> bool:
    """Whether local flags make atom equivalence context-sensitive."""
    if depth > _MAX_PARSE_DEPTH:
        return True
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return True
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name == "SUBPATTERN":
            _number, add_flags, del_flags, body = argument
            if add_flags or del_flags or _contains_scoped_flags(body, depth + 1):
                return True
        elif name in _REPEATS | {"ATOMIC_GROUP", "ASSERT", "ASSERT_NOT"}:
            child = (
                argument[2]
                if name in _REPEATS
                else argument[1]
                if name in {"ASSERT", "ASSERT_NOT"}
                else argument
            )
            if _contains_scoped_flags(child, depth + 1):
                return True
        elif name == "BRANCH" and any(
            _contains_scoped_flags(branch, depth + 1) for branch in argument[1]
        ):
            return True
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if _contains_scoped_flags(yes_arm, depth + 1) or (
                no_arm is not None and _contains_scoped_flags(no_arm, depth + 1)
            ):
                return True
    return False


def _collect_fixed_width_groups(
    sequence: object,
    groups: set[int],
    depth: int = 0,
) -> bool:
    if depth > _MAX_PARSE_DEPTH:
        return False
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return False
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name == "SUBPATTERN":
            number, _add_flags, _del_flags, body = argument
            try:
                minimum, maximum = body.getwidth()
            except AttributeError:
                return False
            if number is not None and minimum == maximum and maximum <= MAX_PATTERN_LENGTH:
                groups.add(number)
            if not _collect_fixed_width_groups(body, groups, depth + 1):
                return False
        elif name in _REPEATS:
            if not _collect_fixed_width_groups(argument[2], groups, depth + 1):
                return False
        elif name == "ATOMIC_GROUP":
            if not _collect_fixed_width_groups(argument, groups, depth + 1):
                return False
        elif name in {"ASSERT", "ASSERT_NOT"}:
            if not _collect_fixed_width_groups(argument[1], groups, depth + 1):
                return False
        elif name == "BRANCH":
            if not all(
                _collect_fixed_width_groups(branch, groups, depth + 1) for branch in argument[1]
            ):
                return False
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if not _collect_fixed_width_groups(yes_arm, groups, depth + 1):
                return False
            if no_arm is not None and not _collect_fixed_width_groups(no_arm, groups, depth + 1):
                return False
    return True


def _repeat_boundary_atoms(
    sequence: object,
    fixed_groups: frozenset[int],
    depth: int = 0,
) -> tuple[tuple[str, object], tuple[str, object]] | None:
    """Return boundaries for a positive, fixed-width character sequence."""
    if depth > _MAX_PARSE_DEPTH:
        return None
    try:
        items = list(sequence)  # type: ignore[call-overload]
        minimum, maximum = sequence.getwidth()  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None
    if minimum <= 0 or minimum != maximum:
        return None

    atoms: list[tuple[str, object]] = []
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name in {"LITERAL", "NOT_LITERAL", "IN", "ANY", "CATEGORY"} or (
            name == "GROUPREF" and argument in fixed_groups
        ):
            atoms.append((name, argument))
        elif name == "SUBPATTERN":
            boundaries = _repeat_boundary_atoms(argument[-1], fixed_groups, depth + 1)
            if boundaries is None:
                return None
            atoms.extend(boundaries if boundaries[0] != boundaries[1] else boundaries[:1])
        else:
            return None
    if not atoms:
        return None
    return atoms[0], atoms[-1]


def _normalise_atom(atom: tuple[str, object]) -> tuple[str, object]:
    name, argument = atom
    if name != "IN":
        return atom
    try:
        members = list(argument)  # type: ignore[call-overload]
    except TypeError:
        return atom
    if len(members) != 1:
        return atom
    opcode, member_argument = members[0]
    member_name = getattr(opcode, "name", str(opcode))
    if member_name in {"LITERAL", "CATEGORY"}:
        return member_name, member_argument
    return atom


def _category_matches_literal(category: object, character: str) -> bool:
    name = getattr(category, "name", str(category))
    if name.endswith("_SPACE"):
        return character.isspace()
    if name.endswith("_DIGIT"):
        return character.isdecimal()
    if name.endswith("_WORD"):
        return character == "_" or character.isalnum()
    if name.endswith("_LINEBREAK"):
        return character in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
    return True


def _atoms_are_disjoint(left: tuple[str, object], right: tuple[str, object]) -> bool:
    left_name, left_argument = _normalise_atom(left)
    right_name, right_argument = _normalise_atom(right)
    if left_name == "LITERAL" and right_name == "LITERAL":
        left_character = chr(left_argument)  # type: ignore[arg-type]
        right_character = chr(right_argument)  # type: ignore[arg-type]
        # A trusted one-literal check preserves Python's full Unicode
        # IGNORECASE equivalence.  Always using the wider equivalence is also
        # conservative for case-sensitive model patterns.
        return (
            re.fullmatch(
                re.escape(left_character),
                right_character,
                re.IGNORECASE,
            )
            is None
        )
    if left_name == "LITERAL" and right_name == "CATEGORY":
        return not _category_matches_literal(right_argument, chr(left_argument))  # type: ignore[arg-type]
    if left_name == "CATEGORY" and right_name == "LITERAL":
        return not _category_matches_literal(left_argument, chr(right_argument))  # type: ignore[arg-type]
    if left_name == "CATEGORY" and right_name == "CATEGORY":
        left_category = getattr(left_argument, "name", str(left_argument))
        right_category = getattr(right_argument, "name", str(right_argument))
        positive = {"CATEGORY_SPACE", "CATEGORY_DIGIT", "CATEGORY_WORD", "CATEGORY_LINEBREAK"}
        if left_category not in positive or right_category not in positive:
            return False
        if left_category == right_category:
            return False
        return {left_category, right_category} not in (
            {"CATEGORY_DIGIT", "CATEGORY_WORD"},
            {"CATEGORY_LINEBREAK", "CATEGORY_SPACE"},
        )
    return False


def _literal_atom(character: str) -> tuple[str, object]:
    return "LITERAL", ord(character)


def _has_ambiguous_repeats(
    sequence: object,
    fixed_groups: frozenset[int],
    allow_repeated_assertions: bool,
    depth: int = 0,
) -> bool:
    if depth > _MAX_PARSE_DEPTH:
        return True
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:
        return True

    outside_repeats = sum(
        1 for opcode, _argument in items if getattr(opcode, "name", str(opcode)) in _REPEATS
    )
    opaque_repeat_regions = sum(
        1
        for opcode, argument in items
        if getattr(opcode, "name", str(opcode)) in {"SUBPATTERN", "ATOMIC_GROUP"}
        and _contains_repeat(
            argument[-1] if getattr(opcode, "name", str(opcode)) == "SUBPATTERN" else argument
        )
    )
    if opaque_repeat_regions and outside_repeats + opaque_repeat_regions > 1:
        return True
    if outside_repeats:
        for opcode, argument in items:
            name = getattr(opcode, "name", str(opcode))
            if name == "BRANCH" and any(_contains_repeat(branch) for branch in argument[1]):
                return True
            if name in {"ASSERT", "ASSERT_NOT"} and _contains_repeat(argument[1]):
                return True
            if name == "ATOMIC_GROUP" and _contains_repeat(argument):
                return True
            if name == "GROUPREF_EXISTS":
                _reference, yes_arm, no_arm = argument
                if _contains_repeat(yes_arm) or (no_arm is not None and _contains_repeat(no_arm)):
                    return True

    previous_repeat: tuple[tuple[str, object], tuple[str, object]] | None = None
    separator_literals: list[str] = []
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name == "LITERAL":
            separator_literals.append(chr(argument))
            continue
        if name == "SUBPATTERN":
            if _has_ambiguous_repeats(
                argument[-1], fixed_groups, allow_repeated_assertions, depth + 1
            ):
                return True
            try:
                subitems = list(argument[-1])
            except TypeError:
                return True
            if all(getattr(op, "name", str(op)) == "LITERAL" for op, _arg in subitems):
                separator_literals.extend(chr(arg) for _op, arg in subitems)
            continue
        if name == "BRANCH":
            if any(
                _has_ambiguous_repeats(branch, fixed_groups, allow_repeated_assertions, depth + 1)
                for branch in argument[1]
            ):
                return True
            continue
        if name in {"ASSERT", "ASSERT_NOT"}:
            if _contains_unbounded_repeat(argument[1]) and not allow_repeated_assertions:
                return True
            if _has_ambiguous_repeats(
                argument[1], fixed_groups, allow_repeated_assertions, depth + 1
            ):
                return True
            continue
        if name == "ATOMIC_GROUP":
            if _has_ambiguous_repeats(argument, fixed_groups, allow_repeated_assertions, depth + 1):
                return True
            continue
        if name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if _contains_repeat(yes_arm) or (no_arm is not None and _contains_repeat(no_arm)):
                return True
            if _has_ambiguous_repeats(yes_arm, fixed_groups, allow_repeated_assertions, depth + 1):
                return True
            if no_arm is not None and _has_ambiguous_repeats(
                no_arm, fixed_groups, allow_repeated_assertions, depth + 1
            ):
                return True
            continue
        if name == "GROUPREF":
            if argument not in fixed_groups:
                return True
            continue
        if name in {"NOT_LITERAL", "IN", "ANY", "CATEGORY", "AT", "FAILURE"}:
            continue
        if name not in _REPEATS:
            return True

        minimum, maximum, body = argument
        if minimum > MAX_SUBJECT_LENGTH:
            return True
        if maximum != regex_parser.MAXREPEAT and maximum > MAX_SUBJECT_LENGTH:
            return True
        boundaries = _repeat_boundary_atoms(body, fixed_groups, depth + 1)
        if boundaries is None:
            return True
        if minimum == maximum:
            separator_literals.clear()
            continue
        if previous_repeat is not None:
            left_boundary = previous_repeat[1]
            right_boundary = boundaries[0]
            hard_separator = any(
                _atoms_are_disjoint(left_boundary, _literal_atom(character))
                and _atoms_are_disjoint(_literal_atom(character), right_boundary)
                for character in separator_literals
            )
            boundary_is_disjoint = not separator_literals and _atoms_are_disjoint(
                left_boundary, right_boundary
            )
            if not hard_separator and not boundary_is_disjoint:
                return True
        previous_repeat = boundaries
        separator_literals.clear()
    return False


def pattern_has_bounded_execution(
    pattern: str,
    flags: int = 0,
    *,
    target_directed: bool = False,
) -> bool:
    """Prove a regex belongs to the verifier's bounded execution subset."""
    try:
        parsed = regex_parser.parse(pattern, flags)
    except Exception:
        return False
    groups: set[int] = set()
    if not _collect_fixed_width_groups(parsed, groups):
        return False
    if _contains_scoped_flags(parsed) and _contains_repeat(parsed):
        return False
    _minimum, maximum = parsed.getwidth()
    effective_flags = parsed.state.flags
    start_anchored = _is_start_anchored(parsed, effective_flags)
    if maximum > 0 and not start_anchored and _contains_unbounded_lazy_repeat(parsed):
        return False
    allow_repeated_assertions = (target_directed and maximum == 0) or start_anchored
    return not _has_ambiguous_repeats(
        parsed,
        frozenset(groups),
        allow_repeated_assertions,
    )
