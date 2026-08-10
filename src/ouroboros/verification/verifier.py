"""SpecVerifier — reads actual source files and checks assertions.

Handles T1 (constant/config) and T2 (structural) verification tiers
by scanning project files with regex patterns. T3/T4 are skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import glob
from itertools import chain
import logging
import os
import re

# The standard-library parser permits inspection without running model regexes.
# It is private but more reliable than maintaining a second regex parser here.
from re import _parser as regex_parser
from typing import NamedTuple

from ouroboros.verification import declaration_binding as decl
from ouroboros.verification.binding import (
    acceptance_polarity,
    acceptance_targets,
    literal_is_bound,
    literal_spans,
)
from ouroboros.verification.models import (
    ACVerificationReport,
    EvidencePolarity,
    SpecAssertion,
    SpecVerificationResult,
    SpecVerificationSummary,
    VerificationOutcome,
    VerificationTier,
)
from ouroboros.verification.regex_safety import pattern_has_bounded_execution
from ouroboros.verification.source_safety import (
    is_known_non_evidence_format,
    mask_non_executable_source,
)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 50 * 1024  # 50KB per file
MAX_FILES_PER_HINT = 100
MAX_PATTERN_LENGTH = 200  # Limit LLM-generated regex length to reduce ReDoS risk
MAX_SCALAR_LENGTH = 4096
_NOISE_DIRECTORY_NAMES = frozenset({"__pycache__", ".git", "node_modules", ".venv", ".tox"})

# Whether a pattern can match the empty string is decided by *reading* it, never
# by running it. Running it is what a hostile pattern is waiting for: `(?:)
# {100000000}` is fifteen characters, passes the length limit, compiles
# instantly, and then takes longer than any timeout to match a subject with
# nothing in it — a stall reached before the verifier has even looked for a file.
# Capping the pattern's length does not cap a repetition count written inside it,
# because that count is one number rather than one character each.
#
# The parse tree is bounded by the pattern's length, and in it a repetition count
# is a number that is read rather than a number of steps that are taken. So the
# analysis is linear in the pattern no matter what the pattern says.
_MAX_PARSE_DEPTH = 40


def _literal_run_binds_target(run: str, target: str, flags: int) -> bool:
    """Apply Python's literal case semantics, then trusted token boundaries."""
    if not flags & re.IGNORECASE:
        return literal_is_bound(run, target)
    literal_flags = re.IGNORECASE | (re.ASCII if flags & re.ASCII else 0)
    for match in re.finditer(re.escape(target), run, literal_flags):
        matched_spelling = run[match.start() : match.end()]
        if (match.start(), match.end()) in literal_spans(run, matched_spelling):
            return True
    return False


def _sequence_requires_target_literal(
    sequence: object,
    target: str,
    flags: int,
    depth: int = 0,
) -> bool:
    """Whether every successful path through a sequence names ``target``.

    A required sequential component is sufficient. A branch is required only
    when every arm requires the target, and a repeat only when it executes at
    least once. Assertions are excluded here and handled as positional finite
    witnesses instead.
    """
    if depth > _MAX_PARSE_DEPTH:
        return False
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:  # pragma: no cover - the parser always supplies a sequence
        return False

    current: list[str] = []

    def current_requires_target() -> bool:
        run = "".join(current)
        current.clear()
        return bool(run) and _literal_run_binds_target(run, target, flags)

    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name == "LITERAL":
            current.append(chr(argument))
            continue
        if current_requires_target():
            return True
        if name == "SUBPATTERN":
            if _sequence_requires_target_literal(argument[-1], target, flags, depth + 1):
                return True
        elif name == "ATOMIC_GROUP":
            if _sequence_requires_target_literal(argument, target, flags, depth + 1):
                return True
        elif name == "BRANCH":
            branches = argument[1]
            if branches and all(
                _sequence_requires_target_literal(branch, target, flags, depth + 1)
                for branch in branches
            ):
                return True
        elif name in _REPEATS:
            minimum = argument[0]
            if minimum > 0 and _sequence_requires_target_literal(
                argument[-1], target, flags, depth + 1
            ):
                return True
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if (
                no_arm is not None
                and _sequence_requires_target_literal(yes_arm, target, flags, depth + 1)
                and _sequence_requires_target_literal(no_arm, target, flags, depth + 1)
            ):
                return True
    return current_requires_target()


_LITERAL_MATCH_FLAGS = re.IGNORECASE | re.ASCII | re.LOCALE | re.UNICODE


def _has_scoped_literal_flags(sequence: object, depth: int = 0) -> bool:
    """Whether scoped flags can change how a fixed target literal is matched."""
    if depth > _MAX_PARSE_DEPTH:
        return True
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:  # pragma: no cover - the parser always supplies a sequence
        return True
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name == "SUBPATTERN":
            _group, add_flags, del_flags, body = argument
            if (add_flags | del_flags) & _LITERAL_MATCH_FLAGS:
                return True
            if _has_scoped_literal_flags(body, depth + 1):
                return True
        elif name in _REPEATS:
            if _has_scoped_literal_flags(argument[-1], depth + 1):
                return True
        elif name == "ATOMIC_GROUP":
            if _has_scoped_literal_flags(argument, depth + 1):
                return True
        elif name == "BRANCH":
            if any(_has_scoped_literal_flags(branch, depth + 1) for branch in argument[1]):
                return True
        elif name in {"ASSERT", "ASSERT_NOT"}:
            if _has_scoped_literal_flags(argument[1], depth + 1):
                return True
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if _has_scoped_literal_flags(yes_arm, depth + 1) or (
                no_arm is not None and _has_scoped_literal_flags(no_arm, depth + 1)
            ):
                return True
    return False


def _pattern_consumes_target_literal(pattern: re.Pattern, target: str) -> bool:
    """Whether every successful consuming path names the complete target."""
    try:
        parsed = regex_parser.parse(pattern.pattern, pattern.flags)
    except Exception:  # pragma: no cover - the pattern is already compiled
        return False
    if _has_scoped_literal_flags(parsed):
        return False
    return _sequence_requires_target_literal(parsed, target, pattern.flags)


def _finite_target_assertions(
    pattern: re.Pattern,
    target: str,
) -> tuple[tuple[int, int, int], ...]:
    """Return finite positive witnesses as ``(direction, offset, width)``."""
    try:
        parsed = regex_parser.parse(pattern.pattern, pattern.flags)
    except Exception:  # pragma: no cover - the pattern is already compiled
        return ()
    if _has_scoped_literal_flags(parsed):
        return ()
    witnesses: list[tuple[int, int, int]] = []
    for direction, offset, assertion_body in _direct_positive_assertions(parsed):
        _minimum, maximum = assertion_body.getwidth()  # type: ignore[attr-defined]
        if maximum >= regex_parser.MAXWIDTH:
            continue
        if _sequence_requires_target_literal(assertion_body, target, pattern.flags):
            witnesses.append((direction, offset, maximum))
    return tuple(witnesses)


def _fixed_item_width(sequence: object, item: tuple[object, object]) -> int | None:
    """Return an item's exact finite width, or ``None`` when it can vary."""
    try:
        state = sequence.state  # type: ignore[attr-defined]
        minimum, maximum = regex_parser.SubPattern(state, [item]).getwidth()
    except (AttributeError, TypeError, ValueError):
        return None
    if minimum != maximum or maximum >= regex_parser.MAXWIDTH:
        return None
    return maximum


def _direct_positive_assertions(
    sequence: object,
    depth: int = 0,
    offset: int = 0,
) -> Iterator[tuple[int, int, object]]:
    """Yield required positive assertions at their exact match-relative offset.

    Plain groups and atomic groups do not move a zero-width match position and
    are safe to unwrap while their preceding width remains exact. Branches and
    repetitions are deliberately not unwrapped: an assertion on an untaken or
    optional path cannot be a required witness. Once a variable-width item is
    crossed, later assertion positions are unknown and fail closed.
    """
    if depth > _MAX_PARSE_DEPTH:
        return
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:  # pragma: no cover - the parser always supplies a sequence
        return
    current_offset: int | None = offset
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if current_offset is not None:
            if name == "ASSERT" and argument[0] in {-1, 1}:
                yield argument[0], current_offset, argument[1]
            elif name == "SUBPATTERN":
                yield from _direct_positive_assertions(argument[-1], depth + 1, current_offset)
            elif name == "ATOMIC_GROUP":
                yield from _direct_positive_assertions(argument, depth + 1, current_offset)
        width = _fixed_item_width(sequence, (opcode, argument))
        if current_offset is None or width is None:
            current_offset = None
        else:
            current_offset += width


def _literal_occurrences(
    subject: str,
    literal: str,
    start: int,
    end: int,
    flags: int,
) -> tuple[tuple[int, int], ...]:
    """Return all raw literal occurrences in a bounded runtime witness window."""
    start = max(0, start)
    end = min(len(subject), end)
    if start >= end or not literal:
        return ()
    window = subject[start:end]
    literal_flags = re.IGNORECASE if flags & re.IGNORECASE else 0
    return tuple(
        (start + match.start(), start + match.end())
        for match in re.finditer(re.escape(literal), window, literal_flags)
    )


def _window_has_one_trusted_target(
    subject: str,
    literal: str,
    start: int,
    end: int,
    trusted_spans: tuple[tuple[int, int], ...],
    flags: int,
) -> bool:
    """Tie a mandatory regex literal to one unambiguous trusted source span."""
    occurrences = _literal_occurrences(subject, literal, start, end, flags)
    return len(occurrences) == 1 and occurrences[0] in trusted_spans


# Consumes at least one character, so a sequence containing one cannot be empty.
_CONSUMING = frozenset({"LITERAL", "NOT_LITERAL", "IN", "ANY", "RANGE", "CATEGORY"})
# Anchors consume nothing, but they still either hold or fail on a subject with
# nothing in it, and which one it is has to be read off the individual anchor.
# `^`, `\A`, `$` and `\Z` hold at the sole position of an empty subject; `\b`
# needs a word character on exactly one side and so can never hold there.
# Calling `\b` zero-width would be the safe error on its own — a pattern wrongly
# called nullable is only refused — but `(?!\b)` negates it into the unsafe one,
# so each anchor is classified by what it actually does.
_ANCHORS_ALWAYS_HOLDING_ON_EMPTY = frozenset(
    {
        "AT_BEGINNING",
        "AT_BEGINNING_STRING",
        "AT_BEGINNING_LINE",
        "AT_END",
        "AT_END_STRING",
        "AT_END_LINE",
    }
)
_BOUNDARY = frozenset({"AT_BOUNDARY", "AT_LOC_BOUNDARY", "AT_UNI_BOUNDARY"})
_NON_BOUNDARY = frozenset({"AT_NON_BOUNDARY", "AT_LOC_NON_BOUNDARY", "AT_UNI_NON_BOUNDARY"})

# `\B` is the one anchor whose answer here is not a fact about regular
# expressions but a fact about this interpreter: before 3.14 it required a
# position between two characters and so failed on an empty subject; from 3.14
# the sole position of an empty subject is a non-boundary and it holds. Asking
# the engine once, with a constant pattern of ours against a constant subject,
# is both correct on every version this runs on and correct on versions that do
# not exist yet — which hard-coding a version number is not. Nothing
# model-supplied is run: the pattern here is two characters and this file's own.
_NON_BOUNDARY_HOLDS_ON_EMPTY = re.search(r"\B", "") is not None

_ANCHORS_HOLDING_ON_EMPTY = _ANCHORS_ALWAYS_HOLDING_ON_EMPTY | (
    _NON_BOUNDARY if _NON_BOUNDARY_HOLDS_ON_EMPTY else frozenset()
)
_ANCHORS_FAILING_ON_EMPTY = _BOUNDARY | (
    frozenset() if _NON_BOUNDARY_HOLDS_ON_EMPTY else _NON_BOUNDARY
)
_REPEATS = frozenset({"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"})

# The two ends of the *subject* a match can be pinned to. Withholding one of
# them asks a different question of the same pattern: not "can this match
# nothing" but "can it match at all without that end of the subject".
#
# Only `\A` and `\Z` are here, and `^` and `$` are not. The parser emits the
# same `AT_BEGINNING` for `^` whether or not `re.MULTILINE` is on — it is the
# compiler, working from flags, that turns it into a line anchor — so a reading
# that counted `^` as pinned to the file would admit `(?m)^$`, which is
# satisfied by a blank line inside a file full of content. `^` is therefore
# read as pinning to nothing at all here, which costs a pattern written that
# way the same formal failure it already gets today.
_START_ANCHORS = frozenset({"AT_BEGINNING_STRING"})
_END_ANCHORS = frozenset({"AT_END_STRING"})


class _Group(NamedTuple):
    """What a capture group did on a match that consumed nothing."""

    empty: bool | None
    """Whether the group's own body can match nothing."""
    took_part: bool | None
    """Whether the group can have participated in such a match."""


def _skippable(on_path: bool | None) -> bool | None:
    """A path the match may have gone around — unless it was never on it at all.

    A repeat that may run zero times, an alternative, and a conditional's arms
    are all paths the match need not have taken. Inside a negative assertion the
    answer is already the stronger `False` and stays there: something the match
    certainly did not walk does not become something it merely might have.
    """
    return False if on_path is False else None


def _participation(body_empty: bool | None, on_path: bool | None) -> bool | None:
    """Whether a group reached here can have taken part in an empty match.

    A group the match certainly never walked certainly did not take part, whether
    or not its body consumes. Otherwise a group whose body certainly consumes
    certainly did not take part either: had it run, the match would not have been
    empty. A group whose body can be empty took part only if the path through it
    is the only path; where the path may have been skipped it may equally have
    taken part or not, and that is not something this reading can settle.
    """
    if on_path is False:
        return False
    if body_empty is False:
        return False
    if body_empty is True and on_path is True:
        return True
    return None


def _all_empty(answers: list[bool | None]) -> bool | None:
    """Can a run of things, taken together, match nothing?

    Only if each of them can. One that certainly cannot settles it for the whole
    run; otherwise a single unknown leaves the run unknown.
    """
    if any(answer is False for answer in answers):
        return False
    if all(answer is True for answer in answers):
        return True
    return None


def _any_empty(answers: list[bool | None]) -> bool | None:
    """Can one of a set of alternatives match nothing?

    Yes as soon as one certainly can; no only if every one certainly cannot.
    """
    if any(answer is True for answer in answers):
        return True
    if all(answer is False for answer in answers):
        return False
    return None


def _agreed(answers: list[bool | None]) -> bool | None:
    """The one answer they all give, or unknown if they do not all give it.

    For a construct where which part runs is itself undecidable — a conditional
    on whether a group took part — the result is only certain when the parts
    cannot disagree.
    """
    if all(answer is True for answer in answers):
        return True
    if all(answer is False for answer in answers):
        return False
    return None


def _can_match_nothing(
    sequence: object,
    depth: int = 0,
    groups: dict[int, _Group] | None = None,
    on_path: bool | None = True,
    withheld: frozenset[str] = frozenset(),
    porous: bool = False,
) -> bool | None:
    """Whether a parsed regex can match the empty string, read rather than run.

    Three answers, not two: True, False, and None for "this reading cannot tell".
    An unknown construct or a tree deeper than `_MAX_PARSE_DEPTH` is None, and so
    is anything built out of a None.

    The third answer is what keeps the doubt pointed at refusal. Refusal is the
    safe direction — a pattern wrongly called nullable costs an honest criterion
    a formal failure, while one wrongly called discriminating is admitted as
    evidence, which is the defect this file exists to close — but "safe" is a
    direction, and `(?!…)` reverses direction. Answering an unreadable inner
    pattern with a plain True and then negating it produces a confident False:
    the guard would report *certainty that the pattern discriminates* on the
    strength of not having understood it. None negates to None, so the doubt
    survives the negation and `_matches_the_empty_string` still refuses.

    `groups` carries, for each capture group once seen, what it can itself match
    and whether it can have taken part in an empty match — the first so a
    backreference can be judged by what it refers to, the second so a conditional
    can be judged by which arm it would run.

    `on_path` says what the match did with the path being walked, and is itself
    three-valued: True for a path the match certainly took, False for one it
    certainly did not — the inside of a negative assertion, which succeeds only
    by failing — and None where it may have gone either way.

    `withheld` names anchors to read as ones that cannot hold, and `porous`
    reads a consuming atom as something a match can cross rather than something
    that ends the question. Together they turn this walk into a second question
    asked of the same tree — "can this match somewhere that is not the start of
    the subject" — whose answer is what tells a pattern pinned to both ends of
    the file apart from one that matches anywhere in it. Defaults leave the
    first question exactly as it was.
    """
    if groups is None:
        groups = {}
    if depth > _MAX_PARSE_DEPTH:
        return None
    answers: list[bool | None] = []
    for opcode, argument in sequence:  # type: ignore[attr-defined]
        name = getattr(opcode, "name", str(opcode))
        if name in _CONSUMING:
            # Recorded rather than returned. One consuming atom already settles
            # this sequence — `_all_empty` answers False as soon as it sees one —
            # but returning here would stop the walk, and captures written after
            # it would never be recorded. A conditional outside can still ask
            # what those captures did, and inside a negative assertion, which is
            # where such a sequence takes part in a match by failing, the answer
            # is that they certainly did not.
            #
            # Unless the walk was asked to be porous, where a consuming atom is
            # something the match crosses rather than something that settles it:
            # the second question is about *where* a match can be, not about how
            # long it is, and a subject free to hold whatever the atom wants is
            # a subject the atom does not rule out.
            answers.append(porous)
        elif name == "FAILURE":
            # An assertion that can never hold. From 3.13 the parser folds `(?!)`
            # and `(?!(?:))` into this single opcode; before that the same source
            # arrives as an `ASSERT_NOT` with an empty body, which this reading
            # already answered. Nothing matches it, a subject with nothing in it
            # included, so the answer is the same False — and reading it as an
            # unknown construct instead refused `(?!)|CameraProvider` on exactly
            # the interpreters that do the folding and nowhere else.
            answers.append(False)
        elif name == "AT":
            anchor = getattr(argument, "name", str(argument))
            # A withheld anchor is read as one that cannot hold. That is not a
            # claim about the anchor — it is how the second and third walks ask
            # their narrower question: with the start of the subject withheld,
            # a `True` answer would mean the pattern can match nothing without
            # ever pinning to the start, and a `False` means every empty match
            # it has needs that end.
            if anchor in withheld:
                answers.append(False)
            elif anchor in _ANCHORS_HOLDING_ON_EMPTY:
                continue
            elif porous:
                # A word boundary says something about the characters on either
                # side, and a subject free to hold whatever it likes is free to
                # hold those. It rules out no position on the porous reading.
                answers.append(True)
            else:
                answers.append(False if anchor in _ANCHORS_FAILING_ON_EMPTY else None)
        elif name in _REPEATS:
            minimum, _maximum, item = argument
            # Walked whatever the count, so that groups inside a repeat that may
            # run zero times are still recorded for a later backreference.
            item_empty = _can_match_nothing(
                item,
                depth + 1,
                groups,
                _skippable(on_path) if minimum == 0 else on_path,
                withheld,
                porous,
            )
            # A repeat that may run zero times is skippable; one that must run
            # is empty only if what it repeats is. The count itself is never
            # counted out.
            answers.append(True if minimum == 0 else item_empty)
        elif name == "SUBPATTERN":
            body_empty = _can_match_nothing(
                argument[-1], depth + 1, groups, on_path, withheld, porous
            )
            number = argument[0]
            if number is not None:
                groups[number] = _Group(body_empty, _participation(body_empty, on_path))
            answers.append(body_empty)
        elif name == "GROUPREF":
            # A backreference repeats whatever its group captured, so it is empty
            # only when that group can be. A reference to a group that certainly
            # did not take part refers to nothing captured, and this interpreter
            # fails such a reference rather than matching nothing with it — so
            # the sequence around it cannot match at all, empty included. A group
            # the walk has not reached is one the match has not reached either,
            # so it has certainly captured nothing here and the reference fails
            # the same way.
            seen = groups.get(argument)
            if seen is None or seen.took_part is False:
                answers.append(False)
            elif porous:
                # A reference that resolves repeats text the subject is free to
                # repeat, so on the porous reading it is crossed like any other
                # consuming atom rather than being empty.
                answers.append(True)
            else:
                answers.append(seen.empty)
        elif name == "GROUPREF_EXISTS":
            # `(?(1)yes|no)` runs the arm the group's participation selects. On a
            # match that consumed nothing, a group whose body consumes cannot have
            # taken part, so `(a)?(?(1)|b)` certainly runs `b` and certainly is
            # not empty. Only when participation itself is undecidable does the
            # conditional fall back on the arms having to agree.
            #
            # Which arm runs is only undecidable while participation is. Once
            # the group settles it, the selected arm is walked on the path the
            # conditional itself is on and the other one certainly does not run,
            # so a capture in the selected arm is as much on the path as the
            # conditional is — reading both arms as skippable left every such
            # capture unknown and refused the conditionals reading them.
            #
            # Nor is a group the walk has not reached an open question. A
            # conditional written before the group it names runs at a point the
            # match has not yet carried the group into, so nothing can have been
            # captured in it and the interpreter takes the `no` arm — every
            # time, on every version. Repetition does not reopen it: a repeat
            # matches nothing only if each of its runs does, and its first run
            # always meets the group unreached. Reading a forward reference as
            # unknown made the arms have to agree, and `(?(1)|a)(a?)` — which
            # plainly discriminates — was refused as evidence.
            reference, yes_arm, no_arm = argument
            seen_by_conditional = groups.get(reference)
            took_part = False if seen_by_conditional is None else seen_by_conditional.took_part
            if took_part is True:
                yes_path, no_path = on_path, False
            elif took_part is False:
                yes_path, no_path = False, on_path
            else:
                yes_path = no_path = _skippable(on_path)
            yes = _can_match_nothing(yes_arm, depth + 1, groups, yes_path, withheld, porous)
            no = (
                True
                if no_arm is None
                else _can_match_nothing(no_arm, depth + 1, groups, no_path, withheld, porous)
            )
            if took_part is True:
                answers.append(yes)
            elif took_part is False:
                answers.append(no)
            else:
                answers.append(_agreed([yes, no]))
        elif name == "ATOMIC_GROUP":
            answers.append(
                _can_match_nothing(argument, depth + 1, groups, on_path, withheld, porous)
            )
        elif name == "BRANCH":
            # Every branch is walked, not just up to the first empty one, so that
            # groups defined in a later branch are recorded too.
            #
            # Which branch runs is undecidable from outside, but not from inside:
            # a capture and a conditional reading it that sit in the same branch
            # stand or fall together, so within a branch the path is the one the
            # branch inherits. Handing every branch a skippable path instead made
            # participation unknown for its own captures, and `()(?(1)x|)` — read
            # correctly on its own — became unreadable the moment an alternative
            # was written beside it.
            #
            # Each branch therefore reads and records over its own copy of the
            # table, so one branch cannot answer a question about a capture that
            # only exists in a sibling it does not run with. What a branch
            # recorded merges back with participation weakened to "may have gone
            # either way", which is what the alternation makes it for anything
            # reading that capture from outside.
            #
            # Except when it is not a choice. On a subject with nothing in it
            # nothing can consume anything, so a branch that cannot match
            # nothing cannot run at all, and when that leaves a single branch
            # standing the alternation decides nothing: its captures took part
            # exactly as much as the alternation itself did. `(?:()|x)(?(1)a|)`
            # is that shape, and weakening it refused a pattern that plainly
            # discriminates.
            walked: list[tuple[bool | None, dict[int, _Group]]] = []
            for branch in argument[1]:
                local = dict(groups)
                walked.append(
                    (_can_match_nothing(branch, depth + 1, local, on_path, withheld, porous), local)
                )
            branch_answers = [answer for answer, _ in walked]
            settled = branch_answers.count(True) == 1 and all(
                answer is not None for answer in branch_answers
            )
            for answer, local in walked:
                for number, group in local.items():
                    if number in groups:
                        continue
                    if answer is False:
                        took = False
                    elif settled:
                        took = group.took_part
                    else:
                        took = _skippable(group.took_part)
                    groups[number] = _Group(group.empty, took)
            answers.append(_any_empty(branch_answers))
        elif name == "ASSERT":
            # On a subject with nothing in it there is nothing to either side, so
            # a lookaround holds exactly when what it looks for can be empty.
            # `(?=.*foo)` cannot, which is why a lookahead-only pattern stays
            # admissible evidence.
            #
            # A positive assertion has to hold for the match to happen, so the
            # path through it is not one the match could have gone around — a
            # capture inside it took part exactly as much as the same capture
            # written outside it would have. Calling it optional here is what
            # made `(?=())(?(1)x|)` unreadable and refused it as evidence.
            answers.append(
                _can_match_nothing(argument[1], depth + 1, groups, on_path, withheld, porous)
            )
        elif name == "ASSERT_NOT":
            # A negative assertion succeeds only where its body fails, and a
            # subpattern that fails leaves nothing captured behind it. So this is
            # the one path the match certainly did not walk, and every capture
            # inside it certainly did not take part.
            #
            # From outside. The body is still attempted, and while it is being
            # attempted its captures are as real as any other — a conditional
            # written inside the body reads the group beside it exactly as it
            # would anywhere else. Declaring them absent before the body had
            # been read made `(?!()(?(1)a|))` take its empty arm, so a pattern
            # that matches every file was admitted as evidence and published as
            # a formal PASS, which is the defect this file exists to close. So
            # the body is walked on the path it inherits, over its own copy of
            # the table, and only what escapes to the outside is `False`.
            local_to_body = dict(groups)
            inner = _can_match_nothing(
                argument[1], depth + 1, local_to_body, on_path, withheld, porous
            )
            for number, group in local_to_body.items():
                if number not in groups:
                    groups[number] = _Group(group.empty, False)
            if porous:
                # The porous reading asks whether the body can match *somewhere*,
                # and the negation of "somewhere" is not "nowhere" — `(?!x)` holds
                # at every position that is not followed by an `x`, however freely
                # an `x` can appear elsewhere. So only a body that can match
                # nowhere at all lets this assertion be read with certainty.
                answers.append(True if inner is False else None)
            else:
                answers.append(None if inner is None else not inner)
        else:
            answers.append(None)
    return _all_empty(answers)


def _matches_the_empty_string(pattern: str, flags: int = 0) -> bool:
    """Whether `pattern` can match a subject with nothing in it. Never runs it.

    Unknown counts as yes, which refuses the pattern as evidence.
    """
    try:
        parsed = regex_parser.parse(pattern, flags)
    except Exception:  # pragma: no cover - re.compile has already accepted this
        return True
    return _can_match_nothing(parsed) is not False


# Whitespace is the one thing a file can hold that no acceptance criterion is
# about, so it is the one thing a pattern may consume and still be read as a
# claim that the file holds nothing. `\s` is exactly this class; a literal space,
# tab or newline is a member of it.
_BLANK_CATEGORIES = frozenset({"CATEGORY_SPACE"})

# Character ranges are read, not enumerated, past this width. Every run of
# whitespace codepoints is a handful long — `\t` through `\r` is five — so a
# wide range is not a blank class that this failed to recognise, it is a class
# with content in it.
_MAX_BLANK_RANGE = 64


def _is_blank_codepoint(value: object) -> bool:
    return isinstance(value, int) and 0 <= value <= 0x10FFFF and chr(value).isspace()


def _is_blank_range(bounds: object) -> bool:
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        return False
    low, high = bounds
    if not isinstance(low, int) or not isinstance(high, int):
        return False
    if high - low > _MAX_BLANK_RANGE:
        return False
    return all(_is_blank_codepoint(code) for code in range(low, high + 1))


def _is_blank_class(items: object) -> bool:
    """Whether a `[...]` class can only ever match whitespace.

    A negated class is not one of them whatever it lists: `[^ ]` is every
    character that is not a space, which is what content is made of.
    """
    try:
        members = list(items)  # type: ignore[call-overload]
    except TypeError:  # pragma: no cover - the parser always hands back a list
        return False
    for opcode, argument in members:
        name = getattr(opcode, "name", str(opcode))
        if name == "LITERAL":
            if not _is_blank_codepoint(argument):
                return False
        elif name == "RANGE":
            if not _is_blank_range(argument):
                return False
        elif name == "CATEGORY":
            if getattr(argument, "name", str(argument)) not in _BLANK_CATEGORIES:
                return False
        else:
            return False
    return True


def _consumes_only_blank(sequence: object, depth: int = 0) -> bool:
    """Whether nothing this can consume is content.

    What a lookaround looks at is not consumed and so is never part of what
    matched; it is skipped here for that reason. A backreference is not, because
    it repeats whatever its group captured, and a group inside a lookaround can
    have captured anything at all.

    False for anything this cannot read, so an unrecognised construct narrows
    nothing.
    """
    if depth > _MAX_PARSE_DEPTH:
        return False
    try:
        items = list(sequence)  # type: ignore[call-overload]
    except TypeError:  # pragma: no cover - the parser always hands back a sequence
        return False
    for opcode, argument in items:
        name = getattr(opcode, "name", str(opcode))
        if name in ("AT", "ASSERT", "ASSERT_NOT", "FAILURE"):
            continue
        if name == "LITERAL":
            if not _is_blank_codepoint(argument):
                return False
        elif name == "IN":
            if not _is_blank_class(argument):
                return False
        elif name == "RANGE":
            if not _is_blank_range(argument):
                return False
        elif name == "CATEGORY":
            if getattr(argument, "name", str(argument)) not in _BLANK_CATEGORIES:
                return False
        elif name in _REPEATS or name == "SUBPATTERN":
            # Both carry the thing they wrap last: a repeat cannot consume what
            # its body does not, and a group consumes exactly its body.
            if not _consumes_only_blank(argument[-1], depth + 1):
                return False
        elif name == "ATOMIC_GROUP":
            if not _consumes_only_blank(argument, depth + 1):
                return False
        elif name == "BRANCH":
            if not all(_consumes_only_blank(branch, depth + 1) for branch in argument[1]):
                return False
        elif name == "GROUPREF_EXISTS":
            _reference, yes_arm, no_arm = argument
            if not _consumes_only_blank(yes_arm, depth + 1):
                return False
            if no_arm is not None and not _consumes_only_blank(no_arm, depth + 1):
                return False
        else:
            return False
    return True


def _matches_only_a_blank_subject(pattern: str, flags: int = 0) -> bool:
    r"""Whether the only file this can match is one with nothing in it.

    `\A\Z` and `\A\s*\Z` match a subject with nothing in it, and the empty-string
    rule refuses them for it. But the reason that rule exists does not reach
    them. A pattern that can match nothing *somewhere in the middle* matches
    every file there is, so its match is evidence of nothing; these match one
    file and no other, which makes them not the weakest evidence available but
    the sharpest. Refusing them failed the one criterion they are the right
    answer to — that a file be left empty — and left it needing a rescue that
    read the criterion's English instead, which is guesswork this file should
    not be doing.

    Three readings, none of which runs the pattern. Nothing it consumes may be
    content, or the file it matched had something in it. No match may begin
    anywhere but the start of the subject, and none may end anywhere but the
    end, or what matched was a blank stretch inside a file rather than the whole
    of one — asked by walking the same tree twice more with one end of the
    subject withheld, which forces every match to justify itself without that
    end. Both walks answering "cannot" is what says both ends were required.

    Only `\A` and `\Z` count as the ends of the subject, so `\A$` and `(?m)^$`
    are refused rather than admitted — see `_START_ANCHORS`. That costs a
    criterion written with `$` the failure it already gets today, and it is the
    only reading that stays sound without inspecting flags the parser has not
    applied.
    """
    try:
        parsed = regex_parser.parse(pattern, flags)
    except Exception:  # pragma: no cover - re.compile has already accepted this
        return False
    if not _consumes_only_blank(parsed):
        return False
    unpinned_at_start = _can_match_nothing(parsed, withheld=_START_ANCHORS, porous=True)
    unpinned_at_end = _can_match_nothing(parsed, withheld=_END_ANCHORS, porous=True)
    return unpinned_at_start is False and unpinned_at_end is False


def _skip_inline_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\f\v":
        index += 1
    return index


def _scan_scalar(text: str, index: int) -> tuple[str, int] | None:
    """Read one bounded scalar without truncating quoted source values."""
    index = _skip_inline_space(text, index)
    if index >= len(text) or text[index] in "\r\n":
        return None

    quote = text[index]
    if quote in {'"', "'"}:
        index += 1
        value: list[str] = []
        consumed = 0
        while index < len(text) and consumed <= MAX_SCALAR_LENGTH:
            char = text[index]
            if char in "\r\n":
                return None
            if char == quote:
                return "".join(value), index + 1
            if char == "\\":
                if index + 1 >= len(text) or text[index + 1] in "\r\n":
                    return None
                escaped = text[index + 1]
                if escaped in {quote, "\\"}:
                    value.append(escaped)
                else:
                    value.extend(("\\", escaped))
                index += 2
                consumed += 2
                continue
            value.append(char)
            index += 1
            consumed += 1
        return None

    start = index
    while (
        index < len(text)
        and index - start <= MAX_SCALAR_LENGTH
        and text[index] not in "\"'\r\n\t ,;)]}{"
    ):
        index += 1
    if index == start or index - start > MAX_SCALAR_LENGTH:
        return None
    return text[start:index], index


def _preceding_assignment_operator(text: str, index: int) -> str | None:
    index -= 1
    while index >= 0 and text[index] in " \t\f\v":
        index -= 1
    return text[index] if index >= 0 and text[index] in "=:" else None


def _has_complete_scalar_terminator(text: str, index: int, operator: str | None) -> bool:
    """Reject a scalar that is only the prefix of an assigned expression."""
    index = _skip_inline_space(text, index)
    if index >= len(text) or text[index] in "\r\n":
        return True
    if text[index] == "#":
        return True
    if operator == "=":
        return text[index] == ";"
    return text[index] in ",;)]}"


def _extract_following_scalar(content: str, index: int) -> str:
    """Extract a direct, assigned, or parenthesized scalar at index."""
    index = _skip_inline_space(content, index)
    if index < len(content) and content[index] in "=:":
        operator = content[index]
        scanned = _scan_scalar(content, index + 1)
        if scanned is None:
            return ""
        value, end = scanned
        return value if _has_complete_scalar_terminator(content, end, operator) else ""
    if index < len(content) and content[index] == "(":
        scanned = _scan_scalar(content, index + 1)
        if scanned is None:
            return ""
        value, end = scanned
        end = _skip_inline_space(content, end)
        if end >= len(content) or content[end] != ")":
            return ""
        return value if _has_complete_scalar_terminator(content, end + 1, None) else ""
    scanned = _scan_scalar(content, index)
    if scanned is None:
        return ""
    value, end = scanned
    operator = _preceding_assignment_operator(content, index)
    return value if _has_complete_scalar_terminator(content, end, operator) else ""


def _has_noise_directory(file_path: str, project_dir: str) -> bool:
    """Whether a path has an exact excluded directory component."""
    relative = os.path.relpath(file_path, project_dir)
    return any(component in _NOISE_DIRECTORY_NAMES for component in relative.split(os.sep)[:-1])


def _criterion_path_kind(ac_text: str, target: str) -> str | None:
    """Return the file-system kind explicitly bound to one criterion target."""
    if not target:
        return None
    literal = re.escape(target)
    quoted_literal = rf"[`'\"]?{literal}[`'\"]?"
    patterns = (
        rf"\b(?P<kind>file|directory)\b(?:\s+(?:named|called))?\s+"
        rf"{quoted_literal}(?![\w./-])",
        rf"(?<![\w./-]){quoted_literal}\s+(?P<kind>file|directory)\b",
    )
    kinds = {
        match.group("kind").casefold()
        for pattern in patterns
        for match in re.finditer(pattern, ac_text, re.IGNORECASE)
    }
    return next(iter(kinds)) if len(kinds) == 1 else None


@dataclass
class SpecVerifier:
    """Verifies spec assertions against actual project files.

    Reads source files and applies regex patterns to check whether
    the expected values/structures actually exist in the codebase. Strict mode
    is the safe default: unavailable and skipped evidence block an approval
    override. Exploratory callers may set ``strict=False`` to observe those
    outcomes without requesting an override; they still never become VERIFIED.
    """

    project_dir: str
    strict: bool = True

    def verify_all(
        self,
        assertions: tuple[SpecAssertion, ...],
        agent_results: dict[int, bool] | None = None,
    ) -> SpecVerificationSummary:
        """Verify all assertions against project files.

        Args:
            assertions: Assertions to verify. Input binding is fail-closed by
                default; False is only an explicit compatibility path for
                trusted direct callers.
            agent_results: Map of ac_index → agent-reported pass/fail.

        Returns:
            SpecVerificationSummary with all results.
        """
        if not assertions:
            return SpecVerificationSummary(project_dir=self.project_dir, strict=self.strict)

        agent_results = agent_results or {}

        # Group assertions by AC index
        by_ac: dict[int, list[SpecAssertion]] = {}
        for a in assertions:
            by_ac.setdefault(a.ac_index, []).append(a)

        reports: list[ACVerificationReport] = []
        for ac_idx in sorted(by_ac.keys()):
            ac_assertions = by_ac[ac_idx]
            ac_text = ac_assertions[0].ac_text if ac_assertions else ""
            agent_pass = agent_results.get(ac_idx, True)

            # One AC index has exactly one caller-authored criterion identity.
            # Mixing assertion text under that index lets evidence for a second
            # criterion inherit the first report label, so reject the whole
            # group before any assertion can contribute positive evidence.
            if any(assertion.ac_text != ac_text for assertion in ac_assertions):
                conflicting = sorted({assertion.ac_text for assertion in ac_assertions})
                result = SpecVerificationResult(
                    assertion=ac_assertions[0],
                    verified=False,
                    discrepancy=agent_pass,
                    detail=(
                        f"Conflicting criterion text for AC {ac_idx}: "
                        + "; ".join(repr(text) for text in conflicting)
                    ),
                )
                reports.append(
                    ACVerificationReport(
                        ac_index=ac_idx,
                        ac_text=ac_text,
                        results=(result,),
                        agent_reported_pass=agent_pass,
                    )
                )
                continue

            results: list[SpecVerificationResult] = []
            for assertion in ac_assertions:
                results.append(self._verify_one(assertion))

            reports.append(
                ACVerificationReport(
                    ac_index=ac_idx,
                    ac_text=ac_text,
                    results=tuple(results),
                    agent_reported_pass=agent_pass,
                )
            )

        return SpecVerificationSummary.from_reports(
            tuple(reports),
            project_dir=self.project_dir,
            strict=self.strict,
        )

    def _compile_or_none(self, pattern: str, flags: int = 0) -> re.Pattern | None:
        """Compile a model-supplied regex, refusing one that is unusable as a regex."""
        if len(pattern) > MAX_PATTERN_LENGTH:
            logger.warning("Regex pattern too long (%d chars), skipping", len(pattern))
            return None
        try:
            return re.compile(pattern, flags)
        except (re.error, OverflowError) as e:
            logger.warning("Invalid regex pattern: %s", e)
            return None

    def _searches_one_named_file(self, file_hint: str | None, candidates: list[str] | None) -> bool:
        """Whether this search is about one file the hint named outright.

        A glob is not that, however few files it happens to match today. `\\A\\Z`
        over `**/*.py` asks whether the project contains *any* empty file, and in
        a Python project it does — some package marker no criterion mentioned. A
        hint that names one path asks about that path.
        """
        if not file_hint or any(char in file_hint for char in "*?["):
            return False
        return candidates is not None and len(candidates) == 1

    def _safe_compile(
        self,
        pattern: str,
        flags: int = 0,
        file_hint: str | None = None,
        candidates: list[str] | None = None,
    ) -> re.Pattern | None:
        """Compile a model-supplied regex, refusing one that cannot be evidence."""
        compiled = self._compile_or_none(pattern, flags)
        if compiled is None:
            return None
        if _matches_the_empty_string(pattern, flags):
            # `.*` and `x?` and `(?:)` and `|` and `^` match anywhere in any file,
            # so they verified whatever criterion they were handed — the match is
            # not evidence of the criterion. `\A\Z` shares the empty match and
            # nothing else: pinned to both ends of the subject and consuming no
            # content, it holds for a file with nothing in it and fails for every
            # other, which is discrimination rather than the absence of it. So it
            # is admitted where it is being asked about one named file, and the
            # search below answers the criterion from that file directly.
            if self._searches_one_named_file(file_hint, candidates) and (
                _matches_only_a_blank_subject(pattern, flags)
            ):
                return compiled
            logger.warning(
                "Regex pattern can match without criterion content, skipping: %r", pattern
            )
            return None
        return compiled

    def _verify_one(self, assertion: SpecAssertion) -> SpecVerificationResult:
        """Verify one assertion, including tiers this scanner deliberately skips."""
        if assertion.tier == VerificationTier.T1_CONSTANT:
            return self._verify_constant(assertion)
        if assertion.tier == VerificationTier.T2_STRUCTURAL:
            return self._verify_structural(assertion)
        if assertion.tier == VerificationTier.T3_BEHAVIORAL:
            detail = "Behavioral assertion requires test execution or semantic analysis"
        else:
            detail = "Subjective assertion is not independently verifiable by source scanning"
        return SpecVerificationResult(
            assertion=assertion,
            outcome=VerificationOutcome.SKIPPED,
            detail=detail,
        )

    def _evidence_targets(self, assertion: SpecAssertion) -> tuple[str, ...]:
        """Return targets derived from the caller-authored criterion.

        The extraction model controls the regex, hint, expected value, and
        description. None of those may redirect evidence away from ``ac_text``.
        For T2 an expected name is preferred only after proving it occurs in the
        actual AC; otherwise targets are derived from that AC directly. T1 binds
        the declaration/key here and checks the scalar separately.
        """
        targets = acceptance_targets(
            assertion.ac_text,
            assertion.expected_value,
            prefer_expected=assertion.tier == VerificationTier.T2_STRUCTURAL,
            pattern=assertion.pattern,
        )
        if assertion.tier == VerificationTier.T2_STRUCTURAL:
            expected = assertion.expected_value.strip()
            targets = tuple(target for target in targets if target == expected)
        if (
            not targets
            and assertion.file_hint
            and not any(char in assertion.file_hint for char in "*?[")
            and assertion.file_hint.casefold() in assertion.ac_text.casefold()
        ):
            targets = (assertion.file_hint,)
        if assertion.evidence_targets and assertion.evidence_targets != targets:
            return ()
        return targets

    def _evidence_polarity(self, assertion: SpecAssertion) -> EvidencePolarity | None:
        """Re-derive polarity from trusted AC text and reject stale/tampered metadata."""
        if not assertion.input_binding_required:
            return EvidencePolarity.REQUIRED
        polarity = acceptance_polarity(assertion.ac_text, self._evidence_targets(assertion))
        if polarity is None or polarity != assertion.evidence_polarity:
            return None
        return polarity

    def _find_bound_match(
        self,
        pattern: re.Pattern,
        searched_text: str,
        assertion: SpecAssertion,
        *,
        binding_text: str | None = None,
        filename_binding: bool = False,
    ) -> tuple[re.Match, str] | None:
        """Find a regex match that is grounded in the acceptance-criterion input.

        A compiling, non-nullable regex is still model output. It becomes
        evidence only when the subject it matched contains a literal target from
        the caller's actual AC. For consuming matches the target must overlap
        the matched span. Zero-width lookarounds are preserved by requiring the
        target in a finite positive lookaround anchored at the match position;
        this keeps discriminating lookaheads valid while global co-presence
        lookaheads cannot verify unrelated evidence.

        ``binding_text`` is used for whole-file blank evidence: the regex proves
        content is absent, while the criterion target binds that proof to the
        project-relative file name.
        """
        return next(
            self._bound_matches(
                pattern,
                searched_text,
                assertion,
                binding_text=binding_text,
                filename_binding=filename_binding,
            ),
            None,
        )

    def _bound_matches(
        self,
        pattern: re.Pattern,
        searched_text: str,
        assertion: SpecAssertion,
        *,
        binding_text: str | None = None,
        filename_binding: bool = False,
    ) -> Iterator[tuple[re.Match, str]]:
        """Return all criterion-bound matches in deterministic search order."""
        if not pattern_has_bounded_execution(
            pattern.pattern,
            pattern.flags,
            externally_anchored=filename_binding,
        ):
            logger.warning(
                "Regex has unbounded backtracking risk, skipping: %r",
                pattern.pattern,
            )
            return
        if not assertion.input_binding_required:
            yield from ((match, "") for match in pattern.finditer(searched_text))
            return
        targets = self._evidence_targets(assertion)
        if not targets:
            return
        bound_subject = searched_text if binding_text is None else binding_text
        target_spans: list[
            tuple[
                str,
                str,
                tuple[tuple[int, int], ...],
                bool,
                tuple[tuple[int, int, int], ...],
            ]
        ] = []
        for target in targets:
            spans = literal_spans(bound_subject, target)
            if not spans:
                continue
            evidence_literal = target.rpartition("/")[2] if filename_binding else target
            if filename_binding:
                evidence_spans = tuple(
                    (target_end - len(evidence_literal), target_end)
                    for _target_start, target_end in spans
                    if bound_subject[target_end - len(evidence_literal) : target_end]
                    == evidence_literal
                )
            else:
                evidence_spans = spans
            if not evidence_spans:
                continue
            consumes_target = _pattern_consumes_target_literal(pattern, evidence_literal)
            finite_witnesses = _finite_target_assertions(pattern, evidence_literal)
            if binding_text is None and not consumes_target and not finite_witnesses:
                continue
            target_spans.append(
                (
                    target,
                    evidence_literal,
                    evidence_spans,
                    consumes_target,
                    finite_witnesses,
                )
            )
        if not target_spans:
            return
        if filename_binding:
            full_match = pattern.fullmatch(searched_text)
            matches = () if full_match is None else (full_match,)
        else:
            matches = pattern.finditer(searched_text)
        for match in matches:
            for (
                target,
                evidence_literal,
                evidence_spans,
                consumes_target,
                finite_witnesses,
            ) in target_spans:
                if binding_text is not None:
                    yield match, target
                    break
                if match.start() == match.end():
                    if self._match_has_target_witness(
                        match,
                        searched_text,
                        evidence_literal,
                        evidence_spans,
                        finite_witnesses,
                        pattern.flags,
                    ):
                        yield match, target
                        break
                    continue
                has_bound_target = _window_has_one_trusted_target(
                    searched_text,
                    evidence_literal,
                    match.start(),
                    match.end(),
                    evidence_spans,
                    pattern.flags,
                )
                if has_bound_target and (
                    consumes_target
                    or self._match_has_target_witness(
                        match,
                        searched_text,
                        evidence_literal,
                        evidence_spans,
                        finite_witnesses,
                        pattern.flags,
                    )
                ):
                    yield match, target
                    break

    @staticmethod
    def _match_has_target_witness(
        match: re.Match,
        searched_text: str,
        evidence_literal: str,
        evidence_spans: tuple[tuple[int, int], ...],
        finite_witnesses: tuple[tuple[int, int, int], ...],
        flags: int,
    ) -> bool:
        """Require a bounded positive assertion to witness the target locally.

        An unbounded lookahead can prove only that a target occurs somewhere in
        the file. It cannot tie that target to a separate condition at this
        match position. The accepted witness therefore has finite width, names
        the trusted target as fixed positive syntax, and overlaps its file span.
        """
        for direction, offset, maximum in finite_witnesses:
            assertion_position = match.start() + offset
            witness_start = assertion_position if direction == 1 else assertion_position - maximum
            witness_end = assertion_position + maximum if direction == 1 else assertion_position
            if _window_has_one_trusted_target(
                searched_text,
                evidence_literal,
                witness_start,
                witness_end,
                evidence_spans,
                flags,
            ):
                return True
        return False

    def _relative_file(self, file_path: str) -> str:
        return os.path.relpath(file_path, self.project_dir).replace(os.sep, "/")

    def _trusted_project_paths(self) -> tuple[tuple[tuple[str, str], ...], str]:
        """Return a complete bounded inventory of regular files and directories."""
        real_project = os.path.realpath(self.project_dir)
        candidates = sorted(
            glob.glob(
                os.path.join(self.project_dir, "**/*"),
                recursive=True,
                include_hidden=True,
            )
        )
        inventory: list[tuple[str, str]] = []
        for path in candidates:
            real_path = os.path.realpath(path)
            if not real_path.startswith(real_project + os.sep):
                continue
            relative = os.path.relpath(path, self.project_dir)
            if any(component in _NOISE_DIRECTORY_NAMES for component in relative.split(os.sep)):
                continue
            if os.path.isfile(path):
                inventory.append((path, "file"))
            elif os.path.isdir(path):
                inventory.append((path, "directory"))
        if len(inventory) > MAX_FILES_PER_HINT:
            return (), (
                f"Trusted forbidden-path inventory exceeds {MAX_FILES_PER_HINT} objects; "
                "absence cannot be proven"
            )
        return tuple(inventory), ""

    def _trusted_project_inventory(
        self,
        *,
        allow_configuration: bool,
    ) -> tuple[tuple[tuple[str, str, str], ...], str]:
        """Read a complete source-aware inventory without model scope input."""
        real_project = os.path.realpath(self.project_dir)
        candidates = sorted(
            glob.glob(
                os.path.join(self.project_dir, "**/*"),
                recursive=True,
                include_hidden=True,
            )
        )
        files = [
            path
            for path in candidates
            if os.path.isfile(path)
            and os.path.realpath(path).startswith(real_project + os.sep)
            and not _has_noise_directory(path, self.project_dir)
        ]
        if len(files) > MAX_FILES_PER_HINT:
            return (), (
                f"Trusted forbidden-evidence inventory exceeds {MAX_FILES_PER_HINT} files; "
                "absence cannot be proven"
            )
        inventory: list[tuple[str, str, str]] = []
        for file_path in files:
            content = self._read_file(file_path)
            if content is None:
                return (), (
                    f"Trusted forbidden-evidence inventory could not read "
                    f"{self._relative_file(file_path)}; absence cannot be proven"
                )
            evidence_content = mask_non_executable_source(
                content,
                file_path,
                allow_configuration=allow_configuration,
            )
            if evidence_content is None:
                if is_known_non_evidence_format(
                    file_path,
                    allow_configuration=allow_configuration,
                ):
                    continue
                return (), (
                    "Trusted forbidden-evidence inventory could not classify "
                    f"{self._relative_file(file_path)}; absence cannot be proven"
                )
            inventory.append((file_path, content, evidence_content))
        return tuple(inventory), ""

    def _verify_forbidden_constant(self, assertion: SpecAssertion) -> SpecVerificationResult:
        """Verify a forbidden scalar from trusted target/value and complete scope."""
        inventory, error = self._trusted_project_inventory(allow_configuration=True)
        if error:
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail=error,
            )
        target = self._evidence_targets(assertion)[0]
        expected = assertion.expected_value.strip()
        for file_path, content, evidence_content in inventory:
            for _start, end in literal_spans(evidence_content, target):
                remainder = content[end : end + MAX_SCALAR_LENGTH]
                annotation = re.match(r"\s*:\s*[^=\n]{1,256}\s*=\s*", remainder)
                value_start = end + annotation.end() if annotation is not None else end
                actual = _extract_following_scalar(content, value_start).strip()
                if not actual:
                    return SpecVerificationResult(
                        assertion=assertion,
                        verified=False,
                        file_path=file_path,
                        discrepancy=True,
                        evidence_source="trusted_project_scan",
                        evidence_target=target,
                        detail=(
                            f"Trusted scan could not parse criterion target '{target}' in "
                            f"{self._relative_file(file_path)}; absence cannot be proven"
                        ),
                    )
                if actual == expected:
                    return SpecVerificationResult(
                        assertion=assertion,
                        verified=False,
                        actual_value=actual,
                        file_path=file_path,
                        discrepancy=True,
                        evidence_source="trusted_project_scan",
                        evidence_target=target,
                        detail=(
                            f"Forbidden value '{expected}' found in "
                            f"{os.path.basename(file_path)}; criterion target '{target}' "
                            "from trusted project scan"
                        ),
                    )
        return SpecVerificationResult(
            assertion=assertion,
            verified=True,
            evidence_source="trusted_project_scan",
            evidence_target=target,
            detail=f"Forbidden criterion target '{target}' with value '{expected}' was not found",
        )

    def _verify_forbidden_structural(self, assertion: SpecAssertion) -> SpecVerificationResult:
        """Verify forbidden target absence without model regex or file-hint scope."""
        target = self._evidence_targets(assertion)[0]
        path_kind = _criterion_path_kind(assertion.ac_text, target)
        if path_kind is not None:
            paths, error = self._trusted_project_paths()
            if error:
                return SpecVerificationResult(
                    assertion=assertion,
                    verified=False,
                    discrepancy=True,
                    detail=error,
                )
            for object_path, object_kind in paths:
                relative = self._relative_file(object_path)
                path_matches = (
                    relative == target if "/" in target else os.path.basename(object_path) == target
                )
                if object_kind == path_kind and path_matches:
                    return SpecVerificationResult(
                        assertion=assertion,
                        verified=False,
                        file_path=object_path,
                        discrepancy=True,
                        evidence_source="trusted_project_scan",
                        evidence_target=target,
                        detail=(
                            f"Forbidden {path_kind} criterion target '{target}' found at "
                            f"{relative} by trusted project scan"
                        ),
                    )
            return SpecVerificationResult(
                assertion=assertion,
                verified=True,
                evidence_source="trusted_project_scan",
                evidence_target=target,
                detail=f"Forbidden {path_kind} criterion target '{target}' was not found",
            )

        inventory, error = self._trusted_project_inventory(allow_configuration=False)
        if error:
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail=error,
            )
        for file_path, content, evidence_content in inventory:
            relative_file = self._relative_file(file_path)
            if decl.matches_criterion(evidence_content, content, target, assertion, file_path):
                return SpecVerificationResult(
                    assertion=assertion,
                    verified=False,
                    file_path=file_path,
                    discrepancy=True,
                    evidence_source="trusted_project_scan",
                    evidence_target=target,
                    detail=(
                        f"Forbidden criterion target '{target}' found in {relative_file} "
                        "file_content by trusted project scan"
                    ),
                )
        return SpecVerificationResult(
            assertion=assertion,
            verified=True,
            evidence_source="trusted_project_scan",
            evidence_target=target,
            detail=f"Forbidden criterion target '{target}' was not found",
        )

    def _find_filename_bound_match(
        self,
        pattern: re.Pattern,
        filename_subject: str,
        assertion: SpecAssertion,
    ) -> tuple[re.Match, str] | None:
        return self._find_bound_match(
            pattern,
            filename_subject,
            assertion,
            filename_binding=assertion.input_binding_required,
        )

    def _verify_constant(self, assertion: SpecAssertion) -> SpecVerificationResult:
        """Verify a T1 constant/config assertion by searching source files."""
        if not assertion.pattern:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail="No pattern to verify",
            )

        blank_subject_contract = _matches_only_a_blank_subject(assertion.pattern)
        if (
            assertion.input_binding_required
            and not blank_subject_contract
            and (
                not assertion.expected_value.strip()
                or not literal_is_bound(assertion.ac_text, assertion.expected_value)
            )
        ):
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail="Expected value is absent from the acceptance-criterion input",
            )
        if assertion.input_binding_required and not self._evidence_targets(assertion):
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail="No criterion-bound target is available for verification",
            )
        polarity = self._evidence_polarity(assertion)
        if polarity is None:
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail="Criterion-bound evidence polarity is ambiguous or stale",
            )
        if polarity is EvidencePolarity.FORBIDDEN:
            return self._verify_forbidden_constant(assertion)

        files = self._find_files(assertion.file_hint)
        if not files:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail=f"No files matched hint: {assertion.file_hint}",
            )

        pattern = self._safe_compile(
            assertion.pattern,
            file_hint=assertion.file_hint,
            candidates=files,
        )
        if pattern is None:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail="Unusable regex pattern: invalid, too long, or able to match a file with no content",
            )

        readable_files = 0
        for file_path in files:
            content = self._read_file(file_path)
            if content is None:
                continue
            evidence_content = mask_non_executable_source(
                content,
                file_path,
                allow_configuration=True,
                allow_plain_text=(
                    not assertion.input_binding_required
                    or all(
                        not target.isidentifier() for target in self._evidence_targets(assertion)
                    )
                ),
            )
            if evidence_content is None:
                continue
            readable_files += 1

            bounds = self._bound_matches(pattern, evidence_content, assertion)
            first_bound = next(bounds, None)
            if first_bound is None and blank_subject_contract:
                bounds = self._bound_matches(
                    pattern,
                    evidence_content,
                    assertion,
                    binding_text=self._relative_file(file_path),
                )
            elif first_bound is not None:
                bounds = chain((first_bound,), bounds)
            for match, evidence_target in bounds:
                # Extract the value after the pattern
                actual = self._extract_value_after_match(content, match)
                if assertion.expected_value:
                    verified = assertion.expected_value.strip() == actual.strip()
                    return SpecVerificationResult(
                        assertion=assertion,
                        outcome=(
                            VerificationOutcome.VERIFIED
                            if verified
                            else VerificationOutcome.DISCREPANCY
                        ),
                        actual_value=actual,
                        file_path=file_path,
                        evidence_source="file_content",
                        evidence_target=evidence_target,
                        detail=(
                            f"Expected '{assertion.expected_value}', "
                            f"found '{actual}' in {os.path.basename(file_path)}"
                            + (
                                f"; criterion target '{evidence_target}' from file content"
                                if evidence_target
                                else ""
                            )
                        ),
                    )
                else:
                    # Pattern found, no expected value to check
                    return SpecVerificationResult(
                        assertion=assertion,
                        outcome=VerificationOutcome.VERIFIED,
                        actual_value=actual,
                        file_path=file_path,
                        evidence_source="file_content",
                        evidence_target=evidence_target,
                        detail=(
                            f"Pattern found in {os.path.basename(file_path)}"
                            + (
                                f"; criterion target '{evidence_target}' from file content"
                                if evidence_target
                                else ""
                            )
                        ),
                    )

        if readable_files == 0:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail=f"Could not read any of {len(files)} matched files",
            )

        # A usable pattern did not match any readable candidate: a real discrepancy.
        return SpecVerificationResult(
            assertion=assertion,
            outcome=VerificationOutcome.DISCREPANCY,
            detail=f"No criterion-bound pattern evidence found in {len(files)} files",
        )

    def _verify_structural(self, assertion: SpecAssertion) -> SpecVerificationResult:
        """Verify a T2 structural assertion (file/class/function exists)."""
        if not assertion.pattern:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail="No pattern to verify",
            )

        if assertion.input_binding_required and not self._evidence_targets(assertion):
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail="No criterion-bound target is available for verification",
            )
        polarity = self._evidence_polarity(assertion)
        if polarity is None:
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail="Criterion-bound evidence polarity is ambiguous or stale",
            )
        if polarity is EvidencePolarity.FORBIDDEN:
            return self._verify_forbidden_structural(assertion)

        files = self._find_files(assertion.file_hint)
        if not files:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail=f"No files matched hint: {assertion.file_hint}",
            )
        evidence_targets = self._evidence_targets(assertion)
        path_kinds = {
            kind
            for target in evidence_targets
            if (kind := _criterion_path_kind(assertion.ac_text, target)) is not None
        }
        requested_path_kind = next(iter(path_kinds)) if len(path_kinds) == 1 else None
        qualified_paths = tuple(target for target in evidence_targets if "/" in target)
        strict_qualified_paths = (
            qualified_paths
            if (
                assertion.input_binding_required
                and assertion.expected_value.strip()
                and re.search(r"\b(?:file|directory)\b", assertion.ac_text, re.IGNORECASE)
            )
            else ()
        )
        blank_subject_contract = _matches_only_a_blank_subject(assertion.pattern)

        # First check: does the pattern match any filename?
        name_pattern = self._safe_compile(
            assertion.pattern,
            re.IGNORECASE,
        )

        if name_pattern:
            for file_path in files:
                basename = os.path.basename(file_path)
                relative_file = self._relative_file(file_path)
                if strict_qualified_paths and not any(
                    relative_file == target for target in strict_qualified_paths
                ):
                    continue
                filename_subject = relative_file if qualified_paths else basename
                bound = self._find_filename_bound_match(
                    name_pattern,
                    filename_subject,
                    assertion,
                )
                if bound:
                    _match, evidence_target = bound
                    if requested_path_kind is None:
                        continue
                    has_requested_kind = (
                        os.path.isfile(file_path)
                        if requested_path_kind == "file"
                        else os.path.isdir(file_path)
                    )
                    if not has_requested_kind:
                        continue
                    return SpecVerificationResult(
                        assertion=assertion,
                        outcome=VerificationOutcome.VERIFIED,
                        file_path=file_path,
                        evidence_source="filename",
                        evidence_target=evidence_target,
                        detail=(
                            f"Found {requested_path_kind}: {basename}"
                            + (
                                f"; criterion target '{evidence_target}' from filename"
                                if evidence_target
                                else ""
                            )
                        ),
                    )

        if requested_path_kind is not None and not blank_subject_contract:
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail=(
                    f"No {requested_path_kind} with matching criterion-bound "
                    "path evidence was found"
                ),
            )

        if strict_qualified_paths and not blank_subject_contract:
            return SpecVerificationResult(
                assertion=assertion,
                verified=False,
                discrepancy=True,
                detail="No exact project-relative path matched the criterion target",
            )

        # Second check: search file contents for class/function/interface
        content_pattern = self._safe_compile(
            assertion.pattern,
            file_hint=assertion.file_hint,
            candidates=files,
        )
        if content_pattern is None:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail="Unusable regex pattern: invalid, too long, or able to match a file with no content",
            )

        readable_files = 0
        for file_path in files:
            if strict_qualified_paths and not any(
                self._relative_file(file_path) == target for target in strict_qualified_paths
            ):
                continue
            content = self._read_file(file_path)
            if content is None:
                continue
            evidence_content = mask_non_executable_source(
                content,
                file_path,
                allow_plain_text=(
                    not assertion.input_binding_required
                    or all(
                        not target.isidentifier() for target in self._evidence_targets(assertion)
                    )
                ),
            )
            if evidence_content is None:
                continue
            readable_files += 1
            bound = next(
                (
                    candidate
                    for candidate in self._bound_matches(
                        content_pattern,
                        evidence_content,
                        assertion,
                    )
                    if decl.match_has_bound_declaration_kind(
                        content_pattern,
                        candidate[0],
                        (evidence_content, content),
                        assertion,
                        candidate[1],
                        _finite_target_assertions(content_pattern, candidate[1]),
                        file_path,
                    )
                ),
                None,
            )
            # Preserve #1837: blank content binds through the exact filename.
            if bound is None and blank_subject_contract:
                bound = self._find_bound_match(
                    content_pattern,
                    evidence_content,
                    assertion,
                    binding_text=self._relative_file(file_path),
                )
            if bound:
                _match, evidence_target = bound
                return SpecVerificationResult(
                    assertion=assertion,
                    outcome=VerificationOutcome.VERIFIED,
                    file_path=file_path,
                    evidence_source="file_content",
                    evidence_target=evidence_target,
                    detail=(
                        f"Pattern found in {os.path.basename(file_path)}"
                        + (
                            f"; criterion target '{evidence_target}' from file content"
                            if evidence_target
                            else ""
                        )
                    ),
                )

        if files and readable_files == 0:
            return SpecVerificationResult(
                assertion=assertion,
                outcome=VerificationOutcome.UNVERIFIABLE,
                detail=f"Could not read any of {len(files)} matched files",
            )

        return SpecVerificationResult(
            assertion=assertion,
            outcome=VerificationOutcome.DISCREPANCY,
            detail=f"No criterion-bound structure evidence found in {len(files)} files",
        )

    def _find_files(self, file_hint: str) -> list[str]:
        """Find project files matching a glob hint.

        Validates that all returned paths are within project_dir to prevent
        path traversal via crafted file_hint patterns (e.g., "../../etc/*").
        """
        if not file_hint:
            file_hint = "**/*.py"

        pattern = os.path.join(self.project_dir, file_hint)
        files = glob.glob(pattern, recursive=True)

        # Canonicalize project_dir for path traversal check
        real_project = os.path.realpath(self.project_dir)

        # Filter: must be within project_dir + exclude noise directories
        filtered = [
            f
            for f in files
            if os.path.realpath(f).startswith(real_project + os.sep)
            and not _has_noise_directory(f, self.project_dir)
        ]

        return filtered[:MAX_FILES_PER_HINT]

    def _read_file(self, file_path: str) -> str | None:
        """Read a file, respecting size limits."""
        try:
            size = os.path.getsize(file_path)
            if size > MAX_FILE_SIZE:
                return None
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except (OSError, PermissionError):
            return None

    def _extract_value_after_match(self, content: str, match: re.Match) -> str:
        """Extract the value immediately following a regex match.

        Handles common patterns:
        - VAR = 10
        - VAR: 10
        - VAR(10)
        - "value"
        """
        return _extract_following_scalar(content, match.end())
