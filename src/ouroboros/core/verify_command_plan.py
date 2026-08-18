"""Run a POSIX ``verify_command`` on a machine that has no POSIX shell.

Most verify commands do not actually need a shell. ``pytest -q``,
``uv run train.py`` and ``python -m pytest tests/x.py`` are simple commands:
an interpreter only performs word splitting on them, which ``shlex`` does
exactly. What genuinely needs a shell is the small set of *builtins* seed
authors reach for — ``exit 0``, ``test -f out.txt``, ``printf READY``,
``true`` — none of which exist as executables on a bare Windows box.

So this module answers one question: **can this exact command be executed
without a shell, with its POSIX meaning preserved?** It never rewrites a
command and never approximates one. Every construct whose meaning it cannot
reproduce exactly — pipes, redirections, expansions, globs, an unsupported
builtin flag — makes it return ``None``, and the caller reports the AC as
unverifiable rather than guessing. A wrong PASS is the one outcome the verify
gate exists to prevent, so refusing is always the correct fallback.

Builtins are evaluated here rather than spawned: they are pure functions of
(argv, cwd) over the filesystem, judged by the orchestrator, exactly like the
``expected_artifacts`` oracle already is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
import shlex
from typing import Literal

# Characters that, unquoted, mean the shell would do something this module
# cannot reproduce: pipelines, redirection, subshells, expansion, globbing,
# job control, comments. `&&`, `||` and `;` are handled before this check.
_UNSUPPORTED_UNQUOTED = frozenset("|<>()$`*?[]{}~#!\n\r")

_ASSIGNMENT_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

RunCondition = Literal["always", "on_success", "on_failure"]

_SUPPORTED_BUILTINS = frozenset({"exit", "true", "false", ":", "echo", "printf", "test", "["})

# Builtins whose exit status and output are fixed by their literal arguments:
# nothing about the workspace, the environment, or any external program can
# change what they report. A verify_command built only from these cannot fail,
# which means it cannot verify. `test` is deliberately absent — it reads the
# filesystem, so it carries real (if weak) evidential force.
_STATUS_CONSTANT_BUILTINS = frozenset({"exit", "true", "false", ":", "echo", "printf"})

# `printf` escape sequences with one unambiguous meaning. A format string with
# a `%` conversion is refused: reproducing printf's conversion semantics is not
# worth the risk of getting one wrong inside a pass/fail decision.
_PRINTF_ESCAPES = {"\\n": "\n", "\\t": "\t", "\\r": "\r", "\\\\": "\\", "\\0": "\0"}

_TEST_FILE_PREDICATES = frozenset({"-e", "-f", "-d", "-s", "-r", "-w", "-x"})
_TEST_STRING_PREDICATES = frozenset({"-z", "-n"})
_TEST_COMPARISONS = frozenset({"=", "!=", "-eq", "-ne", "-lt", "-le", "-gt", "-ge"})


@dataclass(frozen=True, slots=True)
class ShellFreeStep:
    """One simple command, plus when POSIX says it should run."""

    argv: tuple[str, ...]
    run_condition: RunCondition = "always"
    env_overrides: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def is_builtin(self) -> bool:
        """Whether this step is evaluated here instead of being spawned."""
        return self.argv[0] in _SUPPORTED_BUILTINS


@dataclass(frozen=True, slots=True)
class BuiltinResult:
    """Exit status and stdout of one builtin, with POSIX semantics."""

    status: int
    output: str = ""


def _split_on_operators(command: str) -> list[tuple[str, RunCondition]] | None:
    """Split a command line on unquoted ``&&``, ``||`` and ``;``.

    Returns ``None`` when the line contains anything else a shell would act on.
    Quoting state is tracked because `python -c "print('a|b')"` is a simple
    command: the metacharacters inside quotes are literal text.
    """
    segments: list[tuple[str, RunCondition]] = []
    condition: RunCondition = "always"
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = None
            current.append(char)
            index += 1
            continue
        if quote == '"':
            if char == "\\" and index + 1 < len(command):
                current.append(char)
                current.append(command[index + 1])
                index += 2
                continue
            # Inside double quotes these still expand, so the meaning is not
            # ours to reproduce.
            if char in "$`":
                return None
            if char == '"':
                quote = None
            current.append(char)
            index += 1
            continue
        if char in "'\"":
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(command):
            current.append(char)
            current.append(command[index + 1])
            index += 2
            continue
        if char == "&" or char == "|":
            if index + 1 < len(command) and command[index + 1] == char:
                # An empty segment before an operator (`&& x`, `a ; ; b`,
                # `a && || b`) is a shell syntax error (status 2), not an
                # empty command to skip. Silently dropping it would let a
                # malformed contract produce a verdict bash refuses to give.
                if not "".join(current).strip():
                    return None
                segments.append(("".join(current), condition))
                condition = "on_success" if char == "&" else "on_failure"
                current = []
                index += 2
                continue
            # A lone `&` backgrounds; a lone `|` pipes.
            return None
        if char == ";":
            if not "".join(current).strip():
                return None
            segments.append(("".join(current), condition))
            condition = "always"
            current = []
            index += 1
            continue
        if char in _UNSUPPORTED_UNQUOTED:
            return None
        current.append(char)
        index += 1

    if quote is not None:
        return None
    tail = "".join(current)
    if tail.strip():
        segments.append((tail, condition))
    elif condition != "always":
        # A trailing `;` is valid POSIX; a trailing `&&`/`||` leaves the
        # shell waiting for a command and exits 2 under `-c`.
        return None
    return segments or None


def _split_assignment_prefix(tokens: list[str]) -> tuple[tuple[tuple[str, str], ...], list[str]]:
    """Peel `NAME=value` prefixes, which a shell applies to that command only."""
    overrides: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens) and _ASSIGNMENT_PREFIX.match(tokens[index]):
        name, _, value = tokens[index].partition("=")
        overrides.append((name, value))
        index += 1
    return tuple(overrides), tokens[index:]


def _builtin_is_supported(argv: tuple[str, ...]) -> bool:
    """Whether this module reproduces this builtin invocation exactly."""
    name = argv[0]
    operands = list(argv[1:])
    if name in {"true", "false", ":"}:
        return True
    if name == "exit":
        if not operands:
            return True
        # POSIX truncates the operand to 8 bits, so `exit 256` reports 0 and
        # `exit -1` reports 255. Accepting the full signed range (normalized
        # in run_builtin) matters for the vacuity proof: rejecting `exit 256`
        # here would leave a command that always succeeds unproven.
        return len(operands) == 1 and _is_integer(operands[0])
    if name == "echo":
        return all(not operand.startswith("-") or operand == "-n" for operand in operands)
    if name == "printf":
        if not operands:
            return False
        return "%" not in operands[0] and _unescape_printf(operands[0]) is not None
    if name in {"test", "["}:
        if name == "[":
            if not operands or operands[-1] != "]":
                return False
            operands = operands[:-1]
        return _test_expression_is_supported(operands)
    return False


def _test_expression_is_supported(operands: list[str]) -> bool:
    if operands and operands[0] == "!":
        operands = operands[1:]
    if len(operands) == 1:
        return True
    if len(operands) == 2:
        return operands[0] in _TEST_FILE_PREDICATES | _TEST_STRING_PREDICATES
    if len(operands) == 3:
        if operands[1] not in _TEST_COMPARISONS:
            return False
        if operands[1].startswith("-"):
            return _is_integer(operands[0]) and _is_integer(operands[2])
        return True
    return False


def _is_integer(value: str) -> bool:
    return bool(value) and (value.lstrip("+-").isdigit() and value.lstrip("+-") != "")


def _unescape_printf(text: str) -> str | None:
    """Expand the unambiguous escapes, or return None for anything else."""
    result: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "\\":
            result.append(text[index])
            index += 1
            continue
        pair = text[index : index + 2]
        if pair not in _PRINTF_ESCAPES:
            return None
        result.append(_PRINTF_ESCAPES[pair])
        index += 2
    return "".join(result)


def plan_shell_free_execution(command: str) -> tuple[ShellFreeStep, ...] | None:
    """Return an exact shell-free plan for ``command``, or ``None``.

    ``None`` means "this command needs a shell" — never "run it loosely".
    Validation is complete before the first step runs, so a partially
    executable chain cannot half-run and leave the workspace in a state no
    verdict describes.
    """
    if not command.strip():
        return None
    segments = _split_on_operators(command)
    if segments is None:
        return None

    steps: list[ShellFreeStep] = []
    for text, condition in segments:
        try:
            tokens = shlex.split(text, posix=True)
        except ValueError:
            return None
        overrides, argv = _split_assignment_prefix(tokens)
        if not argv:
            return None
        # `command name ...` runs `name` bypassing shell functions — none exist
        # here, so stripping the prefix is exact. Its option forms (`-v`/`-V`
        # print, `-p` swaps PATH) change behavior and are refused. A bare
        # `command` exits 0, same as `true`. Without this, `command true`
        # would read as an external program and escape the vacuity proof.
        while argv and argv[0] == "command":
            if len(argv) > 1 and argv[1].startswith("-"):
                return None
            argv = argv[1:]
        if not argv:
            argv = ["true"]
        step = ShellFreeStep(
            argv=tuple(argv),
            run_condition=condition,
            env_overrides=overrides,
        )
        if step.is_builtin and not _builtin_is_supported(step.argv):
            return None
        steps.append(step)
    return tuple(steps)


def run_builtin(step: ShellFreeStep, *, cwd: str) -> BuiltinResult:
    """Evaluate a supported builtin with POSIX semantics.

    Only ever called for a step ``plan_shell_free_execution`` already admitted.
    """
    name = step.argv[0]
    operands = list(step.argv[1:])
    if name in {"true", ":"}:
        return BuiltinResult(status=0)
    if name == "false":
        return BuiltinResult(status=1)
    if name == "exit":
        # POSIX: the shell reports the operand's low 8 bits.
        return BuiltinResult(status=int(operands[0]) % 256 if operands else 0)
    if name == "echo":
        trailing_newline = True
        if operands and operands[0] == "-n":
            trailing_newline = False
            operands = operands[1:]
        text = " ".join(operands)
        return BuiltinResult(status=0, output=text + ("\n" if trailing_newline else ""))
    if name == "printf":
        expanded = _unescape_printf(operands[0])
        assert expanded is not None  # guaranteed by _builtin_is_supported
        return BuiltinResult(status=0, output=expanded)
    if name == "[":
        operands = operands[:-1]
    return BuiltinResult(status=0 if _evaluate_test(operands, cwd=cwd) else 1)


def _evaluate_test(operands: list[str], *, cwd: str) -> bool:
    negate = False
    if operands and operands[0] == "!":
        negate = True
        operands = operands[1:]
    result = _evaluate_test_expression(operands, cwd=cwd)
    return not result if negate else result


def _evaluate_test_expression(operands: list[str], *, cwd: str) -> bool:
    if len(operands) == 1:
        return operands[0] != ""
    if len(operands) == 2:
        predicate, operand = operands
        if predicate == "-z":
            return operand == ""
        if predicate == "-n":
            return operand != ""
        target = Path(cwd) / operand
        if predicate == "-e":
            return target.exists()
        if predicate == "-f":
            return target.is_file()
        if predicate == "-d":
            return target.is_dir()
        if predicate == "-s":
            return target.is_file() and target.stat().st_size > 0
        # -r / -w / -x
        import os

        mode = {"-r": os.R_OK, "-w": os.W_OK, "-x": os.X_OK}[predicate]
        return os.access(target, mode)
    left, operator, right = operands
    if operator == "=":
        return left == right
    if operator == "!=":
        return left != right
    first, second = int(left), int(right)
    return {
        "-eq": first == second,
        "-ne": first != second,
        "-lt": first < second,
        "-le": first <= second,
        "-gt": first > second,
        "-ge": first >= second,
    }[operator]


@lru_cache(maxsize=512)
def constant_verdict(command: str, output_assertion: str | None = None) -> bool | None:
    """Return the contract's verdict when it cannot depend on the work.

    ``None`` means the verdict is not constant — the command consults the
    filesystem or another program, so it can genuinely fail. Otherwise the
    returned boolean is the verdict this contract will report every single
    time, whatever the worker did.

    The distinction the caller needs is the sign of that constant:

    * ``True`` — ``verify_command: "exit 0"``. The orchestrator runs it, it
      exits 0, and the AC is recorded as independently verified. Worse, merely
      *declaring* a verify_command drops ``commands_run`` and ``tests_passed``
      from the required evidence, so a contract that cannot fail leaves the AC
      checked less than one that declared nothing. The gate built to make
      self-reporting impossible becomes the way around it.
    * ``False`` — ``verify_command: "exit 1"``, or ``printf WAITING`` under
      ``expect: READY``. Information-free too, but it can only ever refuse.
      A broken seed, not an unsound one, so it is left to fail honestly.

    This is a proof, not a heuristic: it commits to an answer only when every
    reachable step is a builtin whose status and output are fixed by its
    literal arguments. Anything unparsed, or touching the filesystem or another
    program, returns ``None``.
    """
    steps = plan_shell_free_execution(command)
    if steps is None:
        return None

    # `bash -c 'exit 0'` is the same constant wearing a wrapper. When the
    # whole command is one shell invocation of a literal string, the verdict
    # is the inner command's verdict, recursively. Only flag letters that
    # cannot change status or output are admitted (`l` login, `e` errexit);
    # `x`/`v` trace into the combined output, so they stay unproven.
    if len(steps) == 1 and not steps[0].env_overrides:
        argv = steps[0].argv
        if (
            len(argv) == 3
            and argv[0] in {"bash", "sh"}
            and argv[1].startswith("-")
            and "c" in argv[1][1:]
            and set(argv[1][1:]) <= set("cle")
        ):
            return constant_verdict(argv[2], output_assertion)

    status = 0
    output: list[str] = []
    for step in steps:
        if step.argv[0] not in _STATUS_CONSTANT_BUILTINS:
            return None
        if step.run_condition == "on_success" and status != 0:
            continue
        if step.run_condition == "on_failure" and status == 0:
            continue
        result = run_builtin(step, cwd="")
        output.append(result.output)
        status = result.status
        if step.argv[0] == "exit":
            # `exit` ends the shell; nothing after it is reachable.
            break
    if status != 0:
        return False
    if output_assertion is None:
        return True
    return output_assertion in "".join(output)


def verify_contract_always_passes(command: str, output_assertion: str | None = None) -> bool:
    """Whether this contract reports a pass no matter what happened."""
    return constant_verdict(command, output_assertion) is True
