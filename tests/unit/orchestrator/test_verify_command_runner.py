"""Running a verify_command on a machine with no POSIX shell.

The planner (`core/verify_command_plan.py`) decides whether a command's meaning
can be reproduced without an interpreter; this module runs what it admits. The
contract these tests pin is that a shell-free run reports exactly what a shell
would have reported — including the failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ouroboros.core.verify_command_plan import plan_shell_free_execution
from ouroboros.orchestrator.verify_command_runner import (
    COMMAND_NOT_EXECUTABLE_STATUS,
    COMMAND_NOT_FOUND_STATUS,
    run_shell_free_plan,
)


async def _run(command: str, cwd: Path, timeout_seconds: float = 30.0) -> Any:
    steps = plan_shell_free_execution(command)
    assert steps is not None, f"planner refused {command!r}"
    return await run_shell_free_plan(
        steps,
        cwd=str(cwd),
        env={"PATH": __import__("os").environ.get("PATH", "")},
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.asyncio
async def test_builtin_status_matches_posix(tmp_path: Path) -> None:
    assert (await _run("exit 0", tmp_path)).returncode == 0
    assert (await _run("exit 3", tmp_path)).returncode == 3
    assert (await _run("true", tmp_path)).returncode == 0
    assert (await _run("false", tmp_path)).returncode == 1


@pytest.mark.asyncio
async def test_test_builtin_reads_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("seed")

    assert (await _run("test -f a.txt", tmp_path)).returncode == 0
    assert (await _run("test -f missing.txt", tmp_path)).returncode == 1
    assert (await _run("test -d .", tmp_path)).returncode == 0


@pytest.mark.asyncio
async def test_printf_and_echo_produce_their_output(tmp_path: Path) -> None:
    assert (await _run("printf READY", tmp_path)).output == "READY"
    assert (await _run("echo done", tmp_path)).output == "done\n"
    assert (await _run("echo -n done", tmp_path)).output == "done"


@pytest.mark.asyncio
async def test_short_circuit_matches_posix(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("seed")

    # `&&` skips the right side when the left failed.
    run = await _run("test -f missing.txt && echo REACHED", tmp_path)
    assert run.returncode == 1
    assert "REACHED" not in run.output

    # `||` runs it only then.
    run = await _run("test -f missing.txt || echo RECOVERED", tmp_path)
    assert run.returncode == 0
    assert "RECOVERED" in run.output

    # `;` runs regardless, and the last status wins.
    run = await _run("echo first; exit 4", tmp_path)
    assert run.returncode == 4
    assert "first" in run.output


@pytest.mark.asyncio
async def test_exit_ends_the_sequence(tmp_path: Path) -> None:
    run = await _run("exit 0; echo UNREACHABLE", tmp_path)

    assert run.returncode == 0
    assert "UNREACHABLE" not in run.output


@pytest.mark.asyncio
async def test_external_command_runs_and_reports_its_status(tmp_path: Path) -> None:
    run = await _run("python -c \"import sys; print('hi'); sys.exit(2)\"", tmp_path)

    assert run.returncode == 2
    assert "hi" in run.output


@pytest.mark.asyncio
async def test_missing_command_reports_127_like_a_shell(tmp_path: Path) -> None:
    run = await _run("nosuchtool --flag", tmp_path)

    assert run.returncode == COMMAND_NOT_FOUND_STATUS
    assert "command not found" in run.output


@pytest.mark.asyncio
async def test_assignment_prefix_applies_to_that_command_only(tmp_path: Path) -> None:
    run = await _run(
        "MARKER=set python -c \"import os; print(os.environ.get('MARKER'))\"",
        tmp_path,
    )

    assert run.returncode == 0
    assert "set" in run.output


@pytest.mark.asyncio
async def test_timeout_is_a_budget_across_the_whole_sequence(tmp_path: Path) -> None:
    run = await _run(
        'python -c "import time; time.sleep(30)" && echo REACHED',
        tmp_path,
        timeout_seconds=1.0,
    )

    assert run.timed_out is True
    assert "REACHED" not in run.output


@pytest.mark.asyncio
async def test_path_bearing_command_resolves_against_the_workspace(tmp_path: Path) -> None:
    """`./check.sh` means the file in the verification cwd, never PATH or the
    orchestrator process directory — a shell running in cwd resolves it there."""
    script = tmp_path / "check.sh"
    script.write_text("#!/bin/sh\necho WORKSPACE-OK\n")
    script.chmod(0o755)

    run = await _run("./check.sh", tmp_path)

    assert run.returncode == 0
    assert "WORKSPACE-OK" in run.output


@pytest.mark.asyncio
async def test_missing_path_bearing_command_reports_127(tmp_path: Path) -> None:
    run = await _run("./nope.sh", tmp_path)

    assert run.returncode == COMMAND_NOT_FOUND_STATUS
    assert "command not found" in run.output


@pytest.mark.asyncio
async def test_non_executable_path_bearing_command_reports_126(tmp_path: Path) -> None:
    script = tmp_path / "check.sh"
    script.write_text("#!/bin/sh\necho NEVER\n")
    script.chmod(0o644)

    run = await _run("./check.sh", tmp_path)

    assert run.returncode == COMMAND_NOT_EXECUTABLE_STATUS
    assert "Permission denied" in run.output
    assert "NEVER" not in run.output


def test_planner_rejects_shell_syntax_errors_instead_of_repairing_them() -> None:
    """Bash exits 2 on these; silently dropping the empty segment would let a
    malformed contract produce a verdict the shell refuses to give."""
    for command in [
        "&& echo PASS",
        "|| echo PASS",
        "; echo PASS",
        "echo PASS &&",
        "echo PASS ||",
        "echo PASS ; ; echo TWO",
        "echo PASS && && echo TWO",
    ]:
        assert plan_shell_free_execution(command) is None, command


@pytest.mark.asyncio
async def test_trailing_semicolon_stays_valid_like_a_shell(tmp_path: Path) -> None:
    run = await _run("echo DONE ;", tmp_path)

    assert run.returncode == 0
    assert "DONE" in run.output


def test_planner_refuses_what_cannot_be_reproduced() -> None:
    """Refusing is the safe direction: the caller reports unverifiable."""
    for command in [
        "grep -q ok f | tee log",
        "echo $(date)",
        "python -c 'x' > out.txt",
        "ls *.py",
    ]:
        assert plan_shell_free_execution(command) is None, command


@pytest.mark.asyncio
async def test_relative_path_entries_resolve_against_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell reads `PATH=./bin` relative to the directory it runs in.

    Resolving it against the orchestrator's own process directory can both miss
    the workspace tool and — the sharp case pinned here — execute a same-named
    binary that happens to sit beside the orchestrator.
    """
    workspace = tmp_path / "workspace" / "bin"
    workspace.mkdir(parents=True)
    (workspace / "check").write_text("#!/bin/sh\necho FROM-WORKSPACE\n")
    (workspace / "check").chmod(0o755)

    decoy = tmp_path / "decoy" / "bin"
    decoy.mkdir(parents=True)
    (decoy / "check").write_text("#!/bin/sh\necho FROM-DECOY\n")
    (decoy / "check").chmod(0o755)

    monkeypatch.chdir(tmp_path / "decoy")
    steps = plan_shell_free_execution("check")
    assert steps is not None

    run = await run_shell_free_plan(
        steps,
        cwd=str(tmp_path / "workspace"),
        env={"PATH": "./bin"},
        timeout_seconds=30.0,
    )

    assert run.returncode == 0
    assert "FROM-WORKSPACE" in run.output
    assert "FROM-DECOY" not in run.output


@pytest.mark.asyncio
async def test_empty_path_entry_means_the_workspace_like_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX defines an empty `PATH` element as the current directory."""
    (tmp_path / "check").write_text("#!/bin/sh\necho HERE\n")
    (tmp_path / "check").chmod(0o755)
    monkeypatch.chdir(tmp_path.parent)

    steps = plan_shell_free_execution("check")
    assert steps is not None
    run = await run_shell_free_plan(
        steps,
        cwd=str(tmp_path),
        env={"PATH": ""},
        timeout_seconds=30.0,
    )

    assert run.returncode == 0
    assert "HERE" in run.output


def test_builtins_this_module_does_not_emulate_are_refused() -> None:
    """A name the shell interprets itself must never become a PATH lookup.

    `type missing` fails in bash; admitting it would run a caller-controlled
    `type` binary that can exit 0 instead — a false pass invented by the
    fallback route. Refusing reports the AC unverifiable, which is honest.
    """
    for command in [
        "type missing",
        "cd sub",
        "export FLAG=1",
        "eval echo hi",
        "exec true",
        "read line",
        "unset FLAG",
        "source ./env.sh",
        "pwd",
        "umask 022",
        "if true",
        "for f in a",
        "while true",
        "function f",
    ]:
        assert plan_shell_free_execution(command) is None, command


@pytest.mark.asyncio
async def test_echo_consumes_every_leading_dash_n_like_bash(tmp_path: Path) -> None:
    """`echo -n -n x` prints `x` in bash; printing `-n x` would let a contract
    asserting on `-n` pass here and fail under a real shell."""
    assert (await _run("echo -n -n x", tmp_path)).output == "x"
    assert (await _run("echo -n -n -n done", tmp_path)).output == "done"
    # A `-n` that is not leading is an ordinary operand.
    assert (await _run("echo x -n", tmp_path)).output == "x -n\n"


@pytest.mark.asyncio
async def test_admitted_commands_match_a_real_bash(tmp_path: Path) -> None:
    """Cross-check the whole admitted surface against the interpreter it claims
    to reproduce. Anything that diverges must be refused, not approximated."""
    import shutil as _shutil
    import subprocess

    bash = _shutil.which("bash")
    if bash is None:  # pragma: no cover - CI always has bash
        pytest.skip("no bash on this machine")

    (tmp_path / "present.txt").write_text("x")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "tool").write_text("#!/bin/sh\necho TOOL\n")
    (tmp_path / "bin" / "tool").chmod(0o755)

    for command in [
        "exit 0",
        "exit 7",
        "true",
        "false",
        "echo hi",
        "echo -n hi",
        "echo -n -n hi",
        "echo hi -n",
        "printf READY",
        "test -f present.txt",
        "test -f missing.txt",
        "test -f present.txt && echo FOUND",
        "test -f missing.txt || echo ABSENT",
        "false ; echo AFTER",
        "./bin/tool",
        "PATH=./bin tool",
    ]:
        steps = plan_shell_free_execution(command)
        assert steps is not None, f"planner refused {command!r}"
        ours = await run_shell_free_plan(
            steps,
            cwd=str(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=30.0,
        )
        theirs = subprocess.run(
            [bash, "-c", command],
            cwd=str(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )

        assert ours.returncode == theirs.returncode, command
        assert ours.output == theirs.stdout, command
