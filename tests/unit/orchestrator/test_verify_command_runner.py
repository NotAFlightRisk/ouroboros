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


def test_planner_refuses_what_cannot_be_reproduced() -> None:
    """Refusing is the safe direction: the caller reports unverifiable."""
    for command in [
        "grep -q ok f | tee log",
        "echo $(date)",
        "python -c 'x' > out.txt",
        "ls *.py",
    ]:
        assert plan_shell_free_execution(command) is None, command
