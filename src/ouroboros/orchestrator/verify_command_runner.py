"""Execute an AC's ``verify_command`` and report only what it did.

Two routes reach the same result type. When a real POSIX shell exists, the
command is handed to it verbatim — that is the contract and the faithful
reading of it. When none exists, a plan from
:mod:`ouroboros.core.verify_command_plan` is executed step by step with POSIX
short-circuit semantics, which is possible only because that planner refuses
any command whose meaning it cannot reproduce exactly.

Nothing here decides pass or fail. It returns an exit status and the combined
output; the gate applies the contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass
import os
import shutil
import time

from ouroboros.core.verify_command_plan import ShellFreeStep, run_builtin

# POSIX exit status for a command name that could not be resolved. Emitted
# rather than raised so a missing tool reads as a failing contract, which is
# what a shell would have reported.
COMMAND_NOT_FOUND_STATUS = 127


@dataclass(frozen=True, slots=True)
class VerifyRun:
    """What running a verify command produced, before any judgment.

    ``start_error`` and ``timed_out`` are kept separate from ``returncode``
    because the gate must not read "could not start" or "ran too long" as an
    ordinary non-zero exit — they are different facts about the run.
    """

    returncode: int
    output: str
    timed_out: bool = False
    start_error: str | None = None


def _spawn_kwargs() -> dict[str, object]:
    # A new session on POSIX so a timeout can kill the whole process group
    # rather than leaving orphaned children behind the shell.
    return {"start_new_session": True} if os.name != "nt" else {}


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if os.name != "nt":
        import signal

        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


async def _run_process(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> VerifyRun:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **_spawn_kwargs(),  # type: ignore[arg-type]
        )
    except Exception as exc:  # pragma: no cover - spawn failure is environmental
        return VerifyRun(returncode=1, output="", start_error=str(exc))

    try:
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        await _terminate(proc)
        return VerifyRun(returncode=1, output="", timed_out=True)

    return VerifyRun(
        returncode=proc.returncode if proc.returncode is not None else 1,
        output=(stdout_bytes or b"").decode("utf-8", errors="replace"),
    )


async def run_with_shell(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> VerifyRun:
    """Run the command through a resolved POSIX shell, unmodified."""
    return await _run_process(argv, cwd=cwd, env=env, timeout_seconds=timeout_seconds)


async def run_shell_free_plan(
    steps: Sequence[ShellFreeStep],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> VerifyRun:
    """Run a planned command with no shell, preserving POSIX semantics.

    ``&&``/``||``/``;`` short-circuiting, per-command ``NAME=value`` prefixes,
    a missing command reporting 127, and ``exit`` ending the sequence all match
    what the shell would have done. The timeout is a budget across the whole
    sequence, not per step, so a chain cannot outlive the gate's bound.
    """
    deadline = time.monotonic() + timeout_seconds
    status = 0
    output: list[str] = []

    for step in steps:
        if step.run_condition == "on_success" and status != 0:
            continue
        if step.run_condition == "on_failure" and status == 0:
            continue

        if step.is_builtin:
            result = run_builtin(step, cwd=cwd)
            output.append(result.output)
            status = result.status
            if step.argv[0] == "exit":
                break
            continue

        step_env = dict(env)
        step_env.update(step.env_overrides)
        executable = shutil.which(step.argv[0], path=step_env.get("PATH"))
        if executable is None:
            output.append(f"{step.argv[0]}: command not found\n")
            status = COMMAND_NOT_FOUND_STATUS
            continue

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return VerifyRun(returncode=status, output="".join(output), timed_out=True)

        run = await _run_process(
            (executable, *step.argv[1:]),
            cwd=cwd,
            env=step_env,
            timeout_seconds=remaining,
        )
        output.append(run.output)
        if run.start_error is not None:
            return VerifyRun(
                returncode=1,
                output="".join(output),
                start_error=run.start_error,
            )
        if run.timed_out:
            return VerifyRun(returncode=status, output="".join(output), timed_out=True)
        status = run.returncode

    return VerifyRun(returncode=status, output="".join(output))
