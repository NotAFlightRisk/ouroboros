"""Pre-effect ownership contract for bounded agent-runtime executions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ouroboros.orchestrator.adapter import AgentMessage


async def force_reap_process(
    process: Any,
    graceful_terminate: Callable[[Any], Awaitable[None]],
    *,
    timeout_seconds: float = 0.1,
) -> bool:
    """Terminate, force-kill if necessary, and finitely wait for an owned subprocess."""
    await _run_operation_bounded(
        graceful_terminate(process),
        timeout_seconds=timeout_seconds,
        label="owned process graceful termination",
    )
    if getattr(process, "returncode", None) is None:
        kill = getattr(process, "kill", None)
        if not callable(kill):
            return False
        try:
            kill()
        except ProcessLookupError:
            pass
    if getattr(process, "returncode", None) is None:
        wait = getattr(process, "wait", None)
        if not callable(wait):
            return False
        _, wait_error = await _run_operation_bounded(
            wait(),
            timeout_seconds=timeout_seconds,
            label="owned process reap",
        )
        if wait_error is not None:
            return False
    return getattr(process, "returncode", None) is not None


class RuntimeExecutionUnavailable(RuntimeError):
    """The selected runtime cannot provide verified termination authority."""


@dataclass(frozen=True, slots=True)
class TerminationReceipt:
    """Evidence that one owned execution has no remaining live work."""

    backend: str
    provider_stopped: bool
    process_reaped: bool
    finalizer_complete: bool

    @property
    def verified(self) -> bool:
        return self.provider_stopped and self.process_reaped and self.finalizer_complete


async def _settle_task_boundary(
    task: asyncio.Task[Any],
    *,
    timeout_seconds: float,
    operation: str,
    cancel_first: bool,
) -> tuple[bool, Exception | None]:
    """Settle one task and forcibly close its awaitable if it resists cancellation."""

    forced = False
    if not task.done() and cancel_first:
        task.cancel()
    if not task.done():
        done, _ = await asyncio.wait((task,), timeout=timeout_seconds)
        if not done:
            task.cancel()
            done, _ = await asyncio.wait((task,), timeout=timeout_seconds)
        if not done:
            close = getattr(task.get_coro(), "close", None)
            if not callable(close):
                raise RuntimeExecutionUnavailable(
                    f"{operation} has no forceable Python task boundary"
                )
            forced = True
            try:
                close()
            except BaseException as exc:
                raise RuntimeExecutionUnavailable(
                    f"{operation} resisted forced Python task closure"
                ) from exc
            task.cancel()
            done, _ = await asyncio.wait((task,), timeout=timeout_seconds)
            if not done:
                raise RuntimeExecutionUnavailable(
                    f"{operation} remained live after forced Python task closure"
                )
    if task.cancelled():
        return forced, None
    error = task.exception()
    if forced:
        return True, None
    if error is None or isinstance(error, Exception):
        return False, error
    raise RuntimeExecutionUnavailable(f"{operation} failed during Python task closure") from error


async def _await_operation[T](operation: Awaitable[T]) -> T:
    """Give every provider awaitable a task-owned coroutine boundary."""

    return await operation


async def _run_operation_bounded[T](
    operation: Awaitable[T],
    *,
    timeout_seconds: float,
    label: str,
) -> tuple[T | None, BaseException | None]:
    """Run one provider operation behind a finite, forceable task boundary."""

    task = asyncio.create_task(_await_operation(operation))
    forced, error = await _settle_task_boundary(
        task,
        timeout_seconds=timeout_seconds,
        operation=label,
        cancel_first=False,
    )
    if forced or error is not None or task.cancelled():
        return None, error
    return task.result(), None


class RuntimeExecutionController:
    """Mutable process authority created before a provider can have effects."""

    def __init__(self, backend: str, *, shutdown_timeout_seconds: float = 0.1) -> None:
        self.backend = backend
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._force_terminate: Callable[[], Awaitable[bool]] | None = None
        self._provider_started = False
        self._process_reaped = True

    def bind_process(self, force_terminate: Callable[[], Awaitable[bool]]) -> None:
        """Publish force authority immediately after spawn, before provider input."""
        if self._provider_started and not self._process_reaped:
            raise RuntimeError("runtime execution already owns a live provider")
        self._provider_started = True
        self._process_reaped = False
        self._force_terminate = force_terminate

    async def acquire_process(
        self,
        spawn: Awaitable[Any],
        terminate_process: Callable[[Any], Awaitable[bool]],
    ) -> Any:
        """Bind kill authority before starting an interruptible process spawn."""
        spawn_task = asyncio.create_task(_await_operation(spawn))

        async def _force_spawned_process() -> bool:
            """Settle a stuck spawn before trying to reap a yielded process."""
            _, spawn_error = await _settle_task_boundary(
                spawn_task,
                timeout_seconds=self._shutdown_timeout_seconds,
                operation=f"{self.backend} process spawn",
                cancel_first=True,
            )
            if not spawn_task.done():
                return False
            if spawn_task.cancelled():
                self.mark_reaped()
                return spawn_error is None
            if spawn_error is not None:
                self.mark_reaped()
                return True
            process = spawn_task.result()
            terminated = await _run_operation_bounded(
                terminate_process(process),
                timeout_seconds=self._shutdown_timeout_seconds,
                label=f"{self.backend} spawned process termination",
            )
            if terminated[1] is not None:
                return False
            return bool(terminated[0])

        self.bind_process(_force_spawned_process)
        try:
            return await asyncio.shield(spawn_task)
        except asyncio.CancelledError:
            stopped = await _force_spawned_process()
            if not stopped:
                raise RuntimeExecutionUnavailable(
                    f"{self.backend} process spawn could not be force-terminated"
                )
            self.mark_reaped()
            raise
        except BaseException:
            self.mark_reaped()
            raise

    def mark_reaped(self) -> None:
        self._process_reaped = True

    async def force_terminate(self) -> bool:
        if self._process_reaped:
            return True
        if not self._provider_started or self._force_terminate is None:
            return False
        stopped = await self._force_terminate()
        if stopped:
            self._process_reaped = True
        return stopped

    @property
    def process_reaped(self) -> bool:
        return self._process_reaped


class RuntimeExecution(AsyncIterator[AgentMessage]):
    """Own a provider stream, its active read, process, and finalizer."""

    @property
    def termination_receipt(self) -> TerminationReceipt | None:
        return self._receipt

    def __init__(
        self,
        *,
        backend: str,
        stream: AsyncIterator[AgentMessage],
        controller: RuntimeExecutionController,
        cooperative_shutdown_seconds: float = 0.1,
    ) -> None:
        self.backend = backend
        self._stream = stream
        self._controller = controller
        self._cooperative_shutdown_seconds = cooperative_shutdown_seconds
        self._active_read: asyncio.Task[AgentMessage] | None = None
        self._closed = False
        self._unwind_error: Exception | None = None
        self._receipt: TerminationReceipt | None = None

    def __aiter__(self) -> RuntimeExecution:
        return self

    async def __anext__(self) -> AgentMessage:
        if self._closed:
            raise StopAsyncIteration
        if self._active_read is not None:
            raise RuntimeError("runtime execution already has an active provider read")
        task: asyncio.Task[AgentMessage] = asyncio.create_task(anext(self._stream))
        self._active_read = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._active_read = None

    async def terminate(self) -> TerminationReceipt:
        """Request cooperative cancellation without abandoning the read task."""
        task = self._active_read
        if task is not None and not task.done():
            task.cancel()
            done, _ = await asyncio.wait((task,), timeout=self._cooperative_shutdown_seconds)
            if done:
                self._active_read = None
                try:
                    await task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                except Exception as exc:
                    self._unwind_error = exc
        return self._build_receipt(finalizer_complete=False)

    async def force_terminate(self) -> TerminationReceipt:
        """Kill/reap provider work behind finite process and read boundaries."""
        provider_stopped_value, force_error = await _run_operation_bounded(
            self._controller.force_terminate(),
            timeout_seconds=self._cooperative_shutdown_seconds,
            label=f"{self.backend} provider force termination",
        )
        if force_error is not None:
            self._unwind_error = force_error
        provider_stopped = bool(provider_stopped_value) and force_error is None
        task = self._active_read
        if task is not None:
            _, read_error = await _settle_task_boundary(
                task,
                timeout_seconds=self._cooperative_shutdown_seconds,
                operation=f"{self.backend} provider read",
                cancel_first=True,
            )
            if read_error is not None and not isinstance(read_error, StopAsyncIteration):
                self._unwind_error = read_error
            self._active_read = None
        return self._build_receipt(
            provider_stopped=provider_stopped,
            finalizer_complete=False,
        )

    async def reap(self) -> TerminationReceipt:
        """Bound provider finalization and return verified no-live-work evidence."""
        if self._closed and self._receipt is not None:
            return self._receipt
        task = self._active_read
        if task is not None:
            _, read_error = await _settle_task_boundary(
                task,
                timeout_seconds=self._cooperative_shutdown_seconds,
                operation=f"{self.backend} provider read",
                cancel_first=True,
            )
            if read_error is not None and not isinstance(read_error, StopAsyncIteration):
                self._unwind_error = read_error
            self._active_read = None
        close = getattr(self._stream, "aclose", None)
        if close is not None:
            _, close_error = await _run_operation_bounded(
                close(),
                timeout_seconds=self._cooperative_shutdown_seconds,
                label=f"{self.backend} provider finalizer",
            )
            if close_error is not None:
                self._unwind_error = close_error
        if self._unwind_error is not None:
            raise self._unwind_error
        self._closed = True
        self._receipt = self._build_receipt(finalizer_complete=True)
        if not self._receipt.verified:
            raise RuntimeExecutionUnavailable(
                f"{self.backend} execution ended without a verified termination receipt"
            )
        return self._receipt

    async def aclose(self) -> None:
        cooperative = await self.terminate()
        if not cooperative.verified:
            forced = await self.force_terminate()
            if not forced.provider_stopped or not forced.process_reaped:
                raise RuntimeExecutionUnavailable(
                    f"{self.backend} execution could not be force-terminated and reaped"
                )
        await self.reap()

    def _build_receipt(
        self,
        *,
        provider_stopped: bool | None = None,
        finalizer_complete: bool,
    ) -> TerminationReceipt:
        if provider_stopped is None:
            provider_stopped = self._controller.process_reaped
        return TerminationReceipt(
            backend=self.backend,
            provider_stopped=provider_stopped,
            process_reaped=self._controller.process_reaped,
            finalizer_complete=finalizer_complete,
        )


def reject_unowned_skill_dispatch(runtime: Any, prompt: str) -> None:
    """Fail before an in-process skill handler can escape process ownership."""
    from ouroboros.router import InvalidSkill, NotHandled, ResolveRequest, resolve_skill_dispatch

    result = resolve_skill_dispatch(
        ResolveRequest(
            prompt=prompt,
            cwd=getattr(runtime, "working_directory", None),
            skills_dir=getattr(runtime, "_skills_dir", None),
        )
    )
    if isinstance(result, (NotHandled, InvalidSkill)):
        return
    raise RuntimeExecutionUnavailable(
        "bounded runtime execution cannot own in-process skill-dispatch effects"
    )


def require_runtime_execution(runtime: Any, **kwargs: Any) -> RuntimeExecution:
    """Acquire termination authority synchronously, before provider entry."""
    acquire = getattr(runtime, "acquire_execution", None)
    if not callable(acquire):
        backend = getattr(runtime, "runtime_backend", type(runtime).__name__)
        raise RuntimeExecutionUnavailable(
            f"runtime {backend!r} cannot provide pre-effect termination authority"
        )
    execution = acquire(**kwargs)
    if not isinstance(execution, RuntimeExecution):
        raise RuntimeExecutionUnavailable(
            "runtime returned an invalid pre-effect execution authority"
        )
    return execution


__all__ = [
    "RuntimeExecution",
    "RuntimeExecutionController",
    "RuntimeExecutionUnavailable",
    "force_reap_process",
    "TerminationReceipt",
    "require_runtime_execution",
    "reject_unowned_skill_dispatch",
]
