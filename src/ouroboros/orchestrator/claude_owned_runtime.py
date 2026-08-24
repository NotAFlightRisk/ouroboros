"""Pre-effect Claude SDK process ownership helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any


@dataclass(slots=True)
class ClaudeRuntimeState:
    """Mutable session state preserved when a stream fails and retries."""

    session_id: str | None
    runtime_handle: Any | None


def acquire_claude_execution(adapter: Any, **kwargs: Any) -> Any:
    """Create a RuntimeExecution before the Claude provider receives input."""
    from ouroboros.orchestrator.runtime_execution import (
        RuntimeExecution,
        RuntimeExecutionController,
    )

    controller = RuntimeExecutionController(adapter._runtime_handle_backend)
    stream = adapter.execute_task(**kwargs, _execution_controller=controller)
    return RuntimeExecution(
        backend=adapter._runtime_handle_backend,
        stream=stream,
        controller=controller,
    )


async def stream_owned_claude_client(
    *,
    client: Any,
    prompt: str,
    controller: Any | None,
    convert_message: Callable[[Any], Any],
    build_handle: Callable[..., Any],
    state: ClaudeRuntimeState,
    approval_mode: str,
    log: Any,
) -> AsyncIterator[Any]:
    """Connect, bind process authority, stream messages, and reap the SDK client."""
    owned_process: Any | None = None
    connect_task = asyncio.create_task(client.connect())

    async def settle_connect_task() -> bool:
        if not connect_task.done():
            connect_task.cancel()
            done, _ = await asyncio.wait((connect_task,), timeout=0.1)
            if not done:
                close = getattr(connect_task.get_coro(), "close", None)
                if not callable(close):
                    return False
                with suppress(BaseException):
                    close()
                connect_task.cancel()
                done, _ = await asyncio.wait((connect_task,), timeout=0.1)
                if not done:
                    return False
        if not connect_task.cancelled():
            with suppress(BaseException):
                connect_task.exception()
        return True

    async def force_owned_client() -> bool:
        if not await settle_connect_task():
            return False
        transport = getattr(client, "_transport", None)
        process = getattr(transport, "_process", None)
        if process is None:
            return True
        if getattr(process, "returncode", None) is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        return getattr(process, "returncode", None) is not None

    if controller is not None:
        controller.bind_process(force_owned_client)
    try:
        await asyncio.shield(connect_task)
        transport = getattr(client, "_transport", None)
        owned_process = getattr(transport, "_process", None)
        if controller is not None and owned_process is None:
            from ouroboros.orchestrator.runtime_execution import RuntimeExecutionUnavailable

            raise RuntimeExecutionUnavailable(
                "Claude SDK did not expose its owned subprocess before dispatch"
            )
        await client.query(prompt)
        async for sdk_message in client.receive_response():
            agent_message = convert_message(sdk_message)
            session_id = getattr(sdk_message, "session_id", None) or agent_message.data.get(
                "session_id"
            )
            if session_id and (session_id != state.session_id or state.runtime_handle is None):
                state.session_id = session_id
                state.runtime_handle = build_handle(
                    session_id,
                    state.runtime_handle,
                    approval_mode=approval_mode,
                )
            if state.runtime_handle:
                data = agent_message.data
                if state.session_id and data.get("session_id") != state.session_id:
                    data = {**data, "session_id": state.session_id}
                agent_message = replace(
                    agent_message,
                    data=data,
                    resume_handle=state.runtime_handle,
                )
            if agent_message.is_final:
                log.info(
                    "orchestrator.adapter.task_completed",
                    success=not agent_message.is_error,
                    session_id=session_id,
                )
            yield agent_message
    finally:
        connect_settled = await settle_connect_task()
        await client.disconnect()
        process_reaped = True
        if owned_process is not None and getattr(owned_process, "returncode", None) is None:
            owned_process.kill()
            await owned_process.wait()
            process_reaped = getattr(owned_process, "returncode", None) is not None
        if controller is not None and connect_settled and process_reaped:
            controller.mark_reaped()
