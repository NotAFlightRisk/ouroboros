"""Pre-effect Claude SDK process ownership helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
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
    connect_task = asyncio.create_task(client.connect())
    shutdown_lock = asyncio.Lock()
    shutdown_verified = False

    async def settle_client_task(task: asyncio.Task[Any], *, operation: str) -> bool:
        from ouroboros.orchestrator.runtime_execution import settle_owned_task

        return await settle_owned_task(task, operation=operation)

    async def settle_task_handle(handle: Any, *, operation: str) -> bool:
        task = getattr(handle, "_task", None)
        if isinstance(task, asyncio.Task):
            return await settle_client_task(task, operation=operation)
        cancel = getattr(handle, "cancel", None)
        wait = getattr(handle, "wait", None)
        done = getattr(handle, "done", None)
        if callable(done) and done():
            return True
        if not callable(cancel) or not callable(wait):
            return False
        cancel()
        wait_task = asyncio.create_task(wait())
        return await settle_client_task(wait_task, operation=operation)

    async def settle_client_children(query: Any | None, transport: Any | None) -> bool:
        handles: list[tuple[Any, str]] = []
        if query is not None:
            read_task = getattr(query, "_read_task", None)
            if read_task is not None:
                handles.append((read_task, "Claude SDK response reader"))
            handles.extend(
                (task, "Claude SDK child task")
                for task in tuple(getattr(query, "_child_tasks", ()))
            )
        stderr_task = getattr(transport, "_stderr_task", None)
        if stderr_task is not None:
            handles.append((stderr_task, "Claude SDK stderr reader"))
        settled = True
        for handle, operation in handles:
            settled = await settle_task_handle(handle, operation=operation) and settled
        return settled


    async def reap_client_process(process: Any | None) -> bool:
        from ouroboros.orchestrator.runtime_execution import force_reap_process

        if process is None or getattr(process, "returncode", None) is not None:
            return True

        async def graceful_terminate(owned_process: Any) -> None:
            try:
                owned_process.terminate()
            except ProcessLookupError:
                return
            wait = getattr(owned_process, "wait", None)
            if callable(wait):
                await wait()

        return await force_reap_process(process, graceful_terminate)

    async def force_owned_client() -> bool:
        nonlocal shutdown_verified
        async with shutdown_lock:
            if shutdown_verified:
                return True

            connect_settled = await settle_client_task(
                connect_task,
                operation="Claude SDK connection setup",
            )
            query = getattr(client, "_query", None)
            transport = getattr(client, "_transport", None)
            process = getattr(transport, "_process", None)
            if not connect_settled:
                await settle_client_children(query, transport)
                await reap_client_process(process)
                return False

            disconnect_task = asyncio.create_task(client.disconnect())
            disconnect_settled = await settle_client_task(
                disconnect_task,
                operation="Claude SDK disconnect",
            )
            children_settled = await settle_client_children(query, transport)
            process_reaped = await reap_client_process(process)
            shutdown_verified = disconnect_settled and children_settled and process_reaped
            return shutdown_verified

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
        client_stopped = await force_owned_client()
        if not client_stopped:
            from ouroboros.orchestrator.runtime_execution import RuntimeExecutionUnavailable

            raise RuntimeExecutionUnavailable(
                "Claude SDK connection work did not reach a verified termination boundary"
            )
        if controller is not None:
            controller.mark_reaped()
