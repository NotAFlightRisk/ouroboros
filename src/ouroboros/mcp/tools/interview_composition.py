"""Shared interview-family wiring for executable MCP composition roots.

Two roots build handlers: ``create_ouroboros_server`` wires the MCP server a
client connects to, and ``get_ouroboros_tools`` wires the lighter set an
in-process runtime dispatches to. They are not copies of each other and should
not be merged -- one injects an event store, interview engines and a job
manager the other has no equivalent for.

What *was* duplicated is the wiring contract for the handlers they share, and
that is what lives here. Both interview handlers are advisory producers: each
takes a fan-out registry, and the PM one also takes the workspace whose
artifacts a lane is pointed at. Those two arguments must arrive together --
a producer holding a registry and no workspace computes the right prior-round
ids and then discards them, which reads exactly like a session with no history.
In #2146 the server passed the workspace to the artifact store and nothing to
the producer, twenty lines apart in one function, and the feature was dead in
production while every lane-level test stayed green.

One function both roots call is what keeps that from recurring by omission:
there is no longer a second place to remember. Where a root genuinely has less
to give -- no event store, no engine, no workspace -- it passes less, and the
handlers' own defaults absorb it. That is the difference between the roots that
is real, kept, and separated from the contract that is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ouroboros.mcp.tools.authoring_handlers import InterviewHandler
from ouroboros.mcp.tools.fanout import FanoutRegistry
from ouroboros.mcp.tools.pm_handler import PMInterviewHandler


def create_interview_handlers(
    *,
    fanout_registry: FanoutRegistry | None = None,
    workspace: Path | str | None = None,
    llm_backend: str | None = None,
    agent_runtime_backend: str | None = None,
    opencode_mode: str | None = None,
    interview_engine: Any | None = None,
    event_store: Any | None = None,
    state_dir: Path | None = None,
    suppress_tool_use_prompt_cues: bool = False,
) -> tuple[InterviewHandler, PMInterviewHandler]:
    """Return the interview and PM-interview handlers, wired the same way.

    ``workspace`` is the resolved project the fan-out artifact store is built
    from, and it reaches the PM producer so a lane can be told where this
    session's earlier answers were published. A root that configures no store
    passes ``None`` and the producer tells its children nothing, which is the
    honest state rather than a degraded one: there is nothing stored to point
    at, and a child sent looking would spend a tool call learning that.

    Returned as a pair rather than registered here, because which tools a root
    exposes and in what order is the root's business; what belongs to this
    module is only how the two are wired.
    """
    interview = InterviewHandler(
        interview_engine=interview_engine,
        event_store=event_store,
        llm_backend=llm_backend,
        agent_runtime_backend=agent_runtime_backend,
        opencode_mode=opencode_mode,
        fanout_registry=fanout_registry,
        suppress_tool_use_prompt_cues=suppress_tool_use_prompt_cues,
    )
    pm_interview = PMInterviewHandler(
        data_dir=state_dir,
        event_store=event_store,
        llm_backend=llm_backend,
        agent_runtime_backend=agent_runtime_backend,
        opencode_mode=opencode_mode,
        fanout_registry=fanout_registry,
        project_dir=Path(workspace) if workspace is not None else None,
    )
    return interview, pm_interview


__all__ = ["create_interview_handlers"]
