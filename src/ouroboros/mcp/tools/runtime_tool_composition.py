"""Configured handler graph for runtime-owned builtin MCP interception.

Capability discovery intentionally uses lightweight handler constructors. A
runtime interceptor, however, executes those handlers and therefore needs the
same configured run -> evaluate -> Ralph -> evolve graph as the MCP server.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_COMPOSING_RUNTIME_TOOLS: ContextVar[bool] = ContextVar(
    "ouroboros_composing_runtime_tools",
    default=False,
)


def configured_runtime_tools(
    *,
    runtime_backend: str | None,
    llm_backend: str | None,
    opencode_mode: str | None,
    include_auto: bool,
    mcp_bridge: Any | None,
) -> tuple[Any, ...] | None:
    """Return the production handler graph, or ``None`` for lightweight fallback.

    ``create_ouroboros_server`` constructs runtime adapters whose registry
    fingerprint calls back into ``get_ouroboros_tools``. The ContextVar makes
    that nested discovery use lightweight definitions while the outer call
    finishes the real graph.
    """

    # The hidden-Seed handoff is specific to passive OpenCode plugin dispatch:
    # the worker re-enters this local registry for parent-owned evaluation.
    # Other runtimes retain their lightweight builtin registry and their
    # existing dispatcher/server ownership model.
    if (
        runtime_backend != "opencode"
        or opencode_mode != "plugin"
        or _COMPOSING_RUNTIME_TOOLS.get()
    ):
        return None

    from ouroboros.mcp.server.adapter import create_ouroboros_server
    from ouroboros.mcp.tools.evaluation_handlers import ChecklistVerifyHandler

    compose_token = _COMPOSING_RUNTIME_TOOLS.set(True)
    try:
        try:
            server = create_ouroboros_server(
                runtime_backend=runtime_backend,
                llm_backend=llm_backend,
                opencode_mode=opencode_mode,
                mcp_bridge=mcp_bridge,
                # Embedded interceptors own their event loop. Preserve this
                # factory's historical in-process JobManager behavior.
                durable_jobs=False,
            )
        except RuntimeError as exc:
            # Capability discovery stays importable when an optional provider
            # extra is absent. Actual calls cannot use that backend either.
            if isinstance(exc.__cause__, ImportError):
                return None
            raise
    finally:
        _COMPOSING_RUNTIME_TOOLS.reset(compose_token)

    configured = dict(server._tool_handlers)  # noqa: SLF001 - composition reuse
    configured["ouroboros_checklist_verify"] = ChecklistVerifyHandler(
        evaluate_handler=configured["ouroboros_evaluate"],
        llm_backend=llm_backend,
    )
    ordered_names = (
        "ouroboros_execute_seed",
        "ouroboros_start_execute_seed",
        *(("ouroboros_auto", "ouroboros_start_auto") if include_auto else ()),
        "ouroboros_session_status",
        "ouroboros_job_status",
        "ouroboros_job_wait",
        "ouroboros_job_result",
        "ouroboros_ac_tree_hud",
        "ouroboros_cancel_job",
        "ouroboros_query_events",
        "ouroboros_query_projection",
        "ouroboros_project_status",
        "ouroboros_generate_seed",
        "ouroboros_measure_drift",
        "ouroboros_interview",
        "ouroboros_evaluate",
        "ouroboros_start_evaluate",
        "ouroboros_checklist_verify",
        "ouroboros_lateral_think",
        "ouroboros_submit_fanout_results",
        "ouroboros_evolve_step",
        "ouroboros_start_evolve_step",
        "ouroboros_ralph",
        "ouroboros_start_ralph",
        "ouroboros_lineage_status",
        "ouroboros_evolve_rewind",
        "ouroboros_cancel_execution",
        "ouroboros_brownfield",
        "ouroboros_pm_interview",
        "ouroboros_qa",
    )
    return tuple(configured[name] for name in ordered_names)
