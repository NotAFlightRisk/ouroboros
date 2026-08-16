"""Both composition roots must agree about where fan-out artifacts live.

Ouroboros builds its handlers in two places, and the split is not accidental:
``create_ouroboros_server`` wires the MCP server a client connects to, with an
event store, interview engines and a job manager behind it, while
``get_ouroboros_tools`` wires the lighter set an in-process runtime dispatches
to without a server at all. Neither list is a copy of the other -- they inject
different services, and merging them would mean one path dragging in what the
other has no equivalent for.

What *is* duplicated is the wiring contract for the handlers they share, and
that is what these tests hold. A fan-out has two sides: the submit handler
publishes each completed run into a project-local artifact store, and the
advisory producer tells a child where those artifacts landed. The two sides are
wired from separate arguments in both roots, so a root can wire one and not the
other -- which is exactly what shipped in #2146, where the server passed its
workspace to the store and nothing to the producer. Every lane-level test kept
passing, because each of them builds its own request and so never travels the
composition that omitted one.

The invariant is a conditional rather than "both roots pass the same thing": a
root that cannot store fan-out results has nothing to point a child at, and
telling it to carry a workspace anyway would be a rule with no defect behind
it. What must not exist is the half -- storage with no address.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ouroboros.mcp.server.adapter import create_ouroboros_server
from ouroboros.mcp.tools.definitions import get_ouroboros_tools
from ouroboros.mcp.tools.fanout_handler import SubmitFanoutResultsHandler
from ouroboros.mcp.tools.pm_handler import PMInterviewHandler


def _artifact_root(handlers: Any) -> Path | None:
    """Return the root the submit side would publish into, if it can publish."""
    submit = next(
        (handler for handler in handlers if isinstance(handler, SubmitFanoutResultsHandler)),
        None,
    )
    if submit is None or submit.disposable_memory is None:
        return None
    return Path(submit.disposable_memory.artifact_store.root)


def _advisory_root(handlers: Any) -> Path | None:
    """Return the root the advisory producer would send a child to."""
    producer = next(
        (handler for handler in handlers if isinstance(handler, PMInterviewHandler)),
        None,
    )
    assert producer is not None, "no PM advisory producer registered"
    if producer.project_dir is None:
        return None
    return Path(producer.project_dir).expanduser().resolve() / ".ouroboros" / "artifacts"


def test_the_server_points_the_producer_at_the_store_it_writes_to(tmp_path: Path) -> None:
    """The composition that ships, compared side against side by value.

    Equality is the assertion, not "both are set". Two roots agreeing that a
    path exists is not the same as agreeing on which one, and a producer sent
    to a neighbouring directory reads exactly like a session with no history.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = create_ouroboros_server(
        project_dir=workspace,
        state_dir=tmp_path / "state",
        durable_jobs=False,
    )
    handlers = list(server._tool_handlers.values())

    stored_at = _artifact_root(handlers)
    assert stored_at is not None, "the server can always publish; this root cannot be absent"
    assert _advisory_root(handlers) == stored_at


def test_the_runtime_tool_set_points_the_producer_at_the_store_it_writes_to(
    tmp_path: Path,
) -> None:
    """The other root, held to the same conditional, with a workspace given."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    handlers = list(get_ouroboros_tools(project_dir=workspace))

    stored_at = _artifact_root(handlers)
    assert stored_at is not None, "a workspace was given, so publication is configured"
    assert _advisory_root(handlers) == stored_at


def test_a_root_that_cannot_store_results_is_not_asked_to_carry_an_address() -> None:
    """No workspace means no store, and no store means nothing to point at.

    Held so the invariant above cannot later be "tightened" into demanding a
    workspace everywhere. A runtime tool set built for inspection has no
    artifacts, and a child sent to look for them would spend a tool call
    learning that.
    """
    handlers = list(get_ouroboros_tools())

    assert _artifact_root(handlers) is None
    assert _advisory_root(handlers) is None
