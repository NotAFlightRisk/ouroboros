"""Opaque parent-owned Seed handoff for passive plugin execution."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import yaml

_HIDDEN_WORKER_KEYS = frozenset({"verify_command", "output_assertion"})


def _worker_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _worker_safe_value(item)
            for key, item in value.items()
            if str(key) not in _HIDDEN_WORKER_KEYS
        }
    if isinstance(value, list):
        return [_worker_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_worker_safe_value(item) for item in value]
    return value


def render_worker_safe_seed(seed_content: str) -> str:
    """Render a Seed without harness-owned verifier commands/assertions.

    Malformed input fails closed: the raw text is never forwarded to a worker.
    """

    try:
        parsed = yaml.safe_load(seed_content)
    except yaml.YAMLError:
        return "# Seed omitted: invalid YAML; ask the parent to retry with a valid Seed.\n"
    if not isinstance(parsed, Mapping):
        return "# Seed omitted: expected a YAML mapping.\n"
    projected = _worker_safe_value(parsed)
    if projected == parsed:
        return seed_content
    return yaml.safe_dump(
        projected,
        sort_keys=False,
        allow_unicode=True,
    )


def plugin_evaluation_instruction(
    *,
    auto_evaluate: bool,
    auto_evolve: bool,
    seed_handoff_id: str | None,
) -> str:
    """Build the plugin child's explicit formal-evaluation successor contract."""

    if not auto_evaluate:
        return (
            "Formal evaluation auto-chain is disabled for this run; preserve the "
            "legacy manual next step `ooo evaluate <session_id>`."
        )
    seed_argument = (
        f"seed_handoff_id `{seed_handoff_id}`"
        if seed_handoff_id
        else "the worker-safe seed_content"
    )
    if not auto_evolve:
        return (
            "After every terminal evaluable execution result, including unsuccessful AC "
            "execution, run formal 3-stage evaluation without host involvement: call "
            "ouroboros_start_evaluate with the session_id, execution "
            f"artifact, {seed_argument}, working directory, and auto_evolve: false. "
            "This opt-out uses plugin delegation: when the call returns "
            "status `delegated_to_plugin` with no job_id, do not poll job tools. Treat "
            "the handoff as terminal and rely on the bridge-rendered evaluation child "
            "in the Task pane to deliver the final APPROVED/not-approved verdict. If "
            "evaluation fails, keep run success intact and report "
            "`ooo evaluate <session_id>` as the manual retry."
        )
    return (
        "After every terminal evaluable execution result, including unsuccessful AC "
        "execution, run formal 3-stage evaluation without host involvement: call "
        "ouroboros_start_evaluate with the session_id, execution "
        f"artifact, {seed_argument}, working directory, and auto_evolve: "
        f"{str(auto_evolve).lower()}; poll the returned job with "
        "ouroboros_job_wait/status. If its terminal result contains "
        "chained_ralph_job_id, poll that Ralph job to terminal too and report its "
        "convergence stop reason; otherwise include the final APPROVED/not-approved "
        "verdict in your report. If evaluation fails or times out, keep run success "
        "intact and report `ooo evaluate <session_id>` as the manual retry."
    )


@dataclass
class SeedHandoffRegistry:
    """Bounded process-local vault keyed by opaque plugin handoff IDs."""

    max_entries: int = 256
    _entries: OrderedDict[str, tuple[str, str]] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def register(self, *, session_id: str, seed_content: str) -> str:
        handoff_id = f"seed_handoff_{uuid4().hex}"
        self._entries[handoff_id] = (session_id, seed_content)
        self._entries.move_to_end(handoff_id)
        while len(self._entries) > max(1, self.max_entries):
            self._entries.popitem(last=False)
        return handoff_id

    def resolve(self, handoff_id: str, *, session_id: str) -> str | None:
        entry = self._entries.get(handoff_id)
        if entry is None or entry[0] != session_id:
            return None
        self._entries.move_to_end(handoff_id)
        return entry[1]
