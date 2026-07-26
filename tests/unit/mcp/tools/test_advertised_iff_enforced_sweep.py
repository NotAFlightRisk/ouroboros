"""Advertised IFF enforced, swept across every scope grammar at once.

Rounds 77-82 each aligned ONE rule where the published grammar and the
semantic validator disagreed — scan field coverage, camel boundaries, letter
case, calendar partitions, letter-hex values, digit widths. Per-rule
alignment does not converge: the next probe samples the next shape.

This module pins the population. Three properties:

1. Every published pattern string IS the enforcement-side compile — the two
   surfaces share one definition and cannot drift.
2. A corpus of schema-valid values per scope field completes end-to-end —
   what the grammar advertises, re-entry accepts.
3. Every schema-valid value re-entry DOES reject belongs to a DECLARED
   semantic class, asserted by its message. The declared classes are the
   deliberate residue the grammar cannot express, each with its reason:

   * entity-key      — grouping/filtering BY an identity column
                       (cross-token English-head rule, round 54)
   * opaque-value    — a 7+-digit / letter-hex identifier under an innocent
                       key (rounds 80-82 boundaries)
   * mutating-verb   — a tool identifier naming a mutator (round 8)
   * credential-name — a source/tool name shaped like a credential
                       (rounds 29-40; metric absorbed its rules in round 79,
                       so the metric field must never appear here)
   * identity-metric — a value-returning aggregation over an identity metric
                       (round 48)

A rejection outside these classes is this test failing, not a future review
round.
"""

from __future__ import annotations

from typing import Any

import pytest

from ouroboros.contracts.data_evidence import (
    _AGGREGATE_DIMENSION,
    _READ_REQUEST_FILTER,
    _READ_REQUEST_GROUPING,
    _READ_REQUEST_METRIC,
    _data_context_answer_contract,
    _data_evidence_boundary_violations,
)


def _defs() -> dict[str, Any]:
    return _data_context_answer_contract()["response_model_schema"]["$defs"]


def test_every_published_pattern_is_the_enforcement_compile() -> None:
    """Property 1: one definition per grammar, on both surfaces."""
    defs = _defs()
    request = defs["read_request"]["properties"]
    assert request["metric"]["pattern"] == _READ_REQUEST_METRIC.pattern
    assert request["filters"]["items"]["pattern"] == _READ_REQUEST_FILTER.pattern
    assert request["grouping"]["items"]["pattern"] == _READ_REQUEST_GROUPING.pattern
    assert defs["aggregate"]["properties"]["dimension"]["pattern"] == _AGGREGATE_DIMENSION.pattern


def _answer(
    *,
    metric: str = "logins",
    filters: list[str] | None = None,
    grouping: list[str] | None = None,
    aggregation: str = "count",
) -> dict[str, Any]:
    request: dict[str, Any] = {"operation": "read", "metric": metric, "aggregation": aggregation}
    if filters is not None:
        request["filters"] = filters
    if grouping is not None:
        request["grouping"] = grouping
    return {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "Counted for the requested scope.",
        "confidence": "reported_by_tool",
        "evidence": [
            {
                "source": "warehouse",
                "request": request,
                "value": {"number": 12},
                "observed_at": "2026-07-25T00:00:00Z",
                "execution_status": "succeeded",
            }
        ],
        "proposed_queries": [],
        "caveats": ["point-in-time"],
        "requires_user_confirmation": True,
    }


def _proposal_answer(*, grouping: list[str] | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {
        "operation": "read",
        "metric": "logins",
        "aggregation": "count",
    }
    if grouping is not None:
        request["grouping"] = grouping
    return {
        "lane_id": "data_context",
        "data_needed": True,
        "finding": "A grouped count is proposed for the parent to run.",
        "confidence": "inferred",
        "evidence": [],
        "proposed_queries": [
            {
                "tool_name": "clickhouse_query",
                "request": request,
                "expected_decision": "Whether enterprise tiers dominate logins.",
                "source_class": "metered_or_uncertain",
            }
        ],
        "caveats": ["point-in-time"],
        "requires_user_confirmation": True,
    }


#: Property 2 corpus: schema-valid, must produce zero violations.
_ACCEPTED: list[tuple[str, dict[str, Any]]] = [
    ("metric-leading-token", _answer(metric="token_usage_v2")),
    ("metric-window", _answer(metric="key_metrics_30d")),
    ("metric-dotted", _answer(metric="api.requests-total")),
    ("metric-long-word", _answer(metric="authentication_failures")),
    ("filter-category", _answer(filters=["cohort=enterprise"])),
    ("filter-month", _answer(filters=["month=202607"])),
    ("filter-date", _answer(filters=["date=20260725"])),
    ("filter-category-code", _answer(filters=["naics_code=541511"])),
    ("filter-version", _answer(filters=["build=v2_1"])),
    ("filter-comparison", _answer(filters=["latency_ms>200"])),
    ("filter-short-hex", _answer(filters=["release=beta7"])),
    # Grouping is advertised on PROPOSALS only — the evidence schema itself
    # rejects a grouped executed request (round-46: one number cannot say
    # which group it came from), which this sweep's first draft rediscovered.
    ("grouping-category", _proposal_answer(grouping=["plan_tier"])),
    ("dimension", _answer()),
]


@pytest.mark.parametrize(("label", "output"), _ACCEPTED, ids=[label for label, _ in _ACCEPTED])
def test_schema_valid_scopes_are_accepted(label: str, output: dict[str, Any]) -> None:
    """Property 2: what the grammar advertises, enforcement accepts."""
    assert _data_evidence_boundary_violations(output) == [], label


#: Property 3 corpus: schema-valid but rejected — each row names its declared
#: class via a stable message fragment.
_DECLARED_REJECTIONS: list[tuple[str, dict[str, Any], str]] = [
    ("entity-filter-key", _answer(filters=["user_id=541511"]), "keys an entity"),
    ("entity-grouping", _proposal_answer(grouping=["email_address"]), "keys an entity"),
    ("opaque-7-digits", _answer(filters=["cohort=9999999"]), "opaque entity identifier"),
    ("opaque-bad-date", _answer(filters=["day=20260231"]), "opaque entity identifier"),
    (
        "identity-metric-max",
        _answer(metric="user_id", aggregation="max"),
        "identity metric may only be counted",
    ),
]


@pytest.mark.parametrize(
    ("label", "output", "declared_class"),
    _DECLARED_REJECTIONS,
    ids=[label for label, _, _ in _DECLARED_REJECTIONS],
)
def test_every_semantic_rejection_belongs_to_a_declared_class(
    label: str,
    output: dict[str, Any],
    declared_class: str,
) -> None:
    """Property 3: the residue beyond the grammar is named, bounded, and stable."""
    violations = _data_evidence_boundary_violations(output)
    assert violations, label
    assert any(declared_class in violation for violation in violations), (label, violations)


def test_the_metric_field_never_needs_the_declared_residue() -> None:
    """Round 79 absorbed the metric's credential rules into its grammar; a
    schema-valid metric must therefore never hit a credential rejection —
    this is the property that distinguishes an absorbed field from one with
    declared residue."""
    for metric in ("token_usage_v2", "key_metrics_30d", "logins", "authentication_failures"):
        assert _READ_REQUEST_METRIC.match(metric), metric
        violations = _data_evidence_boundary_violations(_answer(metric=metric))
        assert violations == [], (metric, violations)
