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

   * opaque-value    — a 7+-digit / letter-hex identifier under an innocent
                       key (rounds 80-82 boundaries)
   * network-address — a dotted-quad value under any key (round 84)
   * mutating-verb   — a tool identifier naming a mutator (round 8)
   * credential-name — a source/tool name shaped like a credential
                       (rounds 29-40; metric absorbed its rules in round 79,
                       so the metric field must never appear here)
   * identity-metric — a value-returning aggregation over an identity metric
                       (round 48)

   The round-85 unverified-grouping class and the round-54 entity-KEY class
   (user_id, ip, passport_number as scope keys) were ABSORBED into the
   grouping, dimension, and filter grammars, like the metric's credential
   rules in round 79: the published patterns compile from the positive head
   sets, so an unverified or identity-named key is schema-invalid rather
   than schema-valid-but-rejected, and those classes no longer appear in
   the declared residue below. Entity-name detection remains in the
   validator for the retained path and as depth behind the grammar.

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
    ("opaque-7-digits", _answer(filters=["cohort=9999999"]), "opaque entity identifier"),
    ("opaque-bad-date", _answer(filters=["day=20260231"]), "opaque entity identifier"),
    (
        "identity-metric-max",
        _answer(metric="user_id", aggregation="max"),
        "identity metric may only be counted",
    ),
    # Round-84 addition to the declared classes. (The key is category-headed
    # so the VALUE rule is what rejects it; entity-named keys like `ip` or
    # `client` are grammar-invalid since round 86.)
    ("network-address-value", _answer(filters=["region=10.0.0.7"]), "network address"),
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


def test_the_scope_grammars_absorbed_the_positive_classification() -> None:
    """Rounds 85-86: positive key classification lives in the published patterns.

    An unverified or identity-named key (credit_card_number, imei,
    email_hash, passport_number, user_id, ip) is schema-INVALID for
    grouping, dimension, AND filters — not schema-valid-but-semantically-
    rejected — so the advertised and enforced surfaces agree by
    construction. Rejection messages never echo the key or value.
    Category-headed keys remain fully attainable end-to-end, and filter
    keys may also be measurement-headed (latency_ms, naics_code,
    created_at, key_metrics_30d).
    """
    for key in ("credit_card_number", "passport_number", "imei", "email_address", "email_hash"):
        assert not _READ_REQUEST_GROUPING.match(key), key
        assert not _AGGREGATE_DIMENSION.match(f"{key}=x1"), key
        violations = _data_evidence_boundary_violations(_proposal_answer(grouping=[key]))
        assert violations, key
        assert key not in " ".join(violations), "the rejected key was echoed"
    for scope in (
        "passport_number=zx123456",
        "user_id=541511",
        "ip=192.168.1.1",
        "client=10.0.0.7",
    ):
        assert not _READ_REQUEST_FILTER.match(scope), scope
        violations = _data_evidence_boundary_violations(_answer(filters=[scope]))
        assert violations, scope
        assert scope.split("=", 1)[1] not in " ".join(violations), "the rejected value was echoed"
    for key in ("plan_tier", "customer_segment", "month", "region", "device_type"):
        assert _READ_REQUEST_GROUPING.match(key), key
        assert _data_evidence_boundary_violations(_proposal_answer(grouping=[key])) == [], key
    for scope in ("latency_ms>200", "naics_code=541511", "created_at>2026-01-01", "build=v2_1"):
        assert _READ_REQUEST_FILTER.match(scope), scope
        assert _data_evidence_boundary_violations(_answer(filters=[scope])) == [], scope


def test_the_category_heads_are_advertised_in_the_policy() -> None:
    """Hosts and children see the same closed sets the grammars compile from."""
    from ouroboros.contracts.data_evidence import (
        _CATEGORY_HEADS,
        _MEASUREMENT_HEADS,
        _data_context_lane_policy,
    )

    policy = _data_context_lane_policy()
    assert policy["category_dimension_heads"] == sorted(_CATEGORY_HEADS)
    assert policy["filter_key_heads"] == sorted(_CATEGORY_HEADS | _MEASUREMENT_HEADS)
    assert "tier" in policy["category_dimension_heads"]
    assert "ms" in policy["filter_key_heads"]


def test_the_metric_field_never_needs_the_declared_residue() -> None:
    """Round 79 absorbed the metric's credential rules into its grammar; a
    schema-valid metric must therefore never hit a credential rejection —
    this is the property that distinguishes an absorbed field from one with
    declared residue."""
    for metric in ("token_usage_v2", "key_metrics_30d", "logins", "authentication_failures"):
        assert _READ_REQUEST_METRIC.match(metric), metric
        violations = _data_evidence_boundary_violations(_answer(metric=metric))
        assert violations == [], (metric, violations)
