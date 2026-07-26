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

   * entity-key      — an identity-MODIFIED preserving head the grammar
                       cannot see (round-88: customer_code — the pattern
                       admits the head, the modifier semantics are
                       cross-token English)

   The round-48 identity-metric class was ABSORBED in round 100: a
   value-returning aggregation now requires a metric that positively names a
   measurement (VALUE_METRIC_PATTERN in the published schema), so
   max(user_id) — and max(credit_card_number), which no identity vocabulary
   recognized — are schema-invalid rather than schema-valid-but-rejected.

   The round-85 unverified-grouping class and the entity-NAMED keys
   (user_id, ip, passport_number as scope keys) were ABSORBED into the
   grouping, dimension, and filter grammars, like the metric's credential
   rules in round 79: the published patterns compile from the positive head
   sets, so an unverified or identity-named key is schema-invalid rather
   than schema-valid-but-rejected. What remains of the round-54 entity-key
   class is the modifier direction only.

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


def _proposal_answer(
    *,
    grouping: list[str] | None = None,
    metric: str = "logins",
    aggregation: str = "count",
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "operation": "read",
        "metric": metric,
        "aggregation": aggregation,
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
                "source_class": "metered",
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
    # An identity WORD as a value names a kind, not an individual — only the
    # word plus a numeric/hex run is a labeled identifier (round-89).
    ("filter-kind-value", _answer(filters=["type=user"])),
    ("filter-category-with-digits", _answer(filters=["cohort=tier_2"])),
    # Round-95: a long numeric category in its canonical spelling — the
    # label in the KEY, the number as a BARE value.
    ("filter-build-number", _answer(filters=["build=12345"])),
    ("filter-short-compound", _answer(filters=["release=build_1234"])),
    # Round-96: a bare number under a MEASUREMENT head is the measurement's
    # value — the opaque-digit rule guards CATEGORY keys (cohort=9999999,
    # still in the declared residue below).
    ("filter-seven-digit-build", _answer(filters=["build=1234567"])),
    ("filter-eight-digit-code", _answer(filters=["naics_code=54151100"])),
    # Grouping is advertised on PROPOSALS only — the evidence schema itself
    # rejects a grouped executed request (round-46: one number cannot say
    # which group it came from), which this sweep's first draft rediscovered.
    ("grouping-category", _proposal_answer(grouping=["plan_tier"])),
    ("dimension", _answer()),
]


def _assert_schema_valid(output: dict[str, Any]) -> None:
    """Both corpora claim schema-validity — assert it with the real validator.

    Round-90: the proposal fixture carried a schema-invalid ``source_class``
    and the sweep never noticed, so "schema-valid but rejected" was being
    asserted about outputs the schema rejects.
    """
    from jsonschema import Draft202012Validator

    schema = _data_context_answer_contract()["response_model_schema"]
    Draft202012Validator(schema).validate(output)


@pytest.mark.parametrize(("label", "output"), _ACCEPTED, ids=[label for label, _ in _ACCEPTED])
def test_schema_valid_scopes_are_accepted(label: str, output: dict[str, Any]) -> None:
    """Property 2: what the grammar advertises, enforcement accepts."""
    _assert_schema_valid(output)
    assert _data_evidence_boundary_violations(output) == [], label


#: Property 3 corpus: schema-valid but rejected — each row names its declared
#: class via a stable message fragment.
_DECLARED_REJECTIONS: list[tuple[str, dict[str, Any], str]] = [
    # Round-88: the grammar admits the `code` HEAD; whether the MODIFIER
    # names an entity (customer_code — a per-customer pseudonym, like
    # email_hash) or a classification standard (naics_code — categorical,
    # in the accepted corpus above) is cross-token semantics the pattern
    # cannot express, so it stays declared residue.
    # (Value shortened in round-91: the long concatenated form became
    # schema-invalid under the positive value grammar; the KEY residue this
    # row pins is unchanged.)
    ("entity-modified-code", _answer(filters=["customer_code=zx12"]), "keys an entity"),
    # Round-89: a category KEY with an identity-LABELED value is one
    # person's row — the label names what the digits index. (Long-digit
    # compounds were absorbed into the grammar in round 95; the declared
    # residue is the identity word beside a SHORT run the grammar admits.)
    (
        "labeled-identifier-value",
        _answer(filters=["segment=user_42"]),
        "labeled entity identifier",
    ),
    ("opaque-7-digits", _answer(filters=["cohort=9999999"]), "opaque entity identifier"),
    ("opaque-bad-date", _answer(filters=["day=20260231"]), "opaque entity identifier"),
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
    _assert_schema_valid(output)
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
        # Round-91: concatenated indexed-entity values are not representable
        # under the positive value-token grammar, whatever the key.
        "segment=employee123456",
        "employee_code=zx123456",
        "cohort=a1b2c3d4e5",
        # Round-95: 5+-digit COMPOUND segments joined them — the round-90
        # semantic rule and the grammar now agree by construction, and the
        # canonical spelling (build=12345) stays attainable above.
        "segment=user_1234567",
        "segment=employee_123456",
        "release=build_12345",
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

    # The cross-token semantics the head grammars cannot express are exposed
    # as a machine-readable rule, from the SAME vocabularies enforcement
    # applies (round-89: the customer_code rejection was declared only in
    # tests, invisible to hosts).
    from ouroboros.contracts.data_evidence import (
        _IDENTITY_KEYS,
        _IDENTITY_PRESERVING_HEADS,
    )

    rules = policy["identity_scope_rules"]
    assert rules["identity_tokens"] == sorted(_IDENTITY_KEYS)
    assert rules["identity_preserving_heads"] == sorted(_IDENTITY_PRESERVING_HEADS)
    assert "code" in rules["identity_preserving_heads"]


def test_the_metric_field_never_needs_the_declared_residue() -> None:
    """Round 79 absorbed the metric's credential rules into its grammar; a
    schema-valid metric must therefore never hit a credential rejection —
    this is the property that distinguishes an absorbed field from one with
    declared residue."""
    for metric in ("token_usage_v2", "key_metrics_30d", "logins", "authentication_failures"):
        assert _READ_REQUEST_METRIC.match(metric), metric
        violations = _data_evidence_boundary_violations(_answer(metric=metric))
        assert violations == [], (metric, violations)


def test_round98_version_quads_and_analytics_heads_are_attainable() -> None:
    """Key-aware address detection and head-last credential reading.

    version=1.2.3.4 was schema-valid yet classified as a network address;
    access_token_usage had no valid spelling at all (schema accepted,
    classifier rejected). The version-key exemption is narrow — region keeps
    the round-84 address rule — and the analytics-head exemption is
    fail-closed: round-40's refresh_token_alphabetic stays rejected.
    """
    from ouroboros.contracts.data_evidence import _identifier_looks_secret

    for scope in ("version=1.2.3.4", "release=1.2.3.4", "build=1.2.3.4"):
        _assert_schema_valid(_answer(filters=[scope]))
        assert _data_evidence_boundary_violations(_answer(filters=[scope])) == [], scope
    violations = _data_evidence_boundary_violations(_answer(filters=["region=10.0.0.7"]))
    assert any("network address" in violation for violation in violations)

    assert not _identifier_looks_secret("access_token_usage")
    assert not _identifier_looks_secret("api_key_metrics")
    assert _identifier_looks_secret("refresh_token_alphabetic")
    assert _identifier_looks_secret("access_token")
    assert _identifier_looks_secret("access_key_abcd1234")


def test_round100_value_returning_requires_a_measurement_metric() -> None:
    """Positive classification on the value-returning surface.

    max(credit_card_number), max(passport_number) and max(token) were
    schema-valid proposals directing the parent to fetch PII or a
    credential — the identity vocabulary recognized none of them. A
    value-returning aggregation now requires a metric that NAMES a
    measurement; counting any metric stays available.
    """
    from ouroboros.contracts.data_evidence import _read_request_shape_problems

    for metric in ("credit_card_number", "passport_number", "token", "user_id", "ssn"):
        request = {"operation": "read", "metric": metric, "aggregation": "max"}
        assert _read_request_shape_problems(request), metric
        assert (
            _read_request_shape_problems(
                {"operation": "read", "metric": metric, "aggregation": "count"}
            )
            == []
        ), f"counting {metric} must stay available"
    for metric in ("latency_ms", "checkout_duration", "revenue_total", "error_rate"):
        assert (
            _read_request_shape_problems(
                {"operation": "read", "metric": metric, "aggregation": "max"}
            )
            == []
        ), metric
