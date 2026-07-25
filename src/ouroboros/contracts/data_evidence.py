"""Data-evidence contract semantics — schema, policy, and their enforcement.

The ``data_context`` lane's guarantees used to live in the transport module
next to payload building and fan-out plumbing, and the schema, the policy
block, and the re-entry checks drifted apart repeatedly (Q00/ouroboros#1703
review rounds: a credential prefix known to one scan and not the other, a
cap the policy advertised and nothing enforced, a fallback weaker than the
contract it stood in for). They are one subject, so they live in one module.

What this module owns:

* the vocabularies the contract declares (aggregation kinds, unit and
  identifier grammars, credential and identity words),
* the semantic checks re-entry runs before anything durable is written,
* the fallback schema used when a lane's declared contract cannot be enforced.

What it deliberately does NOT own: transport (payload building, registry
persistence, correlation) and the contract's published JSON, which stays in
``orchestrator.capabilities.interview_schemas`` as capability metadata. This
module is the single place where that published form is interpreted.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import lru_cache
import json
import math
import re
from typing import Any

CONTRACT_VERSION = "data_evidence_answer.v1"


# Raw-evidence shapes the data_context evidence policy forbids (aggregates
# only, PII-scrubbed): an email-shaped substring is PII, a credential prefix
# glued to an opaque digit-bearing suffix is a leaked secret, and a
# phone-shaped digit group is PII — never an aggregate. Written without
# inline regex flags so the same strings are valid in both Python `re` and
# the ECMA dialect JSON Schema validators use; the re-entry enforcement
# point recompiles the case-sensitive ones case-insensitively.
DATA_EVIDENCE_EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"


# The digit lookahead keeps ordinary hyphenated vocabulary ("token-counts",
# "secret-santa") out: a credential suffix carries digits, a compound noun
# does not.
# Vendor token prefixes are PUBLISHED constants, so they are the one part of
# credential detection that is a fact rather than a heuristic. Declared once
# and shared by the content scan and the identifier classifier (round-42:
# the content scan knew "xox" while the identifier classifier knew "xox[a-z]",
# so the standard xoxb-/xoxp- Slack forms passed content validation).
DATA_EVIDENCE_VENDOR_TOKEN_PREFIX = r"(?:github_pat|ghp|gho|ghu|ghs|ghr|xox[a-z]|sk|pk)[-_]"


# The prefix is recognized after ANY separator, not only at a \b word
# boundary (round-43: "warehouse_xoxb-123456789..." — an underscore is a word
# character, so \b never matched and the compound name hid the token).
DATA_EVIDENCE_SECRET_PATTERN = (
    r"(?:^|[^A-Za-z0-9])"
    + DATA_EVIDENCE_VENDOR_TOKEN_PREFIX
    + r"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{4,}"
    r"|\b(sk|pk|token|secret|bearer|api[_-]?key|ghp|gho|xox)"
    r"[-_=:](?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{4,}"
)


# Standard credential header/assignment forms (bot-review round-6 probe):
# an Authorization/Bearer phrase followed by a digit-bearing opaque value, a
# password assignment (sensitive regardless of digits), or an AWS-style
# access key id.
DATA_EVIDENCE_AUTH_HEADER_PATTERN = (
    # Space-separated form needs a digit-bearing value ("authorization
    # required for X" stays valid); the explicit colon HEADER form is a
    # credential shape regardless of alphabet (round-7 probe: alphabetic
    # Bearer values).
    r"\b(authorization|bearer)\b[:= ]+(?=[^\s]*\d)[A-Za-z0-9_.=/+-]{8,}"
    r"|\b(authorization|bearer)\s*:\s*\S{6,}"
)


DATA_EVIDENCE_PASSWORD_PATTERN = r"\b(password|passwd|pwd)\b\s*[:=]\s*\S{4,}"


# Credential ASSIGNMENTS are secrets regardless of alphabet (round-34:
# "api_key=supersecret"; round-35: alphabetic "bearer=..." assignments) —
# the assignment itself is the signal, no digit entropy required.
# The credential word marks the assignment from ANY position inside a
# compound name (round-39: client_secret=, refresh_token=, private_key=) —
# the same position-independent vocabulary the identifier classifier applies,
# so the content scan and the identifier scan cannot disagree about what a
# credential is named.
DATA_EVIDENCE_CREDENTIAL_ASSIGNMENT_PATTERN = (
    r"\b(?:[A-Za-z0-9]+[_-])*"
    r"(api[_-]?key|access[_-]?key|keys?|secrets?|tokens?|bearer|credentials?|creds)"
    r"(?:[_-][A-Za-z0-9]+)*"
    r"\s*[:=]\s*[A-Za-z0-9_\-/+]{6,}"
)


DATA_EVIDENCE_AWS_KEY_PATTERN = r"(?:^|[^A-Za-z0-9])(AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}"


# RFC 3986 userinfo: a ``scheme://user:password@host`` connection string
# carries the password in the URI itself (round-41 probe:
# "endpoint=https://alice:swordfish@localhost:8443"), which no credential
# WORD appears in. The shape is unambiguous, so it is matched structurally.
DATA_EVIDENCE_URI_USERINFO_PATTERN = r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@"


# US Social Security Number shape (round-7 probe): the phone pattern's group
# widths deliberately do not cover the 3-2-4 split.
DATA_EVIDENCE_SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


# Phone shapes: international (+ then 7+ digits), separator-grouped local
# numbers, or the US parenthesized area-code form. Comma-grouped magnitudes
# ("1,234,567") and ISO dates/times do not match (comma and colon are not in
# the separator class, and date groups are 2-digit).
DATA_EVIDENCE_PHONE_PATTERN = (
    r"\+\d{7,}|\b\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4}\b|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}"
)


# An executed evidence value is a MEASUREMENT: aggregation is numeric by
# definition, so a value with no numeral is a claim or a name roster, never
# query output (qualitative context belongs in finding). This is the
# structural rule that makes digit-free record lists unrepresentable under
# any delimiter.
DATA_EVIDENCE_MEASUREMENT_PATTERN = r"\d"


# A value that opens as a JSON list/object is serialized rows, not an
# aggregate.
DATA_EVIDENCE_ROW_SHAPE_PATTERN = r"^\s*[\[{]"


# An aggregate is a single-line scalar statement: any embedded newline means
# a record list (two customer rows separated by one newline are raw data).
DATA_EVIDENCE_MULTILINE_PATTERN = r"[\r\n]"


def _data_context_answer_contract() -> dict[str, Any]:
    """Return the answer contract for the data_context advisory lane.

    Unlike ``code_fact_investigation_answer.v1`` this contract has NO grade
    clause (``prefix_semantics``) and no auto-confirmed path: every data
    answer requires user confirmation, because data evidence is point-in-time
    and cannot be cheaply re-verified the way a manifest exact-match can
    (Q00/ouroboros#1671). The contract's job is informed consent — the form
    must give the confirming user everything needed to judge: what was
    executed (evidence), what was deliberately NOT executed and why
    (proposed_queries with source_class), and validity caveats.

    Kept intentionally compact: the serialized contract must fit whole inside
    the subagent prompt JSON budget (see the truncation of the code contract
    at ``_INTERVIEW_ADVISORY_MAX_JSON_CHARS``).
    """
    answer_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "lane_id",
            "data_needed",
            "finding",
            "confidence",
            "evidence",
            "proposed_queries",
            "requires_user_confirmation",
        ],
        "properties": {
            "lane_id": {"const": "data_context"},
            "data_needed": {"type": "boolean"},
            "finding": {"type": "string", "minLength": 1, "maxLength": 600},
            "confidence": {"enum": ["reported_by_tool", "inferred", "no_evidence"]},
            "evidence": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    # observed_at is required: executed data evidence is
                    # point-in-time by nature, and an aggregate without its
                    # observation timestamp loses that meaning by the time it
                    # reaches the confirming user and persisted state.
                    # execution_status is required and TYPED (round-42): the
                    # "a failed call is not evidence" rule was enforced only
                    # by recognizing failure vocabulary in prose, which is a
                    # detector, not an invariant. Declaring the outcome makes
                    # it a contract term — evidence may exist only for a
                    # succeeded execution, and any other value is a located
                    # violation instead of a missed phrase.
                    "required": [
                        "source",
                        "request",
                        "value",
                        "observed_at",
                        "execution_status",
                    ],
                    "properties": {
                        "execution_status": {
                            "const": "succeeded",
                        },
                        "source": {
                            "type": "string",
                            "maxLength": 64,
                            "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
                        },
                        # Executed evidence carries ONE number, so a grouped
                        # request cannot describe it: a single value cannot say
                        # which group it came from (round-46). Per-category
                        # evidence is one item per category, narrowed by a
                        # filter; grouping stays available on proposals.
                        # Executed evidence carries a NUMBER into durable
                        # state, so its aggregation must provably not be one
                        # of the inputs (round-51: max(credit_card_number)
                        # returns the card number, and no vocabulary of
                        # column names can decide which columns identify).
                        # count/distinct_count reduce to a cardinality and
                        # sum/avg to a derived total; min/max/median/
                        # percentile return an input verbatim, so they are
                        # requestable as PROPOSALS — which carry no value —
                        # but not reportable as evidence.
                        "request": {
                            "allOf": [
                                {"$ref": "#/$defs/read_request"},
                                {"not": {"required": ["grouping"]}},
                                {
                                    "properties": {
                                        "aggregation": {"$ref": "#/$defs/cardinality_aggregation"}
                                    }
                                },
                            ]
                        },
                        "value": {"$ref": "#/$defs/aggregate"},
                        "observed_at": {
                            "type": "string",
                            "maxLength": 40,
                            # ISO-8601-shaped WITH calendar/clock ranges: a real
                            # date, optionally followed by a real time and zone
                            # offset. Draft 2020-12 validators do not enforce
                            # "format", and a digits-only shape accepted
                            # month 99 / hour 99 (bot-review round-3 probe).
                            "pattern": (
                                r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
                                r"([T ]([01]\d|2[0-3]):[0-5]\d(:[0-5]\d(\.\d{1,6})?)?"
                                r"([Zz]|[+-]([01]\d|2[0-3]):?[0-5]\d)?)?$"
                            ),
                        },
                    },
                },
            },
            "proposed_queries": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["tool_name", "request", "expected_decision", "source_class"],
                    # An unexecuted proposal is only useful if the parent
                    # session can actually run and judge it: empty tool,
                    # request, or decision fields are not a proposal.
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "maxLength": 64,
                            "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
                        },
                        "request": {"$ref": "#/$defs/read_request"},
                        "expected_decision": {"type": "string", "minLength": 1, "maxLength": 300},
                        "source_class": {
                            "enum": [
                                "metered",
                                "external",
                                "side_effect_ambiguous",
                                "unknown",
                            ]
                        },
                    },
                },
            },
            "requires_user_confirmation": {"const": True},
            "caveats": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
            },
        },
        "$defs": {
            "cardinality_aggregation": {
                # A cardinality counts ROWS, so it is provably never one of
                # the values in them. sum/avg looked safe but a singleton
                # cohort makes either return its input verbatim (round-52:
                # sum(credit_card_number) over one row IS the card number).
                # Evidence is the only place a number becomes durable, so it
                # carries only counts; everything else is requestable as a
                # proposal, which carries no value.
                "enum": ["count", "distinct_count"],
            },
            "aggregation_kind": {
                # Each kind names ONE measurement. ratio/share/duration were
                # under-specified (no numerator, denominator, or unit basis),
                # so two identical requests could mean different things — a
                # share is reported as its numerator and its total, which is
                # also what makes it auditable.
                "enum": [
                    "count",
                    "distinct_count",
                    "sum",
                    "avg",
                    "median",
                    "percentile",
                    "min",
                    "max",
                ]
            },
            "aggregate": {
                # An aggregate IS a number with a unit — that is what makes it
                # an aggregate rather than a record. Typing it this way makes
                # raw rows, PII, credentials, and error narratives
                # UNREPRESENTABLE in the durable evidence path instead of
                # something a text classifier must recognize and reject.
                "type": "object",
                "additionalProperties": False,
                "required": ["number", "unit"],
                "properties": {
                    # A cardinality is a whole number of rows. Bound here
                    # rather than left to "is finite" (round-52: a count of
                    # -1.5 validated), since evidence only ever reports one.
                    "number": {"type": "integer", "minimum": 0},
                    "unit": {
                        "type": "string",
                        "maxLength": 24,
                        "pattern": "^[a-z%][a-z_/%]{0,23}$",
                    },
                    "dimension": {
                        "type": "string",
                        "maxLength": 48,
                        "pattern": "^[a-z][a-z0-9_]{0,23}=[a-z0-9][a-z0-9_.-]{0,21}$",
                    },
                },
            },
            "read_request": {
                "allOf": [
                    {
                        "if": {"properties": {"aggregation": {"const": "percentile"}}},
                        "then": {"required": ["percentile"]},
                    },
                    {
                        # Two-way (round-50): a percentile on a count is a
                        # contradictory request, not a harmless extra field.
                        "if": {
                            "properties": {"aggregation": {"not": {"const": "percentile"}}},
                            "required": ["aggregation"],
                        },
                        "then": {"not": {"required": ["percentile"]}},
                    },
                ],
                # A read request is a STRUCTURE, not a query string. The lane
                # names what to measure; the parent session builds and runs the
                # actual query after user confirmation. Mutating statements,
                # side-effecting functions, and prose instructions have no
                # field to live in.
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "metric", "aggregation"],
                "properties": {
                    "operation": {"const": "read"},
                    "metric": {
                        "type": "string",
                        "maxLength": 64,
                        "pattern": "^[A-Za-z][A-Za-z0-9_.*-]{0,63}$",
                    },
                    "aggregation": {"$ref": "#/$defs/aggregation_kind"},
                    # Required when aggregation is percentile: p95 and p50 are
                    # different measurements and must not share a request form.
                    "percentile": {"type": "integer", "minimum": 1, "maximum": 99},
                    "filters": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "maxLength": 48,
                            "pattern": "^[a-z][a-z0-9_]{0,23}(=|!=|<|>|<=|>=)[a-z0-9][a-z0-9_.:+-]{0,21}$",
                        },
                    },
                    "grouping": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "maxLength": 32,
                            "pattern": "^[a-z][a-z0-9_]{0,31}$",
                        },
                    },
                },
            },
        },
        # The DURABLE form is declared, not improvised (round-47): advisory
        # prose is delivered to the host and then dropped, so the retained
        # record is a different — and equally contract-valid — state of the
        # same answer. Declaring it here is what makes a replayed completion
        # validate instead of looking like a mangled submission.
        "allOf": [
            {
                # As SUBMITTED: the advisory narrative is the lane's point, so
                # finding is required and caveats accompany executed evidence.
                # As RETAINED (prose_retained: false): both are absent by
                # design, and a replayed completion is still this contract.
                "if": {"not": {"required": ["prose_retained"]}},
                "then": {"required": ["finding"]},
            },
            {
                "if": {"required": ["prose_retained"]},
                "then": {"not": {"anyOf": [{"required": ["finding"]}, {"required": ["caveats"]}]}},
            },
            {
                "if": {"properties": {"data_needed": {"const": False}}},
                "then": {
                    "properties": {
                        "confidence": {"const": "no_evidence"},
                        "evidence": {"maxItems": 0},
                        "proposed_queries": {"maxItems": 0},
                    }
                },
            },
            {
                "if": {"properties": {"data_needed": {"const": True}}},
                "then": {
                    "anyOf": [
                        {"properties": {"evidence": {"minItems": 1}}, "required": ["evidence"]},
                        {
                            "properties": {"proposed_queries": {"minItems": 1}},
                            "required": ["proposed_queries"],
                        },
                    ]
                },
            },
            {
                # confidence is tied to what was actually executed:
                # "reported_by_tool" without a single executed evidence item
                # (e.g. a proposal-only response) is a category error.
                "if": {"properties": {"confidence": {"const": "reported_by_tool"}}},
                "then": {
                    "properties": {"evidence": {"minItems": 1}},
                    "required": ["evidence"],
                },
            },
            {
                # Executed evidence must carry its point-in-time warning to the
                # confirming user: at least one caveat is required whenever any
                # evidence item exists.
                "if": {
                    "properties": {"evidence": {"minItems": 1}},
                    "required": ["evidence"],
                },
                "then": {
                    "properties": {"caveats": {"minItems": 1}},
                    "required": ["caveats"],
                },
            },
            {
                # The confidence constraint is two-way (round-12): executed
                # evidence alongside confidence="no_evidence" is contradictory
                # informed-consent state, just as reported_by_tool without
                # evidence is.
                "if": {
                    "properties": {"evidence": {"minItems": 1}},
                    "required": ["evidence"],
                },
                "then": {
                    "properties": {"confidence": {"enum": ["reported_by_tool", "inferred"]}},
                },
            },
        ],
    }
    return {
        "contract_id": "data_evidence_answer.v1",
        "scope": "single_data_context_advisory_lane",
        "response_model_schema": answer_schema,
        "proposed_query_semantics": {
            "execution": "parent_session_only_after_user_confirmation",
            "auto_execution": "forbidden",
        },
        "runtime_instruction": (
            "Evidence and proposals are STRUCTURED, not prose: each carries a "
            "read_request; evidence adds the resulting aggregate, observed_at, "
            "and execution_status. There is no free-text value or query field — "
            "what cannot be expressed as an aggregate is a no-evidence finding, "
            "and a lookup you did not run belongs in proposed_queries with its "
            "source_class. Narrative goes in finding and caveats."
        ),
    }


def data_evidence_retained_schema() -> dict[str, Any]:
    """Schema of the DURABLE record — a server-owned lifecycle state.

    Callers submit against the published contract, which requires the advisory
    prose and forbids these markers (round-49: a fresh caller could otherwise
    declare itself "retained" and skip the finding it exists to provide). This
    form is derived from that contract and never advertised as a submission
    target: prose is absent because the server removed it, not because a
    client chose to omit it.
    """
    schema = json.loads(json.dumps(_data_context_answer_contract()["response_model_schema"]))
    schema["required"] = [key for key in schema["required"] if key != "finding"]
    schema["properties"]["prose_retained"] = {"const": False}
    schema["properties"]["proposed_queries"]["items"]["properties"]["decision_retained"] = {
        "const": False
    }
    schema["properties"]["proposed_queries"]["items"]["required"] = [
        key
        for key in schema["properties"]["proposed_queries"]["items"]["required"]
        if key != "expected_decision"
    ]
    key_only = {"type": "string", "maxLength": 24, "pattern": "^[a-z][a-z0-9_]{0,23}$"}
    defs = schema["$defs"]
    defs["aggregate"]["properties"]["dimension"] = key_only
    defs["read_request"]["properties"]["filters"] = {
        "type": "array",
        "maxItems": 5,
        "items": key_only,
    }
    schema["required"] = [*schema["required"], "prose_retained"]
    schema["allOf"] = [
        clause
        for clause in schema.get("allOf", [])
        if "caveats" not in json.dumps(clause.get("then", {}))
    ]
    return schema


def data_evidence_structural_schema() -> dict[str, Any]:
    """The data answer contract's schema, used when a lane's DECLARED contract
    cannot be enforced.

    It is the published schema itself. The fallback is never delivered to a
    child, so it carries no prompt budget and there is no reason for it to
    drop a single required field or conditional (round-45: a trimmed copy let
    a required lane terminalize without data_needed, confidence, evidence, or
    no-op consistency). What a degraded contract loses is the DECLARED form's
    own field grammar — never the invariants this contract states.
    """
    return _data_context_answer_contract()["response_model_schema"]


def _data_context_lane_policy() -> dict[str, Any]:
    """Return the machine-readable data-access policy for the data_context lane.

    Prompt text alone is too weak for a lane that touches production data
    stores (Q00/ouroboros#1671). Hosts with permission systems get this block
    to enforce; the lane prompt restates it as the fallback. The lane is a
    read-only *proposer*: it directly executes only obviously local, free,
    read-only lookups and returns everything else as proposed queries for the
    parent session to run after user confirmation.
    """
    return {
        "read_only": True,
        "aggregate_only": True,
        "relevance_gate": "decide_from_question_text_before_any_tool_call",
        "direct_execution_scope": "local_free_read_only_lookups_only",
        "metered_or_uncertain_sources": "return_proposed_queries_without_executing",
        "error_shaped_tool_output": "return_no_evidence_finding",
        "forbidden_operation_patterns": [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "truncate",
            "create",
            "grant",
            "write",
            "save",
            "upload",
            "publish",
            "upsert",
            "replace",
            "merge",
            "call",
            "exec",
            "execute",
            # Destructive-verb synonyms (bot-review round-37 probe:
            # destroy_database / remove_user / rename_database advertised as
            # preferred tools): the identifier filter matches these as whole
            # tokens, so read-only vocabulary ("removed duplicates in the
            # rollup") is unaffected — prose is scanned by operation SHAPES,
            # not this list.
            "destroy",
            "remove",
            "rename",
            "purge",
            "wipe",
            "erase",
            "revoke",
        ],
        "evidence_policy": {
            "max_evidence_items": 5,
            "max_evidence_chars": 2000,
            "aggregates_only": True,
            "raw_rows_allowed": False,
            "pii_scrub_required": True,
            # How the policy above is ENFORCED rather than merely asserted:
            # evidence values and proposed lookups are typed structures in
            # data_evidence_answer.v1, so raw rows, PII values, credentials,
            # error envelopes, and mutating statements have no field to
            # occupy. Prose survives only in operator-facing narrative.
            "enforcement": "typed_contract_fields",
            "free_text_fields": ["finding", "caveats", "expected_decision"],
            # A unit is a MEASUREMENT unit. Declaring the vocabulary makes
            # "this unit is one of the declared kinds" a decidable claim, so
            # an identity attribute cannot be worn as a unit (round-44 probe:
            # phone digits as the number, "phone" as the unit). Hosts that
            # need another unit extend this list; the engine enforces
            # whatever the snapshot declares.
            "allowed_units": [
                "accounts",
                "users",
                "sessions",
                "events",
                "requests",
                "calls",
                "queries",
                "rows",
                "records",
                "items",
                "orders",
                "messages",
                "errors",
                "ms",
                "s",
                "minutes",
                "hours",
                "days",
                "weeks",
                "months",
                "bytes",
                "kb",
                "mb",
                "gb",
                "%",
                "ratio",
                "count",
                "krw",
                "usd",
            ],
        },
    }


#: Historical alias kept for readability at call sites; the version above is
#: the single definition (round-47 suggestion).
_DATA_EVIDENCE_CONTRACT_ID = CONTRACT_VERSION


@lru_cache(maxsize=1)
def _data_evidence_boundary_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    # Function-level import: subagent.py is imported by orchestrator-side
    # modules, so the canonical pattern strings are pulled lazily.
    from ouroboros.orchestrator.capabilities.interview_schemas import (
        DATA_EVIDENCE_AUTH_HEADER_PATTERN,
        DATA_EVIDENCE_AWS_KEY_PATTERN,
        DATA_EVIDENCE_CREDENTIAL_ASSIGNMENT_PATTERN,
        DATA_EVIDENCE_EMAIL_PATTERN,
        DATA_EVIDENCE_PASSWORD_PATTERN,
        DATA_EVIDENCE_PHONE_PATTERN,
        DATA_EVIDENCE_SECRET_PATTERN,
        DATA_EVIDENCE_SSN_PATTERN,
        DATA_EVIDENCE_URI_USERINFO_PATTERN,
    )

    return (
        ("email-shaped (PII)", re.compile(DATA_EVIDENCE_EMAIL_PATTERN, re.IGNORECASE)),
        ("credential-shaped (secret)", re.compile(DATA_EVIDENCE_SECRET_PATTERN, re.IGNORECASE)),
        (
            "authorization-header-shaped (secret)",
            re.compile(DATA_EVIDENCE_AUTH_HEADER_PATTERN, re.IGNORECASE),
        ),
        (
            "password-assignment-shaped (secret)",
            re.compile(DATA_EVIDENCE_PASSWORD_PATTERN, re.IGNORECASE),
        ),
        (
            "credential-assignment-shaped (secret)",
            re.compile(DATA_EVIDENCE_CREDENTIAL_ASSIGNMENT_PATTERN, re.IGNORECASE),
        ),
        # AWS access-key ids are uppercase by definition; case-insensitive
        # matching would false-positive on ordinary prose.
        ("aws-access-key-shaped (secret)", re.compile(DATA_EVIDENCE_AWS_KEY_PATTERN)),
        (
            "uri-userinfo-shaped (secret)",
            re.compile(DATA_EVIDENCE_URI_USERINFO_PATTERN),
        ),
        ("phone-shaped (PII)", re.compile(DATA_EVIDENCE_PHONE_PATTERN)),
        ("ssn-shaped (PII)", re.compile(DATA_EVIDENCE_SSN_PATTERN)),
    )


@lru_cache(maxsize=1)
def _mutating_tool_verbs() -> frozenset[str]:
    from ouroboros.orchestrator.capabilities.interview_schemas import (
        _data_context_lane_policy,
    )

    return frozenset(
        str(word).lower() for word in _data_context_lane_policy()["forbidden_operation_patterns"]
    )


def _mutating_tool_verb(tool_name: str) -> str | None:
    """Return the mutating verb a tool identifier carries, if any.

    Confirmed proposals are handed to the host for execution, so a proposal
    whose TOOL NAME is a mutator (``delete_database``) is executable payload
    regardless of its query text (bot-review round-8 probe). Identifiers are
    tokenized on non-alphanumerics AND camelCase boundaries because ``\\b``
    splits neither snake_case nor ``SaveReport``.
    """
    decamelled = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", tool_name)
    tokens = re.split(r"[^a-z0-9]+", decamelled.lower())
    verbs = _mutating_tool_verbs()
    return next((token for token in tokens if token in verbs), None)


# Error-envelope shapes: a JSON error key or an HTTP 4xx/5xx status inside an
# evidence value means the tool call FAILED — the policy requires a
# no-evidence finding, never evidence ("error rate 0.2%" prose stays valid;
# only the envelope shapes match).
_DATA_EVIDENCE_ERROR_SHAPE = re.compile(
    r"[\"']error[\"']\s*:|[\"']ok[\"']\s*:\s*[Ff]alse|\bHTTP[ /]?[45]\d{2}\b"
    r"|\btraceback \(most recent call last\)"
    # Assignment-style failure envelopes, = and : forms (rounds 19-26:
    # "status=failed code=502", "status: failed; code: 502",
    # "success=false; status_code=503; retries exhausted").
    # The VALUE class is execution-outcome vocabulary, not just failed/error
    # (round-41 probe: "status=timeout; attempts=3"). These words describe a
    # call's outcome and never name a data category, so the assignment is a
    # failure envelope wherever it appears; domain statuses (status=churned,
    # status=active) are untouched.
    r"|\b(status|state|outcome|result)\s*[:=]\s*"
    r"(failed|failure|error|errored|timeout|timed_out|denied|unauthorized"
    r"|forbidden|refused|rejected|throttled|rate_limited|unavailable"
    r"|unreachable|aborted|exception)\b"
    r"|\b(status_)?code\s*[:=]\s*[45]\d{2}\b"
    # Boolean STATUS assignments close as a class, both polarities (rounds
    # 35-36 probes: "success=no", "failure=true" — word-form variants of
    # success=false): a success-word assigned a negative, or a failure-word
    # assigned an affirmative, is a failure envelope. Metric prose keeps no
    # assignment ("success rate 92%", "failure rate 0.2%") and stays valid.
    r"|\b(success|succeeded)\s*[:=]\s*(false|no)\b"
    r"|\b(fail|failed|failure)\s*[:=]\s*(true|yes|1(?!\d))\b"
    r"|\bretr(y|ies)\s+exhausted\b"
    # Assignment-form error envelopes (round-28: value="error=503"); plural
    # metric forms ("errors: 12", "error rate 0.2%") stay valid.
    r"|\berror\s*[:=]\s*\S"
    # Singular-subject failure narratives (round-27: "request timed out
    # after 3 attempts"); PLURAL forms are metrics ABOUT failures ("1,240
    # requests timed out: 0.8%") and stay valid.
    r"|\b(request|query|connection|call|lookup)\s+timed\s+out\b"
    r"|\b(upstream|gateway|connection|request|query|read|write)\s+timeout\b"
    # Failure prose variants (round-33: "query failed because permission was
    # denied"); plural metric forms ("queries failed: 12") stay valid.
    r"|\b(query|request|call|lookup|connection)\s+failed\b"
    # Negated success is the same statement (round-42: "lookup unsuccessful;
    # attempts=3"). Negation is a closed grammatical form, not another
    # failure synonym.
    r"|\b(query|request|call|lookup|connection|execution)\s+"
    r"(was\s+)?(unsuccessful|not\s+successful)\b"
    r"|\b(unable\s+to|could\s+not|failed\s+to)\s+"
    r"(fetch|query|run|execute|reach|connect|retrieve|read|complete)\b"
    # Bare HTTP failure statuses (round-34: "403 Forbidden", "returned 403").
    r"|\b[45]\d{2}\s+(forbidden|unauthorized|not\s+found|bad\s+request"
    r"|unavailable|error)\b"
    r"|\breturned\s+[45]\d{2}\b"
    r"|\b(permission|access|authorization)\s+(was|is)\s+denied\b"
    r"|\bafter\s+\d+\s+(attempts?|retries)\b"
    # Availability narratives and bare status codes (round-29: "database
    # unavailable, code 503").
    r"|\b(service|database|db|api|endpoint|server|tool|warehouse)\s+"
    r"(unavailable|unreachable|down)\b"
    r"|\bcode\s+[45]\d{2}\b"
    # No-result admissions (round-31: a caveat narrating that nothing came
    # back alongside "executed" evidence).
    r"|\bno\s+(result|results|records|rows|data)\s+(was\s+|were\s+)?"
    r"(returned|found|available)\b",
    re.IGNORECASE,
)


# Embedded serialized-row shapes for PROSE fields (finding, caveats,
# provenance text): a JSON array-of-objects opening, an object chain, or a
# quoted-key object inside prose is smuggled rows. This is the
# field-appropriate subset of the evidence.value row check — prose fields
# legitimately contain newlines, and query text legitimately contains comma
# lists, so neither of those rules applies here.
_EMBEDDED_JSON_ROWS = re.compile(r"\[\s*\{|\}\s*,\s*\{|\{\s*\"[^\"]+\"\s*:")


# An identity-shaped token: an alphabetic label bound to a numeric id
# (user-123, acct_4471, cust-9001). One is a reference; SEVERAL SHARING THE
# SAME LABEL are a column of ids — that repetition is the row signature, and
# it holds under any delimiter (round-42 probe: "user-123 has 12 seats /
# user-456 has 13 seats" used " / " to evade the comma and semicolon forms).
_IDENTITY_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)[-_](\d{2,})\b")


def _repeats_identity_tokens(text: str) -> bool:
    """Whether prose enumerates several ids under one label."""
    by_label: dict[str, set[str]] = {}
    for label, number in _IDENTITY_TOKEN.findall(text):
        by_label.setdefault(label.lower(), set()).add(number)
    return any(len(numbers) >= 2 for numbers in by_label.values())


# A persisted prose field is ONE advisory statement for the human, not a
# layout. Restricting the field's GRAMMAR — no line breaks, no column pipes,
# no spaced-slash separators, at most two sentences — removes the shapes a
# record table needs, rather than trying to recognize the records themselves
# (round-44 probe: "Alice Smith premium 12 seats / Bob Jones free 8 seats",
# which carries no digits inside its identity tokens and so evaded the row
# heuristics). This bounds the residual prose surface; it does not close it —
# the data itself lives in the typed evidence, which is why the finding never
# needs to enumerate anything.
_PROSE_LAYOUT = re.compile(r"[\r\n|]|\s/\s|\s[•·]\s")


def _prose_layout_problem(text: str) -> str | None:
    if _PROSE_LAYOUT.search(text):
        return (
            "uses record-layout separators (line breaks, pipes, spaced "
            "slashes); an advisory statement is one sentence, and the numbers "
            "belong in typed evidence"
        )
    if len(re.findall(r"[.!?](?=\s|$)", text)) > 2:
        return "is more than two sentences; state one advisory finding"
    return None


def _prose_contains_rows(text: str) -> bool:
    """Row-shaped content embedded in a prose field (not evidence.value)."""
    if _EMBEDDED_JSON_ROWS.search(text):
        return True
    if len(re.findall(r"[A-Za-z],[A-Za-z]", text)) >= 2:
        return True
    if _repeats_identity_tokens(text):
        return True
    segments = [segment for segment in text.split(";") if segment.strip()]
    return len(segments) >= 2 and all(segment.count(",") >= 2 for segment in segments)


def _parseable_timestamp(value: str) -> bool:
    """True when ``value`` is a real ISO-8601 calendar date(-time).

    The schema's range regex cannot reject impossible dates like 2026-02-31;
    parsing can.
    """

    normalized = value[:-1] + "+00:00" if value and value[-1] in "zZ" else value
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


#: Fields of ``data_evidence_answer.v1`` that are prose written for a human.
#: They carry no data — the typed evidence does — and the policy's
#: ``pii_scrub_required`` cannot be enforced over them: a name and a street
#: address are PII that no pattern recognizes, and the alternative to
#: recognizing them is not storing them (round-46).
DATA_EVIDENCE_PROSE_FIELDS = ("finding", "caveats")
DATA_EVIDENCE_PROPOSAL_PROSE_FIELDS = ("expected_decision",)


def redact_prose_for_persistence(output: Any) -> Any:
    """Return a data-lane output with its human-facing prose removed.

    The host receives the prose in the response it submitted; the durable
    record keeps only the typed facts. The registry exists for correlation,
    completion, and replay of what was MEASURED — it is not the host's memory
    of advisory sentences, and a durable copy of unconfirmed prose is exactly
    the thing the policy claims to scrub and cannot.
    """
    if not isinstance(output, Mapping):
        return output
    redacted = {
        key: value for key, value in output.items() if key not in DATA_EVIDENCE_PROSE_FIELDS
    }
    proposals = redacted.get("proposed_queries")
    if isinstance(proposals, list):
        redacted["proposed_queries"] = [
            {
                **{
                    key: value
                    for key, value in item.items()
                    if key not in DATA_EVIDENCE_PROPOSAL_PROSE_FIELDS
                },
                "decision_retained": False,
            }
            if isinstance(item, Mapping)
            else item
            for item in proposals
        ]
    # A category VALUE is free-form and can be an address or a name however it
    # is cased (round-50: street=123_main_st). The keys say what the
    # measurement was scoped by, which is what the durable record needs; the
    # values were shown to the user in the response and are not retained.
    evidence = redacted.get("evidence")
    if isinstance(evidence, list):
        redacted["evidence"] = [_scope_keys_only(item) for item in evidence]
    proposals = redacted.get("proposed_queries")
    if isinstance(proposals, list):
        redacted["proposed_queries"] = [_scope_keys_only(item) for item in proposals]
    redacted["prose_retained"] = False
    return redacted


def _predicate_key(text: str) -> str:
    match = _FILTER_OPERATOR.search(text)
    return text[: match.start()] if match else text


def _scope_keys_only(item: Any) -> Any:
    """Drop category VALUES from a retained evidence item or proposal."""
    if not isinstance(item, Mapping):
        return item
    reduced = dict(item)
    request = reduced.get("request")
    if isinstance(request, Mapping):
        request = dict(request)
        filters = request.get("filters")
        if isinstance(filters, list):
            request["filters"] = [
                _predicate_key(text) if isinstance(text, str) else text for text in filters
            ]
        reduced["request"] = request
    value = reduced.get("value")
    if isinstance(value, Mapping) and isinstance(value.get("dimension"), str):
        value = dict(value)
        value["dimension"] = _predicate_key(value["dimension"])
        reduced["value"] = value
    return reduced


def _data_evidence_boundary_violations(
    output: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    *,
    retained: bool = False,
) -> list[str]:
    """Check a data-lane output against the data policy at re-entry.

    ``data_evidence_answer.v1`` carries evidence and proposals as TYPED
    structures — an aggregate is a number with a unit, a proposal is a
    structured read request — so raw rows, PII values, credentials, error
    envelopes, and mutating statements have no field to live in. This function
    therefore checks INVARIANTS, not vocabulary:

    * the typed shapes actually hold (also for records persisted before the
      contract was typed, where the schema may be missing or unenforced),
    * executed evidence declares a succeeded execution and a real timestamp,
    * identifier fields do not name a mutating tool or carry a credential.

    The remaining free text is advisory prose written FOR THE HUMAN
    (``finding``, ``caveats``, ``expected_decision``); it carries no data, so
    the PII/credential/row scan over it is defense-in-depth rather than the
    mechanism. Matched content is never echoed into the messages: violations
    flow back to the host and ride the persisted record.
    """
    try:
        serialized = json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return ["data evidence output is not JSON-serializable"]

    # Identifier FIELDS (evidence source, proposal tool_name) are exempt
    # from the credential-shape scan ONLY when they actually look like
    # identifiers (rounds 29-30): token_usage_v2 is a tool name, but a
    # value like "token=sk-live-123" in the source field is a credential
    # wearing the field, so anything failing the identifier syntax stays
    # fully scanned. Exempted identifiers remain guarded by the
    # mutating-verb checks, and every other pattern scans the full output.
    def _blank_if_identifier(item: Any, field: str) -> Any:
        if not isinstance(item, Mapping):
            return item
        value = item.get(field)
        if (
            isinstance(value, str)
            and _SAFE_IDENTIFIER_SYNTAX.match(value)
            and not _identifier_looks_secret(value)
        ):
            return {**item, field: ""}
        return item

    identifier_scrubbed = dict(output)
    raw_evidence = identifier_scrubbed.get("evidence")
    if isinstance(raw_evidence, list):
        identifier_scrubbed["evidence"] = [
            _blank_if_identifier(item, "source") for item in raw_evidence
        ]
    raw_proposals = identifier_scrubbed.get("proposed_queries")
    if isinstance(raw_proposals, list):
        identifier_scrubbed["proposed_queries"] = [
            _blank_if_identifier(item, "tool_name") for item in raw_proposals
        ]
    try:
        serialized_sans_identifiers = json.dumps(
            identifier_scrubbed, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):
        serialized_sans_identifiers = serialized
    errors = [
        f"data evidence boundary: {label} content is not admissible; the "
        "evidence policy requires typed aggregates, PII-scrubbed"
        for label, pattern in _data_evidence_boundary_patterns()
        if pattern.search(
            serialized_sans_identifiers if label.startswith("credential") else serialized
        )
    ]

    # Confirmation is the lane's defining term, so it is checked HERE rather
    # than only in the schema (round-43): when the declared contract is
    # unenforceable the registry keeps a minimal fallback, and a fallback that
    # carried only {"type": "object"} let requires_user_confirmation=false
    # through. A code-level invariant cannot be degraded by a fallback.
    if output.get("requires_user_confirmation") is not True:
        errors.append(
            "requires_user_confirmation: every data answer requires explicit "
            "user confirmation; the field must be present and true"
        )

    evidence_items = output.get("evidence")
    # Caps the ADVERTISED policy declares are enforced from the persisted
    # snapshot (round-43): a limit that only the prompt knows about is a
    # suggestion, and max_evidence_chars had never been applied.
    evidence_policy = (policy or {}).get("evidence_policy")
    allowed_units = (
        evidence_policy.get("allowed_units") if isinstance(evidence_policy, Mapping) else None
    )
    if isinstance(evidence_policy, Mapping) and isinstance(evidence_items, list):
        max_items = evidence_policy.get("max_evidence_items")
        if isinstance(max_items, int) and len(evidence_items) > max_items:
            errors.append(
                f"evidence: {len(evidence_items)} items exceeds the declared "
                f"max_evidence_items ({max_items})"
            )
        max_chars = evidence_policy.get("max_evidence_chars")
        if isinstance(max_chars, int):
            try:
                measured = len(json.dumps(evidence_items, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                measured = max_chars + 1
            if measured > max_chars:
                errors.append(
                    f"evidence: {measured} serialized chars exceeds the declared "
                    f"max_evidence_chars ({max_chars})"
                )
    for index, item in enumerate(evidence_items if isinstance(evidence_items, list) else ()):
        if not isinstance(item, Mapping):
            continue
        # The TYPED shape is the boundary. A string value is the pre-typed
        # form: accepting it would reopen the free-text path the contract
        # closed, so it fails closed here even when the persisted schema is
        # missing or unenforceable.
        errors.extend(
            f"evidence[{index}].value: {problem}"
            for problem in _aggregate_shape_problems(
                item.get("value"), allowed_units, retained=retained
            )
        )
        errors.extend(
            f"evidence[{index}].request: {problem}"
            for problem in _read_request_shape_problems(
                item.get("request"), executed=True, retained=retained
            )
        )
        # The value states only the number, its unit, and its scope — the
        # aggregation lives in the request alone (round-45), so "a count
        # request reported as an avg" has no field to be expressed in. What
        # remains bindable is the scope: you can only narrow a result to a
        # dimension you actually grouped by.
        executed, reported = item.get("request"), item.get("value")
        if isinstance(executed, Mapping) and isinstance(reported, Mapping):
            dimension = reported.get("dimension")
            if isinstance(dimension, str) and "=" in dimension:
                filters = executed.get("filters")
                applied = {
                    text
                    for text in (filters if isinstance(filters, list) else [])
                    if isinstance(text, str)
                }
                # The whole predicate must match, not just the key (round-47:
                # a request filtered to plan=free reported as plan=growth is a
                # mislabelled measurement, and the label is what the user
                # confirms).
                if dimension not in applied:
                    errors.append(
                        f"evidence[{index}].value.dimension: reports a scope the "
                        "executed request did not apply; it must repeat one of "
                        "the request's filters exactly"
                    )
        # Evidence exists only for a SUCCEEDED execution (round-42): the
        # outcome is declared, so a failed lookup is a located violation
        # rather than a phrase some detector has to recognize.
        if item.get("execution_status") != "succeeded":
            errors.append(
                f"evidence[{index}].execution_status: evidence requires a "
                "succeeded execution; a failed, denied, timed-out, or "
                "undeclared lookup is a no-evidence finding"
            )
        observed_at = item.get("observed_at")
        if not isinstance(observed_at, str) or not _parseable_timestamp(observed_at):
            errors.append(
                f"evidence[{index}].observed_at: not a real ISO-8601 calendar date(-time)"
            )
        # ``source`` names the executed tool. It is a free identifier, so the
        # two identifier rules still apply: a mutating tool name is a
        # read-only violation, and a credential-shaped name is the leak.
        source = item.get("source")
        if isinstance(source, str) and not _SAFE_IDENTIFIER_SYNTAX.match(source):
            errors.append(
                f"evidence[{index}].source: must be the executed tool's identifier, not prose"
            )
        if isinstance(source, str) and (verb := _mutating_tool_verb(source)):
            errors.append(
                f"evidence[{index}].source: names a mutating tool ({verb!r}); "
                "the data lane is read-only"
            )
        if isinstance(source, str) and _identifier_looks_secret(source.strip()):
            errors.append(
                f"evidence[{index}].source: credential-shaped identifier; "
                "name the read-only data tool instead"
            )

    proposals = output.get("proposed_queries")
    for index, item in enumerate(proposals if isinstance(proposals, list) else ()):
        if not isinstance(item, Mapping):
            continue
        # A proposal is executed by the parent after confirmation, so its
        # request is the executable payload. Typed, ``operation`` is a const
        # and every other field is a bounded identifier grammar — a mutating
        # statement, a side-effecting function call, and a prose instruction
        # are all unrepresentable rather than filtered.
        errors.extend(
            f"proposed_queries[{index}].request: {problem}"
            for problem in _read_request_shape_problems(item.get("request"), retained=retained)
        )
        tool_name = item.get("tool_name")
        if isinstance(tool_name, str) and (verb := _mutating_tool_verb(tool_name)):
            errors.append(
                f"proposed_queries[{index}].tool_name: names a mutating tool "
                f"({verb!r}); the data lane is read-only"
            )
        if isinstance(tool_name, str) and _identifier_looks_secret(tool_name.strip()):
            errors.append(
                f"proposed_queries[{index}].tool_name: credential-shaped "
                "identifier; name the read-only data tool instead"
            )
        # ``expected_decision`` is advisory prose for the human.
        decision = item.get("expected_decision")
        if isinstance(decision, str) and _prose_contains_rows(decision):
            errors.append(
                f"proposed_queries[{index}].expected_decision: row-shaped "
                "content is raw evidence and may not ride any persisted field"
            )
        if isinstance(decision, str) and (problem := _prose_layout_problem(decision)):
            errors.append(f"proposed_queries[{index}].expected_decision: {problem}")

    # Advisory prose for the human. It carries no data — the aggregates do —
    # so these checks are defense-in-depth, not the boundary.
    finding = output.get("finding")
    if isinstance(finding, str) and _prose_contains_rows(finding):
        errors.append(
            "finding: row-shaped (serialized-record) content is raw evidence "
            "and may not ride any persisted field; state aggregates only"
        )
    if isinstance(finding, str) and (problem := _prose_layout_problem(finding)):
        errors.append(f"finding: {problem}")
    if (
        isinstance(finding, str)
        and isinstance(evidence_items, list)
        and evidence_items
        and _DATA_EVIDENCE_ERROR_SHAPE.search(finding)
    ):
        errors.append(
            "finding: describes a failed lookup; a failed call is a "
            "no-evidence finding, not evidence with a narrative"
        )
    caveats = output.get("caveats")
    for index, caveat in enumerate(caveats if isinstance(caveats, list) else ()):
        if not isinstance(caveat, str):
            continue
        if _prose_contains_rows(caveat):
            errors.append(
                f"caveats[{index}]: row-shaped (serialized-record) content is "
                "raw evidence and may not ride any persisted field"
            )
        if problem := _prose_layout_problem(caveat):
            errors.append(f"caveats[{index}]: {problem}")
        # A caveat admitting the call produced nothing contradicts EXECUTED
        # evidence it accompanies (rounds 31-32); a no-op legitimately
        # narrates that no lookup was needed.
        if (
            isinstance(evidence_items, list)
            and evidence_items
            and _DATA_EVIDENCE_ERROR_SHAPE.search(caveat)
        ):
            errors.append(
                f"caveats[{index}]: describes a failed lookup; a failed call "
                "is a no-evidence finding, not evidence with a caveat"
            )
    return errors


def _aggregate_shape_problems(
    value: Any, allowed_units: Any = None, *, retained: bool = False
) -> list[str]:
    """Whether an evidence value is a typed aggregate.

    An aggregate is a NUMBER with a unit. Enforcing that here — not only in
    the JSON Schema — keeps the guarantee for records whose persisted contract
    is missing or unenforceable, and makes the pre-typed free-text form fail
    closed instead of silently reopening the old path.
    """
    if not isinstance(value, Mapping):
        return ["must be a typed aggregate object ({number, unit[, dimension]}), not free text"]
    problems: list[str] = []
    number = value.get("number")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        problems.append("number must be a JSON number")
    elif isinstance(number, float) and not number.is_integer():
        problems.append("a cardinality is a whole number of rows")
    elif number < 0:
        problems.append("a cardinality is not negative")
    elif not math.isfinite(number):
        # 1e400 parses as a valid JSON number and becomes float infinity,
        # which json.dumps then writes as non-standard `Infinity` into the
        # response AND the persisted record (round-43). A measurement is
        # finite by definition.
        problems.append("number must be finite")
    unit = value.get("unit")
    if not isinstance(unit, str) or not _AGGREGATE_UNIT.match(unit):
        problems.append("unit must be a short lowercase unit token")
    elif isinstance(allowed_units, (list, tuple)) and unit not in allowed_units:
        # Declared vocabulary, enforced from the persisted policy snapshot
        # (round-44): "is this one of the declared units" is decidable, so an
        # identity attribute cannot be worn as a unit.
        problems.append("unit is not one of the units the data policy declares")
    dimension = value.get("dimension")
    scope_grammar = _RETAINED_SCOPE_KEY if retained else _AGGREGATE_DIMENSION
    if dimension is not None and (
        not isinstance(dimension, str) or not scope_grammar.match(dimension)
    ):
        problems.append("dimension must be a single bounded category scope (key=value)")
    elif isinstance(dimension, str) and (
        problem := _identity_scope_problem(dimension, "dimension")
    ):
        problems.append(problem)
    unknown = set(value) - _aggregate_fields()
    if unknown:
        problems.append("carries fields outside the aggregate shape")
    return problems


def _read_request_shape_problems(
    request: Any, *, executed: bool = False, retained: bool = False
) -> list[str]:
    """Whether a read request is the typed structure the contract declares.

    The lane names WHAT to measure; the parent session builds and runs the
    query after user confirmation. Because every field is a bounded grammar
    and ``operation`` is a const, a mutating statement or a prose instruction
    cannot be expressed — which is why this replaced the statement-head,
    aggregate-projection, side-effecting-function, and imperative-mood
    classifiers that preceded it.
    """
    if not isinstance(request, Mapping):
        return [
            "must be a typed read request "
            "({operation: read, metric, aggregation[, filters, grouping]}), "
            "not a query string"
        ]
    problems: list[str] = []
    if request.get("operation") != "read":
        problems.append("operation must be 'read'")
    metric = request.get("metric")
    if not isinstance(metric, str) or not _READ_REQUEST_METRIC.match(metric):
        problems.append("metric must be a bounded identifier")
    elif _identifier_looks_secret(metric):
        # A metric names what is measured; a credential-shaped name is the
        # leak wearing that field (round-45 probe: password_swordfish).
        problems.append("metric is credential-shaped; name the measurement instead")
    elif _entity_key(metric) and request.get("aggregation") not in _cardinality_aggregations():
        # Counting identities yields a cardinality; taking their min/max/median
        # yields an identity (round-48 probe: max(ssn)). The typed form makes
        # that decidable per field instead of per SQL projection.
        problems.append(
            "an identity metric may only be counted; a value-returning "
            "aggregation over it reports the identity itself"
        )
    if request.get("aggregation") not in _aggregation_kinds():
        problems.append("aggregation is not one of the declared kinds")
    elif request.get("aggregation") == "percentile" and not isinstance(
        request.get("percentile"), int
    ):
        problems.append("a percentile request must state which percentile")
    elif request.get("aggregation") != "percentile" and "percentile" in request:
        problems.append("percentile applies only to a percentile aggregation")
    for list_field, pattern, limit in (
        ("filters", _RETAINED_SCOPE_KEY if retained else _READ_REQUEST_FILTER, 5),
        ("grouping", _READ_REQUEST_GROUPING, 3),
    ):
        items = request.get(list_field)
        if items is None:
            continue
        if not isinstance(items, list) or len(items) > limit:
            problems.append(f"{list_field} must be a bounded list")
            continue
        if any(not isinstance(item, str) or not pattern.match(item) for item in items):
            problems.append(f"{list_field} entries must match the declared grammar")
            continue
        problems.extend(
            problem
            for item in items
            if (problem := _identity_scope_problem(item, list_field)) is not None
        )
    unknown = set(request) - _read_request_fields()
    if unknown:
        problems.append("carries fields outside the read-request shape")
    if executed and request.get("aggregation") not in _cardinality_aggregations():
        problems.append(
            "executed evidence may only report a cardinality "
            "(count/distinct_count); every other aggregation can reproduce a "
            "value from the rows it read"
        )
    if executed and request.get("grouping"):
        # A grouped query returns one row per group; an evidence item carries
        # ONE number (round-46). The two cannot describe each other — a single
        # value cannot say which group it came from, and a second grouping key
        # is not expressible at all. Per-category evidence is one item per
        # category, each narrowed by a filter.
        problems.append(
            "executed evidence reports a single number, so its request may not "
            "group; narrow with filters and report one item per category"
        )
    return problems


# The typed vocabularies the contract declares. Kept beside the re-entry
# checks so the schema and the enforcement point cannot disagree about what a
# typed aggregate or read request IS.
@lru_cache(maxsize=1)
def _schema_defs() -> Mapping[str, Any]:
    return _data_context_answer_contract()["response_model_schema"]["$defs"]


@lru_cache(maxsize=1)
def _aggregation_kinds() -> frozenset[str]:
    """The aggregation vocabulary, read from the SCHEMA (round-49).

    Maintaining it twice is how the validator came to reject a percentile the
    schema advertised while still accepting kinds the schema had dropped.
    """
    return frozenset(_schema_defs()["aggregation_kind"]["enum"])


@lru_cache(maxsize=1)
def _read_request_fields() -> frozenset[str]:
    """Fields the schema's read_request declares — the validator's allowlist."""
    return frozenset(_schema_defs()["read_request"]["properties"])


@lru_cache(maxsize=1)
def _aggregate_fields() -> frozenset[str]:
    return frozenset(_schema_defs()["aggregate"]["properties"])


def _data_evidence_fallback_schema() -> dict[str, Any]:
    """Structural schema kept when the DECLARED data contract is unenforceable.

    This form is never delivered to a child, so it carries no prompt budget
    and there is no reason for it to be weaker than the real contract's
    structure (round-44: a permissive fallback accepted
    ``{requires_user_confirmation: true, raw_rows: [...]}``). It reuses the
    published ``$defs`` and closes the object, so a degraded contract loses
    the field-level descriptions and nothing else.
    """
    from ouroboros.orchestrator.capabilities.interview_schemas import (
        data_evidence_structural_schema,
    )

    return data_evidence_structural_schema()


_AGGREGATE_UNIT = re.compile(r"^[a-z%][a-z_/%]{0,23}$")


_AGGREGATE_DIMENSION = re.compile(r"^[a-z][a-z0-9_]{0,23}=[A-Za-z0-9_.-]{1,22}$")


_READ_REQUEST_METRIC = re.compile(r"^[A-Za-z][A-Za-z0-9_.*-]{0,63}$")


_READ_REQUEST_FILTER = re.compile(r"^[a-z][a-z0-9_]{0,23}[=<>!]{1,2}[A-Za-z0-9_.:+-]{1,22}$")


_READ_REQUEST_GROUPING = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
#: The RETAINED form keeps a scope's key and drops its value, so its grammar
#: is the bare key. Re-checking a carried-forward result against the
#: submission grammar is the same category error as validating it against the
#: submission schema (round-50).
_RETAINED_SCOPE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


# An ENTITY key identifies a person or account; grouping or scoping by one
# produces per-identity results, which the aggregate-only policy forbids
# (round-43 probes: grouping ["user_id"], dimension "user_id=847291").
# Mechanically decidable from the key alone: the `*_id`/`*_uuid` suffix is the
# universal entity-key convention, plus the identity columns the policy names.
_ENTITY_KEY_SUFFIX = re.compile(r"(?:^|_)(id|ids|uuid|guid|key|keys)$")


_IDENTITY_KEYS = frozenset(
    {
        "email",
        "emails",
        "phone",
        "address",
        "ssn",
        "name",
        "first_name",
        "last_name",
        "full_name",
        "username",
        "user",
        "customer",
        "account",
        "member",
    }
)


# A category VALUE is a label ("growth", "kr", "2026-01"); an opaque entity
# identifier is a bare number or a UUID/hash. Scoping an aggregate to one of
# those makes it a single-entity row however the key is spelled.
_OPAQUE_ENTITY_VALUE = re.compile(
    r"^(?:\d{3,}|[0-9a-fA-F]{8,}|[0-9a-fA-F-]{16,})$",
)


def _entity_key(key: str) -> bool:
    """Whether a grouping/filter/dimension key identifies an entity.

    Matched per TOKEN (round-45 probe: ``email_address``, whose parts are both
    identity words while the whole string is neither) — the same
    position-independent rule the credential classifier uses, so a compound
    name cannot hide the word that matters.
    """
    lowered = key.lower()
    if lowered in _IDENTITY_KEYS or _ENTITY_KEY_SUFFIX.search(lowered):
        return True
    return any(token in _IDENTITY_KEYS for token in re.split(r"[-_.]", lowered) if token)


# The public filter grammar accepts <, >, ! and = in one- or two-character
# combinations, so identity-scope parsing splits on the OPERATOR rather than
# on "=" alone (round-46 warning: tenant_id=847291 was rejected while
# tenant_id>847291 walked through the same check).
_FILTER_OPERATOR = re.compile(r"[=<>!]{1,2}")


#: Aggregations that reduce their input to a cardinality. Every other kind
#: returns one of the input VALUES, which is why an identity metric may only
#: be counted.
@lru_cache(maxsize=1)
def _cardinality_aggregations() -> frozenset[str]:
    """Aggregations whose result is provably not one of the inputs.

    Read from the schema (round-49 rule: one vocabulary). A cardinality counts
    rows; every other kind can reproduce a value from them — verbatim for
    min/max/median/percentile, and for sum/avg whenever the cohort is a single
    row (round-52).
    """
    return frozenset(_schema_defs()["cardinality_aggregation"]["enum"])


def _identity_scope_problem(text: str, label: str) -> str | None:
    """Whether ``key<op>value`` scopes an aggregate to an identified entity."""
    match = _FILTER_OPERATOR.search(text)
    if match:
        key, value = text[: match.start()], text[match.end() :]
    else:
        key, value = text, ""
    key, value = key.strip(), value.strip()
    if _entity_key(key):
        return f"{label} keys an entity ({key!r}); aggregate by category instead"
    # Round-48: grouping=["password"], filters=["access_token!=huntertwo"] —
    # the parsed key and the compared value are identifiers too, so they get
    # the same credential classification every other identifier field has.
    if _identifier_looks_secret(key) or (value and _identifier_looks_secret(value)):
        return f"{label} names or compares a credential; it may not ride a data request"
    if value and _OPAQUE_ENTITY_VALUE.match(value):
        return f"{label} scopes to an opaque entity identifier; aggregate by category instead"
    return None


# A safe identifier is one token of word/dot/dash characters — no spaces,
# no '=', no ':' — so a credential can never wear the exemption.
_SAFE_IDENTIFIER_SYNTAX = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


@lru_cache(maxsize=1)
def _vendor_secret_prefix() -> re.Pattern[str]:
    """Vendor token prefixes, from the SAME published vocabulary the content
    scan uses (round-42): two copies drifted, and the copy that knew the
    standard Slack forms was not the one guarding persisted content."""
    from ouroboros.orchestrator.capabilities.interview_schemas import (
        DATA_EVIDENCE_VENDOR_TOKEN_PREFIX,
    )

    # Recognized at the START or after any separator (round-43 probes:
    # warehouse_xoxb-…, tool_ghp_…, reader_AKIA…): a vendor token does not
    # stop being one because a word was glued in front of it.
    return re.compile(
        r"(?:^|[-_.])(?:" + DATA_EVIDENCE_VENDOR_TOKEN_PREFIX + r"|AKIA|ASIA|ABIA|ACCA)"
    )


# Credential words that NEVER name a read-only data tool. No suffix
# analysis applies to them: an identifier containing one is a credential
# wherever it sits (round-40 probes: password_swordfish,
# client_secret_huntertwo — an alphabetic suffix is not tool vocabulary when
# the word itself only ever names a secret).
_ABSOLUTE_CREDENTIAL_WORDS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "secrets",
        "bearer",
        "credential",
        "credentials",
        "creds",
        "apikey",
        "apikeys",
    }
)


# Credential words that DO appear in analytics vocabulary — "token usage",
# "key metrics". English compounds put the head last, so a qualifier BEFORE
# the word makes the word the head and the identifier a credential name
# (refresh_token, access_key, private_key); the word LEADING makes it a
# modifier and the identifier a tool name (token_usage_v2, key_metrics_30d).
# That position rule is the whole exemption: it is granted only to a leading
# credential word with a word-like tail, and everything else fails closed.
_QUALIFIABLE_CREDENTIAL_WORDS = frozenset({"token", "tokens", "key", "keys"})


_CREDENTIAL_WORDS = _ABSOLUTE_CREDENTIAL_WORDS | _QUALIFIABLE_CREDENTIAL_WORDS


def _identifier_looks_secret(value: str) -> bool:
    if _vendor_secret_prefix().search(value):
        return True
    tokens = [tok for tok in re.split(r"[-_.]", value) if tok]
    lowered = [tok.lower() for tok in tokens]
    if any(tok in _ABSOLUTE_CREDENTIAL_WORDS for tok in lowered):
        return True
    credential_index = next(
        (index for index, tok in enumerate(lowered) if tok in _QUALIFIABLE_CREDENTIAL_WORDS),
        None,
    )
    if credential_index is None:
        return False
    # A qualifier before the credential word makes it the compound's HEAD —
    # refresh_token_*, access_key_*, private_key_* are credential names, not
    # tool names (round-38: access_key_abcd1234; round-40:
    # refresh_token_alphabetic, whose word-like tail carried the exemption).
    if credential_index > 0:
        return True
    # A LEADING credential word is a modifier, and the identifier stays
    # exempt only while every remaining token is recognizably word-like —
    # the fail-closed rule from round-36 (api_key_staging_XYZ12345, whose
    # non-hex opaque tail evaded the gibberish list; rounds 32-33: 123abc,
    # live/supersecret markers). Word-like means a plain alphabetic
    # non-marker word, a version tag (v2, v12), or a short window tag
    # (7d, 30d) — so token_usage_v2 and key_metrics_30d keep naming
    # read-only tools while any opaque tail fails closed.
    suffix_tokens = tokens[credential_index + 1 :]

    # Secret-marker words mark alphabetic credentials too (round-33:
    # api_key_live_supersecret) — live/test-key conventions and the word
    # secret itself never name read-only data tools.
    def _secret_marker(tok: str) -> bool:
        lowered = tok.lower()
        return lowered in ("live", "prod", "priv", "private") or "secret" in lowered

    def _word_like(tok: str) -> bool:
        # Real identifier words are short; a 13+-char alphabetic run after a
        # credential prefix is an opaque value, not vocabulary (round-37
        # probe: api_key_abcdefghijklmnop). "clickhouse", "warehouse",
        # "aggregation" all fit; nothing named like a data tool does not.
        if re.fullmatch(r"[A-Za-z]{1,12}", tok):
            return not _secret_marker(tok)
        return bool(re.fullmatch(r"v?\d{1,3}|\d{1,3}[dhwmy]", tok, re.IGNORECASE))

    return bool(suffix_tokens) and not all(_word_like(tok) for tok in suffix_tokens)
