"""End-to-end regressions for extracted assertions promoted to formal verdicts."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ouroboros.core.lineage import EvaluationSummary, TaskResult
from ouroboros.core.types import Result
from ouroboros.mcp.server.adapter import _evaluation_summary_from_spec_verification
from ouroboros.providers.base import CompletionResponse
from ouroboros.verification.binding import _EFFECT_EQUIVALENCE_CLASSES
from ouroboros.verification.extractor import AssertionExtractor
from ouroboros.verification.models import EvidencePolarity, SpecAssertion
from ouroboros.verification.verifier import SpecVerifier


async def _extract(ac_text: str, payload: list[dict[str, Any]]) -> tuple[SpecAssertion, ...]:
    adapter = AsyncMock()
    adapter.complete = AsyncMock(
        return_value=Result.ok(
            CompletionResponse(
                content=json.dumps(payload),
                model="test",
                usage={"input": 0, "output": 0},
            )
        )
    )
    result = await AssertionExtractor(llm_adapter=adapter).extract("seed", (ac_text,))
    assert result.is_ok
    return result.value


def _formal_verdict(
    ac_text: str,
    verifier: Any,
    *,
    agent_reported_pass: bool = True,
) -> Any:
    mechanical = EvaluationSummary(
        final_approved=agent_reported_pass,
        highest_stage_passed=2,
        task_results=(
            TaskResult(
                task_index=0,
                task_content=ac_text,
                status="completed" if agent_reported_pass else "failed",
                completed=agent_reported_pass,
                source_ac_index=0,
                execution_method="legacy_parallel_report",
            ),
        ),
        execution_completion_status="completed",
        approval_status="approved" if agent_reported_pass else "not_evaluated",
    )
    result = _evaluation_summary_from_spec_verification(mechanical, verifier)
    assert result is not None
    return result


@pytest.mark.asyncio
async def test_positive_structure_survives_extractor_verifier_and_formal_adapter(
    tmp_path: Any,
) -> None:
    ac_text = "MUST define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+\w+",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "CameraProvider exists",
            }
        ],
    )
    (tmp_path / "main.py").write_text("class CameraProvider:\n    pass\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.REQUIRED
    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "approved"),
    [
        ("class CameraProvider:\n    pass\n", False),
        ("class Unrelated:\n    pass\n", True),
    ],
    ids=["forbidden-present", "forbidden-absent"],
)
async def test_negative_structure_polarity_reaches_formal_verdict(
    tmp_path: Any,
    content: str,
    approved: bool,
) -> None:
    ac_text = "MUST NOT define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+\w+",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "CameraProvider is forbidden",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(
        assertions,
        agent_results={0: False},
    )
    formal = _formal_verdict(ac_text, verification, agent_reported_pass=False)

    assert assertions[0].evidence_polarity is EvidencePolarity.FORBIDDEN
    assert verification.reports[0].verified_pass is approved
    assert formal.final_approved is approved
    assert formal.ac_results[0].final_verdict == ("pass" if approved else "fail")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "tier", "pattern", "expected", "hidden_path", "hidden_content"),
    [
        (
            "MUST NOT define a CameraProvider class",
            "t2_structural",
            r"interface\s+\w+",
            "CameraProvider",
            "hidden.py",
            "class CameraProvider:\n    pass\n",
        ),
        (
            "MUST NOT set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*5",
            "10",
            "hidden.py",
            "RETRIES = 10\n",
        ),
        (
            "MUST NOT define a CameraProvider class",
            "t2_structural",
            r"interface\s+\w+",
            "CameraProvider",
            ".hidden.py",
            "class CameraProvider:\n    pass\n",
        ),
        (
            "MUST NOT set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*5",
            "10",
            ".config/settings.py",
            "RETRIES: int = 10\n",
        ),
        (
            "MUST NOT set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*5",
            "10",
            ".github/settings.py",
            "RETRIES = 10\n",
        ),
    ],
    ids=[
        "hidden-structure",
        "hidden-constant",
        "dotfile-structure",
        "typed-dotdir-constant",
        "dot-github-is-not-git",
    ],
)
async def test_forbidden_scan_ignores_model_predicate_and_scope(
    tmp_path: Any,
    ac_text: str,
    tier: str,
    pattern: str,
    expected: str,
    hidden_path: str,
    hidden_content: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": tier,
                "pattern": pattern,
                "expected_value": expected,
                "file_hint": "safe.py",
                "description": "Model-selected narrow negative scan",
            }
        ],
    )
    (tmp_path / "safe.py").write_text("# safe\n")
    hidden_file = tmp_path / hidden_path
    hidden_file.parent.mkdir(parents=True, exist_ok=True)
    hidden_file.write_text(hidden_content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.FORBIDDEN
    assert verification.reports[0].verified_pass is False
    assert verification.reports[0].results[0].evidence_source == "trusted_project_scan"
    assert formal.final_approved is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix",
    [
        "to prevent outages",
        "to omit decorator boilerplate",
        "because errors must not remain",
        "so failures must never remain",
        "for failures that must not remain",
        "for transient failures that must not remain",
        "to ensure errors must not remain",
        "to ensure runtime errors must not remain",
        "in order to ensure errors must not remain",
    ],
    ids=[
        "prevent-purpose",
        "omit-purpose",
        "because-reason",
        "so-reason",
        "for-reason",
        "for-modified-effect-reason",
        "to-purpose-subject",
        "to-modified-effect-subject",
        "in-order-purpose",
    ],
)
@pytest.mark.parametrize(
    ("content", "approved"),
    [("class CameraProvider:\n    pass\n", True), ("class Unrelated:\n    pass\n", False)],
    ids=["required-present", "required-absent"],
)
async def test_negative_purpose_words_after_target_do_not_flip_positive_polarity(
    tmp_path: Any,
    suffix: str,
    content: str,
    approved: bool,
) -> None:
    ac_text = f"MUST define a CameraProvider class {suffix}"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+\w+",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "CameraProvider is required",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.REQUIRED
    assert formal.final_approved is approved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ac_text",
    [
        "The CameraProvider class must not exist",
        "The CameraProvider implementation class must not exist",
        "The CameraProvider service class must never be present",
    ],
    ids=["direct-class", "implementation-class", "service-class"],
)
@pytest.mark.parametrize(
    ("content", "approved"),
    [("class CameraProvider:\n    pass\n", False), ("class Unrelated:\n    pass\n", True)],
    ids=["forbidden-present", "forbidden-absent"],
)
async def test_postfix_modal_negation_controls_the_target_clause(
    tmp_path: Any,
    ac_text: str,
    content: str,
    approved: bool,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+\w+",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "CameraProvider must not exist",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.FORBIDDEN
    assert formal.final_approved is approved


_UNKNOWN_SUFFIX_NEGATIONS = (
    "The CameraProvider mysterious widget must not exist",
    "The CameraProvider class for legacy clients must not exist",
    "The CameraProvider class to be removed must not exist",
    "The CameraProvider class to ensure this class must not remain",
    "The CameraProvider class for this class that must not remain",
    "The CameraProvider class to ensure it must not remain",
    "The CameraProvider class to ensure the same class must not remain",
    "The CameraProvider class for the target class that must not remain",
    "The CameraProvider class for the CameraProvider class that must not remain",
    "The CameraProvider class to ensure its class must not remain",
    "The CameraProvider class to ensure the provider must not remain",
    "The CameraProvider class for the provider that must not remain",
    "The CameraProvider class to ensure said class must not remain",
    "The CameraProvider class for the referenced component that must not remain",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ac_text",
    _UNKNOWN_SUFFIX_NEGATIONS,
    ids=[
        "unknown-widget",
        "for-target-qualifier",
        "to-target-qualifier",
        "to-purpose-target-referent",
        "for-purpose-target-referent",
        "to-purpose-target-pronoun",
        "to-purpose-same-target",
        "for-purpose-explicit-target",
        "for-purpose-named-target",
        "to-purpose-possessive-target",
        "to-purpose-provider-referent",
        "for-purpose-provider-referent",
        "to-purpose-said-class-referent",
        "for-purpose-referenced-component",
    ],
)
async def test_unknown_noncausal_postfix_modifier_fails_closed(ac_text: str) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Unknown target-relative negation grammar",
            }
        ],
    )

    assert assertions == ()


@pytest.mark.parametrize(
    "ac_text",
    _UNKNOWN_SUFFIX_NEGATIONS,
    ids=[
        "unknown-widget",
        "for-target-qualifier",
        "to-target-qualifier",
        "to-purpose-target-referent",
        "for-purpose-target-referent",
        "to-purpose-target-pronoun",
        "to-purpose-same-target",
        "for-purpose-explicit-target",
        "for-purpose-named-target",
        "to-purpose-possessive-target",
        "to-purpose-provider-referent",
        "for-purpose-provider-referent",
        "to-purpose-said-class-referent",
        "for-purpose-referenced-component",
    ],
)
@pytest.mark.parametrize(
    "content",
    ["class CameraProvider:\n    pass\n", "class Unrelated:\n    pass\n"],
    ids=["target-present", "target-absent"],
)
def test_unknown_target_suffix_never_publishes_a_direct_formal_pass(
    tmp_path: Any,
    ac_text: str,
    content: str,
) -> None:
    assertion = SpecAssertion(
        ac_index=0,
        ac_text=ac_text,
        tier="t2_structural",
        pattern=r"CameraProvider",
        expected_value="CameraProvider",
        file_hint="*.py",
        evidence_targets=("CameraProvider",),
        evidence_polarity=EvidencePolarity.REQUIRED,
        input_binding_required=True,
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all((assertion,))
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].results[0].verified is False
    assert "polarity is ambiguous or stale" in verification.reports[0].results[0].detail
    assert formal.final_approved is False


_EFFECT_CLASS_CASES = tuple(
    pytest.param(effect_class, id="-".join(sorted(effect_class)))
    for effect_class in _EFFECT_EQUIVALENCE_CLASSES
)


def _effect_target_shapes(alias: str) -> tuple[str, ...]:
    titled = alias.title()
    return (
        alias.casefold(),
        f"{titled}Mode",
        f"Runtime{titled}",
        f"{alias.casefold()}_mode",
        f"runtime-{alias.casefold()}",
        f"{alias.casefold()}mode",
        f"runtime{alias.casefold()}",
    )


_EFFECT_SUBJECT_CASES = tuple(
    pytest.param(subject, id=subject)
    for effect_class in _EFFECT_EQUIVALENCE_CLASSES
    for subject in sorted(effect_class)
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subject",
    _EFFECT_SUBJECT_CASES,
)
@pytest.mark.parametrize(
    ("content", "approved"),
    [("class CameraProvider:\n    pass\n", True), ("class Unrelated:\n    pass\n", False)],
    ids=["target-present", "target-absent"],
)
async def test_disjoint_causal_effect_subject_remains_required(
    tmp_path: Any,
    subject: str,
    content: str,
    approved: bool,
) -> None:
    ac_text = f"The class CameraProvider to ensure {subject} must not remain"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Disjoint target prevents a causal effect",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.REQUIRED
    assert formal.final_approved is approved


@pytest.mark.parametrize(
    "effect_class",
    _EFFECT_CLASS_CASES,
)
@pytest.mark.parametrize("target_present", [True, False], ids=["present", "absent"])
def test_stale_required_effect_target_cross_product_never_publishes_formal_pass(
    tmp_path: Any,
    effect_class: frozenset[str],
    target_present: bool,
) -> None:
    for subject in sorted(effect_class):
        for alias in sorted(effect_class):
            for target in _effect_target_shapes(alias):
                ac_text = f"The class {target} to ensure {subject} must not remain"
                assertion = SpecAssertion(
                    ac_index=0,
                    ac_text=ac_text,
                    tier="t2_structural",
                    pattern=rf"class\s+{target}",
                    expected_value=target,
                    file_hint="*.py",
                    evidence_targets=(target,),
                    evidence_polarity=EvidencePolarity.REQUIRED,
                    input_binding_required=True,
                )
                content = (
                    f"class {target}:\n    pass\n"
                    if target_present
                    else "class Unrelated:\n    pass\n"
                )
                (tmp_path / "main.py").write_text(content)

                verification = SpecVerifier(str(tmp_path)).verify_all((assertion,))
                formal = _formal_verdict(ac_text, verification)
                result = verification.reports[0].results[0]

                assert result.verified is False, (subject, target, target_present)
                assert any(
                    message in result.detail
                    for message in (
                        "polarity is ambiguous or stale",
                        "No criterion-bound target is available",
                    )
                ), (subject, target, target_present, result.detail)
                assert formal.final_approved is False, (subject, target, target_present)


_UNKNOWN_PREFIX_NEGATIONS = (
    "MUST avoid ever defining a CameraProvider class",
    "MUST prevent accidental creation of a CameraProvider class",
    "MUST omit any use of a CameraProvider class",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ac_text",
    _UNKNOWN_PREFIX_NEGATIONS,
    ids=["avoid-ever", "prevent-accidental", "omit-any-use"],
)
async def test_unknown_target_prefix_negation_fails_closed_before_verifier(
    ac_text: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Unknown target-prefix negation grammar",
            }
        ],
    )

    assert assertions == ()


@pytest.mark.parametrize(
    "ac_text",
    _UNKNOWN_PREFIX_NEGATIONS,
    ids=["avoid-ever", "prevent-accidental", "omit-any-use"],
)
@pytest.mark.parametrize(
    "content",
    ["class CameraProvider:\n    pass\n", "class Unrelated:\n    pass\n"],
    ids=["target-present", "target-absent"],
)
def test_unknown_target_prefix_never_publishes_a_direct_formal_pass(
    tmp_path: Any,
    ac_text: str,
    content: str,
) -> None:
    assertion = SpecAssertion(
        ac_index=0,
        ac_text=ac_text,
        tier="t2_structural",
        pattern=r"CameraProvider",
        expected_value="CameraProvider",
        file_hint="*.py",
        evidence_targets=("CameraProvider",),
        evidence_polarity=EvidencePolarity.REQUIRED,
        input_binding_required=True,
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all((assertion,))
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].results[0].verified is False
    assert "polarity is ambiguous or stale" in verification.reports[0].results[0].detail
    assert formal.final_approved is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ac_text",
    [
        "MUST avoid defining a CameraProvider class",
        "MUST prevent defining a CameraProvider class",
        "MUST omit a CameraProvider class",
    ],
    ids=["avoid", "prevent", "omit"],
)
async def test_avoid_wording_is_forbidden_instead_of_defaulting_to_required(
    tmp_path: Any,
    ac_text: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+\w+",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Avoid CameraProvider",
            }
        ],
    )
    (tmp_path / "main.py").write_text("class CameraProvider:\n    pass\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.FORBIDDEN
    assert formal.final_approved is False


@pytest.mark.asyncio
async def test_forbidden_absence_fails_closed_when_project_inventory_is_truncated(
    tmp_path: Any,
) -> None:
    ac_text = "MUST NOT define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "CameraProvider is forbidden",
            }
        ],
    )
    for index in range(101):
        (tmp_path / f"safe_{index}.py").write_text("# safe\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert result.verified is False
    assert "inventory exceeds 100 files" in result.detail
    assert formal.final_approved is False


@pytest.mark.asyncio
async def test_repeated_constant_values_keep_clause_identity_through_formal_adapter(
    tmp_path: Any,
) -> None:
    ac_text = "MUST set WARMUP=10 and RETRIES=10"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"WARMUP\s*=\s*",
                "expected_value": "10",
                "file_hint": "*.py",
                "description": "Warmup value",
            },
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"RETRIES\s*=\s*",
                "expected_value": "10",
                "file_hint": "*.py",
                "description": "Retry value",
            },
        ],
    )
    (tmp_path / "config.py").write_text("WARMUP = 10\nRETRIES = 10\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert [assertion.evidence_targets for assertion in assertions] == [
        ("WARMUP",),
        ("RETRIES",),
    ]
    assert [result.evidence_target for result in verification.reports[0].results] == [
        "WARMUP",
        "RETRIES",
    ]
    assert formal.final_approved is True


@pytest.mark.asyncio
async def test_forbidden_constant_scan_does_not_stop_at_an_allowed_earlier_value(
    tmp_path: Any,
) -> None:
    ac_text = "MUST NOT set RETRIES=10"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"RETRIES\s*=\s*",
                "expected_value": "10",
                "file_hint": "*.py",
                "description": "Retries must not equal ten",
            }
        ],
    )
    (tmp_path / "config.py").write_text("RETRIES = 5\nRETRIES = 10\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.FORBIDDEN
    assert verification.reports[0].verified_pass is False
    assert "Forbidden value '10'" in verification.reports[0].results[0].detail
    assert formal.final_approved is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [r".", r".+", r"\b", r"(?=m)"],
    ids=["dot", "consume-all", "word-boundary", "single-letter-lookahead"],
)
async def test_irrelevant_filename_regex_cannot_reach_formal_pass(
    tmp_path: Any,
    pattern: str,
) -> None:
    ac_text = "MUST create file CameraProvider.py"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider.py",
                "file_hint": "*.py",
                "description": "Find the requested file",
            }
        ],
    )
    (tmp_path / "CameraProvider.py").write_text("# unrelated contents\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
async def test_target_specific_filename_regex_still_reaches_formal_pass(tmp_path: Any) -> None:
    ac_text = "MUST create file CameraProvider.py"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"CameraProvider\.py",
                "expected_value": "CameraProvider.py",
                "file_hint": "*.py",
                "description": "Find CameraProvider.py",
            }
        ],
    )
    (tmp_path / "CameraProvider.py").write_text("# provider\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].results[0].evidence_source == "filename"
    assert formal.final_approved is True


@pytest.mark.asyncio
async def test_palindrome_filename_keeps_exact_filename_evidence(tmp_path: Any) -> None:
    ac_text = "MUST create file aba.py"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"aba\.py",
                "expected_value": "aba.py",
                "file_hint": "*.py",
                "description": "Find aba.py",
            }
        ],
    )
    (tmp_path / "aba.py").write_text("# palindrome\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].results[0].evidence_source == "filename"
    assert formal.final_approved is True
