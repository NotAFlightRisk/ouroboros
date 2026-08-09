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
