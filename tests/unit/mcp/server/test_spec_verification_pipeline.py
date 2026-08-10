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
from ouroboros.verification.models import EvidencePolarity, SpecAssertion, VerificationTier
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
                "pattern": r"class\s+CameraProvider",
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
    ("ac_text", "tier", "pattern", "expected", "filename", "content"),
    [
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.lua",
            "-- class CameraProvider\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "settings.sql",
            "-- RETRIES = 10\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.py",
            'message = f"class CameraProvider"\n',
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.py",
            'message = rf"class CameraProvider"\n',
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.py",
            'message = f"{prefix} class CameraProvider"\n',
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.py",
            'message = f"""\nclass CameraProvider\n"""\n',
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.rs",
            "/* outer /* inner */ class CameraProvider */\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.hs",
            "{- outer {- inner -} class CameraProvider -}\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.lua",
            "--[=[\nclass CameraProvider\n]=]\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "README.md",
            "class CameraProvider should be implemented later\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "README.txt",
            "class CameraProvider should be implemented later\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "manifest.json",
            '{"todo": "class CameraProvider"}\n',
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "config.yaml",
            "todo: class CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.js",
            "const declaration = /class CameraProvider/;\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.js",
            "const makePattern = () => /class CameraProvider/;\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "main.cpp",
            'const char* decoy = R"TAG(foo " class CameraProvider)TAG";\n',
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "Main.java",
            'String decoy = """\nfoo " class CameraProvider\n""";\n',
        ),
    ],
    ids=[
        "comment-only-structure",
        "comment-only-constant",
        "python-fstring",
        "python-raw-fstring",
        "python-interpolated-fstring",
        "python-multiline-fstring",
        "rust-nested-comment",
        "haskell-nested-comment",
        "lua-extended-comment",
        "markdown-prose",
        "plain-text-prose",
        "json-string",
        "yaml-scalar",
        "javascript-regex-literal",
        "javascript-arrow-regex-literal",
        "cpp-raw-string",
        "java-text-block",
    ],
)
async def test_comment_only_evidence_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    tier: str,
    pattern: str,
    expected: str,
    filename: str,
    content: str,
) -> None:
    """Extractor-bound assertions reject comments before formal promotion."""
    suffix = filename[filename.rfind(".") :]
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": tier,
                "pattern": pattern,
                "expected_value": expected,
                "file_hint": f"*{suffix}",
                "description": "Comment text is not executable evidence",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].input_binding_required is True
    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "tier", "pattern", "expected", "filename", "content"),
    [
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "build.sh",
            "cat <<'EOF'\nclass CameraProvider:\n    pass\nEOF\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"class\s+CameraProvider",
            "CameraProvider",
            "build.sh",
            "cat <<'123'\nclass CameraProvider:\n    pass\n123\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "config.yaml",
            "notes: |\n  RETRIES=10\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "config.yaml",
            "- |\n  RETRIES=10\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "config.ini",
            "notes = decoy\n  RETRIES=10\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "config.toml",
            'notes = """\nRETRIES=10\n"""\n',
        ),
    ],
    ids=[
        "shell-heredoc",
        "numeric-shell-heredoc",
        "yaml-block-scalar",
        "yaml-sequence-block-scalar",
        "ini-continuation",
        "toml-multiline",
    ],
)
async def test_container_body_evidence_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    tier: str,
    pattern: str,
    expected: str,
    filename: str,
    content: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": tier,
                "pattern": pattern,
                "expected_value": expected,
                "file_hint": filename,
                "description": "Container body is not executable evidence",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("Main.swift", "let pattern = /class CameraProvider/\n"),
        ("Main.swift", "let pattern = #/class CameraProvider/#\n"),
        ("main.pl", "$pattern = qr/class CameraProvider/;\n"),
        ("main.pl", "$pattern = qr{class CameraProvider};\n"),
        ("Main.hs", "[r| class CameraProvider |]\n"),
        ("Main.hs", "[| class CameraProvider |]\n"),
        ("main.rb", "pattern = /class CameraProvider/\n"),
        ("main.rb", "pattern = %r{class CameraProvider}\n"),
        ("Main.cs", 'var pattern = @$"class CameraProvider";\n'),
        ("main.pl", "=pod\nclass CameraProvider\n=cut\n"),
        ("main.pl", "__DATA__\nclass CameraProvider\n"),
        ("main.rb", "=begin\nclass CameraProvider\n=end\n"),
        ("main.rb", "__END__\nclass CameraProvider\n"),
        ("main.sql", "SELECT $tag$class CameraProvider$tag$;\n"),
        ("main.r", 'pattern <- r"(foo " class CameraProvider)"\n'),
        ("main.r", 'pattern <- r"---[foo " class CameraProvider]---"\n'),
        ("main.jsx", "const view = <div>class CameraProvider</div>;\n"),
        ("main.tsx", "const view = <>class CameraProvider</>;\n"),
        ("main.pl", "format STDOUT =\nclass CameraProvider\n.\n"),
    ],
    ids=[
        "swift-bare-regex",
        "swift-extended-regex",
        "perl-qr-slash",
        "perl-qr-brace",
        "haskell-quasiquote",
        "haskell-template-quote",
        "ruby-slash-regex",
        "ruby-percent-regex",
        "csharp-interpolated-verbatim",
        "perl-pod",
        "perl-data",
        "ruby-block-comment",
        "ruby-data",
        "sql-dollar-quote",
        "r-raw-string",
        "r-delimited-raw-string",
        "jsx-text",
        "tsx-fragment-text",
        "perl-format",
    ],
)
async def test_unclassified_language_literals_cannot_reach_formal_pass(
    tmp_path: Any, filename: str, content: str
) -> None:
    ac_text = "MUST define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Literal body is not executable evidence",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


def test_mixed_criterion_text_cannot_borrow_a_report_identity_for_formal_pass(
    tmp_path: Any,
) -> None:
    ac_text = "MUST define a CameraProvider class"
    trusted = SpecAssertion(
        ac_index=0,
        ac_text=ac_text,
        tier=VerificationTier.T3_BEHAVIORAL,
    )
    conflicting = SpecAssertion(
        ac_index=0,
        ac_text="MUST define class Unrelated",
        tier=VerificationTier.T2_STRUCTURAL,
        pattern=r"class\s+Unrelated",
        expected_value="Unrelated",
        file_hint="main.py",
        evidence_targets=("Unrelated",),
    )
    (tmp_path / "main.py").write_text("class Unrelated:\n    pass\n")

    verification = SpecVerifier(str(tmp_path)).verify_all((trusted, conflicting))
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert "Conflicting criterion text" in verification.reports[0].results[0].detail
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


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
                "pattern": r"class\s+CameraProvider",
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
    "content",
    [
        "# class CameraProvider:\nclass Unrelated:\n    pass\n",
        'message = "class CameraProvider"\nclass Unrelated:\n    pass\n',
    ],
    ids=["comment", "string"],
)
async def test_forbidden_structure_ignores_non_executable_source(
    tmp_path: Any,
    content: str,
) -> None:
    ac_text = "MUST NOT define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "CameraProvider is forbidden",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "# RETRIES=10\nRETRIES = 5\n",
        'message = "RETRIES=10"\nRETRIES = 5\n',
    ],
    ids=["comment", "string"],
)
async def test_forbidden_constant_ignores_non_executable_source(
    tmp_path: Any,
    content: str,
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
    (tmp_path / "config.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "tier", "expected", "filename", "content"),
    [
        (
            "MUST NOT define a CameraProvider class",
            "t2_structural",
            "CameraProvider",
            "build.sh",
            "cat <<'123'\nsafe text\n123\n",
        ),
        (
            "MUST NOT set RETRIES=10",
            "t1_constant",
            "10",
            "config.yaml",
            "- |\n  safe text\n",
        ),
    ],
    ids=["shell-heredoc", "yaml-sequence-block-scalar"],
)
async def test_forbidden_absence_fails_closed_for_unsupported_containers(
    tmp_path: Any,
    ac_text: str,
    tier: str,
    expected: str,
    filename: str,
    content: str,
) -> None:
    target = "CameraProvider" if tier == "t2_structural" else "RETRIES"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": tier,
                "pattern": target,
                "expected_value": expected,
                "file_hint": filename,
                "description": "Forbidden target must be absent",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert result.verified is False
    assert "could not classify" in result.detail
    assert "absence cannot be proven" in result.detail
    assert formal.final_approved is False


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
                "pattern": r"class\s+CameraProvider",
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


_SENTENCE_BOUNDARY_CONTRADICTIONS = (
    "MUST define a CameraProvider class! The class must not exist",
    "MUST define a CameraProvider class? The class must not exist",
    "MUST define a CameraProvider class. The class must not exist",
    "MUST define a CameraProvider class?! The class must not exist",
    "MUST define a CameraProvider class!\nThe class must not exist",
    "MUST define a CameraProvider class\nThe class must not exist",
    "MUST define a CameraProvider class!The deeply referenced target class must not exist",
    "MUST define a CameraProvider class?The class does not exist",
    "MUST define a CameraProvider class.The class must not exist",
    "MUST define a CameraProvider class.the class must not exist",
    "MUST define a CameraProvider class (required).the class must not exist",
    "MUST define a CameraProvider class。the class must not exist",
    "MUST define a CameraProvider class！the class must not exist",
    "MUST define a CameraProvider class？the class must not exist",
    "MUST define a CameraProvider class…the class must not exist",
    "MUST define a CameraProvider class — it must not exist",
    "MUST define a CameraProvider class–it must not exist",
    "MUST define a CameraProvider class (it must not exist)",
    "MUST define a CameraProvider class [it must not exist]",
    "MUST define a CameraProvider class / it must not exist",
    "MUST define a CameraProvider class: it must not exist",
    'MUST define a CameraProvider class "it must not exist"',
    "MUST define a CameraProvider class -> it must not exist",
    "MUST define a CameraProvider class, it must not exist",
    "MUST define a CameraProvider class; it must not exist",
    "MUST define a CameraProvider class and it must not exist",
    "MUST define a CameraProvider class but it must not exist",
    "MUST define a CameraProvider class -- it must not exist",
    "MUST define a CameraProvider class · it must not exist",
    "MUST define a CameraProvider class\u2028it must not exist",
    "MUST define a CameraProvider class plus it must not exist",
    "MUST define a CameraProvider class although it must not exist",
    "MUST define a CameraProvider class to be removed and must not exist",
    "MUST define a CameraProvider class to ensure this class must not remain",
    "MUST define a CameraProvider class because it must not exist",
    "MUST define a CameraProvider class so it must not exist",
    "MUST define a CameraProvider class because the object must not exist",
    "MUST define a CameraProvider class because the object's errors must not remain",
    "MUST define a CameraProvider class to prevent the widget from existing",
    "MUST define a CameraProvider class because errors must not remain although that one must not exist",
    "MUST define a CameraProvider class so failures must not remain although the former must not exist",
    "MUST define a CameraProvider class because handler errors must not remain although the handler must not exist",
    "MUST define a CameraProvider class to prevent module outages although the module must not exist",
    "MUST define a CameraProvider class and the deeply referenced target class definitely must not exist",
    "MUST define a CameraProvider class\r\nThe class must not exist",
    "MUST NOT define a CameraProvider class! The class must exist",
    "MUST NOT define a CameraProvider class.the class must exist",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("ac_text", _SENTENCE_BOUNDARY_CONTRADICTIONS)
@pytest.mark.parametrize("target_present", [True, False], ids=["present", "absent"])
async def test_sentence_boundary_contradiction_never_mints_formal_evidence(
    tmp_path: Any,
    ac_text: str,
    target_present: bool,
) -> None:
    """A later target-relative predicate makes the whole bound polarity ambiguous."""
    payload = [
        {
            "ac_index": 0,
            "tier": "t2_structural",
            "pattern": r"class\s+CameraProvider",
            "expected_value": "CameraProvider",
            "file_hint": "*.py",
            "description": "Punctuation cannot hide contradictory target semantics",
        }
    ]

    assertions = await _extract(ac_text, payload)
    assert assertions == ()

    stale_assertion = SpecAssertion(
        ac_index=0,
        ac_text=ac_text,
        tier="t2_structural",
        pattern=r"class\s+CameraProvider",
        expected_value="CameraProvider",
        file_hint="*.py",
        evidence_targets=("CameraProvider",),
        evidence_polarity=EvidencePolarity.REQUIRED,
        input_binding_required=True,
    )
    content = (
        "class CameraProvider:\n    pass\n" if target_present else "class Unrelated:\n    pass\n"
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all((stale_assertion,))
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert result.verified is False
    assert "criterion-bound" in result.detail.casefold()
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminator",
    ["!", "?", ".", "...", "?!", "\n", "！", "？", "。", "…"],
)
async def test_terminal_sentence_punctuation_preserves_one_positive_requirement(
    tmp_path: Any,
    terminator: str,
) -> None:
    """A terminal boundary is harmless when no second predicate follows it."""
    ac_text = f"MUST define a CameraProvider class{terminator}"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "One explicit positive sentence",
            }
        ],
    )
    (tmp_path / "main.py").write_text("class CameraProvider:\n    pass\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.REQUIRED
    assert verification.reports[0].results[0].verified is True
    assert formal.final_approved is True


@pytest.mark.asyncio
async def test_decimal_constant_does_not_look_like_a_second_sentence(tmp_path: Any) -> None:
    """A numeric dot inside one scalar value is not prose punctuation."""
    ac_text = "MUST set RATIO = 3.14"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"RATIO\s*=\s*",
                "expected_value": "3.14",
                "file_hint": "*.py",
                "description": "The ratio has one decimal value",
            }
        ],
    )
    (tmp_path / "config.py").write_text("RATIO = 3.14\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.REQUIRED
    assert verification.reports[0].results[0].verified is True
    assert formal.final_approved is True


_VALUE_TRAILING_CONTRADICTIONS = (
    "MUST set RETRIES=10 because it must not exist",
    "MUST set RETRIES=10 so the setting must not exist",
    "MUST set RETRIES=10 plus it must not exist",
    "MUST set RETRIES=10 although it must not exist",
    "MUST set RETRIES=10 despite it must not exist",
    "MUST set RETRIES=10 nevertheless it must not exist",
    "MUST set RETRIES=10 where it must not exist",
    "MUST set RETRIES=10 ~ it must not exist",
    "MUST set RETRIES=10 and RETRIES=20",
    "MUST set RETRIES=10 and RETRIES=10",
    "MUST set RETRIES=10 and RETRIES=absent",
    "MUST set RETRIES=10 and FPS=30 and RETRIES=20",
    "MUST set RETRIES=10 and setting=absent",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("ac_text", _VALUE_TRAILING_CONTRADICTIONS)
async def test_value_trailing_clause_never_mints_formal_evidence(
    tmp_path: Any,
    ac_text: str,
) -> None:
    """A valid scalar prefix cannot bypass complete-tail polarity validation."""
    payload = [
        {
            "ac_index": 0,
            "tier": "t1_constant",
            "pattern": r"RETRIES\s*=\s*",
            "expected_value": "10",
            "file_hint": "*.py",
            "description": "Trailing value clauses cannot mint evidence",
        }
    ]

    assertions = await _extract(ac_text, payload)
    assert assertions == ()

    stale_assertion = SpecAssertion(
        ac_index=0,
        ac_text=ac_text,
        tier="t1_constant",
        pattern=r"RETRIES\s*=\s*",
        expected_value="10",
        file_hint="*.py",
        evidence_targets=("RETRIES",),
        evidence_polarity=EvidencePolarity.REQUIRED,
        input_binding_required=True,
    )
    (tmp_path / "config.py").write_text("RETRIES = 10\n")

    verification = SpecVerifier(str(tmp_path)).verify_all((stale_assertion,))
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert result.verified is False
    assert "polarity is ambiguous or stale" in result.detail
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


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


_ARBITRARY_CAUSAL_TARGETS = (
    "FailureMode",
    "DefectHandler",
    "PreErrorPost",
    "FailingState",
    "CameraProvider",
    "QuartzBeacon",
    "FrobNicator",
)

_UNSUPPORTED_BARE_IMPERATIVES = (
    "Delete",
    "Remove",
    "Deprecate",
    "Document",
    "Mention",
    "Rename",
    "Test",
)

_CAUSAL_COMMAND_SPOOFS = (
    'MUST say "MUST define a CameraProvider class to ensure errors must not remain"',
    "MUST document the phrase 'MUST define a CameraProvider class to ensure errors must not remain'",
    "`MUST define a CameraProvider class to ensure errors must not remain` is an example only",
    "The documentation quotes MUST define a CameraProvider class to ensure errors must not remain",
    "The documentation records an example, MUST define a CameraProvider class to ensure errors must not remain",
    "MUST document this example, MUST define a CameraProvider class to ensure errors must not remain",
    "MUST document this example; MUST define a CameraProvider class to ensure errors must not remain",
    "MUST explain because, MUST define a CameraProvider class to ensure errors must not remain",
    "MUST say MUST define a CameraProvider class to ensure errors must not remain",
    "MUST explain because MUST define a CameraProvider class to ensure errors must not remain",
    "MUST document (MUST define a CameraProvider class to ensure errors must not remain)",
    "The CameraProvider class because errors must not remain",
    "The CameraProvider class so failures must never remain",
    "The CameraProvider class in order to ensure errors must not remain",
    "The CameraProvider class while errors must not remain",
    "The CameraProvider class whereas failures must never remain",
    "The CameraProvider class; errors must not remain",
    "The CameraProvider class\nerrors must not remain",
    "MUST avoid using this component, the CameraProvider class",
    "MUST prevent creation of this component; the CameraProvider class",
    "MUST omit this implementation, the CameraProvider class",
    "MUST NOT define the provider, the CameraProvider class",
    "WITHOUT any provider, the CameraProvider class",
    "MUST remove the CameraProvider class",
    "MUST delete the CameraProvider class",
    "MUST eliminate the CameraProvider class",
    "MUST drop the CameraProvider class",
    "MUST deprecate the CameraProvider class",
    "MUST erase the CameraProvider class",
    "MUST get rid of the CameraProvider class",
    "The CameraProvider class must be removed",
    "The CameraProvider class must disappear",
    "The CameraProvider class should be deleted",
    "The CameraProvider class shall be eliminated",
    "The CameraProvider class is deprecated",
    "The CameraProvider class is banned",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("ac_text", _CAUSAL_COMMAND_SPOOFS)
async def test_quoted_or_nested_positive_command_cannot_prove_target(ac_text: str) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Quoted or nested command is not a target requirement",
            }
        ],
    )

    assert assertions == ()


@pytest.mark.parametrize("ac_text", _CAUSAL_COMMAND_SPOOFS)
@pytest.mark.parametrize("target_present", [True, False], ids=["present", "absent"])
def test_stale_nested_causal_command_never_publishes_formal_pass(
    tmp_path: Any,
    ac_text: str,
    target_present: bool,
) -> None:
    assertion = SpecAssertion(
        ac_index=0,
        ac_text=ac_text,
        tier="t2_structural",
        pattern=r"class\s+CameraProvider",
        expected_value="CameraProvider",
        file_hint="*.py",
        evidence_targets=("CameraProvider",),
        evidence_polarity=EvidencePolarity.REQUIRED,
        input_binding_required=True,
    )
    content = (
        "class CameraProvider:\n    pass\n" if target_present else "class Unrelated:\n    pass\n"
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all((assertion,))
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].results[0].verified is False
    assert formal.final_approved is False


@pytest.mark.asyncio
@pytest.mark.parametrize("target", _ARBITRARY_CAUSAL_TARGETS)
async def test_unproven_causal_target_fails_closed_without_name_semantics(
    target: str,
) -> None:
    ac_text = f"The class {target} to ensure errors must not remain"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": rf"class\s+{target}",
                "expected_value": target,
                "file_hint": "*.py",
                "description": "Target has no independent positive predicate",
            }
        ],
    )

    assert assertions == ()


@pytest.mark.parametrize("target", _ARBITRARY_CAUSAL_TARGETS)
@pytest.mark.parametrize("target_present", [True, False], ids=["present", "absent"])
def test_stale_unproven_causal_target_never_publishes_formal_pass(
    tmp_path: Any,
    target: str,
    target_present: bool,
) -> None:
    ac_text = f"The class {target} to ensure errors must not remain"
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
    content = f"class {target}:\n    pass\n" if target_present else "class Unrelated:\n    pass\n"
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all((assertion,))
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].results[0].verified is False
    assert formal.final_approved is False


@pytest.mark.asyncio
@pytest.mark.parametrize("predicate", _UNSUPPORTED_BARE_IMPERATIVES)
@pytest.mark.parametrize("target", _ARBITRARY_CAUSAL_TARGETS)
@pytest.mark.parametrize("target_present", [True, False], ids=["present", "absent"])
async def test_unsupported_bare_imperative_never_mints_formal_evidence(
    tmp_path: Any,
    predicate: str,
    target: str,
    target_present: bool,
) -> None:
    """Unknown commands fail at extraction and again at verifier re-derivation."""
    ac_text = f"{predicate} the {target} class"
    payload = [
        {
            "ac_index": 0,
            "tier": "t2_structural",
            "pattern": rf"class\s+{target}",
            "expected_value": target,
            "file_hint": "*.py",
            "description": "Unsupported command cannot require target presence",
        }
    ]

    assertions = await _extract(ac_text, payload)
    assert assertions == ()

    stale_assertion = SpecAssertion(
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
    content = f"class {target}:\n    pass\n" if target_present else "class Unrelated:\n    pass\n"
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all((stale_assertion,))
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert result.verified is False
    assert "polarity is ambiguous or stale" in result.detail
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    _ARBITRARY_CAUSAL_TARGETS,
)
@pytest.mark.parametrize(
    ("content", "approved"),
    [("class CameraProvider:\n    pass\n", True), ("class Unrelated:\n    pass\n", False)],
    ids=["target-present", "target-absent"],
)
async def test_explicit_positive_predicate_proves_arbitrary_causal_target(
    tmp_path: Any,
    target: str,
    content: str,
    approved: bool,
) -> None:
    ac_text = f"MUST define a {target} class to ensure errors must not remain"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": rf"class\s+{target}",
                "expected_value": target,
                "file_hint": "*.py",
                "description": "Explicit predicate requires the target",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content.replace("CameraProvider", target))

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert assertions[0].evidence_polarity is EvidencePolarity.REQUIRED
    assert formal.final_approved is approved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ac_text",
    [
        "The CameraProvider class to ensure errors must not remain",
        "A CameraProvider class is required for failures that must not remain",
    ],
    ids=["bare-declarative", "implicit-required"],
)
async def test_implicit_positive_causal_prose_accepts_conservative_false_negative(
    ac_text: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Implicit prose is not a bounded positive proof",
            }
        ],
    )

    assert assertions == ()


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
@pytest.mark.parametrize(
    ("ac_text", "tier", "pattern", "expected", "content"),
    [
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?=class)",
            "CameraProvider",
            "class Unrelated:\n    pass\n# CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"[\s\S]+",
            "CameraProvider",
            "class Unrelated:\n    pass\n# CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?=[\s\S]*CameraProvider)(?=class)",
            "CameraProvider",
            "class Unrelated:\n    pass\n# CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?=[\s\S]*CameraProvider)[\s\S]+",
            "CameraProvider",
            "class Unrelated:\n    pass\n# CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"CameraProvider|[\s\S]+",
            "CameraProvider",
            "class Unrelated:\n    pass\n# CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?=(?:CameraProvider|class))",
            "CameraProvider",
            "class X: # CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?i)cameraprovider[\s\S]+",
            "CameraProvider",
            "cameraprovider unrelated\n# CameraProvider\n",
        ),
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?i:CameraProvider)[\s\S]+",
            "CameraProvider",
            "cameraprovider unrelated\n# CameraProvider\n",
        ),
        (
            "MUST define a StraßeProvider class",
            "t2_structural",
            r"(?i)STRASSEPROVIDER[\s\S]+",
            "StraßeProvider",
            "STRASSEPROVIDER_fake = 1\n# StraßeProvider\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"(?=10)(?=[\s\S]*RETRIES)",
            "10",
            "10\n# RETRIES is not assigned\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"(?=10)",
            "10",
            "# RETRIES is not assigned\n10\n",
        ),
    ],
    ids=[
        "t2-zero-width",
        "t2-consuming",
        "t2-split-lookaheads",
        "t2-global-target-consuming",
        "t2-target-free-branch",
        "t2-zero-width-target-free-branch",
        "t2-ignorecase-runtime-decoy",
        "t2-scoped-ignorecase-addition",
        "t2-unicode-casefold-expansion",
        "t1-split-lookaheads",
        "t1-zero-width",
    ],
)
async def test_copresent_regex_and_target_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    tier: str,
    pattern: str,
    expected: str,
    content: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": tier,
                "pattern": pattern,
                "expected_value": expected,
                "file_hint": "*.py",
                "description": "Model regex and target are independently co-present",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "tier", "pattern", "expected", "content"),
    [
        (
            "MUST define a CameraProvider class",
            "t2_structural",
            r"(?=class CameraProvider)",
            "CameraProvider",
            "class CameraProvider:\n    pass\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "RETRIES = 10\n",
        ),
    ],
    ids=["t2-target-lookahead", "t1-target-assignment"],
)
async def test_target_causal_regex_controls_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    tier: str,
    pattern: str,
    expected: str,
    content: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": tier,
                "pattern": pattern,
                "expected_value": expected,
                "file_hint": "*.py",
                "description": "Regex satisfaction depends on the trusted target",
            }
        ],
    )
    (tmp_path / "main.py").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


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
