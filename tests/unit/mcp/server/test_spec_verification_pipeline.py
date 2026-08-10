"""End-to-end regressions for extracted assertions promoted to formal verdicts."""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ouroboros.core.lineage import EvaluationSummary, TaskResult
from ouroboros.core.types import Result
from ouroboros.mcp.server.adapter import _evaluation_summary_from_spec_verification
from ouroboros.providers.base import CompletionResponse
from ouroboros.verification.extractor import AssertionExtractor
from ouroboros.verification.models import (
    EvidencePolarity,
    SpecAssertion,
    VerificationOutcome,
    VerificationTier,
)
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
    return result.value if result.is_ok else ()


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
            "config.yaml",
            "notes: &payload !text |\n  RETRIES=10\n",
        ),
        (
            "MUST set RETRIES=10",
            "t1_constant",
            r"RETRIES\s*=\s*",
            "10",
            "config.yaml",
            "- !text &payload >-\n  RETRIES=10\n",
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
        "yaml-mapping-node-properties",
        "yaml-sequence-node-properties",
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
        ("main.sql", "SELECT q'[ignore ' class CameraProvider]' FROM dual;\n"),
        ("view.xml", '<class CameraProvider=""/>\n'),
        ("main.m", "% class CameraProvider\n"),
        ("styles.css", "class CameraProvider { color: red; }\n"),
        ("styles.scss", "class CameraProvider { color: red; }\n"),
        ("main.r", 'pattern <- r"(foo " class CameraProvider)"\n'),
        ("main.r", 'pattern <- r"---[foo " class CameraProvider]---"\n'),
        ("main.jsx", "const view = <div>class CameraProvider</div>;\n"),
        ("main.tsx", "const view = <>class CameraProvider</>;\n"),
        ("main.pl", "format STDOUT =\nclass CameraProvider\n.\n"),
        ("main.pl", "format =\nclass CameraProvider\n.\n"),
        ("main.pl", "format Foo::Bar =\nclass CameraProvider\n.\n"),
        ("main.pl", "format\nSTDOUT =\nclass CameraProvider\n.\nwrite;\n"),
        ("main.pl", "format Foo::Bar\n=\nclass CameraProvider\n.\n"),
        ("main.cpp", "#if 0\nclass CameraProvider {};\n#endif\n"),
        ("main.cpp", "#i\\\nf 0\nclass CameraProvider {};\n#e\\\nndif\n"),
        ("main.cpp", "%:if 0\nclass CameraProvider {};\n%:endif\n"),
        ("main.cpp", "%:i\\\nf 0\nclass CameraProvider {};\n%:e\\\nndif\n"),
        ("main.cpp", "??=if 0\nclass CameraProvider {};\n??=endif\n"),
        ("main.cpp", "#define UNUSED_DECL class CameraProvider\n"),
        ("main.cpp", "#define UNUSED_DECL \\\nclass CameraProvider\n"),
        ("main.cpp", "#de\\\nfine UNUSED_DECL class CameraProvider\n"),
        ("main.cpp", "// ignored \\\nclass CameraProvider {};\n"),
        ("main.cpp", "/\\\n/ class CameraProvider {};\n"),
        ("main.mm", 'const char* s = R"TAG(foo " class CameraProvider)TAG";\n'),
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
        "sql-oracle-q-quote",
        "xml-element-attribute",
        "matlab-comment",
        "css-declaration",
        "scss-declaration",
        "r-raw-string",
        "r-delimited-raw-string",
        "jsx-text",
        "tsx-fragment-text",
        "perl-format",
        "perl-anonymous-format",
        "perl-package-format",
        "perl-multiline-format",
        "perl-package-multiline-format",
        "cpp-disabled-preprocessor-region",
        "cpp-spliced-disabled-preprocessor-region",
        "cpp-digraph-disabled-preprocessor-region",
        "cpp-spliced-digraph-disabled-preprocessor-region",
        "cpp-trigraph-disabled-preprocessor-region",
        "cpp-macro-replacement-list",
        "cpp-continued-macro-replacement-list",
        "cpp-spliced-macro-name",
        "cpp-spliced-line-comment",
        "cpp-splice-created-line-comment",
        "objective-cpp-raw-string",
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        r"\u002f\u002f class CameraProvider {}" "\n",
        r"String s = \u0022class CameraProvider {}\u0022;" "\n",
        r"// ignored \u000a class CameraProvider {}" "\n",
        r"cl\u0061ss CameraProvider {}" "\n",
    ],
    ids=["comment", "quote", "line-terminator", "token"],
)
async def test_java_prelexical_unicode_escape_cannot_reach_formal_pass(
    tmp_path: Any,
    content: str,
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
                "file_hint": "*.java",
                "description": "Translated Java text is not physical source evidence",
            }
        ],
    )
    (tmp_path / "Provider.java").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider interface",
            "provider.go",
            "//go:build ignore && !ignore\npackage provider\ntype CameraProvider interface {}\n",
            r"type\s+CameraProvider\s+interface",
        ),
        (
            "MUST define a CameraProvider trait",
            "provider.rs",
            "#[cfg(any())]\ntrait CameraProvider {}\n",
            r"trait\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.hs",
            "{-# LANGUAGE CPP #-}\n#if 0\nclass CameraProvider a\n#endif\n",
            r"class\s+CameraProvider",
        ),
    ],
    ids=["go-build-constraint", "rust-cfg", "haskell-cpp"],
)
async def test_build_excluded_declarations_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Build-excluded declaration is not evidence",
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
        (
            "provider.go",
            "package provider\nvar a = `ends with \\`\nvar b = `func CameraProvider`\n",
        ),
        ("provider.rs", "const _: &str = stringify!(fn CameraProvider);\n"),
        (
            "provider.rs",
            "macro_rules! hidden { () => { fn CameraProvider() {} }; }\n",
        ),
    ],
    ids=["go-raw-string", "rust-macro-invocation", "rust-macro-definition"],
)
async def test_arbitrary_token_containers_cannot_reach_formal_pass(
    tmp_path: Any,
    filename: str,
    content: str,
) -> None:
    ac_text = "MUST define a CameraProvider function"
    pattern = r"(?:func|fn)\s+CameraProvider"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Token container is not a function declaration",
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
async def test_model_cannot_substitute_function_evidence_for_a_class_criterion(
    tmp_path: Any,
) -> None:
    ac_text = "MUST define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"def\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.py",
                "description": "Substitute a function for the requested class",
            }
        ],
    )
    (tmp_path / "provider.py").write_text("def CameraProvider():\n    pass\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content", "pattern", "approved"),
    [
        ("Provider.java", "def CameraProvider(): {}\n", r"def\s+CameraProvider", False),
        (
            "Provider.java",
            "class Provider {\n    public void CameraProvider(int value) {}\n}\n",
            r"void\s+CameraProvider",
            True,
        ),
        ("Provider.kt", "fun CameraProvider() {}\n", r"fun\s+CameraProvider", True),
        ("provider.c", "void CameraProvider(void) {}\n", r"void\s+CameraProvider", True),
        (
            "provider.cpp",
            "int CameraProvider(int value) { return value; }\n",
            r"int\s+CameraProvider",
            True,
        ),
        ("provider.go", "func CameraProvider() {}\n", r"func\s+CameraProvider", True),
        (
            "provider.ts",
            "function CameraProvider(): void {}\n",
            r"function\s+CameraProvider",
            True,
        ),
        ("provider.swift", "func CameraProvider() {}\n", r"func\s+CameraProvider", True),
        ("provider.rs", "fn CameraProvider() {}\n", r"fn\s+CameraProvider", True),
    ],
    ids=["invalid-java-def", "java", "kotlin", "c", "cpp", "go", "typescript", "swift", "rust"],
)
async def test_declaration_kind_is_bound_to_the_file_language_before_formal_promotion(
    tmp_path: Any,
    filename: str,
    content: str,
    pattern: str,
    approved: bool,
) -> None:
    ac_text = "MUST define a CameraProvider function"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Language-specific function declaration",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is approved
    assert formal.final_approved is approved
    assert formal.ac_results[0].final_verdict == ("pass" if approved else "fail")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "enum class CameraProvider { A };\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.cpp",
            "enum struct CameraProvider { A };\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.c",
            "typedef void CameraProvider(void);\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.c",
            "typedef\nvoid CameraProvider(void);\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "public delegate void CameraProvider();\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "public delegate\nvoid CameraProvider();\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "template <class CameraProvider> void use(CameraProvider*);\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "struct Widget { explicit Widget(int); }; Widget CameraProvider(42);\n",
            r"Widget\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "Widget CameraProvider(foo.value);\n",
            r"Widget\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.ts",
            "function CameraProvider(): void;\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.kt",
            "expect fun CameraProvider(): Unit\n",
            r"fun\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.swift",
            "protocol Provider { func CameraProvider() }\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "trait Provider { fn CameraProvider(); }\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.hpp",
            "class CameraProvider;\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.hpp",
            "struct CameraProvider;\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class CameraProvider { public CameraProvider() {} }\n",
            r"CameraProvider\s*\(",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "expect class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.py",
            "def CameraProvider():\n",
            r"def\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.py",
            "class CameraProvider:\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.js",
            "class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.rb",
            "class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.swift",
            "class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "class CameraProvider {\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.hs",
            "class CameraProvider a\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.go",
            "func CameraProvider() {\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "public void CameraProvider() {\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "fn CameraProvider() {\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.js",
            "class CameraProvider = {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider = {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider = {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.swift",
            "class CameraProvider = {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider() {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.js",
            "function CameraProvider() = {}\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.go",
            "func CameraProvider() = {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "fn CameraProvider() = {}\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.swift",
            "func CameraProvider() = {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.js",
            "function CameraProvider(not valid) {}\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.go",
            "func CameraProvider(not valid) {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.kt",
            "fun CameraProvider(not valid) {}\n",
            r"fun\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.swift",
            "func CameraProvider(not valid) {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider<not valid> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider<not valid> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.rs",
            "struct CameraProvider<T T> {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.js",
            "function CameraProvider() { const = ; }\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.ts",
            "function CameraProvider(): ??? {}\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "fn CameraProvider() -> ??? {}\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.swift",
            "func CameraProvider() -> ??? {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.kt",
            "fun CameraProvider(): ??? {}\n",
            r"fun\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.go",
            "func CameraProvider() + {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class X {\n    public void CameraProvider() where ??? {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "auto CameraProvider() -> ??? {}\n",
            r"auto\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider implements A extends B {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider interface",
            "Provider.java",
            "interface CameraProvider implements A {}\n",
            r"interface\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider implements A extends B {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "private class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "private class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider extends int {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider implements int {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider permits Other {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "final class CameraProvider permits Other {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class Provider {\n    abstract void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class Provider {\n    native void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "public class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class Provider {\n    public int CameraProvider() {}\n}\n",
            r"int\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider<T, T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider<T, T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider interface",
            "Provider.java",
            "final interface CameraProvider {}\n",
            r"interface\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "Provider.cs",
            "static struct CameraProvider {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "Provider.cs",
            "abstract struct CameraProvider {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.rs",
            "unsafe struct CameraProvider {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider interface",
            "provider.kt",
            "data interface CameraProvider {}\n",
            r"interface\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "class CameraProvider : public int {};\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "class CameraProvider : public Base, private Base {};\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class Provider {\n    public void CameraProvider(int value, long value) {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class Provider {\n    public void CameraProvider(int int) {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "class Provider {\n    public void CameraProvider() throws int {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            (
                "class Provider {\n"
                "    public void CameraProvider() throws IOException, IOException {}\n"
                "}\n"
            ),
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "CameraProvider.java",
            "public non-sealed class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "CameraProvider.java",
            "public sealed class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider<class> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider implements class {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "data class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "value class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "class CameraProvider : public CameraProvider {};\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider extends CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider interface",
            "Provider.java",
            "interface CameraProvider extends CameraProvider {}\n",
            r"interface\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider implements Base, Base {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "abstract sealed class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "sealed open class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "main.js",
            "class CameraProvider extends CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "main.ts",
            "class CameraProvider extends CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "class CameraProvider : CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.swift",
            "class CameraProvider : CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "class CameraProvider : CameraProvider() {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "static class CameraProvider : Base {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "Provider.cs",
            "struct CameraProvider : int {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.rs",
            "struct CameraProvider<T, T> {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "class CameraProvider<T, T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "class CameraProvider<T, T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.swift",
            "class CameraProvider<T, T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.kt",
            "open fun CameraProvider() {}\n",
            r"fun\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.swift",
            "static func CameraProvider() {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "virtual void CameraProvider() {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class Provider {\n    public static virtual void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class Provider {\n    public sealed virtual void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class Provider {\n    private virtual void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "abstract enum Provider {\n    X;\n    public void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "static struct Provider {\n    public void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "abstract struct Provider {\n    public void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "record Provider {\n    public void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "enum Provider {\n    public void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "annotation enum class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "sealed enum class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "open annotation class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "async const fn CameraProvider() {}\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "unsafe pub fn CameraProvider() {}\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider trait",
            "provider.rs",
            "unsafe pub trait CameraProvider {}\n",
            r"trait\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "class CameraProvider : Int() {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.swift",
            "class CameraProvider: Int {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class Provider {\n    public int CameraProvider() {}\n}\n",
            r"int\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class Provider {\n    public abstract void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "class Provider {\n    public extern void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider { void value; }\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider { var value; }\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.js",
            "default export class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.js",
            "default export function CameraProvider() {}\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.js",
            "async export function CameraProvider() {}\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.js",
            "if (true) class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "if (true)\nclass CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "if (true)\nclass CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "public private class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.rs",
            "nonsense fn CameraProvider() {}\n",
            r"fn\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "public private void CameraProvider() {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "public void CameraProvider() {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "interface X {\n    public void CameraProvider() {}\n}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.js",
            "class X {\n    function CameraProvider() {}\n}\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.go",
            "func outer() {\n    func CameraProvider() {}\n}\n",
            r"func\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "public void CameraProvider() {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            "if (true) class X { public void CameraProvider() {} }\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.cs",
            "if (true) class X { public void CameraProvider() {} }\n",
            r"void\s+CameraProvider",
        ),
    ],
    ids=[
        "enum-class",
        "enum-struct",
        "c-typedef",
        "c-multiline-typedef",
        "csharp-delegate",
        "csharp-multiline-delegate",
        "cpp-template-parameter",
        "cpp-direct-initializer",
        "cpp-member-direct-initializer",
        "typescript-signature",
        "kotlin-expect",
        "swift-protocol-requirement",
        "rust-trait-signature",
        "cpp-class-forward-declaration",
        "cpp-struct-forward-declaration",
        "java-constructor",
        "kotlin-expect-class",
        "python-incomplete-function",
        "python-incomplete-class",
        "java-incomplete-class",
        "javascript-incomplete-class",
        "ruby-incomplete-class",
        "csharp-incomplete-class",
        "swift-incomplete-class",
        "typescript-incomplete-class",
        "cpp-unclosed-class",
        "haskell-incomplete-class",
        "go-unclosed-function",
        "java-unclosed-function",
        "rust-unclosed-function",
        "javascript-assigned-class-body",
        "java-assigned-class-body",
        "typescript-assigned-class-body",
        "swift-assigned-class-body",
        "java-parenthesized-class-header",
        "javascript-assigned-function-body",
        "go-assigned-function-body",
        "rust-assigned-function-body",
        "swift-assigned-function-body",
        "javascript-invalid-parameters",
        "go-invalid-parameters",
        "kotlin-invalid-parameters",
        "swift-invalid-parameters",
        "java-invalid-generic-clause",
        "typescript-invalid-generic-clause",
        "rust-invalid-generic-clause",
        "javascript-invalid-body",
        "typescript-invalid-return-type",
        "rust-invalid-return-type",
        "swift-invalid-return-type",
        "kotlin-invalid-return-type",
        "go-invalid-result-clause",
        "csharp-invalid-constraint-clause",
        "cpp-invalid-trailing-return-type",
        "java-invalid-class-clause-order",
        "java-interface-implements-clause",
        "typescript-invalid-class-clause-order",
        "java-illegal-top-level-modifier",
        "csharp-illegal-top-level-modifier",
        "java-primitive-extends",
        "java-primitive-implements",
        "java-unsealed-permits",
        "java-final-permits",
        "java-abstract-method-body",
        "java-native-method-body",
        "java-public-type-filename-mismatch",
        "java-empty-nonvoid-method",
        "java-duplicate-generic-parameters",
        "typescript-duplicate-generic-parameters",
        "java-final-interface",
        "csharp-static-struct",
        "csharp-abstract-struct",
        "rust-unsafe-struct",
        "kotlin-data-interface",
        "cpp-builtin-base",
        "cpp-duplicate-direct-base",
        "java-duplicate-parameter-name",
        "java-keyword-parameter-name",
        "java-keyword-throws-type",
        "java-duplicate-throws-type",
        "java-nonsealed-without-parent",
        "java-sealed-without-permits",
        "typescript-keyword-generic-parameter",
        "typescript-keyword-implements-type",
        "kotlin-empty-data-class",
        "kotlin-empty-value-class",
        "cpp-self-inheritance",
        "java-class-self-inheritance",
        "java-interface-self-inheritance",
        "java-duplicate-direct-supertype",
        "csharp-abstract-sealed-class",
        "kotlin-sealed-open-class",
        "javascript-self-inheritance",
        "typescript-self-inheritance",
        "csharp-self-inheritance",
        "swift-self-inheritance",
        "kotlin-self-inheritance",
        "csharp-static-class-base",
        "csharp-struct-builtin-base",
        "rust-duplicate-generic-parameter",
        "csharp-duplicate-generic-parameter",
        "kotlin-duplicate-generic-parameter",
        "swift-duplicate-generic-parameter",
        "kotlin-top-level-open-function",
        "swift-top-level-static-function",
        "cpp-top-level-virtual-function",
        "csharp-static-virtual-method",
        "csharp-sealed-virtual-method",
        "csharp-private-virtual-method",
        "java-method-in-abstract-enum",
        "csharp-method-in-static-struct",
        "csharp-method-in-abstract-struct",
        "java-method-in-record-without-header",
        "java-method-in-enum-without-separator",
        "kotlin-annotation-enum-class",
        "kotlin-sealed-enum-class",
        "kotlin-open-annotation-class",
        "rust-async-const-function",
        "rust-unsafe-before-pub-function",
        "rust-unsafe-before-pub-trait",
        "kotlin-final-builtin-base",
        "swift-final-builtin-base",
        "csharp-empty-nonvoid-method",
        "csharp-abstract-method-body",
        "csharp-extern-method-body",
        "java-void-field",
        "java-var-field",
        "javascript-default-export-class",
        "javascript-default-export-function",
        "javascript-async-export-function",
        "javascript-conditional-class-prefix",
        "java-multiline-conditional-class",
        "csharp-multiline-conditional-class",
        "java-conflicting-class-modifiers",
        "rust-unsupported-function-prefix",
        "java-conflicting-function-modifiers",
        "java-top-level-method",
        "java-interface-concrete-method",
        "javascript-class-function-declaration",
        "go-nested-named-function",
        "csharp-top-level-method",
        "java-method-in-invalid-type-shell",
        "csharp-method-in-invalid-type-shell",
    ],
)
async def test_type_declaration_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Type declarations are not requested declarations",
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
    ("filename", "content", "pattern"),
    [
        ("provider.rs", "pub(foo) fn CameraProvider() {}\n", r"fn\s+CameraProvider"),
        ("provider.kt", "public(foo) fun CameraProvider() {}\n", r"fun\s+CameraProvider"),
        (
            "provider.swift",
            "public(foo) func CameraProvider() {}\n",
            r"func\s+CameraProvider",
        ),
        (
            "provider.c",
            "extern static void CameraProvider(void) {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "provider.c",
            "static register void CameraProvider(void) {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "provider.cpp",
            "extern static void CameraProvider() {}\n",
            r"void\s+CameraProvider",
        ),
    ],
    ids=[
        "rust-invalid-visibility-argument",
        "kotlin-invalid-modifier-argument",
        "swift-invalid-modifier-argument",
        "c-extern-static-function",
        "c-static-register-function",
        "cpp-extern-static-function",
    ],
)
async def test_invalid_modifier_syntax_cannot_reach_formal_pass(
    tmp_path: Any,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    ac_text = "MUST define a CameraProvider function"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Invalid modifier syntax cannot prove a function",
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
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "signed unsigned void CameraProvider() {}\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "long short int CameraProvider() { return 0; }\n",
            r"int\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "char float CameraProvider() { return 0; }\n",
            r"float\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "CameraProvider.java",
            "public class CameraProvider { int class; }\n",
            r"class\s+CameraProvider",
        ),
    ],
    ids=[
        "cpp-conflicting-sign-specifiers",
        "cpp-conflicting-width-specifiers",
        "cpp-conflicting-scalar-specifiers",
        "java-keyword-field-name",
    ],
)
async def test_invalid_language_tokens_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Invalid language tokens cannot prove a declaration",
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
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider function",
            "provider.c",
            "void CameraProvider(void) { return 1; }\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "void CameraProvider() { return 1; }\n",
            r"void\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "int CameraProvider() { return nullptr; }\n",
            r"int\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.cpp",
            "int *CameraProvider() { return 1; }\n",
            r"CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider extends String {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider implements Object {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider extends Enum {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.js",
            "class CameraProvider extends true {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.rs",
            "struct CameraProvider { type: type, }\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.rs",
            "struct CameraProvider { fn: fn, }\n",
            r"struct\s+CameraProvider",
        ),
    ],
    ids=[
        "c-void-return-value",
        "cpp-void-return-value",
        "cpp-nonpointer-nullptr-return",
        "cpp-pointer-nonzero-return",
        "java-final-class-base",
        "java-class-as-interface",
        "java-special-nonextendable-base",
        "javascript-primitive-base",
        "rust-keyword-field-and-type",
        "rust-function-keyword-field-and-type",
    ],
)
async def test_intrinsically_invalid_declarations_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Intrinsic declaration semantics must be valid",
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
        ("provider.py", "class CameraProvider:\n    pass\n"),
        ("Provider.java", "class CameraProvider {}\n"),
        ("provider.js", "class CameraProvider {}\n"),
        ("provider.rb", "class CameraProvider\nend\n"),
        ("Provider.cs", "class CameraProvider {}\n"),
        ("provider.kt", "class CameraProvider\n"),
        ("provider.swift", "class CameraProvider {}\n"),
        ("provider.ts", "class CameraProvider {}\n"),
        ("provider.cpp", "class CameraProvider {};\n"),
        ("Provider.hs", "class CameraProvider a where\n"),
    ],
    ids=[
        "python",
        "java",
        "javascript",
        "ruby",
        "csharp",
        "kotlin-bodyless",
        "swift",
        "typescript",
        "cpp",
        "haskell",
    ],
)
async def test_complete_class_definitions_can_reach_formal_pass(
    tmp_path: Any,
    filename: str,
    content: str,
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
                "description": "Complete language-specific class definition",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider<T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider<T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.kt",
            "class CameraProvider<T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.swift",
            "class CameraProvider<T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "Provider.cs",
            "class CameraProvider<T> {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider trait",
            "provider.rs",
            "trait CameraProvider<T> {}\n",
            r"trait\s+CameraProvider",
        ),
    ],
    ids=["java", "typescript", "kotlin", "swift", "csharp", "rust"],
)
async def test_valid_generic_type_definitions_can_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Complete generic type definition",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider class",
            "Provider.java",
            "class CameraProvider { int value; }\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "provider.rs",
            "struct CameraProvider { value: i32, }\n",
            r"struct\s+CameraProvider",
        ),
    ],
    ids=["java-field", "rust-field"],
)
async def test_bounded_nonempty_types_can_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Complete bounded non-empty type definition",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
async def test_forbidden_function_finds_parameterized_java_method(tmp_path: Any) -> None:
    ac_text = "MUST NOT define a CameraProvider function"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"void\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "*.java",
                "description": "CameraProvider function is forbidden",
            }
        ],
    )
    (tmp_path / "Provider.java").write_text(
        "class Provider { public void CameraProvider(int value) {} }\n"
    )

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "filename", "content", "pattern"),
    [
        (
            "class",
            "Provider.java",
            "class CameraProvider { int value; }\n",
            r"class\s+CameraProvider",
        ),
        (
            "struct",
            "provider.rs",
            "struct CameraProvider { value: i32, }\n",
            r"struct\s+CameraProvider",
        ),
    ],
    ids=["java-class", "rust-struct"],
)
async def test_forbidden_type_finds_bounded_nonempty_definition(
    tmp_path: Any,
    kind: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    ac_text = f"MUST NOT define a CameraProvider {kind}"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": f"CameraProvider {kind} is forbidden",
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
    ("ac_text", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider class",
            "class CameraProvider { void start() {} }\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "class Provider {\n    public void CameraProvider(String value) {}\n}\n",
            r"void\s+CameraProvider",
        ),
    ],
    ids=["nonempty-class-body", "reference-typed-method-parameter"],
)
async def test_unclassified_valid_declaration_shape_is_not_a_formal_discrepancy(
    tmp_path: Any,
    ac_text: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": "*.java",
                "description": "Valid declaration outside the parser-free subset",
            }
        ],
    )
    (tmp_path / "Provider.java").write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions, agent_results={0: True})
    formal = _formal_verdict(ac_text, verification, agent_reported_pass=True)

    assert verification.reports[0].results[0].outcome is VerificationOutcome.UNVERIFIABLE
    assert formal.final_approved is False
    assert formal.ac_results[0].ac_verdict_state == "not_evaluated"
    assert formal.ac_results[0].rendered_verdict == "NOT_EVALUATED"


@pytest.mark.asyncio
async def test_t1_comparison_cannot_reach_formal_pass_as_assignment(tmp_path: Any) -> None:
    ac_text = "MUST set RETRIES=3"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"RETRIES\s*==\s*",
                "expected_value": "3",
                "file_hint": "*.py",
                "description": "RETRIES is set to 3",
            }
        ],
    )
    (tmp_path / "main.py").write_text("assert RETRIES == 3\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is False
    assert formal.final_approved is False
    assert formal.ac_results[0].final_verdict == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content", "verified"),
    [
        ("main.py", "const RETRIES = 3\n", False),
        ("main.py", "RETRIES: int = 3\n", True),
        ("Config.java", "class Config {\n    static final RETRIES = 3;\n}\n", False),
        (
            "Config.java",
            "class Config {\n    static final int RETRIES = 3;\n}\n",
            True,
        ),
        (
            "Config.java",
            "class Config {\n    public public int RETRIES = 3;\n}\n",
            False,
        ),
        (
            "Config.java",
            "class Config {\n    public private int RETRIES = 3;\n}\n",
            False,
        ),
        (
            "Config.java",
            "interface Config {\n    private int RETRIES = 3;\n}\n",
            False,
        ),
        (
            "Config.java",
            "interface Config {\n    int RETRIES = 3;\n}\n",
            True,
        ),
        (
            "Config.java",
            "public public class Config {\n    int RETRIES = 3;\n}\n",
            False,
        ),
        (
            "Config.java",
            "class Config<T> {\n    int RETRIES = 3;\n}\n",
            True,
        ),
        (
            "Config.java",
            "record Config {\n    int RETRIES = 3;\n}\n",
            False,
        ),
    ],
    ids=[
        "python-const",
        "python-annotated",
        "java-untyped-field",
        "java-typed-field",
        "java-duplicate-field-modifier",
        "java-conflicting-field-modifiers",
        "java-private-interface-field",
        "java-implicit-public-interface-field",
        "java-invalid-enclosing-type-modifiers",
        "java-generic-enclosing-class",
        "java-record-without-parameters",
    ],
)
async def test_t1_assignment_syntax_is_bound_to_file_language_before_formal_prompt(
    tmp_path: Any,
    filename: str,
    content: str,
    verified: bool,
) -> None:
    ac_text = "MUST set RETRIES=3"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"RETRIES(?:: int)? = ",
                "expected_value": "3",
                "file_hint": filename,
                "description": "RETRIES is set to 3",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is verified
    assert formal.final_approved is verified
    assert formal.ac_results[0].final_verdict == ("pass" if verified else "fail")


@pytest.mark.asyncio
async def test_t1_valid_unsupported_java_field_context_is_not_formal_discrepancy(
    tmp_path: Any,
) -> None:
    ac_text = "MUST set RETRIES=3"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t1_constant",
                "pattern": r"RETRIES = ",
                "expected_value": "3",
                "file_hint": "Config.java",
                "description": "Valid unsupported record field context",
            }
        ],
    )
    (tmp_path / "Config.java").write_text(
        "record Config(int value) {\n    static final int RETRIES = 3;\n}\n"
    )

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions, agent_results={0: True})
    formal = _formal_verdict(ac_text, verification, agent_reported_pass=True)

    assert verification.reports[0].results[0].outcome is VerificationOutcome.UNVERIFIABLE
    assert formal.final_approved is False
    assert formal.ac_results[0].ac_verdict_state == "not_evaluated"
    assert formal.ac_results[0].rendered_verdict == "NOT_EVALUATED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider interface",
            "Provider.java",
            "abstract interface CameraProvider {}\n",
            r"interface\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider struct",
            "Provider.cs",
            "partial struct CameraProvider {}\n",
            r"struct\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider trait",
            "provider.rs",
            "unsafe trait CameraProvider {}\n",
            r"trait\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider interface",
            "provider.kt",
            "sealed interface CameraProvider {}\n",
            r"interface\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.cpp",
            "class CameraProvider : public Base {};\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "class CameraProvider<T> implements Base {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            ("class Provider {\n    public void CameraProvider(int value, long count) {}\n}\n"),
            r"void\s+CameraProvider",
        ),
    ],
    ids=[
        "java-abstract-interface",
        "csharp-partial-struct",
        "rust-unsafe-trait",
        "kotlin-sealed-interface",
        "cpp-type-like-base",
        "typescript-generic-implements",
        "java-unique-parameters",
    ],
)
async def test_type_modifier_and_cpp_base_positive_controls_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Valid kind-specific modifier or C++ base",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider class",
            "CameraProvider.java",
            (
                "public sealed class CameraProvider permits Other {}\n"
                "final class Other extends CameraProvider {}\n"
            ),
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "Provider.java",
            (
                "class Provider {\n"
                "    public void CameraProvider() throws java.lang.Exception {}\n"
                "}\n"
            ),
            r"void\s+CameraProvider",
        ),
    ],
    ids=["java-sealed-relationship", "java-throws-clause"],
)
async def test_valid_unsupported_java_declaration_is_not_formal_discrepancy(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Valid unsupported Java declaration",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions, agent_results={0: True})
    formal = _formal_verdict(ac_text, verification, agent_reported_pass=True)

    assert verification.reports[0].results[0].outcome is VerificationOutcome.UNVERIFIABLE
    assert formal.final_approved is False
    assert formal.ac_results[0].ac_verdict_state == "not_evaluated"
    assert formal.ac_results[0].rendered_verdict == "NOT_EVALUATED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("Provider.java", "class CameraProvider {}\n"),
        ("CameraProvider.java", "public class CameraProvider {}\n"),
    ],
    ids=["package-private-other-filename", "public-matching-filename"],
)
async def test_java_declaration_context_positive_controls_reach_formal_pass(
    tmp_path: Any,
    filename: str,
    content: str,
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
                "description": "Valid Java declaration context",
            }
        ],
    )
    (tmp_path / filename).write_text(content)

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
async def test_cpp_template_class_declaration_can_reach_formal_pass(tmp_path: Any) -> None:
    ac_text = "MUST define a CameraProvider class"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"class\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": "provider.cpp",
                "description": "Class template declaration",
            }
        ],
    )
    (tmp_path / "provider.cpp").write_text("template <class T>\nclass CameraProvider {};\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ac_text", "filename", "content", "pattern"),
    [
        (
            "MUST define a CameraProvider class",
            "provider.pyi",
            "class CameraProvider: ...\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.d.ts",
            "declare class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider function",
            "provider.d.ts",
            "declare function CameraProvider(): void;\n",
            r"function\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider class",
            "provider.ts",
            "declare class CameraProvider {}\n",
            r"class\s+CameraProvider",
        ),
        (
            "MUST define a CameraProvider interface",
            "provider.ts",
            "interface CameraProvider {}\n",
            r"interface\s+CameraProvider",
        ),
    ],
    ids=[
        "python-stub",
        "typescript-stub-class",
        "typescript-stub-function",
        "ambient",
        "erased-interface",
    ],
)
async def test_declaration_only_sources_cannot_reach_formal_pass(
    tmp_path: Any,
    ac_text: str,
    filename: str,
    content: str,
    pattern: str,
) -> None:
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": pattern,
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Declaration-only source is not executable evidence",
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
    "filename",
    ["provider_windows.go", "provider_test.go", "_provider.go", ".provider.go"],
    ids=["goos", "test", "underscore", "dot"],
)
async def test_implicitly_excluded_go_file_cannot_reach_formal_pass(
    tmp_path: Any,
    filename: str,
) -> None:
    ac_text = "MUST define a CameraProvider function"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"func\s+CameraProvider",
                "expected_value": "CameraProvider",
                "file_hint": filename,
                "description": "Implicitly excluded Go source is not evidence",
            }
        ],
    )
    (tmp_path / filename).write_text("func CameraProvider() {}\n")

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
        ("def CameraProvider():\n    pass\n", True),
        ("CameraProvider = object()\n", True),
        ("class Unrelated:\n    pass\n", True),
    ],
    ids=["forbidden-present", "same-name-function", "same-name-variable", "forbidden-absent"],
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
    ("requested_kind", "target"),
    [("file", "README.md"), ("directory", "pkg/cache")],
)
async def test_forbidden_path_inventory_includes_non_source_files_and_directories(
    tmp_path: Any,
    requested_kind: str,
    target: str,
) -> None:
    ac_text = f"MUST NOT create {requested_kind} {target}"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": re.escape(target),
                "expected_value": target,
                "file_hint": "safe.py",
                "description": f"The {requested_kind} is forbidden",
            }
        ],
    )
    forbidden_path = tmp_path / target
    forbidden_path.parent.mkdir(parents=True, exist_ok=True)
    if requested_kind == "directory":
        forbidden_path.mkdir()
    else:
        forbidden_path.write_text("documentation\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert result.verified is False
    assert result.evidence_target == target
    assert f"Forbidden {requested_kind}" in result.detail
    assert formal.final_approved is False


@pytest.mark.asyncio
async def test_forbidden_declaration_ignores_declaration_like_filename(tmp_path: Any) -> None:
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
    (tmp_path / "CameraProvider.py").write_text("VALUE = 1\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    assert verification.reports[0].verified_pass is True
    assert formal.final_approved is True
    assert formal.ac_results[0].final_verdict == "pass"


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
    ("requested_kind", "actual_kind", "approved"),
    [
        ("file", "file", True),
        ("file", "directory", False),
        ("directory", "directory", True),
        ("directory", "file", False),
    ],
)
async def test_filename_evidence_binds_the_requested_filesystem_kind(
    tmp_path: Any,
    requested_kind: str,
    actual_kind: str,
    approved: bool,
) -> None:
    ac_text = f"MUST create {requested_kind} pkg/artifact"
    assertions = await _extract(
        ac_text,
        [
            {
                "ac_index": 0,
                "tier": "t2_structural",
                "pattern": r"pkg/artifact",
                "expected_value": "pkg/artifact",
                "file_hint": "**/*",
                "description": f"Find the requested {requested_kind}",
            }
        ],
    )
    artifact = tmp_path / "pkg" / "artifact"
    artifact.parent.mkdir()
    if actual_kind == "directory":
        artifact.mkdir()
    else:
        artifact.write_text("contents\n")

    verification = SpecVerifier(str(tmp_path)).verify_all(assertions)
    formal = _formal_verdict(ac_text, verification)

    result = verification.reports[0].results[0]
    assert verification.reports[0].verified_pass is approved
    assert formal.final_approved is approved
    assert result.evidence_source == ("filename" if approved else "")
    if approved:
        assert result.detail.startswith(f"Found {requested_kind}:")


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
