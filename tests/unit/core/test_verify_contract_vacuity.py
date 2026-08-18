"""A verify_command that cannot fail must not certify an AC.

`verify_command: "exit 0"` is the sharp case. It does two things at once: the
orchestrator runs it, it exits 0, and the AC is recorded as independently
verified — and merely *declaring* a verify_command drops `commands_run` and
`tests_passed` from the required evidence. So a contract that always passes
leaves an AC checked *less* than one that declared no contract at all, turning
the gate built to make self-reporting impossible into the way around it.

The line drawn here is the sign of the constant, not its presence: a contract
that can only ever refuse (`exit 1`) is information-free too, but it cannot
manufacture a pass, so it is left alone to fail honestly.
"""

from __future__ import annotations

import pytest

from ouroboros.core.seed import AcceptanceCriterionSpec
from ouroboros.core.verify_command_plan import constant_verdict


@pytest.mark.parametrize(
    "command",
    ["exit 0", "true", ":", "echo done", "exit 0 && pytest -q", "false || true"],
)
def test_always_passing_contracts_are_proven(command: str) -> None:
    assert constant_verdict(command) is True
    spec = AcceptanceCriterionSpec(description="d", verify_command=command)
    assert spec.verify_contract_always_passes is True


@pytest.mark.parametrize("command", ["exit 1", "false", "echo ok && exit 3"])
def test_always_failing_contracts_are_left_alone(command: str) -> None:
    """They cannot fake a pass, so the gate should still run and refuse."""
    assert constant_verdict(command) is False
    spec = AcceptanceCriterionSpec(description="d", verify_command=command)
    assert spec.verify_contract_always_passes is False
    assert spec.has_evidential_success_contract is True


@pytest.mark.parametrize(
    "command",
    ["pytest -q", "test -f out.txt", "uv run train.py", 'python -c "import app"', "grep -q ok f"],
)
def test_real_checks_are_never_accused(command: str) -> None:
    """The check may only ever accuse a command it is certain about."""
    assert constant_verdict(command) is None
    spec = AcceptanceCriterionSpec(description="d", verify_command=command)
    assert spec.verify_contract_always_passes is False


def test_output_assertion_participates_in_the_verdict() -> None:
    # Constant output that satisfies its own assertion is still a foregone pass.
    assert constant_verdict("printf READY", "READY") is True
    # Constant output that never satisfies it can only refuse.
    assert constant_verdict("printf WAITING", "READY") is False


def test_always_passing_contract_stops_buying_the_evidence_exemption() -> None:
    vacuous = AcceptanceCriterionSpec(description="d", verify_command="exit 0")

    # Still declared, so the field round-trips through the persisted Seed.
    assert vacuous.has_success_contract is True
    assert vacuous.to_seed_value()["verify_command"] == "exit 0"
    # But it buys nothing: the AC is judged like one with no contract.
    assert vacuous.has_evidential_success_contract is False


def test_expected_artifacts_keep_a_contract_alive() -> None:
    """A vacuous command does not void a criterion's other evidence."""
    spec = AcceptanceCriterionSpec(
        description="d",
        verify_command="exit 0",
        expected_artifacts=("report.md",),
    )

    assert spec.verify_contract_always_passes is True
    assert spec.has_evidential_success_contract is True


def test_a_malformed_assertion_only_contract_still_reaches_the_gate() -> None:
    """An assertion with no command must be rejected, not skipped."""
    malformed = AcceptanceCriterionSpec.model_construct(
        description="d",
        verify_command=None,
        expected_artifacts=(),
        output_assertion="READY",
    )

    assert malformed.has_evidential_success_contract is True


def test_unparseable_commands_are_not_accused() -> None:
    assert constant_verdict("echo $(date)") is None
    assert constant_verdict("echo ok | tee log") is None


@pytest.mark.parametrize(
    "command",
    [
        "exit 256",
        "exit -0",
        "exit +0",
        "exit 512",
        "bash -c 'exit 0'",
        'sh -c "true"',
        "bash -lc 'exit 0'",
        "bash -c 'exit 256'",
        "command true",
        "command exit 0",
    ],
)
def test_disguised_constants_are_still_proven(command: str) -> None:
    """POSIX truncation, `command` prefixes, and literal shell wrappers are the
    same constant wearing a costume; the proof must see through all of them."""
    assert constant_verdict(command) is True
    spec = AcceptanceCriterionSpec(description="d", verify_command=command)
    assert spec.has_evidential_success_contract is False


@pytest.mark.parametrize("command", ["exit -1", "bash -c 'exit 1'", "sh -c 'false'"])
def test_disguised_constant_failures_stay_left_alone(command: str) -> None:
    assert constant_verdict(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'pytest -q'",
        "sh -c 'test -f out.txt'",
        "bash -xc 'exit 0'",
        "command -v pytest",
        "bash -c 'exit 0' extra-arg",
    ],
)
def test_wrappers_that_could_do_real_work_stay_unproven(command: str) -> None:
    assert constant_verdict(command) is not True
