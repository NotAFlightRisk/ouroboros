"""Finite atomic-attempt budget contracts."""

from __future__ import annotations

import math

import pytest

from ouroboros.core.attempt_budget import (
    AttemptBudgetExhaustion,
    AttemptBudgetKind,
    validate_attempt_budget,
)


def test_validate_attempt_budget_preserves_exact_values() -> None:
    assert validate_attempt_budget(
        max_iterations_per_ac=7,
        timeout_seconds=12.5,
    ) == (7, 12.5)


@pytest.mark.parametrize(
    ("steps", "timeout"),
    ((0, 1.0), (True, 1.0), (1, 0.0), (1, math.inf), (1, True)),
)
def test_validate_attempt_budget_rejects_unbounded_or_ambiguous_values(
    steps: object,
    timeout: object,
) -> None:
    with pytest.raises(ValueError):
        validate_attempt_budget(  # type: ignore[arg-type]
            max_iterations_per_ac=steps,
            timeout_seconds=timeout,
        )


def test_attempt_budget_exhaustion_rejects_invalid_measurement() -> None:
    with pytest.raises(ValueError):
        AttemptBudgetExhaustion(
            kind=AttemptBudgetKind.WALL_CLOCK,
            limit=1,
            observed=math.nan,
        )
