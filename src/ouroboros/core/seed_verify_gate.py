"""Deterministic gate over acceptance criteria that nothing can verify.

An AC carrying a ``verify_command`` is judged by the orchestrator running that
command itself (``_run_ac_verify_gate``): immune both to a worker grading its
own homework and to a lost transcript. An AC without one falls back to matching
the worker's typed evidence against its runtime transcript — the only path that
can be gamed, and the only path that collapses when transcript collection
fails.

So the gate pushes criteria toward ``verify_command`` and asks for an explicit
per-AC reason when that is genuinely not feasible. It is deliberately staged:
``warn`` surfaces violations without changing behavior, ``block`` refuses the
run. Nothing here inspects or rewrites the command — it only asks whether a
deterministic judgment exists at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from ouroboros.core.seed import AcceptanceCriterionSpec, Seed

_MAX_RENDERED_DESCRIPTION_CHARS = 80


@dataclass(frozen=True, slots=True)
class UnverifiableCriterion:
    """One AC the orchestrator has no way to decide.

    ``vacuous`` separates the two ways that happens. A criterion with no
    command at all falls back to transcript evidence, which is weak but real.
    A criterion whose command cannot fail is worse than either: it reports a
    pass unconditionally *and* the mere declaration drops ``commands_run`` and
    ``tests_passed`` from the required evidence, so the gate built to make
    self-reporting impossible becomes the way around it.
    """

    ac_index: int
    description: str
    vacuous: bool = False
    verify_command: str | None = None

    @property
    def display_index(self) -> int:
        """1-based position as operators see it in reports."""
        return self.ac_index + 1


def unverifiable_criteria(seed: Seed) -> tuple[UnverifiableCriterion, ...]:
    """Return criteria the orchestrator has no deterministic way to judge.

    Bare-string criteria count: they carry neither a command nor a reason.
    ``expected_artifacts`` alone does not exempt a criterion — a file existing
    is not the same claim as the criterion being satisfied.
    """
    findings: list[UnverifiableCriterion] = []
    for index, criterion in enumerate(seed.acceptance_criteria):
        if isinstance(criterion, AcceptanceCriterionSpec):
            if criterion.verify_contract_always_passes:
                findings.append(
                    UnverifiableCriterion(
                        ac_index=index,
                        description=criterion.description,
                        vacuous=True,
                        verify_command=criterion.verify_command,
                    )
                )
                continue
            # An exemption reason excuses the absence of a command, never a
            # command that cannot fail — that is not an admission, it is a
            # claim of verification that isn't one.
            if criterion.verify_command or criterion.verify_exemption_reason:
                continue
            description = criterion.description
        else:
            description = str(criterion).strip()
        findings.append(UnverifiableCriterion(ac_index=index, description=description))
    return tuple(findings)


def render_verify_command_gate_warning(
    findings: tuple[UnverifiableCriterion, ...],
) -> str:
    """Render the operator-facing warning for a warn-stage gate result."""
    if not findings:
        return ""
    vacuous = [finding for finding in findings if finding.vacuous]
    missing = [finding for finding in findings if not finding.vacuous]
    lines: list[str] = []

    def render(finding: UnverifiableCriterion) -> str:
        description = finding.description
        if len(description) > _MAX_RENDERED_DESCRIPTION_CHARS:
            description = description[: _MAX_RENDERED_DESCRIPTION_CHARS - 1] + "…"
        return f"  - AC {finding.display_index}: {description}"

    if vacuous:
        lines.append(
            f"{len(vacuous)} acceptance criteria declare a verify_command that cannot fail, "
            "so it reports a pass no matter what happened:"
        )
        lines.extend(f"{render(finding)}  [{finding.verify_command}]" for finding in vacuous)
        lines.append(
            "Replace it with a command whose exit status depends on the work, "
            "or drop it and give a verify_exemption_reason."
        )
    if missing:
        if lines:
            lines.append("")
        lines.append(
            f"{len(missing)} acceptance criteria declare no verify_command, so they can only "
            "be judged from the worker's own transcript:"
        )
        lines.extend(render(finding) for finding in missing)
        lines.append(
            "Add a verify_command, or a verify_exemption_reason saying why one is not feasible."
        )
    return "\n".join(lines)


def verify_command_gate_mode() -> str:
    """Return the configured gate mode, defaulting to the non-blocking stage.

    A missing or unreadable config must not turn into a hard run refusal, so
    resolution failures fall back to ``warn``.
    """
    try:
        from ouroboros.config.loader import load_config

        return load_config().seed.verify_command_gate
    except Exception:
        return "warn"
