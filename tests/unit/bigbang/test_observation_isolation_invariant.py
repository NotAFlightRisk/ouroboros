"""The observation-isolation invariant, pinned as a population rather than probes.

Rounds 73-82 found six entrances to the requirement-authority surfaces one
round at a time: deterministic candidate promotion, the in-process extractor,
the PM extractor, the plugin extractor, generated questions, the
oversized-context summary, and the initial-goal candidate. Each fix answered
that round's probe; the next round found the complement.

This module pins the POPULATION instead. The enumeration criterion: every
consumer of interview text (``state.rounds`` / ``state.initial_context``)
whose output feeds requirement extraction or deterministic requirement
promotion. Those are exactly the three extraction formatters and the
distillation builder — the conversational surfaces (question generation,
advisory prompts) deliberately see observations and are OUT of scope by the
round-75 distinction. Adding a new consumer of interview text that feeds an
extractor without registering it here should feel wrong in review; that is
this file's job.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ouroboros.bigbang.interview import (
    INITIAL_CONTEXT_SUMMARY_QUESTION,
    MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS,
    InterviewRound,
    InterviewState,
)
from ouroboros.bigbang.pm_interview import PMInterviewEngine
from ouroboros.bigbang.requirement_distillation import build_requirement_distillation
from ouroboros.bigbang.seed_generator import SeedGenerator
from ouroboros.mcp.tools.authoring_handlers import _format_extraction_transcript

_OBSERVATION = "Confirmed: 42 enterprise accounts require SSO today."
_DECISION = "Enterprise tier must include SSO."


def _dev_context(state: InterviewState) -> str:
    return SeedGenerator.__new__(SeedGenerator)._build_interview_context(state)


def _pm_context(state: InterviewState) -> str:
    return PMInterviewEngine.__new__(PMInterviewEngine)._build_interview_context(state)


def _pm_document_context(state: InterviewState) -> str:
    """The PM document prompt surface (round-84: found OUTSIDE this table).

    PMDocumentGenerator reconstructs raw Q&A for a prompt that instructs the
    LLM to preserve transcript information in a durable requirements
    document — an extraction surface by this file's own criterion, missed in
    the first enumeration. Its sanitizer is pinned here so the miss cannot
    recur silently.
    """
    from ouroboros.bigbang.pm_document import extraction_safe_qa_pairs

    pairs = [(r.question, r.user_response or "") for r in state.rounds]
    if state.initial_context:
        pairs = [("What is the initial context?", state.initial_context), *pairs]
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in extraction_safe_qa_pairs(pairs))


#: The extraction surfaces. THE enumeration — a new extractor belongs here.
_EXTRACTORS: list[tuple[str, Callable[[InterviewState], str]]] = [
    ("dev", _dev_context),
    ("pm", _pm_context),
    ("plugin", _format_extraction_transcript),
    ("pm-document", _pm_document_context),
]


def _rounds(*pairs: tuple[str, str | None]) -> list[InterviewRound]:
    return [
        InterviewRound(round_number=index + 1, question=question, user_response=answer)
        for index, (question, answer) in enumerate(pairs)
    ]


def _entrance_states() -> list[tuple[str, InterviewState]]:
    """One state per known entrance, each carrying the observation."""
    states: list[tuple[str, InterviewState]] = []

    states.append(
        (
            "round-answer",
            InterviewState(
                interview_id="iv_inv_a",
                initial_context="Build the reporting lane",
                rounds=_rounds(
                    ("What did the data show?", f"[from-data] {_OBSERVATION}"),
                    ("What must the product guarantee?", _DECISION),
                ),
            ),
        )
    )
    states.append(
        (
            "tainted-question",
            InterviewState(
                interview_id="iv_inv_b",
                initial_context="Build the reporting lane",
                rounds=_rounds(
                    ("What did the data show?", f"[from-research] {_OBSERVATION}"),
                    (f"Given that {_OBSERVATION}, what tier?", _DECISION),
                ),
            ),
        )
    )
    states.append(
        (
            "summary-answer",
            InterviewState(
                interview_id="iv_inv_c",
                initial_context="x" * (MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS + 1),
                rounds=_rounds(
                    (INITIAL_CONTEXT_SUMMARY_QUESTION, f"[from-data] {_OBSERVATION}"),
                    ("What must the product guarantee?", _DECISION),
                ),
            ),
        )
    )
    states.append(
        (
            "initial-context",
            InterviewState(
                interview_id="iv_inv_d",
                initial_context=f"[from-data] {_OBSERVATION}",
                rounds=_rounds(("What must the product guarantee?", _DECISION)),
            ),
        )
    )
    return states


@pytest.mark.parametrize(("surface", "build"), _EXTRACTORS, ids=[s for s, _ in _EXTRACTORS])
@pytest.mark.parametrize(
    ("entrance", "state"), _entrance_states(), ids=[e for e, _ in _entrance_states()]
)
def test_no_extraction_surface_receives_the_observation(
    surface: str,
    build: Callable[[InterviewState], str],
    entrance: str,
    state: InterviewState,
) -> None:
    """(surface x entrance): the observation never reaches an extractor.

    The user's own decision always survives — withholding that costs nothing
    where there is nothing to withhold is only half the invariant.
    """
    context = build(state)

    assert "42 enterprise accounts" not in context, (surface, entrance)
    assert _DECISION in context, (surface, entrance)


@pytest.mark.parametrize(
    ("entrance", "state"), _entrance_states(), ids=[e for e, _ in _entrance_states()]
)
def test_no_deterministic_candidate_carries_the_observation(
    entrance: str,
    state: InterviewState,
) -> None:
    """The deterministic surface: no promoted candidate text carries it."""
    distillation = build_requirement_distillation(state)

    for candidate in distillation.candidates:
        assert "42 enterprise accounts" not in candidate.text, (entrance, candidate.candidate_id)
