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
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ouroboros.bigbang.interview import (
    INITIAL_CONTEXT_SUMMARY_QUESTION,
    MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS,
    InterviewRound,
    InterviewState,
    InterviewStatus,
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
    from ouroboros.bigbang.interview import prompt_safe_initial_context_with_provenance
    from ouroboros.bigbang.pm_document import extraction_safe_qa_pairs

    pairs = [(r.question, r.user_response or "") for r in state.rounds]
    provenances = [r.answer_provenance for r in state.rounds]
    if state.initial_context:
        # Any inclusion of interview context in a durable prompt goes
        # through the AUTHORITATIVE value with its provenance (round-92) —
        # the raw oversized blob can carry the observation mid-text.
        context, context_provenance = prompt_safe_initial_context_with_provenance(state)
        pairs = [("What is the initial context?", context), *pairs]
        provenances = [context_provenance, *provenances]
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in extraction_safe_qa_pairs(pairs, provenances))


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
            # Provenance as a FIELD, marker stripped from the text (round-85):
            # in-band markers can be lost to paraphrase or a forgetful relay;
            # the ingestion-time field withholds regardless.
            "field-only",
            InterviewState(
                interview_id="iv_inv_e",
                initial_context="Build the reporting lane",
                rounds=[
                    InterviewRound(
                        round_number=1,
                        question="What did the data show?",
                        user_response=_OBSERVATION,
                        answer_provenance="data_fact",
                    ),
                    InterviewRound(
                        round_number=2,
                        question="What must the product guarantee?",
                        user_response=_DECISION,
                    ),
                ],
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
    states.append(
        (
            # Round-92: the observation sits MID-TEXT in an oversized raw
            # context where no marker leads — only the authoritative summary
            # substitution keeps it out, and the plugin transcript rendered
            # the raw blob. A human decision round keeps the readiness gate
            # out of the picture: this pins the extraction surface itself.
            "oversized-embedded",
            InterviewState(
                interview_id="iv_inv_g",
                initial_context=(
                    "Background notes. "
                    + _OBSERVATION
                    + " "
                    + "x" * MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS
                ),
                rounds=[
                    InterviewRound(
                        round_number=1,
                        question=INITIAL_CONTEXT_SUMMARY_QUESTION,
                        user_response="Reporting-lane context, observation adopted separately.",
                        answer_provenance="data_fact",
                    ),
                    InterviewRound(
                        round_number=2,
                        question="What must the product guarantee?",
                        user_response=_DECISION,
                    ),
                ],
            ),
        )
    )
    states.append(
        (
            # Round-90: the oversized-context SUMMARY substitute is a round
            # answer whose typed field is the record — a markerless summary
            # typed data_fact reached the Dev and PM extractors because both
            # sanitized the substitute by marker only.
            "summary-field-only",
            InterviewState(
                interview_id="iv_inv_f",
                initial_context="x" * (MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS + 1),
                rounds=[
                    InterviewRound(
                        round_number=1,
                        question=INITIAL_CONTEXT_SUMMARY_QUESTION,
                        user_response=_OBSERVATION,
                        answer_provenance="data_fact",
                    ),
                    InterviewRound(
                        round_number=2,
                        question="What must the product guarantee?",
                        user_response=_DECISION,
                    ),
                ],
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


# --------------------------------------------------------------------------- #
# The generation gates — the second population (round-85).
#
# Withholding (above) empties the extractor's input; the gates refuse to run
# the extractor at all when NOTHING contentful would remain. Round-85 found
# the gate present on one path (reference-aware distillation), absent on the
# plain dev path, bypassed by the plugin path, and never built for PM. The
# enumeration criterion: every entry point that turns an InterviewState into
# a requirement artifact (Seed, plugin Seed dispatch, PMSeed). A new
# generator belongs in this table.
# --------------------------------------------------------------------------- #


def _observation_only_state() -> InterviewState:
    return InterviewState(
        interview_id="iv_inv_gate",
        initial_context=f"[from-data] {_OBSERVATION}",
        rounds=_rounds(("What did the data show?", f"[from-data] {_OBSERVATION}")),
        status=InterviewStatus.COMPLETED,
    )


async def _dev_generate(state: InterviewState, tmp_path: Path) -> object:
    from ouroboros.bigbang.ambiguity import AmbiguityScore

    generator = SeedGenerator(llm_adapter=AsyncMock(), output_dir=tmp_path / "seeds")
    return await generator.generate(state, AmbiguityScore(overall_score=0.1, breakdown=None))


async def _plugin_generate(state: InterviewState, tmp_path: Path) -> object:
    from unittest.mock import patch

    from ouroboros.core.types import Result
    from ouroboros.mcp.tools.authoring_handlers import GenerateSeedHandler

    handler = GenerateSeedHandler(agent_runtime_backend="opencode", opencode_mode="plugin")
    with patch(
        "ouroboros.mcp.tools.authoring_handlers._plugin_load_state",
        AsyncMock(return_value=Result.ok(state)),
    ):
        return await handler.handle({"session_id": state.interview_id})


async def _pm_generate(state: InterviewState, tmp_path: Path) -> object:
    from unittest.mock import MagicMock

    from ouroboros.bigbang.pm_interview import PMInterviewEngine
    from ouroboros.bigbang.question_classifier import QuestionClassifier

    adapter = MagicMock()
    adapter.complete = AsyncMock()
    engine = PMInterviewEngine(
        inner=MagicMock(),
        classifier=QuestionClassifier(llm_adapter=adapter),
        llm_adapter=adapter,
    )
    return await engine.generate_pm_seed(state)


async def _pm_plugin_generate(state: InterviewState, tmp_path: Path) -> object:
    from unittest.mock import patch

    from ouroboros.core.types import Result
    from ouroboros.mcp.tools.pm_handler import PMInterviewHandler

    handler = PMInterviewHandler(agent_runtime_backend="opencode", opencode_mode="plugin")
    with patch(
        "ouroboros.mcp.tools.authoring_handlers._plugin_load_state",
        AsyncMock(return_value=Result.ok(state)),
    ):
        return await handler.handle({"session_id": state.interview_id, "action": "generate"})


#: THE enumeration of generation entry points.
_GENERATORS: list[tuple[str, Callable[..., object]]] = [
    ("dev", _dev_generate),
    ("plugin", _plugin_generate),
    ("pm", _pm_generate),
    # Round-93: the PM plugin transport duplicated its own gating and
    # skipped the shared readiness question entirely.
    ("pm-plugin", _pm_plugin_generate),
]


@pytest.mark.parametrize(("path_name", "generate"), _GENERATORS, ids=[n for n, _ in _GENERATORS])
def test_every_generation_path_refuses_an_observation_only_interview(
    path_name: str,
    generate: Callable[..., object],
    tmp_path: Path,
) -> None:
    """(generator): observation-only input is refused with the ONE message.

    The shared message is part of the invariant: three different phrasings
    would be three gates drifting apart again.
    """
    import asyncio

    from ouroboros.bigbang.requirement_distillation import (
        OBSERVATION_ONLY_INTERVIEW_MESSAGE,
    )

    outcome = asyncio.run(generate(_observation_only_state(), tmp_path))

    assert outcome.is_err, path_name
    assert OBSERVATION_ONLY_INTERVIEW_MESSAGE in str(outcome.error), path_name


# ---------------------------------------------------------------------------
# One provenance authority for initial_context
# ---------------------------------------------------------------------------
#
# ``InterviewRound`` decides provenance once at ingestion and carries it as a
# typed field, so consumers read instead of re-parsing. ``initial_context``
# never got that treatment — it enters at session creation, not through
# ``record_response`` — so consumers re-derived it while
# ``prompt_safe_initial_context_with_provenance`` answered a hardcoded
# ``human`` for any context short enough to use as-is. Two authorities, two
# answers, same string. This pins their agreement across the whole marker
# population rather than the one marker a probe happens to use.

_PROVENANCE_BY_MARKER = [
    ("[from-data] Enterprise plan has 412 active accounts.", "data_fact"),
    ("[from-research] The spec was ratified in 2019.", "research_fact"),
    ("[from-code] AuthService reads the session cookie.", "repo_fact"),
    ("[from-auto] Enterprise tier must include SSO.", "generated"),
    ("Build an SSO dashboard for enterprise admins.", "human"),
]


@pytest.mark.parametrize(
    ("context", "expected"),
    _PROVENANCE_BY_MARKER,
    ids=[expected for _, expected in _PROVENANCE_BY_MARKER],
)
def test_initial_context_provenance_authorities_agree(context: str, expected: str) -> None:
    """The state property and the prompt-safe authority report the same class."""
    from ouroboros.bigbang.interview import prompt_safe_initial_context_with_provenance

    state = InterviewState(interview_id="provenance-authority", initial_context=context)

    _, authoritative = prompt_safe_initial_context_with_provenance(state)

    assert state.initial_context_provenance == expected
    assert authoritative == expected


def test_oversized_context_still_reports_the_summary_round_provenance() -> None:
    """Oversized contexts keep deferring to the substitute round's typed field.

    The agreement above must not be won by making the short path authoritative
    everywhere: when a summary stands in for the raw context, the SUBSTITUTE's
    provenance is the record, even though the raw context reads as human.
    """
    from ouroboros.bigbang.interview import prompt_safe_initial_context_with_provenance

    state = InterviewState(
        interview_id="provenance-authority-oversized",
        initial_context="A" * (MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS + 1),
        rounds=[
            InterviewRound(
                round_number=1,
                question=INITIAL_CONTEXT_SUMMARY_QUESTION,
                user_response="Enterprise plan has 412 active accounts.",
                answer_provenance="data_fact",
            )
        ],
    )

    text, provenance = prompt_safe_initial_context_with_provenance(state)

    assert provenance == "data_fact"
    assert "412" in text
    assert state.initial_context_provenance == "human"


# ---------------------------------------------------------------------------
# The round provenance field is correct whoever built the round
# ---------------------------------------------------------------------------
#
# ``record_response`` is the door and decides provenance from the UNDECORATED
# answer. But the MCP handlers also build rounds directly, and each of those
# sites had to remember to classify — the same "caller must remember" shape
# that let a decorated answer be stamped human. The field now classifies
# itself when the constructor did not decide, so forgetting is no longer
# expressible; an explicit value still wins, which is what keeps the
# decoration case and a user's own observation-quoting answer correct.


def test_round_built_without_provenance_classifies_itself() -> None:
    """A construction site that forgets still gets the right class."""
    round_data = InterviewRound(
        round_number=1,
        question="How many enterprise accounts?",
        user_response=f"{_OBSERVATION}",
    )

    assert round_data.answer_provenance == "human"

    observed = InterviewRound(
        round_number=1,
        question="How many enterprise accounts?",
        user_response="[from-data] 412 enterprise accounts are active.",
    )

    assert observed.answer_provenance == "data_fact"


def test_explicit_provenance_wins_over_the_text() -> None:
    """Decoration and user-authored citation both depend on the explicit value.

    ``PM answer: [from-data] ...`` must stay ``data_fact`` (the decoration
    must not launder it), and a human decision that QUOTES an observation
    must stay ``human`` (the user authored the decision).
    """
    decorated = InterviewRound(
        round_number=1,
        question="Q",
        user_response="PM answer: [from-data] 412 enterprise accounts are active.",
        answer_provenance="data_fact",
    )

    assert decorated.answer_provenance == "data_fact"

    cited = InterviewRound(
        round_number=2,
        question="Q",
        user_response="412 of them, 37 asked for SSO, so this is P0. [from-data] source",
        answer_provenance="human",
    )

    assert cited.answer_provenance == "human"


def test_legacy_round_without_the_field_is_classified_on_load() -> None:
    """A persisted round predating the typed field migrates on deserialization."""
    legacy = InterviewRound.model_validate(
        {
            "round_number": 1,
            "question": "Q",
            "user_response": "[from-research] The spec was ratified in 2019.",
        }
    )

    assert legacy.answer_provenance == "research_fact"


# ---------------------------------------------------------------------------
# The oversized-context substitute inherits what it substitutes for
# ---------------------------------------------------------------------------
#
# The engine hands consumers a user-written summary in place of an oversized
# `initial_context`. That substitute is not an independent answer, so it
# cannot carry more authority than the context it replaces — otherwise the
# same observation is an observation when short and a requirement when long,
# and the difference is a character count.


def _oversized(marker: str, observation: str) -> str:
    return (marker + observation + " " + ("padding. " * 3000)).strip()


_SUBSTITUTE_CASES = [
    ("[from-data] ", "data_fact", True),
    ("[from-research] ", "research_fact", True),
    ("", "human", False),
]


@pytest.mark.parametrize(
    ("marker", "expected", "gated"),
    _SUBSTITUTE_CASES,
    ids=[expected for _, expected, _ in _SUBSTITUTE_CASES],
)
def test_summary_substitute_inherits_context_provenance(
    marker: str, expected: str, gated: bool
) -> None:
    """A markerless summary of an observation is still an observation."""
    import asyncio

    from ouroboros.bigbang.interview import (
        InterviewEngine,
        prompt_safe_initial_context_with_provenance,
    )
    from ouroboros.bigbang.requirement_distillation import (
        interview_has_no_promotable_requirement,
    )

    observation = "Enterprise plan has 412 active accounts and 37 SSO requests."
    state = InterviewState(
        interview_id="substitute-provenance",
        initial_context=_oversized(marker, observation),
    )
    engine = InterviewEngine(llm_adapter=AsyncMock())

    asyncio.run(engine.record_response(state, observation, INITIAL_CONTEXT_SUMMARY_QUESTION))

    _, authoritative = prompt_safe_initial_context_with_provenance(state)

    assert state.rounds[-1].answer_provenance == expected
    assert authoritative == expected
    assert interview_has_no_promotable_requirement(state) is gated


def test_a_summary_may_still_declare_a_stronger_class_than_its_context() -> None:
    """Inheritance raises authority-withholding, it never lowers it.

    A human-authored context summarized with an explicit `[from-data]` answer
    stays an observation: the substitute's own marker is not overridden by
    the milder class of what it replaces.
    """
    import asyncio

    from ouroboros.bigbang.interview import InterviewEngine

    state = InterviewState(
        interview_id="substitute-stronger",
        initial_context=_oversized("", "Build an SSO dashboard for enterprise admins."),
    )
    engine = InterviewEngine(llm_adapter=AsyncMock())

    asyncio.run(
        engine.record_response(
            state,
            "[from-data] Enterprise plan has 412 active accounts.",
            INITIAL_CONTEXT_SUMMARY_QUESTION,
        )
    )

    assert state.rounds[-1].answer_provenance == "data_fact"


def test_inheritance_is_scoped_to_the_summary_question() -> None:
    """An ordinary answer in an observation-context interview stays human.

    The substitute rule is about standing in for `initial_context`. A normal
    round is the user's own decision and keeps its own class, which is what
    lets a withheld interview become generatable again.
    """
    import asyncio

    from ouroboros.bigbang.interview import InterviewEngine

    state = InterviewState(
        interview_id="substitute-scope",
        initial_context="[from-data] Enterprise plan has 412 active accounts.",
    )
    engine = InterviewEngine(llm_adapter=AsyncMock())

    asyncio.run(
        engine.record_response(
            state,
            "Build an SSO dashboard for enterprise admins.",
            "What should we build?",
        )
    )

    assert state.rounds[-1].answer_provenance == "human"


# ---------------------------------------------------------------------------
# Consumers render the isolated transcript; they do not compute it
# ---------------------------------------------------------------------------
#
# The enumeration this file opens with named the extraction surfaces. It could
# not stop a NEW one from being written without the rule, because the rule
# lived in each surface: three byte-identical copies of the walk plus a fourth
# in pair form, and round-84 found one of them a round late. There is one walk
# now, so a surface that renders `extraction_safe_transcript` inherits the
# rule and a surface that does not cannot see an observation to leak.

_ISOLATION_INTERNALS = (
    "extraction_safe_answer",
    "extraction_safe_question",
    "observation_seen",
    "OBSERVATION_WITHHELD_NOTE",
    "QUESTION_WITHHELD_NOTE",
)

_EXTRACTION_SURFACES = (
    "src/ouroboros/bigbang/seed_generator.py",
    "src/ouroboros/bigbang/pm_interview.py",
    "src/ouroboros/bigbang/pm_document.py",
    "src/ouroboros/mcp/tools/authoring_handlers.py",
)


@pytest.mark.parametrize("module_path", _EXTRACTION_SURFACES)
def test_extraction_surface_does_not_reimplement_isolation(module_path: str) -> None:
    """No extraction surface names the isolation internals."""
    source = Path(module_path).read_text(encoding="utf-8")

    leaked = [name for name in _ISOLATION_INTERNALS if name in source]

    assert leaked == [], (
        f"{module_path} names {leaked}: render extraction_safe_transcript "
        f"(or isolate_observation_pairs for explicit pairs) instead of "
        f"re-applying the rule."
    )


def test_the_isolation_walk_has_exactly_one_implementation() -> None:
    """`observation_seen` is advanced in one place in the whole tree."""
    import subprocess

    hits = subprocess.run(
        ["grep", "-rn", "observation_seen = True", "src/ouroboros/"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()

    assert len(hits) == 1, f"expected one isolation walk, found {len(hits)}:\n" + "\n".join(hits)


def test_the_view_still_withholds_what_the_copies_withheld() -> None:
    """An observation answer is withheld and taints the questions after it."""
    from ouroboros.bigbang.interview import extraction_safe_transcript

    state = InterviewState(
        interview_id="view-withholds",
        initial_context="Build an SSO dashboard for enterprise admins.",
        rounds=[
            InterviewRound(
                round_number=1,
                question="How many enterprise accounts are there?",
                user_response=f"[from-data] {_OBSERVATION}",
            ),
            InterviewRound(
                round_number=2,
                question=f"Given {_OBSERVATION}, should SSO be P0?",
                user_response="Yes, make it P0.",
            ),
        ],
    )

    transcript = extraction_safe_transcript(state)
    rendered = "\n".join(f"{item.question}\n{item.answer or ''}" for item in transcript.rounds)

    assert _OBSERVATION not in rendered
    assert "Yes, make it P0." in rendered
    assert transcript.rounds[0].question == "How many enterprise accounts are there?"


# ---------------------------------------------------------------------------
# A round filled after construction still gets its provenance
# ---------------------------------------------------------------------------
#
# The MCP transports emit a question, persist a round carrying only that
# question, and fill the answer on a later call. Provenance is derived at
# construction and pydantic does not re-validate on assignment, so assigning
# `user_response` afterwards left the field at its `human` default while the
# text said otherwise. Both fill sites re-stamped it by hand — the shape that
# a third fill site would have to remember.


def test_filling_a_placeholder_round_decides_provenance_with_the_answer() -> None:
    """The field cannot stay `human` by omission on the fill path."""
    state = InterviewState(
        interview_id="fill-provenance",
        rounds=[InterviewRound(round_number=1, question="How many accounts?")],
    )

    assert state.rounds[-1].answer_provenance == "human"

    state.fill_pending_answer("[from-data] Enterprise plan has 412 active accounts.")

    assert state.rounds[-1].answer_provenance == "data_fact"


def test_filling_a_placeholder_applies_the_summary_substitute_rule() -> None:
    """The fill path uses the door's rule, inheritance included."""
    state = InterviewState(
        interview_id="fill-summary",
        initial_context=_oversized("[from-data] ", _OBSERVATION),
        rounds=[InterviewRound(round_number=1, question=INITIAL_CONTEXT_SUMMARY_QUESTION)],
    )

    state.fill_pending_answer(_OBSERVATION)

    assert state.rounds[-1].answer_provenance == "data_fact"


def test_filling_a_placeholder_replaces_a_stale_question() -> None:
    """A stale placeholder question is replaced, and taint follows the new one."""
    state = InterviewState(
        interview_id="fill-question",
        rounds=[InterviewRound(round_number=1, question="(continued from subagent)")],
    )

    state.fill_pending_answer("Build an SSO dashboard.", question="What should we build?")

    assert state.rounds[-1].question == "What should we build?"
    assert state.rounds[-1].answer_provenance == "human"


def test_no_transport_stamps_provenance_next_to_a_manual_answer_assignment() -> None:
    """Fill paths go through the state, not through paired assignments.

    Two sites assigned `user_response` and `answer_provenance` in sequence and
    stayed correct only because both remembered. Pinning their absence is what
    stops a third from being added the same way.
    """
    for module_path in (
        "src/ouroboros/mcp/tools/authoring_handlers.py",
        "src/ouroboros/mcp/tools/pm_handler.py",
        "src/ouroboros/cli/commands/pm.py",
        "src/ouroboros/cli/commands/init.py",
    ):
        source = Path(module_path).read_text(encoding="utf-8")
        assert ".answer_provenance = " not in source, (
            f"{module_path} stamps provenance by assignment: use "
            f"InterviewState.fill_pending_answer so the answer and its "
            f"provenance are decided together."
        )
