"""Tests for generic reference-aware requirement distillation and filtering."""

from unittest.mock import AsyncMock

import pytest

from ouroboros.bigbang.ambiguity import AmbiguityScore, ComponentScore, ScoreBreakdown
from ouroboros.bigbang.interview import InterviewRound, InterviewState
from ouroboros.bigbang.requirement_distillation import (
    apply_requirement_distillation,
    build_promoted_reference_seed,
    build_requirement_distillation,
)
from ouroboros.bigbang.seed_generator import SeedGenerator
from ouroboros.core.requirement_candidate import (
    CandidateContentSource,
    CandidateResolution,
    ConfirmationAuthority,
    RequirementCandidate,
    RequirementDistillation,
    RequirementEvidence,
    RequirementEvidenceKind,
    RequirementSection,
)
from ouroboros.core.types import Result
from ouroboros.interview_adapters import (
    InterviewTurnContext,
    ReferenceContrastResolution,
    ReferenceCue,
    ReferenceOrigin,
    ReferenceResolutionStatus,
    build_reference_contrast_question,
)
from ouroboros.providers.base import CompletionResponse, UsageInfo


def _reference_state(*, confirmation: str | None = None) -> InterviewState:
    cue = ReferenceCue(
        reference_id="linear",
        label="Linear-like",
        origin=ReferenceOrigin.USER_TEXT,
    )
    contrast_question = build_reference_contrast_question(cue)
    rounds = [
        InterviewRound(
            round_number=1,
            question="What outcome matters most?",
            user_response="Fast issue triage.",
        ),
        InterviewRound(
            round_number=2,
            question=contrast_question,
            user_response="Copy the workflow speed, not the command menu.",
        ),
    ]
    if confirmation:
        rounds.append(
            InterviewRound(
                round_number=3,
                question="Which reference traits are actual requirements?",
                user_response=confirmation,
            )
        )
    return InterviewState(
        interview_id="reference-test",
        initial_context="Build a Linear-like issue tool",
        rounds=rounds,
        reference_cues=(cue,),
        reference_resolutions=(
            ReferenceContrastResolution(
                reference_id="linear",
                status=ReferenceResolutionStatus.RESOLVED,
                asked_question=contrast_question,
                answer="Copy the workflow speed, not the command menu.",
            ),
        ),
    )


def _requirements() -> dict[str, object]:
    return {
        "goal": "Build an issue tool",
        "constraints": "Python",
        "acceptance_criteria": (
            "Keyboard-first command menu | Queue navigation | Fast issue triage"
        ),
        "ontology_name": "IssueTool",
        "ontology_description": "Issue workflow",
    }


def _low_ambiguity() -> AmbiguityScore:
    return AmbiguityScore(
        overall_score=0.1,
        breakdown=ScoreBreakdown(
            goal_clarity=ComponentScore(
                name="Goal",
                clarity_score=0.9,
                weight=0.4,
                justification="clear",
            ),
            constraint_clarity=ComponentScore(
                name="Constraints",
                clarity_score=0.9,
                weight=0.3,
                justification="clear",
            ),
            success_criteria_clarity=ComponentScore(
                name="Success",
                clarity_score=0.9,
                weight=0.3,
                justification="clear",
            ),
        ),
    )


def _extraction_response() -> CompletionResponse:
    return CompletionResponse(
        content="""GOAL: Build an issue tool
CONSTRAINTS: ["Python"]
ACCEPTANCE_CRITERIA:
AC: Keyboard-first command menu | verify: NONE | artifacts: NONE | expect: NONE
AC: Queue navigation | verify: NONE | artifacts: NONE | expect: NONE
ONTOLOGY_NAME: IssueTool
ONTOLOGY_DESCRIPTION: Issue workflow
ONTOLOGY_FIELDS: issue:string:Issue
EVALUATION_PRINCIPLES: correctness:Requirements are met:1.0
EXIT_CONDITIONS: done:All criteria met:All criteria pass
PROJECT_TYPE: greenfield""",
        model="test-model",
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def test_reference_contrast_does_not_promote_inferred_acceptance_criteria() -> None:
    distillation = build_requirement_distillation(_reference_state())

    applied = apply_requirement_distillation(_requirements(), distillation)

    assert applied.promotion.is_ready_for_seed
    assert applied.requirements["acceptance_criteria"] == ()


def test_unresolved_reference_blocks_seed_generation() -> None:
    cue = ReferenceCue(
        reference_id="linear",
        label="Linear-like",
        origin=ReferenceOrigin.USER_TEXT,
    )
    state = InterviewState(
        interview_id="unresolved-reference",
        initial_context="Build an issue tool",
        rounds=[
            InterviewRound(
                round_number=1,
                question="What outcome matters most?",
                user_response="Fast issue triage.",
            )
        ],
        reference_cues=(cue,),
    )

    applied = apply_requirement_distillation({}, build_requirement_distillation(state))

    assert not applied.promotion.is_ready_for_seed
    blocker = applied.promotion.blockers[0]
    assert blocker.candidate.content_source is CandidateContentSource.REFERENCE_DERIVED
    assert blocker.candidate.resolution is CandidateResolution.UNKNOWN
    assert blocker.reason == "required_unknown"


def test_explicit_reference_confirmation_promotes_exact_user_statement() -> None:
    state = _reference_state(
        confirmation="For the Linear-like reference, keyboard-first navigation is required."
    )
    distillation = build_requirement_distillation(state)

    applied = apply_requirement_distillation(_requirements(), distillation)

    assert applied.promotion.is_ready_for_seed
    assert applied.requirements["acceptance_criteria"] == (
        "For the Linear-like reference, keyboard-first navigation is required.",
    )
    promoted = [
        candidate
        for candidate in applied.promotion.promoted
        if candidate.candidate_id == "round-3:requirement"
    ][0]
    assert promoted.content_source.value == "reference_derived"
    assert promoted.confirmation_authority.value == "user"


@pytest.mark.parametrize(
    "confirmation",
    [
        "확인된 요구사항은 키보드만으로 탐색할 수 있어야 한다는 것입니다.",
        "確認済みの要件は、キーボードだけで移動できることです。",
    ],
)
def test_non_english_confirmation_after_reference_contrast_is_preserved(
    confirmation: str,
) -> None:
    distillation = build_requirement_distillation(_reference_state(confirmation=confirmation))

    applied = apply_requirement_distillation(_requirements(), distillation)

    assert applied.promotion.is_ready_for_seed
    assert applied.requirements["acceptance_criteria"] == (confirmation,)
    promoted = [
        candidate
        for candidate in applied.promotion.promoted
        if candidate.candidate_id == "round-3:requirement"
    ][0]
    assert promoted.content_source is CandidateContentSource.USER_STATED
    assert promoted.confirmation_authority is ConfirmationAuthority.USER


def test_ordinary_follow_up_after_reference_contrast_is_not_promoted() -> None:
    distillation = build_requirement_distillation(
        _reference_state(confirmation="Maybe blue is nice.")
    )

    applied = apply_requirement_distillation(_requirements(), distillation)

    assert applied.promotion.is_ready_for_seed
    assert applied.requirements["acceptance_criteria"] == ()
    assert all(
        candidate.candidate_id != "round-3:requirement" for candidate in distillation.candidates
    )


def test_non_reference_interview_preserves_legacy_extraction() -> None:
    state = InterviewState(
        interview_id="legacy",
        initial_context="Build a CLI",
        rounds=[
            InterviewRound(
                round_number=1,
                question="What matters?",
                user_response="It must print hello.",
            )
        ],
    )
    requirements = _requirements()

    applied = apply_requirement_distillation(
        requirements,
        build_requirement_distillation(state),
    )

    assert applied.requirements == requirements


def test_promoted_reference_seed_preserves_literal_pipe_in_constraint() -> None:
    state = _reference_state(
        confirmation="The CLI must accept only --lang ko|en as the language flag."
    )
    distillation = build_requirement_distillation(state)

    seed = build_promoted_reference_seed(state, distillation, ambiguity_score=0.1)

    assert seed.constraints == ("The CLI must accept only --lang ko|en as the language flag.",)


def test_promoted_reference_seed_preserves_literal_pipe_in_acceptance_criterion() -> None:
    state = _reference_state(
        confirmation="The confirmed requirement is that output must show ko|en verbatim."
    )
    distillation = build_requirement_distillation(state)

    seed = build_promoted_reference_seed(state, distillation, ambiguity_score=0.1)

    assert tuple(str(item) for item in seed.acceptance_criteria) == (
        "The confirmed requirement is that output must show ko|en verbatim.",
    )


def test_reference_cue_merge_changes_fingerprint_and_revision() -> None:
    state = InterviewState(interview_id="test", initial_context="Build a tool")
    before = build_requirement_distillation(state)
    state.merge_turn_context(
        InterviewTurnContext(
            references=(
                ReferenceCue(
                    reference_id="linear",
                    label="Linear",
                    origin=ReferenceOrigin.USER_TEXT,
                ),
            )
        )
    )
    after = build_requirement_distillation(state)

    assert after.input_revision == before.input_revision + 1
    assert after.input_fingerprint != before.input_fingerprint


@pytest.mark.asyncio
async def test_seed_generator_filters_unconfirmed_reference_acs(tmp_path) -> None:
    adapter = AsyncMock()
    adapter.complete.return_value = Result.ok(_extraction_response())
    generator = SeedGenerator(
        llm_adapter=adapter,
        model="test-model",
        output_dir=tmp_path,
    )

    result = await generator.generate(_reference_state(), _low_ambiguity())

    assert result.is_ok
    assert result.value.acceptance_criteria == ()


@pytest.mark.asyncio
async def test_seed_generator_keeps_explicitly_confirmed_reference_ac(tmp_path) -> None:
    adapter = AsyncMock()
    adapter.complete.return_value = Result.ok(_extraction_response())
    generator = SeedGenerator(
        llm_adapter=adapter,
        model="test-model",
        output_dir=tmp_path,
    )
    state = _reference_state(
        confirmation="For the Linear-like reference, keyboard-first navigation is required."
    )

    result = await generator.generate(state, _low_ambiguity())

    assert result.is_ok
    assert tuple(str(item) for item in result.value.acceptance_criteria) == (
        "For the Linear-like reference, keyboard-first navigation is required.",
    )


@pytest.mark.asyncio
async def test_seed_generator_returns_typed_reopen_error_for_conflict(tmp_path) -> None:
    adapter = AsyncMock()
    generator = SeedGenerator(
        llm_adapter=adapter,
        model="test-model",
        output_dir=tmp_path,
    )
    state = InterviewState(interview_id="conflict", initial_context="Build a tool")
    fingerprint = state.requirement_input_fingerprint()
    state.requirement_distillation = RequirementDistillation(
        candidates=(
            RequirementCandidate(
                candidate_id="conflict-1",
                section=RequirementSection.ACCEPTANCE_CRITERION,
                text="Automate approval but require manual approval.",
                content_source=CandidateContentSource.USER_STATED,
                resolution=CandidateResolution.CONFLICTING,
                confirmation_authority=ConfirmationAuthority.NONE,
                evidence_ids=("user-1",),
                required=True,
            ),
        ),
        evidence=(
            RequirementEvidence(
                evidence_id="user-1",
                kind=RequirementEvidenceKind.USER_STATEMENT,
                text="Automate approval but require manual approval.",
            ),
        ),
        input_revision=state.requirement_input_revision,
        input_fingerprint=fingerprint,
    )

    result = await generator.generate(state, _low_ambiguity())

    assert result.is_err
    assert result.error.details["code"] == "interview_reopen_required"
    assert result.error.details["blockers"][0]["reason"] == "conflict_requires_tradeoff"
    adapter.complete.assert_not_awaited()


def _round_candidates(distillation: RequirementDistillation) -> list[RequirementCandidate]:
    """Only the candidates promoted from interview ROUNDS.

    The initial goal always yields its own candidate; the round-73 property
    is about answers.
    """
    return [c for c in distillation.candidates if c.candidate_id.startswith("round-")]


def _state_with_answer(answer: str) -> InterviewState:
    return InterviewState(
        interview_id="iv_provenance",
        initial_context="Build the reporting lane",
        rounds=[
            InterviewRound(
                round_number=1,
                question="What must the system guarantee?",
                user_response=answer,
            )
        ],
    )


def test_round73_data_observation_is_not_promoted_to_a_requirement() -> None:
    """A [from-data] answer is a confirmed OBSERVATION, not a product decision.

    Its narrative routinely contains this gate's own trigger words —
    "confirmed", "required" — so the deterministic promotion manufactured a
    requirement with confirmation_authority=USER out of a decision the user
    never made. End-to-end through build_requirement_distillation, as the
    review asked.
    """
    marked = "[from-data] Confirmed: 42 enterprise accounts require SSO today."
    distillation = build_requirement_distillation(_state_with_answer(marked))

    assert _round_candidates(distillation) == []

    # The same sentence in the user's own words IS a decision, and promotes.
    unmarked = "Confirmed: 42 enterprise accounts require SSO today."
    promoted = _round_candidates(build_requirement_distillation(_state_with_answer(unmarked)))

    assert len(promoted) == 1
    assert promoted[0].text == unmarked


def test_round73_research_observation_is_the_same_class() -> None:
    """[from-research] carries the same user-adopted-observation provenance;
    the intent guard already groups them, and a second grouping here is how
    the two surfaces would drift."""
    marked = "[from-research] The payment provider requires 3DS for EU cards."
    assert _round_candidates(build_requirement_distillation(_state_with_answer(marked))) == []


def test_round73_users_own_words_around_data_still_decide() -> None:
    """A mixed answer the USER leads is the user's decision.

    The marker is a provenance stamp on what the answer leads with; an answer
    that opens in the user's own words and cites data inline promotes.
    """
    mixed = "We must support 100 concurrent sessions given [from-data] 42 accounts."
    promoted = _round_candidates(build_requirement_distillation(_state_with_answer(mixed)))
    assert len(promoted) == 1


def test_round74_observation_content_never_reaches_the_extractor() -> None:
    """The LLM path is enforced at its INPUT, where enforcement is decidable.

    Round 73 gated the deterministic candidates; the ordinary interview hands
    the transcript to an LLM extractor, which paraphrased the observation into
    a Seed AC. A paraphrase cannot be provenance-checked afterwards, so the
    content must not reach the prompt at all.
    """
    from ouroboros.bigbang.seed_generator import SeedGenerator
    from ouroboros.core.requirement_candidate import OBSERVATION_WITHHELD_NOTE

    state = _state_with_answer("[from-data] Confirmed: 42 enterprise accounts require SSO today.")
    state.rounds.append(
        InterviewRound(
            round_number=2,
            question="So what must the product guarantee?",
            user_response="Enterprise accounts must be able to use SSO.",
        )
    )

    generator = SeedGenerator.__new__(SeedGenerator)
    context = generator._build_interview_context(state)

    assert "42 enterprise accounts" not in context
    assert "[from-data]" not in context
    assert OBSERVATION_WITHHELD_NOTE in context
    # The user's own decision still reaches the extractor verbatim.
    assert "Enterprise accounts must be able to use SSO." in context


def test_round74_pm_transcript_withholds_the_same_class() -> None:
    """A PMSeed is durable too, and the PM extractor paraphrases the same way."""
    from ouroboros.bigbang.pm_interview import PMInterviewEngine
    from ouroboros.core.requirement_candidate import OBSERVATION_WITHHELD_NOTE

    state = _state_with_answer("[from-research] The provider requires 3DS for EU cards.")

    engine = PMInterviewEngine.__new__(PMInterviewEngine)
    transcript = engine._build_interview_context(state)

    assert "3DS" not in transcript
    assert OBSERVATION_WITHHELD_NOTE in transcript


def test_round76_a_pre_change_cache_cannot_bypass_the_provenance_gate() -> None:
    """The derivation-policy version participates in cache validity.

    Rounds 73-74 changed what the distillation derives, but a cache computed
    before the change matches on fingerprint and revision — a probe reused
    one and emitted its research observation as a Seed acceptance criterion.
    The version bump is the migration: v1 caches are recomputed, not reused.
    """
    from ouroboros.core.requirement_candidate import (
        REQUIREMENT_DISTILLATION_SCHEMA_VERSION,
        RequirementDistillation,
    )

    state = _state_with_answer("[from-research] The provider requires 3DS for EU cards.")
    # A cache distilled under the pre-change policy: same inputs, old
    # version, and the observation promoted the way v1 promoted it.
    stale = RequirementDistillation(
        candidates=(
            RequirementCandidate(
                candidate_id="round-1:requirement",
                section=RequirementSection.ACCEPTANCE_CRITERION,
                text="[from-research] The provider requires 3DS for EU cards.",
                content_source=CandidateContentSource.USER_STATED,
                resolution=CandidateResolution.CONFIRMED,
                confirmation_authority=ConfirmationAuthority.USER,
                evidence_ids=("round-1:answer",),
                required=True,
            ),
        ),
        evidence=(
            RequirementEvidence(
                evidence_id="round-1:answer",
                kind=RequirementEvidenceKind.USER_STATEMENT,
                text="[from-research] The provider requires 3DS for EU cards.",
            ),
        ),
        schema_version="requirement-distillation.v1",
        input_revision=state.requirement_input_revision,
        input_fingerprint=state.requirement_input_fingerprint(),
    )
    state.requirement_distillation = stale
    assert REQUIREMENT_DISTILLATION_SCHEMA_VERSION != "requirement-distillation.v1"

    rebuilt = build_requirement_distillation(state)

    assert rebuilt is not stale, "the pre-change cache was reused"
    assert _round_candidates(rebuilt) == []
    assert rebuilt.schema_version == REQUIREMENT_DISTILLATION_SCHEMA_VERSION


def test_round76_a_current_cache_is_still_reused() -> None:
    """The bump must not disable the cache — a v2 cache stays a cache."""
    state = _state_with_answer("We must ship the reporting lane this quarter.")
    first = build_requirement_distillation(state)
    state.requirement_distillation = first

    assert build_requirement_distillation(state) is first


def test_round80_questions_after_an_observation_are_withheld() -> None:
    """The reviewer's probe: a later question restates the withheld observation.

    The interviewer legitimately sees observations in conversational context,
    so a question generated after one can carry it verbatim — and the
    extractors derive requirements from the whole conversation. Taint
    provenance is the decidable line: questions before the first observation
    keep their interpretive value; every question after it is withheld.
    """
    from ouroboros.bigbang.seed_generator import SeedGenerator
    from ouroboros.core.requirement_candidate import (
        OBSERVATION_WITHHELD_NOTE,
        QUESTION_WITHHELD_NOTE,
    )

    state = _state_with_answer("[from-data] 42 enterprise accounts require SSO.")
    state.rounds[0].question = "What does the data show about SSO?"
    state.rounds.append(
        InterviewRound(
            round_number=2,
            question="Given that 42 enterprise accounts require SSO, what tier is needed?",
            user_response="Enterprise tier must include SSO.",
        )
    )

    generator = SeedGenerator.__new__(SeedGenerator)
    context = generator._build_interview_context(state)

    # The pre-observation question keeps its interpretive value.
    assert "What does the data show about SSO?" in context
    # The post-observation question — which restates the observation — is out.
    assert "42 enterprise accounts" not in context
    assert QUESTION_WITHHELD_NOTE in context
    assert OBSERVATION_WITHHELD_NOTE in context
    # The user's own decision survives verbatim.
    assert "Enterprise tier must include SSO." in context


def test_round80_pm_and_plugin_paths_withhold_tainted_questions() -> None:
    """All three extraction surfaces apply the same taint rule."""
    from ouroboros.bigbang.pm_interview import PMInterviewEngine
    from ouroboros.core.requirement_candidate import QUESTION_WITHHELD_NOTE
    from ouroboros.mcp.tools.authoring_handlers import _format_extraction_transcript

    state = _state_with_answer("[from-research] The provider requires 3DS.")
    state.rounds.append(
        InterviewRound(
            round_number=2,
            question="Since the provider requires 3DS, how should checkout flow?",
            user_response="Checkout must support 3DS redirects.",
        )
    )

    pm_engine = PMInterviewEngine.__new__(PMInterviewEngine)
    for label, transcript in (
        ("pm", pm_engine._build_interview_context(state)),
        ("plugin", _format_extraction_transcript(state)),
    ):
        assert "provider requires 3DS, how should checkout" not in transcript, label
        assert QUESTION_WITHHELD_NOTE in transcript, label
        assert "Checkout must support 3DS redirects." in transcript, label


def test_round80_observation_free_interviews_keep_every_question() -> None:
    """No observation, no taint — the interpretive context is untouched."""
    from ouroboros.bigbang.seed_generator import SeedGenerator
    from ouroboros.core.requirement_candidate import QUESTION_WITHHELD_NOTE

    state = _state_with_answer("We must ship the reporting lane this quarter.")
    state.rounds.append(
        InterviewRound(
            round_number=2,
            question="What is the acceptance bar?",
            user_response="All exports finish under a minute.",
        )
    )

    generator = SeedGenerator.__new__(SeedGenerator)
    context = generator._build_interview_context(state)

    assert QUESTION_WITHHELD_NOTE not in context
    assert "What is the acceptance bar?" in context


def test_round81_summary_answers_cannot_bypass_the_withholding() -> None:
    """The oversized-context path substitutes the summary ANSWER for the
    initial context — the fourth entrance to the extraction contexts.

    The reviewer's probe: a summary answer leading with [from-data] reached
    both the Seed and PM extraction contexts verbatim, and — because the
    summary round is skipped in the loop — never set the question taint.
    """
    from ouroboros.bigbang.interview import (
        INITIAL_CONTEXT_SUMMARY_QUESTION,
        MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS,
    )
    from ouroboros.bigbang.pm_interview import PMInterviewEngine
    from ouroboros.bigbang.seed_generator import SeedGenerator
    from ouroboros.core.requirement_candidate import (
        OBSERVATION_WITHHELD_NOTE,
        QUESTION_WITHHELD_NOTE,
    )

    state = InterviewState(
        interview_id="iv_81",
        # Oversized, so prompt_safe_initial_context falls through to the
        # summary answer.
        initial_context="x" * (MAX_PROMPT_SAFE_INITIAL_CONTEXT_CHARS + 1),
        rounds=[
            InterviewRound(
                round_number=1,
                question=INITIAL_CONTEXT_SUMMARY_QUESTION,
                user_response=("[from-data] Confirmed: 42 enterprise accounts require SSO today."),
            ),
            InterviewRound(
                round_number=2,
                question="Given the 42 enterprise accounts, what tier is needed?",
                user_response="Enterprise tier must include SSO.",
            ),
        ],
    )

    dev = SeedGenerator.__new__(SeedGenerator)._build_interview_context(state)
    pm = PMInterviewEngine.__new__(PMInterviewEngine)._build_interview_context(state)

    for label, context in (("dev", dev), ("pm", pm)):
        assert "42 enterprise accounts" not in context, label
        assert OBSERVATION_WITHHELD_NOTE in context, label
        # The summary was the observation, so every question is tainted.
        assert QUESTION_WITHHELD_NOTE in context, label
        assert "Enterprise tier must include SSO." in context, label


def test_round81_plugin_initial_context_is_sanitized() -> None:
    """The plugin transcript's own header applies the same rule."""
    from ouroboros.core.requirement_candidate import OBSERVATION_WITHHELD_NOTE
    from ouroboros.mcp.tools.authoring_handlers import _format_extraction_transcript

    state = _state_with_answer("Enterprise tier must include SSO.")
    state.initial_context = "[from-data] 42 enterprise accounts require SSO."

    transcript = _format_extraction_transcript(state)

    assert "42 enterprise accounts" not in transcript
    assert OBSERVATION_WITHHELD_NOTE in transcript


def test_round82_observation_context_is_not_promoted_as_the_goal() -> None:
    """The initial context takes the same provenance gate as every answer.

    An observation-marked context was promoted as a CONFIRMED goal with user
    authority, so [from-data] observations became the runnable Seed goal
    through the reference-aware path while every extraction surface withheld
    them — the fifth entrance.
    """
    state = _state_with_answer("Enterprise tier must include SSO.")
    state.initial_context = "[from-data] 42 enterprise accounts require SSO today."

    distillation = build_requirement_distillation(state)

    assert all(c.candidate_id != "initial-goal" for c in distillation.candidates)

    # An ordinary user-authored context still yields the goal candidate.
    plain = _state_with_answer("Enterprise tier must include SSO.")
    plain.initial_context = "Build SSO for enterprise accounts"
    assert any(
        c.candidate_id == "initial-goal" for c in build_requirement_distillation(plain).candidates
    )
