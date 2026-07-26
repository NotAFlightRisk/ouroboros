"""Reference-aware pre-Seed requirement candidates and promotion policy.

The models in this module are deliberately weaker than the auto-mode
``SeedDraftLedger``. They describe derived interview material before it becomes
an executable Seed contract. Canonical authority remains with the interview
transcript, bounded reference cues, and repository evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# v3 (round-82): the initial-goal candidate now takes the provenance gate
# too — a v2 cache may carry an observation-marked context promoted as a
# confirmed goal. v2 (round-76): the derivation policy changed — observation-marked answers
# ([from-data]/[from-research]) are no longer promoted to requirement
# candidates (round-73). The version participates in is_current(), so a
# cache distilled under v1 with an observation-derived candidate is
# invalidated by this bump instead of being reused past the new gate.
REQUIREMENT_DISTILLATION_SCHEMA_VERSION = "requirement-distillation.v3"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def redacted_segment(text: str) -> str:
    """Digest form of a value that may not ride a report.

    The single definition (round-76): the transport's error paths and the
    contract's semantic diagnostics both need it, and the round-69 lesson was
    that a value rejected FOR BEING secret-shaped must not then appear
    verbatim in the rejection.
    """
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"<redacted-key sha256:{digest}>"


def classify_answer_provenance(answer: str) -> str:
    """Return a coarse provenance class for an interview answer.

    The marker an answer LEADS with is its provenance stamp: ``data_fact`` /
    ``research_fact`` are user-adopted external facts — the human confirmed
    them before forwarding, but confirmed them AS OBSERVATIONS. This is the
    single definition (round-73): the intent guard and the deterministic
    requirement promotion both key off it, and a second prefix list in either
    place is how the two would drift.
    """
    lowered = str(answer).lstrip().casefold()
    if lowered.startswith("[from-code]") or lowered.startswith("[from-repo]"):
        return "repo_fact"
    if lowered.startswith("[from-data]"):
        return "data_fact"
    if lowered.startswith("[from-research]"):
        return "research_fact"
    if lowered.startswith("[from-auto]") or lowered.startswith("[from-safe-default]"):
        return "generated"
    if lowered.startswith("[from-assumption]") or lowered.startswith("[from-inference]"):
        return "generated"
    return "human"


#: What a withheld observation is replaced with in extraction inputs. A fixed
#: string: nothing of the observation survives to be paraphrased.
OBSERVATION_WITHHELD_NOTE = (
    "[point-in-time observation withheld from requirement extraction: "
    "data/research observations are not requirements. Any requirement the "
    "user derived from it appears in their own words in their own answers.]"
)


#: What a withheld interviewer question is replaced with in extraction inputs.
QUESTION_WITHHELD_NOTE = (
    "[interviewer question withheld from requirement extraction: it was "
    "generated after a data/research observation entered the conversation "
    "and may restate it. Requirements derive from the user's answers.]"
)


def extraction_safe_question(question: str, *, observation_seen: bool) -> str:
    """The form of an interviewer QUESTION that may enter requirement extraction.

    Withholding answers was not enough (round-80): the interviewer legitimately
    sees observations in conversational context, so a question generated AFTER
    an observation entered the history can restate it verbatim — and the
    extractors are told to derive requirements from the conversation. A
    paraphrase in a question is as unmatchable as one in a Seed AC, so the
    decidable line is taint provenance: a question generated BEFORE the first
    observation cannot contain it and keeps its interpretive value; every
    question after it is withheld.
    """
    if observation_seen:
        return QUESTION_WITHHELD_NOTE
    return question


#: The closed range of classify_answer_provenance. Shared so the declared
#: field's authority is bounded by the same set the classifier can produce.
_KNOWN_PROVENANCE = frozenset({"human", "data_fact", "research_fact", "repo_fact", "generated"})


def effective_answer_provenance(answer: str, declared: str = "human") -> str:
    """Field-first provenance: the ingestion-time field wins when set.

    The marker is the display/legacy projection; the field is the record
    (round-85). A round whose field says data_fact is an observation even if
    the marker was stripped from the text.

    A declared value OUTSIDE the closed range carries no authority and falls
    back to marker classification — an open passthrough failed open: any
    unknown string was neither "human" nor an observation class, so it
    overrode the marker and un-withheld a `[from-data]` answer. The
    InterviewRound field is enum-constrained at model validation; this
    fallback bounds every other caller the same way.
    """
    if declared in _KNOWN_PROVENANCE and declared != "human":
        return declared
    return classify_answer_provenance(answer)


def extraction_safe_answer(answer: str, declared_provenance: str = "human") -> str:
    """The form of an answer that may enter requirement extraction.

    A ``[from-data]`` / ``[from-research]`` answer is an observation the user
    confirmed AS AN OBSERVATION. The deterministic promotion gate skips it
    (round-73), but the ordinary interview path hands the whole transcript to
    an LLM extractor, which paraphrases — `[from-data] Confirmed: 42
    enterprise accounts require SSO today` came back as the Seed AC
    "Support SSO for enterprise accounts" (round-74). A paraphrase cannot be
    matched against the marker afterwards, so the enforcement point is the
    extraction INPUT: the observation's content is replaced with a fixed
    note, and what the extractor never receives it cannot promote. A Seed is
    a durable artifact; a point-in-time measurement decays inside it.
    """
    if effective_answer_provenance(answer, declared_provenance) in {
        "data_fact",
        "research_fact",
    }:
        return OBSERVATION_WITHHELD_NOTE
    return answer


class CandidateContentSource(StrEnum):
    """Where candidate content originated."""

    USER_STATED = "user_stated"
    REFERENCE_DERIVED = "reference_derived"
    MODEL_INFERRED = "model_inferred"
    REPO_OBSERVED = "repo_observed"


class CandidateResolution(StrEnum):
    """Current resolution state of a candidate."""

    CONFIRMED = "confirmed"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"


class ConfirmationAuthority(StrEnum):
    """Authority that confirmed the candidate, independent of its source."""

    USER = "user"
    REPO_EVIDENCE = "repo_evidence"
    NONE = "none"


class RequirementSection(StrEnum):
    """Seed-facing section targeted by a candidate."""

    GOAL = "goal"
    CONSTRAINT = "constraint"
    EXISTING_CONSTRAINT = "existing_constraint"
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    ONTOLOGY = "ontology"
    EVALUATION_PRINCIPLE = "evaluation_principle"
    EXIT_CONDITION = "exit_condition"
    NON_GOAL = "non_goal"
    CONTEXT = "context"


class RequirementEvidenceKind(StrEnum):
    """Trusted channel that supplied evidence for a candidate."""

    USER_STATEMENT = "user_statement"
    REFERENCE_CUE = "reference_cue"
    REFERENCE_CONTRAST = "reference_contrast"
    MODEL_INFERENCE = "model_inference"
    REPO_EVIDENCE = "repo_evidence"


class PromotionDisposition(StrEnum):
    """How a candidate participates in Seed construction."""

    PROMOTE = "promote"
    OMIT = "omit"
    BLOCK = "block"


class RequirementEvidence(BaseModel, frozen=True):
    """One tagged evidence item used by the distillation read model."""

    evidence_id: str = Field(min_length=1, max_length=128)
    kind: RequirementEvidenceKind
    text: str = Field(default="", max_length=8000)
    reference_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("evidence_id", "reference_id")
    @classmethod
    def _validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("identifier contains unsupported characters")
        return value

    @model_validator(mode="after")
    def _require_reference_id_for_reference_evidence(self) -> RequirementEvidence:
        if (
            self.kind
            in {
                RequirementEvidenceKind.REFERENCE_CUE,
                RequirementEvidenceKind.REFERENCE_CONTRAST,
            }
            and self.reference_id is None
        ):
            raise ValueError("reference evidence requires reference_id")
        return self


class RequirementCandidate(BaseModel, frozen=True):
    """One derived pre-Seed requirement candidate."""

    candidate_id: str = Field(min_length=1, max_length=128)
    section: RequirementSection
    text: str = Field(min_length=1, max_length=8000)
    content_source: CandidateContentSource
    resolution: CandidateResolution = CandidateResolution.NEEDS_CONFIRMATION
    confirmation_authority: ConfirmationAuthority = ConfirmationAuthority.NONE
    reference_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    required: bool = False

    @field_validator("candidate_id")
    @classmethod
    def _validate_candidate_id(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("candidate_id contains unsupported characters")
        return value

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("candidate text must not be blank")
        return value

    @field_validator("reference_ids", "evidence_ids")
    @classmethod
    def _validate_id_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("identifier lists must not contain duplicates")
        if any(not _ID_PATTERN.fullmatch(item) for item in value):
            raise ValueError("identifier list contains unsupported characters")
        return value

    @model_validator(mode="after")
    def _validate_confirmation_state(self) -> RequirementCandidate:
        if (
            self.resolution is CandidateResolution.CONFIRMED
            and self.confirmation_authority is ConfirmationAuthority.NONE
        ):
            raise ValueError("confirmed candidates require confirmation authority")
        if (
            self.resolution is not CandidateResolution.CONFIRMED
            and self.confirmation_authority is not ConfirmationAuthority.NONE
        ):
            raise ValueError("unconfirmed candidates cannot declare confirmation authority")
        return self

    @property
    def display_prefix(self) -> str:
        """Render the issue vocabulary without using it as the data model."""
        if self.resolution is CandidateResolution.CONFLICTING:
            return "[CONFLICT]"
        if self.resolution is CandidateResolution.UNKNOWN:
            return "[UNKNOWN]"
        return {
            CandidateContentSource.USER_STATED: "[USER_STATED]",
            CandidateContentSource.REFERENCE_DERIVED: "[REFERENCE_DERIVED]",
            CandidateContentSource.MODEL_INFERRED: "[MODEL_INFERRED]",
            CandidateContentSource.REPO_OBSERVED: "[REPO_OBSERVED]",
        }[self.content_source]

    def render(self) -> str:
        """Return the compact prefixed representation used in handoffs."""
        return f"{self.display_prefix} {self.text}"

    def confirm_by_user(self) -> RequirementCandidate:
        """Confirm without rewriting truthful content provenance."""
        return self.model_copy(
            update={
                "resolution": CandidateResolution.CONFIRMED,
                "confirmation_authority": ConfirmationAuthority.USER,
            }
        )


class CandidateLineage(BaseModel, frozen=True):
    """Validated candidate plus evidence-lineage diagnostics."""

    candidate: RequirementCandidate
    valid: bool
    missing_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)


class PromotionDecision(BaseModel, frozen=True):
    """Deterministic promotion decision for one normalized candidate."""

    candidate: RequirementCandidate
    disposition: PromotionDisposition
    reason: str


class PromotionResult(BaseModel, frozen=True):
    """Promotion decisions for one distillation."""

    decisions: tuple[PromotionDecision, ...]

    @property
    def promoted(self) -> tuple[RequirementCandidate, ...]:
        return tuple(
            decision.candidate
            for decision in self.decisions
            if decision.disposition is PromotionDisposition.PROMOTE
        )

    @property
    def omitted(self) -> tuple[RequirementCandidate, ...]:
        return tuple(
            decision.candidate
            for decision in self.decisions
            if decision.disposition is PromotionDisposition.OMIT
        )

    @property
    def blockers(self) -> tuple[PromotionDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.disposition is PromotionDisposition.BLOCK
        )

    @property
    def is_ready_for_seed(self) -> bool:
        return not self.blockers


class RequirementDistillation(BaseModel, frozen=True):
    """Versioned, invalidatable read model derived from canonical inputs."""

    candidates: tuple[RequirementCandidate, ...] = Field(default_factory=tuple)
    evidence: tuple[RequirementEvidence, ...] = Field(default_factory=tuple)
    schema_version: str = REQUIREMENT_DISTILLATION_SCHEMA_VERSION
    input_revision: int = Field(default=0, ge=0)
    input_fingerprint: str = Field(min_length=64, max_length=64)

    @field_validator("input_fingerprint")
    @classmethod
    def _validate_fingerprint(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> RequirementDistillation:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        return self

    def is_current(self, *, input_revision: int, input_fingerprint: str) -> bool:
        """Return whether this cache matches the current canonical inputs."""
        return (
            self.schema_version == REQUIREMENT_DISTILLATION_SCHEMA_VERSION
            and self.input_revision == input_revision
            and self.input_fingerprint == input_fingerprint
        )


def compute_requirement_input_fingerprint(canonical_inputs: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for JSON-compatible canonical inputs."""
    normalized = _normalize_json_value(canonical_inputs)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_candidate_lineage(
    candidate: RequirementCandidate,
    evidence: Sequence[RequirementEvidence],
) -> CandidateLineage:
    """Derive authoritative content source from tagged evidence channels."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    missing = tuple(item_id for item_id in candidate.evidence_ids if item_id not in evidence_by_id)
    resolved = [
        evidence_by_id[item_id] for item_id in candidate.evidence_ids if item_id in evidence_by_id
    ]

    kinds = {item.kind for item in resolved}
    if RequirementEvidenceKind.MODEL_INFERENCE in kinds:
        source = CandidateContentSource.MODEL_INFERRED
    elif kinds & {
        RequirementEvidenceKind.REFERENCE_CUE,
        RequirementEvidenceKind.REFERENCE_CONTRAST,
    }:
        source = CandidateContentSource.REFERENCE_DERIVED
    elif RequirementEvidenceKind.REPO_EVIDENCE in kinds:
        source = CandidateContentSource.REPO_OBSERVED
    elif RequirementEvidenceKind.USER_STATEMENT in kinds:
        source = CandidateContentSource.USER_STATED
    else:
        source = CandidateContentSource.MODEL_INFERRED

    derived_reference_ids = {
        item.reference_id for item in resolved if item.reference_id is not None
    }
    reference_ids = tuple(sorted(set(candidate.reference_ids) | derived_reference_ids))
    valid = not missing and bool(resolved)
    if source is CandidateContentSource.REFERENCE_DERIVED and not reference_ids:
        valid = False

    normalized = candidate.model_copy(
        update={
            "content_source": source,
            "reference_ids": reference_ids,
        }
    )
    return CandidateLineage(
        candidate=normalized,
        valid=valid,
        missing_evidence_ids=missing,
    )


def evaluate_promotion(distillation: RequirementDistillation) -> PromotionResult:
    """Apply the deterministic pre-Seed promotion and blocking policy."""
    decisions: list[PromotionDecision] = []
    for raw_candidate in distillation.candidates:
        lineage = validate_candidate_lineage(raw_candidate, distillation.evidence)
        candidate = lineage.candidate

        if not lineage.valid:
            decisions.append(
                PromotionDecision(
                    candidate=candidate,
                    disposition=(
                        PromotionDisposition.BLOCK
                        if candidate.required
                        else PromotionDisposition.OMIT
                    ),
                    reason="invalid_evidence_lineage",
                )
            )
            continue

        if candidate.resolution is CandidateResolution.CONFLICTING:
            decisions.append(
                PromotionDecision(
                    candidate=candidate,
                    disposition=PromotionDisposition.BLOCK,
                    reason="conflict_requires_tradeoff",
                )
            )
            continue

        if candidate.resolution is CandidateResolution.UNKNOWN:
            decisions.append(
                PromotionDecision(
                    candidate=candidate,
                    disposition=(
                        PromotionDisposition.BLOCK
                        if candidate.required
                        else PromotionDisposition.OMIT
                    ),
                    reason="required_unknown" if candidate.required else "optional_unknown",
                )
            )
            continue

        if candidate.resolution is not CandidateResolution.CONFIRMED:
            decisions.append(
                PromotionDecision(
                    candidate=candidate,
                    disposition=(
                        PromotionDisposition.BLOCK
                        if candidate.required
                        else PromotionDisposition.OMIT
                    ),
                    reason=(
                        "confirmation_required" if candidate.required else "optional_unconfirmed"
                    ),
                )
            )
            continue

        if candidate.content_source in {
            CandidateContentSource.USER_STATED,
            CandidateContentSource.REFERENCE_DERIVED,
            CandidateContentSource.MODEL_INFERRED,
        }:
            promotable = candidate.confirmation_authority is ConfirmationAuthority.USER
        elif candidate.section in {
            RequirementSection.CONTEXT,
            RequirementSection.EXISTING_CONSTRAINT,
        }:
            promotable = candidate.confirmation_authority in {
                ConfirmationAuthority.REPO_EVIDENCE,
                ConfirmationAuthority.USER,
            }
        else:
            promotable = candidate.confirmation_authority is ConfirmationAuthority.USER

        decisions.append(
            PromotionDecision(
                candidate=candidate,
                disposition=(
                    PromotionDisposition.PROMOTE
                    if promotable
                    else (
                        PromotionDisposition.BLOCK
                        if candidate.required
                        else PromotionDisposition.OMIT
                    )
                ),
                reason=(
                    "promoted"
                    if promotable
                    else (
                        "user_confirmation_required"
                        if candidate.required
                        else "optional_without_authority"
                    )
                ),
            )
        )

    return PromotionResult(decisions=tuple(decisions))


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize_json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"canonical input contains unsupported type: {type(value).__name__}")


__all__ = [
    "CandidateContentSource",
    "CandidateLineage",
    "CandidateResolution",
    "ConfirmationAuthority",
    "PromotionDecision",
    "PromotionDisposition",
    "PromotionResult",
    "REQUIREMENT_DISTILLATION_SCHEMA_VERSION",
    "RequirementCandidate",
    "RequirementDistillation",
    "RequirementEvidence",
    "RequirementEvidenceKind",
    "RequirementSection",
    "compute_requirement_input_fingerprint",
    "evaluate_promotion",
    "validate_candidate_lineage",
]
