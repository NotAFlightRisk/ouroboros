from types import SimpleNamespace

from ouroboros.auto.seed_qa_contract import inherited_parent_seed_id


def _seed(goal: str) -> SimpleNamespace:
    return SimpleNamespace(goal=goal)


def test_parent_seed_uses_explicit_positive_inheritance_clause() -> None:
    seed = _seed("Reference seed_old for context; inherit seed_parent.")

    assert inherited_parent_seed_id(seed) == "seed_parent"


def test_parent_seed_ignores_negated_reference() -> None:
    seed = _seed("Do not inherit seed_old; inherit seed_parent.")

    assert inherited_parent_seed_id(seed) == "seed_parent"


def test_parent_seed_ignores_adverb_qualified_negation() -> None:
    for goal in (
        "Do not ever inherit seed_bad.",
        "Never directly inherit seed_bad.",
        "The Seed must not inherit seed_bad.",
        "Cannot inherit seed_bad.",
        "Continue without inheriting seed_bad.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) is None


def test_parent_seed_uses_positive_correction_after_negated_candidate() -> None:
    seed = _seed("Do not inherit seed_bad, instead inherit seed_good.")

    assert inherited_parent_seed_id(seed) == "seed_good"


def test_parent_seed_scopes_conjunction_negation_to_each_candidate() -> None:
    for goal in (
        "Inherit seed_good and do not copy its obsolete constraints.",
        "Do not inherit seed_bad and instead inherit seed_good.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) == "seed_good"


def test_parent_seed_survives_adjacent_negative_constraint() -> None:
    for goal in (
        "Inherit seed_good without copying obsolete constraints.",
        "Derive from seed_good although we must not reuse its runtime settings.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) == "seed_good"


def test_parent_seed_ignores_quoted_and_historical_references() -> None:
    for goal in (
        'The phrase "inherit seed_bad" is an example, not a requirement.',
        "We discussed inherit seed_bad in the rejected proposal.",
        "Do not copy obsolete constraints because this Seed should inherit seed_good.",
    ):
        expected = "seed_good" if "should inherit" in goal else None
        assert inherited_parent_seed_id(_seed(goal)) == expected


def test_parent_seed_scopes_possessives_and_historical_governors() -> None:
    assert (
        inherited_parent_seed_id(
            _seed("The previous proposal was rejected, but it said inherit seed_bad for reference.")
        )
        is None
    )
    assert inherited_parent_seed_id(_seed("Inherit seed_good for John's project.")) == ("seed_good")


def test_parent_seed_rejects_ordinary_negative_inheritance_language() -> None:
    for goal in (
        "It is false that we inherit seed_bad.",
        "Start fresh instead of inheriting seed_bad.",
        "Rather than inherit seed_bad, start fresh.",
        "We no longer inherit seed_bad.",
        "In the previous proposal, inherit seed_bad.",
        "It is not true that we inherit seed_bad.",
        "Inherit seed_bad will not be used.",
        "Inherit seed_bad is no longer required.",
        "We won't inherit seed_bad.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) is None


def test_parent_seed_resets_governors_at_punctuation_boundary() -> None:
    goal = "Rather than copy old constraints, start the repair; inherit seed_good."

    assert inherited_parent_seed_id(_seed(goal)) == "seed_good"


def test_parent_seed_accepts_ordinary_positive_wording() -> None:
    for goal in (
        "Inherit from seed_parent.",
        "Set parent_seed_id to seed_parent.",
        "Use seed_parent as the parent seed.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) == "seed_parent"


def test_parent_seed_requires_inheritance_semantics() -> None:
    seed = _seed("Compare seed_old with seed_candidate.")

    assert inherited_parent_seed_id(seed) is None


def test_parent_seed_preserves_korean_inheritance_contract() -> None:
    seed = _seed("seed_parent를 계승해 문서형 Seed로 명세한다.")

    assert inherited_parent_seed_id(seed) == "seed_parent"


def test_parent_seed_rejects_typographic_quoted_conditional_and_ambiguous_language() -> None:
    for goal in (
        "We won’t inherit seed_bad.",
        "Inherit seed_bad should be avoided.",
        'The old docs say "Inherit seed_bad for migrations." Replace that guidance.',
        "If we inherit seed_bad, copy settings; otherwise start fresh.",
        "Inherit seed_one or seed_two after review.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) is None


def test_parent_seed_rejects_explanatory_optional_and_single_quoted_language() -> None:
    for goal in (
        "Add migration notes explaining how to inherit seed_old.",
        "The docs say 'inherit seed_old' for migrations.",
        "Only if approved, inherit seed_old.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) is None


def test_parent_seed_scopes_mixed_clause_authority_to_contract() -> None:
    assert (
        inherited_parent_seed_id(
            _seed("Whether to copy settings or rebuild them is undecided, but inherit seed_good.")
        )
        == "seed_good"
    )


def test_parent_seed_rejects_modal_or_optional_contracts() -> None:
    for goal in (
        "We may inherit seed_bad.",
        "Inheriting from seed_bad is optional.",
        "It does not inherit from seed_bad.",
        "Inherit seed_one and inherit seed_two.",
        "Inherit seed_one; inherit seed_two.",
    ):
        assert inherited_parent_seed_id(_seed(goal)) is None
    assert (
        inherited_parent_seed_id(_seed("Inherit from seed_good for an optional migration note."))
        == "seed_good"
    )
    assert (
        inherited_parent_seed_id(_seed("Inherit seed_bad, but that requirement was rejected."))
        is None
    )
