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


def test_parent_seed_requires_inheritance_semantics() -> None:
    seed = _seed("Compare seed_old with seed_candidate.")

    assert inherited_parent_seed_id(seed) is None


def test_parent_seed_preserves_korean_inheritance_contract() -> None:
    seed = _seed("seed_parent를 계승해 문서형 Seed로 명세한다.")

    assert inherited_parent_seed_id(seed) == "seed_parent"
