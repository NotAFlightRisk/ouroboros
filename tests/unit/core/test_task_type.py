from ouroboros.core.task_type import explicit_task_type_from_goal


def test_explicit_task_type_ignores_question_and_uses_answer() -> None:
    transcript = "Q: Should task_type be code or document?\nA: task_type must be document."

    assert explicit_task_type_from_goal(transcript) == "document"


def test_explicit_task_type_uses_final_correction() -> None:
    goal = "task_type must be code. Correction: task_type must be document."

    assert explicit_task_type_from_goal(goal) == "document"


def test_explicit_task_type_ignores_superseded_clause() -> None:
    goal = "Ignore superseded task_type: code. task_type: research."

    assert explicit_task_type_from_goal(goal) == "research"


def test_explicit_task_type_ignores_non_binding_mentions() -> None:
    for goal in (
        "Should task_type: document?",
        "Document the literal example `task_type: code` for users.",
        "Do not use task_type: document.",
        "We cannot use task_type: document.",
        "We can't use task_type: document.",
        "Without using task_type: document, describe the migration.",
        "The task_type: document value is not allowed.",
        "The task_type: document value must not be used.",
    ):
        assert explicit_task_type_from_goal(goal) is None


def test_explicit_task_type_uses_positive_correction_after_negated_candidate() -> None:
    goal = "Do not use task_type: code, instead use task_type: document."

    assert explicit_task_type_from_goal(goal) == "document"


def test_explicit_task_type_scopes_conjunction_negation_to_each_candidate() -> None:
    for goal in (
        "Use task_type: document and do not modify source code.",
        "Do not use task_type: code and instead use task_type: document.",
    ):
        assert explicit_task_type_from_goal(goal) == "document"


def test_explicit_task_type_survives_adjacent_negative_constraint() -> None:
    for goal in (
        "task_type must be document without changing repository files.",
        "Use task_type: document although we must not produce code.",
        "Use task_type: document because source code must not change.",
    ):
        assert explicit_task_type_from_goal(goal) == "document"


def test_explicit_task_type_ignores_causal_prefix_and_historical_contracts() -> None:
    assert (
        explicit_task_type_from_goal(
            "Do not modify source code because task_type must be document."
        )
        == "document"
    )
    for goal in (
        "We discussed task_type: document in the rejected proposal. Build a CLI.",
        'The phrase "task_type: document" is an example, not a requirement.',
    ):
        assert explicit_task_type_from_goal(goal) is None


def test_explicit_task_type_scopes_quotes_and_historical_governors_to_contract() -> None:
    assert (
        explicit_task_type_from_goal(
            "The previous proposal was rejected, but its task_type: document "
            "should be recorded for audit."
        )
        is None
    )
    for goal in (
        "We'll use task_type: document for the final plan.",
        "Use task_type: document for the historical archive.",
    ):
        assert explicit_task_type_from_goal(goal) == "document"


def test_explicit_task_type_rejects_ordinary_negative_contract_language() -> None:
    for goal in (
        "The task_type: document requirement was rejected.",
        "Use the default code task instead of task_type: document.",
        "Rather than use task_type: document, keep the code task.",
        "We no longer use task_type: document.",
        "In the previous proposal, task_type: document.",
        "It is not true that task_type: document.",
        "We are not using task_type: document.",
        "task_type: document will not be used.",
        "task_type: document is no longer required.",
        "We won't use task_type: document.",
        "The task type isn't document.",
    ):
        assert explicit_task_type_from_goal(goal) is None
    assert explicit_task_type_from_goal("Use task_type: document rather than code.") == "document"


def test_explicit_task_type_resets_governors_at_punctuation_boundary() -> None:
    assert (
        explicit_task_type_from_goal(
            "Rather than change source code, write a plan; task_type: document."
        )
        == "document"
    )


def test_explicit_task_type_accepts_ordinary_positive_wording() -> None:
    for goal in (
        "The task type is document.",
        "Set the task type to document.",
        "The task type must remain document.",
        "Keep the task type as document.",
        "Use document as the task type.",
        "The task type must be a document.",
        "The task type should be a document.",
    ):
        assert explicit_task_type_from_goal(goal) == "document"


def test_explicit_task_type_rejects_typographic_quoted_and_conditional_language() -> None:
    for goal in (
        "We won’t use task_type: document.",
        "task_type: document should be avoided.",
        "If task_type is document, write a guide; otherwise keep code.",
        "Choose between task_type: document or code later.",
        'The docs currently say "Set task_type: document for exports." Replace that guidance.',
    ):
        assert explicit_task_type_from_goal(goal) is None


def test_explicit_task_type_rejects_explanatory_optional_and_single_quoted_text() -> None:
    assert (
        explicit_task_type_from_goal(
            "Set the task type to document. Add a section explaining task_type: code."
        )
        == "document"
    )
    for goal in (
        "Only if approved, use task_type: document.",
        "The docs say 'task_type: code' for legacy exports.",
    ):
        assert explicit_task_type_from_goal(goal) is None
