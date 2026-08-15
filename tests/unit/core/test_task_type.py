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
