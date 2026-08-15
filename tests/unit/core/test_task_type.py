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
