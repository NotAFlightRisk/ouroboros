"""Every file that holds interview or data content is owner-only.

The protection cannot be an instruction to each writer to remember: a call
site that forgets is invisible. These tests pin the property at each site that
persists this class of content, including the case that mattered — a state
directory inherited at 0755 from an earlier version.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import stat
from typing import Any

from ouroboros.core.owner_only import secure_directory, write_owner_only


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_write_owner_only_never_exists_at_the_umask_default(tmp_path: Path) -> None:
    target = tmp_path / "secret.json"
    write_owner_only(target, '{"answer": "confirmed"}')
    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == '{"answer": "confirmed"}'


def test_secure_directory_repairs_an_inherited_open_directory(tmp_path: Path) -> None:
    directory = tmp_path / "data"
    directory.mkdir(mode=0o755)
    os.chmod(directory, 0o755)
    secure_directory(directory)
    assert _mode(directory) == 0o700


def test_interview_state_is_owner_only(tmp_path: Path) -> None:
    """The transcript holds confirmed data answers and lives indefinitely."""
    from ouroboros.bigbang.interview import InterviewEngine, InterviewState

    state_dir = tmp_path / "data"
    state_dir.mkdir(mode=0o755)
    os.chmod(state_dir, 0o755)

    engine: Any = InterviewEngine.__new__(InterviewEngine)
    engine.state_dir = state_dir
    state = InterviewState(interview_id="iv_owner_only", initial_context="ctx")
    asyncio.run(engine.save_state(state))

    saved = state_dir / "interview_iv_owner_only.json"
    assert _mode(saved) == 0o600
    assert _mode(state_dir) == 0o700


def test_overwriting_an_existing_open_file_upgrades_it(tmp_path: Path) -> None:
    """The creation mode is ignored for a file that already exists (round-60).

    A Seed or transcript left at 0644 by an earlier version must not keep that
    mode just because the write reuses the inode.
    """
    target = tmp_path / "existing.yaml"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o644)

    write_owner_only(target, "new")

    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == "new"


def test_writing_does_not_re_permission_the_callers_directory(tmp_path: Path) -> None:
    """A Seed goes wherever the caller asks — often a shared project directory.

    Narrowing that directory would be this package changing something that is
    not its own (round-60). Only the file it writes is its business.
    """
    project = tmp_path / "project"
    project.mkdir(mode=0o755)
    os.chmod(project, 0o755)

    write_owner_only(project / "seed.yaml", "seed: {}")

    assert _mode(project) == 0o755
    assert _mode(project / "seed.yaml") == 0o600


def test_seed_save_leaves_the_target_directory_alone(tmp_path: Path) -> None:
    """The Seed writer must not chmod a caller-controlled parent."""
    from ouroboros.bigbang.seed_generator import save_seed_sync
    from ouroboros.core.seed import (
        EvaluationPrinciple,
        ExitCondition,
        OntologySchema,
        Seed,
        SeedMetadata,
    )

    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    os.chmod(shared, 0o755)
    seed_path = shared / "seed.yaml"

    seed = Seed(
        metadata=SeedMetadata(),
        goal="ship the lane",
        task_type="code",
        constraints=("Python 3.14+",),
        acceptance_criteria=("The lane answers",),
        ontology_schema=OntologySchema(name="lane", description="lane domain"),
        evaluation_principles=(
            EvaluationPrinciple(name="completeness", description="all done", weight=1.0),
        ),
        exit_conditions=(
            ExitCondition(name="done", description="criteria pass", criteria="100% satisfied"),
        ),
    )
    result = save_seed_sync(seed, seed_path)
    assert result.is_ok, result.error if result.is_err else None

    assert _mode(shared) == 0o755
    assert _mode(seed_path) == 0o600
