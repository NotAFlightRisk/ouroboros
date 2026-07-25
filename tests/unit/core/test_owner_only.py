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

import pytest

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


def test_the_mode_is_established_not_repaired(tmp_path: Path) -> None:
    """Content reaches disk only through a file created owner-only (round-61).

    The earlier version chmod'd an existing file and suppressed failure, so a
    filesystem or ownership that refused the repair got the new secret at the
    old mode. Establishing the mode at creation removes the failure branch
    entirely, and the replacement is atomic so no reader sees a partial file.
    """
    target = tmp_path / "seed.yaml"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o644)

    write_owner_only(target, "SECRET")

    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == "SECRET"
    # No temporary is left behind on success.
    assert [entry.name for entry in tmp_path.iterdir() if entry.name.startswith(".")] == []


def test_a_failed_write_leaves_nothing_behind(tmp_path: Path, monkeypatch: Any) -> None:
    """A write that cannot complete must not deposit the content anywhere."""
    import ouroboros.core.owner_only as owner_only

    target = tmp_path / "seed.yaml"
    original_write = os.fdopen

    def _explode(*args: Any, **kwargs: Any) -> Any:
        handle = original_write(*args, **kwargs)
        handle.close()
        raise OSError("disk full")

    monkeypatch.setattr(owner_only.os, "fdopen", _explode)
    try:
        owner_only.write_owner_only(target, "SECRET")
    except OSError:
        pass
    else:  # pragma: no cover - the monkeypatch always raises
        raise AssertionError("expected the write to fail")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_every_transport_that_persists_a_transcript_is_owner_only(tmp_path: Path) -> None:
    """The plugin interview path writes the same transcript the stdio one does.

    Round-62 found it still on ``write_text``: a fourth persistence site behind
    three that had been converted. The class is "files carrying interview,
    PM, Seed, or fan-out content", and every member of it goes through the
    owner-only writer.
    """
    import asyncio

    from ouroboros.bigbang.interview import InterviewState
    from ouroboros.mcp.tools.authoring_handlers import _plugin_save_state

    state_dir = tmp_path / "data"
    state_dir.mkdir(mode=0o755)
    os.chmod(state_dir, 0o755)

    state = InterviewState(interview_id="iv_plugin", initial_context="ctx")
    result = asyncio.run(_plugin_save_state(state_dir, state))
    assert result.is_ok, result.error if result.is_err else None

    saved = Path(result.value)
    assert _mode(saved) == 0o600
    assert _mode(saved.parent) == 0o700


def test_owner_only_write_reports_durability(tmp_path: Path) -> None:
    """The owner-only write is also the durable write.

    Round-64 merged two independently-added guarantees: main gained an atomic
    write that fsyncs and reports durability, this branch gained owner-only
    creation. Resolving the conflict either way would have dropped one of
    them silently, so the write returns the durability signal it replaced.
    """
    target = tmp_path / "state.json"
    assert write_owner_only(target, "{}") is True
    assert _mode(target) == 0o600


def test_owner_only_write_does_not_inherit_an_existing_open_mode(tmp_path: Path) -> None:
    """The mode is established at creation, never carried over from the target.

    An atomic-write helper conventionally preserves the previous mode. That is
    exactly what would keep a 0644 transcript written by an older version
    world-readable forever.
    """
    target = tmp_path / "legacy.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o644)
    assert _mode(target) == 0o644

    write_owner_only(target, '{"round": 64}')

    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == '{"round": 64}'


def test_durability_is_reported_not_swallowed(tmp_path: Path, monkeypatch: Any) -> None:
    """A directory fsync that genuinely fails is surfaced, not suppressed.

    The caller logs on an unconfirmed write; if the failure were swallowed the
    log would claim a durability the filesystem never gave.
    """
    import errno as _errno

    from ouroboros.core import owner_only

    real_fsync = os.fsync
    target = tmp_path / "state.json"

    def _fail_directory_fsync(fd: int) -> None:
        if os.path.isdir(f"/dev/fd/{fd}") or stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(_errno.EIO, "directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(owner_only.os, "fsync", _fail_directory_fsync)

    assert write_owner_only(target, "{}") is False
    # The content is still written and still owner-only — only the durability
    # claim is withheld.
    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == "{}"


def test_no_descriptor_leaks_when_the_file_object_cannot_be_created(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The raw descriptor is closed when nothing takes ownership of it.

    Between ``os.open`` and ``os.fdopen`` the descriptor belongs to no file
    object, so a failure there leaks it — a long-lived server would exhaust
    its descriptors one failed transcript write at a time.
    """
    from ouroboros.core import owner_only

    opened: list[int] = []
    real_open = os.open

    def _recording_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        fd = real_open(path, flags, *args, **kwargs)
        if flags & os.O_EXCL:
            opened.append(fd)
        return fd

    def _failing_fdopen(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fdopen failed")

    monkeypatch.setattr(owner_only.os, "open", _recording_open)
    monkeypatch.setattr(owner_only.os, "fdopen", _failing_fdopen)

    target = tmp_path / "state.json"
    try:
        write_owner_only(target, "{}")
    except RuntimeError:
        pass
    else:  # pragma: no cover - the write must not succeed here
        raise AssertionError("expected the fdopen failure to propagate")

    assert opened, "the temporary was never created"
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
