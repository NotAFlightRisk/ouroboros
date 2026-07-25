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
