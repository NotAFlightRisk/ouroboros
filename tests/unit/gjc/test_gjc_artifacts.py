"""Tests for GJC-native Ouroboros skill projection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ouroboros.gjc import install_gjc_skills, remove_gjc_skills


def _skill(root: Path, name: str, *, body: str = "# Skill\n") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f'---\nname: {name}\ndescription: "{name} description"\n---\n\n{body}',
        encoding="utf-8",
    )


def _frontmatter(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    closing = raw.find("\n---\n", 4)
    parsed = yaml.safe_load(raw[4:closing])
    assert isinstance(parsed, dict)
    return parsed


def test_installs_namespaced_skills_and_rewrites_cross_skill_references(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    agent_dir = tmp_path / "agent"
    _skill(
        source,
        "ooo",
        body="Read `../welcome/SKILL.md` or invoke /ouroboros:welcome.\n",
    )
    _skill(source, "welcome")

    result = install_gjc_skills(agent_dir=agent_dir, skills_dir=source)

    projected = agent_dir / "skills" / "ouroboros-ooo" / "SKILL.md"
    assert result.skill_paths == (
        agent_dir / "skills" / "ouroboros-ooo",
        agent_dir / "skills" / "ouroboros-welcome",
    )
    assert _frontmatter(projected)["name"] == "ouroboros-ooo"
    assert "bare `ooo`" in str(_frontmatter(projected)["description"])
    content = projected.read_text(encoding="utf-8")
    assert "../ouroboros-welcome/SKILL.md" in content
    assert "/skill:ouroboros-welcome" in content
    assert "/ouroboros:welcome" not in content


def test_refresh_is_idempotent_prunes_only_managed_namespace_and_preserves_user_skills(
    tmp_path: Path,
) -> None:
    old_source = tmp_path / "old-source"
    source = tmp_path / "source"
    agent_dir = tmp_path / "agent"
    _skill(old_source, "stale")
    _skill(source, "interview")
    user_skill = agent_dir / "skills" / "my-skill"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user", encoding="utf-8")
    install_gjc_skills(agent_dir=agent_dir, skills_dir=old_source)
    stale = agent_dir / "skills" / "ouroboros-stale"
    custom_namespaced = agent_dir / "skills" / "ouroboros-custom"
    custom_namespaced.mkdir()
    (custom_namespaced / "SKILL.md").write_text(
        "---\nname: ouroboros-custom\ndescription: user-owned\n---\n",
        encoding="utf-8",
    )

    first = install_gjc_skills(agent_dir=agent_dir, skills_dir=source)
    second = install_gjc_skills(agent_dir=agent_dir, skills_dir=source)

    assert first.skill_paths == second.skill_paths
    assert user_skill.exists()
    assert not stale.exists()
    assert custom_namespaced.exists()


def test_remove_deletes_only_intact_generated_skills(tmp_path: Path) -> None:
    source = tmp_path / "source"
    agent_dir = tmp_path / "agent"
    _skill(source, "interview")
    managed = install_gjc_skills(
        agent_dir=agent_dir, skills_dir=source
    ).skill_paths[0]
    custom_namespaced = agent_dir / "skills" / "ouroboros-custom"
    custom_namespaced.mkdir()
    (custom_namespaced / "SKILL.md").write_text(
        "---\nname: ouroboros-custom\ndescription: user-owned\n---\n",
        encoding="utf-8",
    )
    user_skill = agent_dir / "skills" / "interview"
    user_skill.mkdir()

    removed = remove_gjc_skills(agent_dir=agent_dir)

    assert removed == (managed,)
    assert not managed.exists()
    assert user_skill.exists()
    assert custom_namespaced.exists()


def test_refresh_and_remove_preserve_modified_generated_skill(tmp_path: Path) -> None:
    source = tmp_path / "source"
    agent_dir = tmp_path / "agent"
    _skill(source, "interview")
    projected = install_gjc_skills(
        agent_dir=agent_dir, skills_dir=source
    ).skill_paths[0]
    skill_md = projected / "SKILL.md"
    modified = skill_md.read_text(encoding="utf-8") + "\nOperator notes.\n"
    skill_md.write_text(modified, encoding="utf-8")

    with pytest.raises(OSError, match="non-Ouroboros GJC skill"):
        install_gjc_skills(agent_dir=agent_dir, skills_dir=source)

    assert remove_gjc_skills(agent_dir=agent_dir) == ()
    assert skill_md.read_text(encoding="utf-8") == modified


def test_install_refuses_to_replace_user_owned_namespaced_skill(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    agent_dir = tmp_path / "agent"
    _skill(source, "interview")
    collision = agent_dir / "skills" / "ouroboros-interview"
    collision.mkdir(parents=True)
    (collision / "SKILL.md").write_text(
        "---\nname: ouroboros-interview\ndescription: user-owned\n---\n",
        encoding="utf-8",
    )

    try:
        install_gjc_skills(agent_dir=agent_dir, skills_dir=source)
    except OSError as exc:
        assert "non-Ouroboros GJC skill" in str(exc)
    else:
        raise AssertionError("expected user-owned skill collision to fail closed")
