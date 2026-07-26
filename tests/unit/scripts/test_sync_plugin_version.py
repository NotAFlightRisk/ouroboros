"""Tests for sync-plugin-version script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "sync-plugin-version.py"
_spec = importlib.util.spec_from_file_location("sync_plugin_version", str(_SCRIPT_PATH))
assert _spec is not None
sync_plugin_version = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sync_plugin_version)


def test_git_describe_fallback_uses_hatch_vcs_next_dev_version() -> None:
    assert sync_plugin_version.version_from_git_describe("v0.39.1-28-gc05024d6") == ("0.39.2.dev28")


def test_git_describe_fallback_preserves_exact_tag_version() -> None:
    assert sync_plugin_version.version_from_git_describe("v0.39.1") == "0.39.1"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.39.2.dev28", "0.39.2"),
        ("1.2.3a1", "1.2.3a1"),
        ("1.2.3b4", "1.2.3b4"),
        ("1.2.3rc2", "1.2.3rc2"),
        ("1.2.3b4.dev1", "1.2.3b4"),
    ],
)
def test_plugin_metadata_version_normalizes_supported_versions(
    version: str,
    expected: str,
) -> None:
    assert sync_plugin_version.normalize_version(version) == expected


@pytest.mark.parametrize(
    "version",
    ["not-a-version", "1.2", "1.2.3garbage", "١.٢.٣", "1.2.3\n"],
)
def test_plugin_metadata_version_rejects_unsupported_versions(version: str) -> None:
    with pytest.raises(ValueError, match="unsupported version"):
        sync_plugin_version.normalize_version(version)


def test_main_write_updates_both_setup_skill_markers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)

    source_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.4"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.4"}]}\n')

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    sync_plugin_version.main()

    captured = capsys.readouterr()
    assert "WRITE skills/setup/SKILL.md (0.39.1 -> 1.2.4)" in captured.out
    assert "WRITE .claude-plugin/skills/setup/SKILL.md (0.39.1 -> 1.2.4)" in captured.out
    assert source_skill.read_text() == "<!-- ooo:VERSION:1.2.4 -->\nsource\n"
    assert bundled_skill.read_text() == "<!-- ooo:VERSION:1.2.4 -->\nbundled\n"


def test_update_version_marker_preserves_unrelated_bytes_with_bom_and_crlf(tmp_path: Path) -> None:
    marker = b"<!-- ooo:VERSION:1.2.3 -->"
    target = tmp_path / "SKILL.md"
    original = b"\xef\xbb\xbfheading\r\n" + marker + "\r\nbody café\r\n".encode()
    target.write_bytes(original)

    assert sync_plugin_version.update_version_marker(target, "1.2.4") is True

    assert target.read_bytes() == original.replace(marker, b"<!-- ooo:VERSION:1.2.4 -->")


def test_main_write_fails_when_required_setup_skill_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nsource\n")
    plugin_json.write_text('{"version": "1.2.4"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.4"}]}\n')

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    try:
        sync_plugin_version.main()
    except SystemExit as exc:
        assert "required setup skill not found" in str(exc)
    else:
        raise AssertionError("missing required setup skill must fail")


def test_main_write_fails_when_setup_marker_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("source without marker\n")
    bundled_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.4"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.4"}]}\n')
    original_plugin_json = plugin_json.read_text()
    original_marketplace_json = marketplace_json.read_text()

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    try:
        sync_plugin_version.main()
    except SystemExit as exc:
        assert "expected exactly one version marker" in str(exc)
        assert plugin_json.read_text() == original_plugin_json
        assert marketplace_json.read_text() == original_marketplace_json
    else:
        raise AssertionError("missing version marker must fail")


def test_main_write_preflights_json_targets_before_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [}\n')
    original_plugin_json = plugin_json.read_text()

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    try:
        sync_plugin_version.main()
    except SystemExit as exc:
        assert "could not validate" in str(exc)
        assert plugin_json.read_text() == original_plugin_json
    else:
        raise AssertionError("invalid marketplace JSON must fail before mutation")


def test_update_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    target = tmp_path / "plugin.json"
    original = b'{"version": "1.2.3", "version": "9.9.9"}\n'
    target.write_bytes(original)

    with pytest.raises(ValueError, match="duplicate JSON object key: version"):
        sync_plugin_version.update_json(target, "1.2.4")

    assert target.read_bytes() == original


def test_main_write_preflight_rejects_duplicate_json_keys_before_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills/setup/SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin/skills/setup/SKILL.md"
    plugin_json = tmp_path / ".claude-plugin/plugin.json"
    marketplace_json = tmp_path / ".claude-plugin/marketplace.json"
    source_skill.parent.mkdir(parents=True)
    bundled_skill.parent.mkdir(parents=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_bytes(b"<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_bytes(b"<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_bytes(b'{"version":"1.2.3"}\n')
    marketplace_json.write_bytes(b'{"plugins":[{"version":"1.2.3","version":"9.9.9"}]}\n')
    originals = {path: path.read_bytes() for path in (source_skill, bundled_skill, plugin_json)}
    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    with pytest.raises(SystemExit, match="duplicate JSON object key: version"):
        sync_plugin_version.main()

    assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize("missing_target", ["plugin", "marketplace"])
def test_main_write_fails_before_mutation_when_required_json_is_missing(
    tmp_path: Path,
    monkeypatch,
    missing_target: str,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:0.39.1 -->\nbundled\n")
    if missing_target != "plugin":
        plugin_json.write_text('{"version": "1.2.3"}\n')
    if missing_target != "marketplace":
        marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')

    original_source = source_skill.read_text()
    original_bundled = bundled_skill.read_text()
    existing_json = marketplace_json if missing_target == "plugin" else plugin_json
    original_json = existing_json.read_text()

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    with pytest.raises(SystemExit, match="required plugin metadata not found"):
        sync_plugin_version.main()

    assert source_skill.read_text() == original_source
    assert bundled_skill.read_text() == original_bundled
    assert existing_json.read_text() == original_json


@pytest.mark.parametrize(
    ("plugin_version", "marketplace_version"),
    [
        (None, "1.2.3"),
        (123, "1.2.3"),
        ("1.2.3", None),
        ("1.2.3", 123),
    ],
)
def test_main_write_rejects_missing_or_non_string_json_version_before_mutation(
    tmp_path: Path,
    monkeypatch,
    plugin_version: object,
    marketplace_version: object,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")

    plugin_payload = {} if plugin_version is None else {"version": plugin_version}
    marketplace_plugin = {} if marketplace_version is None else {"version": marketplace_version}
    plugin_json.write_text(__import__("json").dumps(plugin_payload) + "\n")
    marketplace_json.write_text(__import__("json").dumps({"plugins": [marketplace_plugin]}) + "\n")
    original_plugin = plugin_json.read_text()
    original_marketplace = marketplace_json.read_text()
    original_source = source_skill.read_text()
    original_bundled = bundled_skill.read_text()

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    with pytest.raises(SystemExit, match="version must be a string"):
        sync_plugin_version.main()

    assert plugin_json.read_text() == original_plugin
    assert marketplace_json.read_text() == original_marketplace
    assert source_skill.read_text() == original_source
    assert bundled_skill.read_text() == original_bundled


def test_main_write_rejects_unsupported_source_version_before_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')
    originals = {
        path: path.read_text()
        for path in (source_skill, bundled_skill, plugin_json, marketplace_json)
    }

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.3garbage"],
    )

    with pytest.raises(SystemExit, match="unsupported version"):
        sync_plugin_version.main()

    for path, original in originals.items():
        assert path.read_text() == original


@pytest.mark.parametrize("marker_version", ["stale.version", "abc", "1..2", "1.2.3.4"])
def test_main_write_rejects_malformed_single_marker_value_before_mutation(
    tmp_path: Path,
    monkeypatch,
    marker_version: str,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text(f"<!-- ooo:VERSION:{marker_version} -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')
    originals = {
        path: path.read_bytes()
        for path in (source_skill, bundled_skill, plugin_json, marketplace_json)
    }

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    with pytest.raises(SystemExit, match="expected exactly one version marker"):
        sync_plugin_version.main()

    for path, original in originals.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize(
    "source_text",
    [
        "<!-- ooo:VERSION:stale-version -->\n<!-- ooo:VERSION:1.2.3 -->\nsource\n",
        "<!-- ooo:VERSION:stale\nversion -->\n<!-- ooo:VERSION:1.2.3 -->\nsource\n",
        "<!-- ooo:VERSION:1.2.3 -->\n<!-- ooo:VERSION:stale\nversion-->\nsource\n",
        "<!-- ooo:VERSION:1.2.3 -->\n<!-- ooo:VERSION:stale\nversion\nsource\n",
    ],
)
def test_main_write_rejects_malformed_duplicate_marker_before_mutation(
    tmp_path: Path,
    monkeypatch,
    source_text: str,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text(source_text)
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')
    originals = {
        path: path.read_text()
        for path in (source_skill, bundled_skill, plugin_json, marketplace_json)
    }

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    with pytest.raises(SystemExit, match="expected exactly one version marker"):
        sync_plugin_version.main()

    for path, original in originals.items():
        assert path.read_text() == original


def test_atomic_write_keeps_target_intact_when_temp_fsync_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target.json"
    original = b'{"version": "1.2.3"}\n'
    target.write_bytes(original)

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(sync_plugin_version.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        sync_plugin_version._atomic_write_bytes(target, b'{"version": "1.2.4"}\n')

    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_applies_mode_to_open_temp_before_file_fsync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"old")
    target.chmod(0o640)
    calls: list[tuple[str, int]] = []
    real_fchmod = sync_plugin_version.os.fchmod
    real_fsync = sync_plugin_version.os.fsync

    def record_fchmod(fd: int, mode: int) -> None:
        calls.append(("fchmod", mode))
        real_fchmod(fd, mode)

    def record_fsync(fd: int) -> None:
        calls.append(("fsync", fd))
        real_fsync(fd)

    monkeypatch.setattr(sync_plugin_version.os, "fchmod", record_fchmod)
    monkeypatch.setattr(sync_plugin_version.os, "fsync", record_fsync)

    sync_plugin_version._atomic_write_bytes(target, b"new")

    assert calls[0] == ("fchmod", 0o640)
    assert calls[1][0] == "fsync"


def test_atomic_write_rejects_target_changed_since_preflight(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    original = b'{"version":"1.2.3"}\n'
    external = b'{"version":"1.2.3","note":"external"}\n'
    target.write_bytes(external)

    with pytest.raises(RuntimeError, match="write conflict"):
        sync_plugin_version._atomic_write_bytes(
            target,
            b'{"version":"1.2.4"}\n',
            expected_current=original,
        )

    assert target.read_bytes() == external


def test_main_write_rolls_back_when_later_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')
    originals = {
        path: path.read_bytes()
        for path in (source_skill, bundled_skill, plugin_json, marketplace_json)
    }

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    original_update_json = sync_plugin_version.update_json
    calls = 0

    def fail_second_json_write(
        path: Path,
        version: str,
        *,
        nested_key=None,
        expected_current=None,
    ) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected later write failure")
        return original_update_json(
            path,
            version,
            nested_key=nested_key,
            expected_current=expected_current,
        )

    monkeypatch.setattr(sync_plugin_version, "update_json", fail_second_json_write)

    with pytest.raises(OSError, match="injected later write failure"):
        sync_plugin_version.main()

    for path, original in originals.items():
        assert path.read_bytes() == original


def test_main_write_does_not_clobber_external_edit_during_rollback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills/setup/SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin/skills/setup/SKILL.md"
    plugin_json = tmp_path / ".claude-plugin/plugin.json"
    marketplace_json = tmp_path / ".claude-plugin/marketplace.json"
    source_skill.parent.mkdir(parents=True)
    bundled_skill.parent.mkdir(parents=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version":"1.2.3"}\n')
    marketplace_json.write_text('{"plugins":[{"version":"1.2.3"}]}\n')
    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )
    original_update_json = sync_plugin_version.update_json
    calls = 0
    external = b'{"version":"external"}\n'

    def fail_after_external_edit(
        path: Path,
        version: str,
        *,
        nested_key=None,
        expected_current=None,
    ) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            plugin_json.write_bytes(external)
            raise OSError("injected later write failure")
        return original_update_json(
            path,
            version,
            nested_key=nested_key,
            expected_current=expected_current,
        )

    monkeypatch.setattr(sync_plugin_version, "update_json", fail_after_external_edit)

    with pytest.raises(BaseExceptionGroup, match="rollback was incomplete") as raised:
        sync_plugin_version.main()

    assert plugin_json.read_bytes() == external
    assert any("rollback conflict" in str(exc) for exc in raised.value.exceptions)


def test_main_write_rejects_external_edit_after_preflight(tmp_path: Path, monkeypatch) -> None:
    source_skill = tmp_path / "skills/setup/SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin/skills/setup/SKILL.md"
    plugin_json = tmp_path / ".claude-plugin/plugin.json"
    marketplace_json = tmp_path / ".claude-plugin/marketplace.json"
    source_skill.parent.mkdir(parents=True)
    bundled_skill.parent.mkdir(parents=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version":"1.2.3"}\n')
    marketplace_json.write_text('{"plugins":[{"version":"1.2.3"}]}\n')
    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )
    original_update_json = sync_plugin_version.update_json
    external = b'{"version":{"external":"must-survive"}}\n'
    edited = False

    def edit_after_preflight(
        path: Path,
        version: str,
        *,
        nested_key=None,
        expected_current=None,
    ) -> bool:
        nonlocal edited
        if not edited:
            edited = True
            plugin_json.write_bytes(external)
        return original_update_json(
            path,
            version,
            nested_key=nested_key,
            expected_current=expected_current,
        )

    monkeypatch.setattr(sync_plugin_version, "update_json", edit_after_preflight)

    with pytest.raises(BaseExceptionGroup, match="rollback was incomplete") as raised:
        sync_plugin_version.main()

    assert plugin_json.read_bytes() == external
    assert any("write conflict" in str(exc) for exc in raised.value.exceptions)


def test_main_write_rolls_back_only_successfully_replaced_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    original_atomic_write = sync_plugin_version._atomic_write_bytes
    rollback_attempts: list[Path] = []

    def fail_primary_and_two_rollbacks(
        path: Path,
        content: bytes,
        *,
        expected_current: bytes | None = None,
    ) -> None:
        if b"1.2.4" in content and path == marketplace_json:
            raise OSError("primary marketplace failure")
        if b"1.2.3" in content:
            rollback_attempts.append(path)
            if path == plugin_json:
                raise OSError("plugin rollback failure")
            if path == source_skill:
                raise OSError("source rollback failure")
        original_atomic_write(path, content, expected_current=expected_current)

    monkeypatch.setattr(
        sync_plugin_version,
        "_atomic_write_bytes",
        fail_primary_and_two_rollbacks,
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        sync_plugin_version.main()

    assert [str(exc) for exc in raised.value.exceptions] == [
        "primary marketplace failure",
        f"failed to roll back {plugin_json}: plugin rollback failure",
    ]
    assert rollback_attempts == [plugin_json]


def test_main_write_ignores_untrusted_legacy_transaction_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills/setup/SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin/skills/setup/SKILL.md"
    plugin_json = tmp_path / ".claude-plugin/plugin.json"
    marketplace_json = tmp_path / ".claude-plugin/marketplace.json"
    source_skill.parent.mkdir(parents=True)
    bundled_skill.parent.mkdir(parents=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version":"1.2.3"}\n')
    marketplace_json.write_text('{"plugins":[{"version":"1.2.3"}]}\n')
    forged = tmp_path / ".sync-plugin-version.transaction.json"
    forged.write_text('{"originals":{".claude-plugin/plugin.json":"Zm9yZ2Vk"}}\n')
    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )

    sync_plugin_version.main()

    assert forged.read_text() == '{"originals":{".claude-plugin/plugin.json":"Zm9yZ2Vk"}}\n'
    assert __import__("json").loads(plugin_json.read_text())["version"] == "1.2.4"
    assert __import__("json").loads(marketplace_json.read_text())["plugins"][0]["version"] == (
        "1.2.4"
    )


def test_concurrent_invocations_are_serialized_by_os_lock(tmp_path: Path) -> None:
    source_skill = tmp_path / "skills/setup/SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin/skills/setup/SKILL.md"
    plugin_json = tmp_path / ".claude-plugin/plugin.json"
    marketplace_json = tmp_path / ".claude-plugin/marketplace.json"
    source_skill.parent.mkdir(parents=True)
    bundled_skill.parent.mkdir(parents=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version":"1.2.3"}\n')
    marketplace_json.write_text('{"plugins":[{"version":"1.2.3"}]}\n')
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    runner = """
import importlib.util
from pathlib import Path
import sys
import time
script, root, version, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location("sync_plugin_version_process", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = root
module.PLUGIN_JSON = root / ".claude-plugin/plugin.json"
module.MARKETPLACE_JSON = root / ".claude-plugin/marketplace.json"
module.SETUP_SKILL_MD = root / "skills/setup/SKILL.md"
module.BUNDLED_SETUP_SKILL_MD = root / ".claude-plugin/skills/setup/SKILL.md"
if mode == "hold":
    original = module.update_json
    def hold_first_write(path, value, *, nested_key=None, expected_current=None):
        if not (root / "ready").exists():
            (root / "ready").touch()
            while not (root / "release").exists():
                time.sleep(0.01)
        return original(
            path,
            value,
            nested_key=nested_key,
            expected_current=expected_current,
        )
    module.update_json = hold_first_write
module.sys.argv = ["sync-plugin-version.py", "--write", "--version", version]
module.main()
"""
    first = subprocess.Popen(
        [sys.executable, "-c", runner, str(_SCRIPT_PATH), str(tmp_path), "1.2.4", "hold"]
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    second = subprocess.Popen(
        [sys.executable, "-c", runner, str(_SCRIPT_PATH), str(tmp_path), "1.2.5", "normal"]
    )
    time.sleep(0.25)
    try:
        assert second.poll() is None
    finally:
        release.touch()
        first.wait(timeout=5)
        second.wait(timeout=5)

    assert first.returncode == second.returncode == 0
    assert __import__("json").loads(plugin_json.read_text())["version"] == "1.2.5"
    assert __import__("json").loads(marketplace_json.read_text())["plugins"][0]["version"] == (
        "1.2.5"
    )
    assert b"VERSION:1.2.5" in source_skill.read_bytes()
    assert b"VERSION:1.2.5" in bundled_skill.read_bytes()


def test_lock_file_is_outside_checkout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)

    lock_path = sync_plugin_version._lock_path()

    assert lock_path.parent != tmp_path


def test_release_workflow_checks_tag_metadata_before_build() -> None:
    workflow = (_SCRIPT_PATH.parents[1] / ".github/workflows/release.yml").read_text()

    validation = workflow.index("python scripts/sync-plugin-version.py")
    build = workflow.index("uv build")
    tui_job = workflow[workflow.index("  build-tui:") : workflow.index("  attach-tui-binaries:")]

    assert "${REF_NAME#v}" in workflow
    assert validation < build
    assert "needs: release" in tui_job


def test_main_write_rolls_back_when_marker_update_reports_no_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_skill = tmp_path / "skills" / "setup" / "SKILL.md"
    bundled_skill = tmp_path / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
    plugin_json = tmp_path / ".claude-plugin" / "plugin.json"
    marketplace_json = tmp_path / ".claude-plugin" / "marketplace.json"

    source_skill.parent.mkdir(parents=True, exist_ok=True)
    bundled_skill.parent.mkdir(parents=True, exist_ok=True)
    plugin_json.parent.mkdir(parents=True, exist_ok=True)
    source_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nsource\n")
    bundled_skill.write_text("<!-- ooo:VERSION:1.2.3 -->\nbundled\n")
    plugin_json.write_text('{"version": "1.2.3"}\n')
    marketplace_json.write_text('{"plugins": [{"version": "1.2.3"}]}\n')
    originals = {
        path: path.read_bytes()
        for path in (source_skill, bundled_skill, plugin_json, marketplace_json)
    }

    monkeypatch.setattr(sync_plugin_version, "ROOT", tmp_path)
    monkeypatch.setattr(sync_plugin_version, "PLUGIN_JSON", plugin_json)
    monkeypatch.setattr(sync_plugin_version, "MARKETPLACE_JSON", marketplace_json)
    monkeypatch.setattr(sync_plugin_version, "SETUP_SKILL_MD", source_skill)
    monkeypatch.setattr(sync_plugin_version, "BUNDLED_SETUP_SKILL_MD", bundled_skill)
    monkeypatch.setattr(
        sync_plugin_version.sys,
        "argv",
        ["sync-plugin-version.py", "--write", "--version", "1.2.4"],
    )
    monkeypatch.setattr(
        sync_plugin_version,
        "update_version_marker",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(SystemExit, match="failed to update"):
        sync_plugin_version.main()

    for path, original in originals.items():
        assert path.read_bytes() == original
