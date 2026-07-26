#!/usr/bin/env python3
"""Sync .claude-plugin/ version fields with hatch-vcs (git tag) version.

Usage:
    python scripts/sync-plugin-version.py          # dry-run
    python scripts/sync-plugin-version.py --write  # actually update files

Called by CI (dev-publish.yml) before build to keep plugin metadata in sync.

Each target is replaced atomically. Catchable failures roll back files already
replaced, but an uncatchable process or machine crash can leave a mixed version
across targets; ordinary files cannot provide a durable multi-file transaction.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_JSON = ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_JSON = ROOT / ".claude-plugin" / "marketplace.json"
SETUP_SKILL_MD = ROOT / "skills" / "setup" / "SKILL.md"
BUNDLED_SETUP_SKILL_MD = ROOT / ".claude-plugin" / "skills" / "setup" / "SKILL.md"
VERSION_MARKER_RE = re.compile(r"<!-- ooo:VERSION:([0-9A-Za-z.]+) -->")
VERSION_MARKER_ENVELOPE_RE = re.compile(r"<!-- ooo:VERSION:(.*?) -->", re.DOTALL)


def get_version() -> str:
    """Get version from hatch-vcs (same source as the Python package)."""
    # Try hatch first
    try:
        result = subprocess.run(
            ["hatch", "version"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback: parse git describe like hatch-vcs does.
    # dev-publish.yml intentionally runs this script before installing hatch,
    # so this branch must preserve the same next-dev source version that the
    # subsequent hatch-vcs package build will produce.
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--match", "v*"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        return version_from_git_describe(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    sys.exit("Error: cannot determine version (no hatch, no git tags)")


def version_from_git_describe(desc: str) -> str:
    """Return a hatch-vcs compatible version from ``git describe`` output."""
    normalized = desc.removeprefix("v")
    match = re.fullmatch(r"(?P<base>.+)-(?P<distance>\d+)-g[0-9a-f]+(?:-dirty)?", normalized)
    if match is None:
        return normalized
    next_version = _guess_next_dev_base(match.group("base"))
    return f"{next_version}.dev{match.group('distance')}"


def _guess_next_dev_base(version: str) -> str:
    """Approximate hatch-vcs/setuptools-scm ``guess-next-dev`` for tags."""
    match = re.fullmatch(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)", version)
    if match is not None:
        patch = int(match.group("patch")) + 1
        return f"{match.group('major')}.{match.group('minor')}.{patch}"

    prerelease = re.fullmatch(
        r"(?P<prefix>\d+\.\d+\.\d+(?P<label>a|alpha|b|beta|rc))(?P<num>\d+)",
        version,
    )
    if prerelease is not None:
        return f"{prerelease.group('prefix')}{int(prerelease.group('num')) + 1}"

    return version


def normalize_version(v: str) -> str:
    """Normalize version for plugin metadata.

    Keeps pre-release tags (alpha/beta/rc) but strips dev suffixes.
    e.g. 0.26.0b4 -> 0.26.0b4, 0.26.0.dev3 -> 0.26.0, 0.26.0b4.dev1 -> 0.26.0b4
    """
    # Match semver + optional pre-release (a/alpha/b/beta/rc + number)
    # and an optional hatch-vcs development suffix.
    match = re.fullmatch(
        r"(?P<public>[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|alpha|b|beta|rc)[0-9]*)?)"
        r"(?:\.dev[0-9]+)?",
        v,
    )
    if match is None:
        raise ValueError(f"unsupported version: {v}")
    return match.group("public")


def _read_version_marker(text: str, path: Path) -> str:
    """Return one well-formed marker value, rejecting malformed duplicates."""
    if text.count("<!-- ooo:VERSION:") != 1:
        raise ValueError(f"expected exactly one version marker in {path}")
    envelopes = list(VERSION_MARKER_ENVELOPE_RE.finditer(text))
    matches = list(VERSION_MARKER_RE.finditer(text))
    if len(envelopes) != 1 or len(matches) != 1 or envelopes[0].span() != matches[0].span():
        raise ValueError(f"expected exactly one version marker in {path}")
    marker_value = matches[0].group(1)
    try:
        normalized_marker = normalize_version(marker_value)
    except ValueError as exc:
        raise ValueError(f"expected exactly one version marker in {path}") from exc
    if normalized_marker != marker_value:
        raise ValueError(f"expected exactly one version marker in {path}")
    return marker_value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    return json.loads(
        path.read_bytes().decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    expected_current: bytes | None = None,
) -> None:
    """Durably replace one target without exposing a partial file."""
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fchmod(temp_file.fileno(), mode)
            os.fsync(temp_file.fileno())
        if expected_current is not None and path.read_bytes() != expected_current:
            raise RuntimeError(f"write conflict for {path}: file changed since preflight")
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def update_version_marker(
    path: Path,
    version: str,
    *,
    expected_current: bytes | None = None,
) -> bool:
    """Update <!-- ooo:VERSION:X.Y.Z --> marker in a text file."""
    original = expected_current if expected_current is not None else path.read_bytes()
    text = original.decode("utf-8")
    _read_version_marker(text, path)

    old_marker = VERSION_MARKER_RE.search(text)
    assert old_marker is not None
    old_marker_bytes = old_marker.group(0).encode("ascii")
    new_marker_bytes = f"<!-- ooo:VERSION:{version} -->".encode("ascii")
    content = original.replace(old_marker_bytes, new_marker_bytes, 1)
    if original == content:
        return False
    _atomic_write_bytes(path, content, expected_current=original)

    updated = path.read_bytes()
    if updated != content:
        raise RuntimeError(f"failed to verify version marker update in {path}")
    updated_matches = list(VERSION_MARKER_RE.finditer(updated.decode("utf-8")))
    if len(updated_matches) != 1 or updated_matches[0].group(1) != version:
        raise RuntimeError(f"failed to verify version marker update in {path}")
    return True


def update_json(
    path: Path,
    version: str,
    *,
    nested_key: str | None = None,
    expected_current: bytes | None = None,
) -> bool:
    """Update version in a JSON file. Returns True if changed."""
    original = expected_current if expected_current is not None else path.read_bytes()
    data = json.loads(
        original.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(data, dict):
        raise TypeError("top-level JSON value must be an object")

    if nested_key:
        # marketplace.json: plugins[0].version
        target = data
        for key in nested_key.split("."):
            if key.isdigit():
                target = target[int(key)]
            else:
                target = target[key]
        old = target.get("version")
        target["version"] = version
    else:
        old = data.get("version")
        data["version"] = version

    if old == version:
        return False

    content = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(path, content, expected_current=original)
    return True


def _run() -> None:
    write = "--write" in sys.argv

    # Allow explicit version override (e.g. --version 0.26.0b6)
    # Used by the release process to sync BEFORE tagging.
    explicit_version = None
    for i, arg in enumerate(sys.argv):
        if arg == "--version" and i + 1 < len(sys.argv):
            explicit_version = sys.argv[i + 1]

    raw_version = explicit_version or get_version()
    try:
        version = normalize_version(raw_version)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    print(f"Source version: {raw_version}")
    print(f"Plugin version: {version}")
    print()

    targets = [
        (PLUGIN_JSON, None),
        (MARKETPLACE_JSON, "plugins.0"),
    ]
    originals: dict[Path, bytes] = {}
    setup_markers: dict[Path, tuple[str, str]] = {}
    for path in (SETUP_SKILL_MD, BUNDLED_SETUP_SKILL_MD):
        if not path.exists():
            sys.exit(f"Error: required setup skill not found: {path.relative_to(ROOT)}")
        original = path.read_bytes()
        originals[path] = original
        text = original.decode("utf-8")
        try:
            old_marker = _read_version_marker(text, path)
        except ValueError:
            sys.exit(f"Error: expected exactly one version marker in {path.relative_to(ROOT)}")
        setup_markers[path] = (text, old_marker)

    json_targets: list[tuple[Path, str | None, object, str]] = []
    expected_writes: dict[Path, bytes] = {}
    for path, nested in targets:
        if not path.exists():
            sys.exit(f"Error: required plugin metadata not found: {path.relative_to(ROOT)}")

        try:
            original = path.read_bytes()
            originals[path] = original
            data = json.loads(
                original.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
            if not isinstance(data, dict):
                raise TypeError("top-level JSON value must be an object")
            target: object = data
            if nested:
                for key in nested.split("."):
                    if key.isdigit():
                        if not isinstance(target, list):
                            raise TypeError("numeric path component requires an array")
                        target = target[int(key)]
                    else:
                        if not isinstance(target, dict):
                            raise TypeError("named path component requires an object")
                        target = target[key]
            if not isinstance(target, dict):
                raise TypeError("version target must be an object")
            old = target.get("version")
            if not isinstance(old, str):
                raise TypeError("version must be a string")
            target["version"] = version
            expected_writes[path] = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            sys.exit(f"Error: could not validate {path.relative_to(ROOT)}: {exc}")
        json_targets.append((path, nested, data, old))

    changed = False
    for path, (_text, old_marker) in setup_markers.items():
        expected_writes[path] = originals[path].replace(
            f"<!-- ooo:VERSION:{old_marker} -->".encode("ascii"),
            f"<!-- ooo:VERSION:{version} -->".encode("ascii"),
            1,
        )
    attempted: list[Path] = []
    try:
        for path, nested, _data, old in json_targets:
            if old == version:
                print(f"  OK    {path.relative_to(ROOT)} ({old})")
            elif write:
                attempted.append(path)
                update_json(
                    path,
                    version,
                    nested_key=nested,
                    expected_current=originals[path],
                )
                print(f"  WRITE {path.relative_to(ROOT)} ({old} -> {version})")
                changed = True
            else:
                print(f"  DRIFT {path.relative_to(ROOT)} ({old} != {version})")
                changed = True

        for path, (_text, old_marker) in setup_markers.items():
            if old_marker == version:
                print(f"  OK    {path.relative_to(ROOT)} ({old_marker})")
            elif write:
                attempted.append(path)
                updated = update_version_marker(
                    path,
                    version,
                    expected_current=originals[path],
                )
                if not updated:
                    sys.exit(f"Error: failed to update {path.relative_to(ROOT)}")
                print(f"  WRITE {path.relative_to(ROOT)} ({old_marker} -> {version})")
                changed = True
            else:
                print(f"  DRIFT {path.relative_to(ROOT)} ({old_marker} != {version})")
                changed = True
    except BaseException as primary_error:
        if not write:
            raise
        rollback_failures: list[BaseException] = []
        for path in attempted:
            try:
                current = path.read_bytes()
                if current == originals[path]:
                    continue
                if current != expected_writes[path]:
                    raise RuntimeError(
                        f"rollback conflict for {path}: file changed after version sync write"
                    )
                _atomic_write_bytes(
                    path,
                    originals[path],
                    expected_current=expected_writes[path],
                )
            except BaseException as rollback_error:
                failure = RuntimeError(f"failed to roll back {path}: {rollback_error}")
                failure.__cause__ = rollback_error
                rollback_failures.append(failure)
        if rollback_failures:
            raise BaseExceptionGroup(
                "version synchronization failed and rollback was incomplete",
                [primary_error, *rollback_failures],
            ) from None
        raise

    if changed and not write:
        print("\nRun with --write to update files.")
        sys.exit(1)


def _lock_path() -> Path:
    root_key = hashlib.sha256(str(ROOT.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"ouroboros-sync-plugin-version-{root_key}.lock"


def main() -> None:
    """Serialize invocations; the OS releases this lock if the process crashes."""
    lock_path = _lock_path()
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _run()


if __name__ == "__main__":
    main()
