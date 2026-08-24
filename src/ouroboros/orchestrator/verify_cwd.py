"""Deterministic working-directory resolution for verification commands."""

from __future__ import annotations

from pathlib import Path
import shlex

from ouroboros.core.seed import AcceptanceCriterionSpec
from ouroboros.orchestrator.evidence.shell_parsing import (
    _single_command_after_safe_shell_preamble,
    _strip_env_prefix,
)

_IGNORED_MANIFEST_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "node_modules"}
)
_NODE_PACKAGE_RUNNERS = frozenset({"npm", "npx", "yarn", "pnpm"})


def _sole_node_manifest_directory(root: Path) -> Path | None:
    if (root / "package.json").is_file():
        return None
    candidates: list[Path] = []
    for manifest in root.rglob("package.json"):
        if (
            any(part in _IGNORED_MANIFEST_DIRECTORIES for part in manifest.relative_to(root).parts)
            or not manifest.is_file()
        ):
            continue
        candidates.append(manifest.parent)
        if len(candidates) > 1:
            return None
    return candidates[0] if candidates else None


def _verify_command_executable(command: str) -> str:
    try:
        parts = _strip_env_prefix(shlex.split(command))
    except ValueError:
        return ""
    if parts and Path(parts[0]).name in {"command", "exec"}:
        parts = parts[1:]
        if parts and parts[0] == "--":
            parts = parts[1:]
        parts = _strip_env_prefix(parts)
    inner_command = _single_command_after_safe_shell_preamble(shlex.join(parts))
    if inner_command is not None:
        try:
            parts = _strip_env_prefix(shlex.split(inner_command))
        except ValueError:
            return ""
        if parts and Path(parts[0]).name in {"command", "exec"}:
            parts = parts[1:]
            if parts and parts[0] == "--":
                parts = parts[1:]
            parts = _strip_env_prefix(parts)
    return Path(parts[0]).name if parts else ""


def resolve_verify_command_cwd(
    root_cwd: str, spec: AcceptanceCriterionSpec
) -> tuple[str, str | None]:
    """Resolve explicit verify_cwd or the sole nested Node manifest directory."""
    root = Path(root_cwd).expanduser().resolve(strict=False)
    if spec.verify_cwd:
        target = (root / spec.verify_cwd).resolve(strict=False)
        if not target.is_relative_to(root):
            return root_cwd, f"verify_cwd escapes the workspace: {spec.verify_cwd!r}"
        if not target.is_dir():
            return root_cwd, f"verify_cwd does not exist in the workspace: {spec.verify_cwd!r}"
        return str(target), None
    if _verify_command_executable(spec.verify_command or "") in _NODE_PACKAGE_RUNNERS:
        try:
            manifest_dir = _sole_node_manifest_directory(root)
        except OSError:
            manifest_dir = None
        if manifest_dir is not None:
            return str(manifest_dir), None
    return root_cwd, None
