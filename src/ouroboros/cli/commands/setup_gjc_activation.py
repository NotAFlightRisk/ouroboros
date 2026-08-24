"""Ownership-safe GJC runtime artifact activation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def install_gjc_runtime_artifacts(
    gjc_path: str,
    *,
    registration_state: dict[str, bool] | None,
    bridge_source: str,
    agent_dir: Path,
    instruction_path: Path,
    bridge_config_path: Path,
    atomic_write_text: Callable[[Path, str], None],
    snapshot_path: Callable[..., Any],
    restore_path_snapshot: Callable[[Path, Any], None],
    install_bridge_config: Callable[[], bool],
    install_skills: Callable[[], bool],
    install_instruction: Callable[[], bool],
    register_server: Callable[..., bool],
    remove_legacy_bridge: Callable[[], bool],
    warn: Callable[[str], None],
) -> bool:
    """Activate one complete GJC frontdoor before retiring any prior route."""
    from ouroboros.cli.gjc_setup import (
        gjc_bridge_path,
        gjc_native_mcp_autoload_support,
        install_gjc_compatibility_bridge,
        rollback_gjc_activation,
    )
    from ouroboros.gjc import setup_owned_gjc_skill_paths

    state = registration_state if registration_state is not None else {}
    paths = (
        *setup_owned_gjc_skill_paths(agent_dir=agent_dir),
        instruction_path,
        bridge_config_path,
        gjc_bridge_path(),
    )
    try:
        snapshots = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
        native_support = gjc_native_mcp_autoload_support(gjc_path)
        if native_support is None:
            succeeded = False
            expected = snapshots
        elif not native_support:
            succeeded = install_gjc_compatibility_bridge(bridge_source, atomic_write_text)
            expected = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
        else:
            installed = install_bridge_config() and install_skills() and install_instruction()
            expected = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
            succeeded = bool(installed and register_server(gjc_path, registration_state=state))
            if succeeded:
                succeeded = remove_legacy_bridge()
                expected = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
    except OSError as exc:
        warn(f"Could not install GJC runtime artifacts: {exc}")
        succeeded = False
    if succeeded:
        return True
    if "snapshots" in locals():
        rollback_gjc_activation(
            snapshots,
            expected if "expected" in locals() else snapshots,
            restore_path_snapshot=restore_path_snapshot,
            snapshot_path=snapshot_path,
            registration_state=state,
        )
        for directory in (
            agent_dir / "skills",
            agent_dir / "rules",
            agent_dir / "ouroboros",
            agent_dir / "extensions" / "ouroboros-ooo-bridge",
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    return False
