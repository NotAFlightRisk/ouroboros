"""Implementation of the ``ouroboros setup refresh`` command."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import typer

from ouroboros.hermes.artifacts import HERMES_SKILL_CATEGORY, HERMES_SKILL_NAME
from ouroboros.runtime_instruction_artifacts import (
    copilot_instruction_path,
    gemini_instruction_path,
    gjc_agent_dir,
    gjc_instruction_path,
    has_managed_section,
    kiro_instruction_path,
    opencode_instruction_path,
)


def refresh_runtime_artifacts() -> None:
    """Refresh artifacts that a previous setup already installed."""
    # Import lazily to keep setup.py's command registration and test patch
    # points stable while avoiding a module-import cycle.
    from ouroboros.cli.commands import setup

    refreshed: list[str] = []
    failed: list[str] = []

    codex_dir = setup._codex_home_candidate_for_setup()
    if codex_dir.exists() or shutil.which("codex"):
        from ouroboros.codex import install_codex_artifacts

        try:
            result = install_codex_artifacts(codex_dir=codex_dir, prune=False)
        except OSError as exc:
            setup.print_warning(f"Could not refresh Codex artifacts: {exc}")
            failed.append("codex")
        else:
            setup.print_success(f"Installed Codex rules → {result.rules_path}")
            setup.print_success(
                f"Installed {len(result.skill_paths)} Codex skills → {codex_dir / 'skills'}"
            )
            refreshed.append("codex")

    hermes_skill_dir = Path.home() / ".hermes" / "skills" / HERMES_SKILL_CATEGORY
    if os.path.lexists(hermes_skill_dir / HERMES_SKILL_NAME) or any(
        hermes_skill_dir.glob(f".{HERMES_SKILL_NAME}.old.*.intent")
    ):
        if not setup._install_hermes_artifacts():
            failed.append("hermes")
        else:
            refreshed.append("hermes")

    opencode_dir = setup.opencode_config_dir()
    opencode_expected = False
    opencode_succeeded = True
    if setup._opencode_bridge_dest().exists():
        opencode_expected = True
        opencode_succeeded = setup._install_opencode_bridge_plugin() and opencode_succeeded
    if has_managed_section(opencode_instruction_path(opencode_dir)):
        opencode_expected = True
        opencode_succeeded = (
            setup._install_runtime_instruction_artifact("opencode", config_dir=opencode_dir)
            and opencode_succeeded
        )
    if opencode_expected:
        (refreshed if opencode_succeeded else failed).append("opencode")

    if has_managed_section(gemini_instruction_path()):
        target = refreshed if setup._install_runtime_instruction_artifact("gemini") else failed
        target.append("gemini")

    if kiro_instruction_path().exists():
        target = refreshed if setup._install_runtime_instruction_artifact("kiro") else failed
        target.append("kiro")

    if copilot_instruction_path().exists():
        target = refreshed if setup._install_runtime_instruction_artifact("copilot") else failed
        target.append("copilot")

    pi_bridge = Path.home() / ".pi" / "agent" / "extensions" / setup._PI_OOO_BRIDGE_FILENAME
    if pi_bridge.exists():
        if setup._install_pi_ooo_bridge():
            refreshed.append("pi")
        else:
            failed.append("pi")

    gjc_expected = False
    gjc_succeeded = True
    gjc_root = gjc_agent_dir()
    gjc_skills = gjc_root / "skills"
    gjc_bridge = gjc_root / "extensions" / "ouroboros-ooo-bridge" / "index.ts"
    has_projected_skill = gjc_skills.is_dir() and any(
        path.name.startswith("ouroboros-") for path in gjc_skills.iterdir()
    )
    from ouroboros.cli.gjc_setup import (
        gjc_mcp_bridge_config_path,
        is_setup_managed_gjc_mcp_bridge_config,
        is_setup_managed_gjc_mcp_entry,
        persisted_gjc_mcp_entry,
    )

    bridge_config = gjc_mcp_bridge_config_path()
    has_managed_bridge_config = is_setup_managed_gjc_mcp_bridge_config(bridge_config)
    has_managed_registration = is_setup_managed_gjc_mcp_entry(persisted_gjc_mcp_entry())
    gjc_expected = (
        gjc_bridge.exists()
        or has_projected_skill
        or gjc_instruction_path().exists()
        or has_managed_bridge_config
        or has_managed_registration
    )
    if gjc_expected:
        from ouroboros.config import get_gjc_cli_path

        gjc_path = get_gjc_cli_path() or shutil.which("gjc")
        if gjc_path is None:
            setup.print_warning("Could not refresh the installed GJC route: GJC CLI was not found.")
            gjc_succeeded = False
        else:
            gjc_succeeded = setup._install_gjc_runtime_artifacts(gjc_path)
        (refreshed if gjc_succeeded else failed).append("gjc")

    if refreshed:
        setup.print_success(f"Refreshed runtime artifacts: {', '.join(refreshed)}")
    elif not failed:
        setup.print_info("No installed runtime artifacts found to refresh.")
    if failed:
        setup.print_warning(f"Runtime artifact refresh incomplete: {', '.join(failed)}")
        raise typer.Exit(1)
