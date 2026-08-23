"""GJC runtime setup primitives shared by setup, refresh, and uninstall."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from ouroboros.cli.formatters.panels import print_error, print_info, print_success, print_warning
from ouroboros.runtime_instruction_artifacts import gjc_agent_dir

_GJC_MCP_BRIDGE_CONFIG_CONTENT = "# Managed by ouroboros setup --runtime gjc\nmcp_servers: []\n"


_GJC_AUTOLOAD_HELP_MARKERS = (
    "ordinary standalone sessions load at startup",
    "registrations are consumed by ordinary standalone gjc sessions at startup",
    "conventional autoload",
)
_GJC_STORAGE_ONLY_HELP_MARKERS = (
    "storage-only",
    "standalone sessions do not load stored registrations",
    "standalone sessions don't load stored registrations",
)


def gjc_supports_standalone_mcp_autoload(
    gjc_path: str,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> bool:
    """Prove that this GJC build loads stored MCP registrations at runtime."""
    try:
        completed = run_command(
            [gjc_path, "mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print_error(f"Could not verify GJC standalone MCP support: {exc}")
        return False
    help_text = f"{completed.stdout}\n{completed.stderr}".lower()
    if completed.returncode == 0 and not any(
        marker in help_text for marker in _GJC_STORAGE_ONLY_HELP_MARKERS
    ) and any(marker in help_text for marker in _GJC_AUTOLOAD_HELP_MARKERS):
        return True
    if any(marker in help_text for marker in _GJC_STORAGE_ONLY_HELP_MARKERS):
        print_error(
            "This GJC release stores MCP registrations but does not load them in ordinary "
            "standalone sessions. Ouroboros setup stopped before changing the GJC projection "
            "or removing the legacy input bridge. Upgrade to a GJC release whose `gjc mcp "
            "--help` documents conventional autoload; until then, run Ouroboros commands "
            "from the `ouroboros` CLI with `--runtime gjc`."
        )
    else:
        print_error(
            "GJC did not provide a verifiable standalone MCP autoload contract. Ouroboros "
            "setup stopped before changing the GJC projection or removing the legacy input "
            "bridge."
        )
    return False


def gjc_mcp_bridge_config_path() -> Path:
    """Return the setup-owned empty upstream bridge config for GJC sessions."""
    return gjc_agent_dir() / "ouroboros" / "mcp-bridge.yaml"


def install_gjc_mcp_bridge_config(
    atomic_write_text: Callable[..., object],
) -> bool:
    """Install the empty bridge config without replacing user-managed content."""
    path = gjc_mcp_bridge_config_path()
    if path.is_symlink() or (path.exists() and not is_setup_managed_gjc_mcp_bridge_config(path)):
        print_info(f"Preserved user-managed GJC MCP bridge config at {path}")
        return True
    try:
        atomic_write_text(path, _GJC_MCP_BRIDGE_CONFIG_CONTENT, mode=0o600)
    except OSError as exc:
        print_warning(f"Could not install GJC MCP bridge config: {exc}")
        return False
    return True


def is_setup_managed_gjc_mcp_bridge_config(path: Path) -> bool:
    """Return whether *path* is the exact setup-owned empty bridge config."""
    try:
        return (
            not path.is_symlink()
            and path.read_text(encoding="utf-8") == _GJC_MCP_BRIDGE_CONFIG_CONTENT
        )
    except (OSError, UnicodeDecodeError):
        return False


def _is_exact_launcher_args(command: str, args: Sequence[object]) -> bool:
    """Match only launcher argv generations emitted by Ouroboros GJC setup."""
    runtime_suffix = ["--runtime", "gjc"]
    package_spec = "ouroboros-ai[mcp]"
    if command == "uvx":
        expected = [
            "--isolated",
            "--python",
            ">=3.12",
            "--from",
            package_spec,
            "ouroboros",
            "mcp",
            "serve",
            *runtime_suffix,
        ]
    elif command == "pipx":
        expected = [
            "run",
            "--spec",
            package_spec,
            "ouroboros",
            "mcp",
            "serve",
            *runtime_suffix,
        ]
    else:
        return False
    return list(args) == expected


def is_setup_managed_gjc_mcp_entry(entry: object) -> bool:
    """Return whether *entry* exactly matches a setup-owned GJC launcher."""
    if not isinstance(entry, dict):
        return False
    config = entry.get("config")
    if not isinstance(config, dict) or config.get("type") != "stdio":
        return False
    command = config.get("command")
    args = config.get("args")
    return (
        isinstance(command, str)
        and isinstance(args, list)
        and _is_exact_launcher_args(command, args)
    )


def is_runtime_loaded_gjc_mcp_entry(entry: object) -> bool:
    """Return whether GJC reports the exact endpoint as runtime-loaded."""
    return (
        is_setup_managed_gjc_mcp_entry(entry)
        and isinstance(entry, dict)
        and entry.get("runtimeStatus") == "autoload"
        and entry.get("runtimeLoadedByStandalone", True) is not False
    )


def build_gjc_mcp_add_command(
    gjc_path: str,
    *,
    command: str,
    server_args: Sequence[str],
) -> list[str]:
    """Build an argv accepted by GJC versions before and after MCP autoload."""
    return [
        gjc_path,
        "mcp",
        "add",
        "ouroboros",
        "--command",
        command,
        *(f"--arg={arg}" for arg in server_args),
        f"--env=OUROBOROS_MCP_CONFIG={gjc_mcp_bridge_config_path()}",
        "--timeout",
        "30000",
        "--json",
    ]


def _listed_gjc_mcp_entry(
    gjc_path: str,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[bool, dict[str, object] | None]:
    """Read GJC's Ouroboros MCP entry without conflating absence with failure."""
    try:
        listed = run_command(
            [gjc_path, "mcp", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print_warning(f"Could not inspect GJC MCP registrations: {exc}")
        return False, None
    if listed.returncode != 0:
        print_warning(f"Could not inspect GJC MCP registrations: {listed.stderr.strip()}")
        return False, None
    try:
        payload = json.loads(listed.stdout)
    except json.JSONDecodeError:
        print_warning("GJC MCP list returned malformed JSON; leaving registrations untouched.")
        return False, None
    servers = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(servers, list):
        print_warning("GJC MCP list returned an invalid server collection.")
        return False, None
    return True, next(
        (
            entry
            for entry in servers
            if isinstance(entry, dict) and entry.get("name") == "ouroboros"
        ),
        None,
    )


def remove_gjc_mcp_server(
    gjc_path: str,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> bool:
    """Remove the Ouroboros GJC MCP entry through GJC's public CLI."""
    try:
        removed = run_command(
            [gjc_path, "mcp", "remove", "ouroboros", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print_warning(f"Could not roll back Ouroboros MCP registration in GJC: {exc}")
        return False
    if removed.returncode != 0:
        print_warning(
            f"Could not roll back Ouroboros MCP registration in GJC: {removed.stderr.strip()}"
        )
        return False
    return True


def register_gjc_mcp_server(
    gjc_path: str,
    *,
    detect_mcp_entry: Callable[..., dict[str, object] | None],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    detected: dict[str, object] | None = None,
    registration_state: dict[str, bool] | None = None,
) -> bool:
    """Register the isolated server only when standalone GJC can load it."""
    if registration_state is not None:
        registration_state.update(created=False, changed=False)
    if not gjc_supports_standalone_mcp_autoload(gjc_path, run_command=run_command):
        return False
    detected = detected or detect_mcp_entry(package_spec="ouroboros-ai[mcp]")
    if detected is None:
        print_error(
            "GJC setup requires an isolated MCP 2 launcher. "
            "Install uv/uvx or pipx, then re-run setup."
        )
        return False

    listed_ok, existing = _listed_gjc_mcp_entry(gjc_path, run_command)
    if not listed_ok:
        return False
    if existing is not None:
        if not is_setup_managed_gjc_mcp_entry(existing):
            print_error(
                "GJC already has a user-managed MCP server named 'ouroboros'. "
                "Preserved it and aborted activation because its tool contract cannot be verified."
            )
            return False
        if not is_runtime_loaded_gjc_mcp_entry(existing):
            print_error(
                "GJC did not report the stored Ouroboros MCP server as loaded by ordinary "
                "standalone sessions. The existing registration and legacy input bridge "
                "were preserved."
            )
            return False
        print_info("Ouroboros MCP server in GJC is already active and up to date.")
        return True
    command = detected.get("command")
    raw_args = detected.get("args")
    if (
        not isinstance(command, str)
        or not isinstance(raw_args, list)
        or not all(isinstance(arg, str) for arg in raw_args)
    ):
        print_warning("Detected Ouroboros MCP launcher is invalid; GJC registration skipped.")
        return False
    server_args = [*raw_args, "--runtime", "gjc"]
    add_command = build_gjc_mcp_add_command(
        gjc_path,
        command=command,
        server_args=server_args,
    )
    try:
        added = run_command(
            add_command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print_warning(f"Could not register Ouroboros MCP server in GJC: {exc}")
        return False
    if added.returncode != 0:
        print_warning(f"Could not register Ouroboros MCP server in GJC: {added.stderr.strip()}")
        return False

    if registration_state is not None:
        registration_state.update(created=existing is None, changed=True)
    validated_ok, validated = _listed_gjc_mcp_entry(gjc_path, run_command)
    if not validated_ok or not is_runtime_loaded_gjc_mcp_entry(validated):
        print_warning(
            "GJC accepted the registration but did not report it as a runtime-loaded "
            "standalone endpoint."
        )
        if existing is None and remove_gjc_mcp_server(gjc_path, run_command=run_command):
            if registration_state is not None:
                registration_state.update(created=False, changed=False)
        return False
    print_success("Registered runtime-loaded Ouroboros MCP server in GJC.")
    return True


def remove_legacy_gjc_bridge() -> bool:
    """Remove the obsolete setup-owned input bridge without touching custom files."""
    bridge = gjc_agent_dir() / "extensions" / "ouroboros-ooo-bridge" / "index.ts"
    try:
        source = bridge.read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    except OSError as exc:
        print_warning(f"Could not inspect legacy GJC bridge: {exc}")
        return False
    signatures = (
        "const COMMAND_RE = /^\\s*ooo(?:\\s+|$)/i;",
        '"dispatch", "--runtime", "gjc"',
        "_OUROBOROS_GJC_BRIDGE_DEPTH",
        "export default function ouroborosBridge",
    )
    if not all(signature in source for signature in signatures):
        print_info(f"Preserved custom GJC extension at {bridge}")
        return True
    try:
        bridge.unlink()
    except OSError as exc:
        print_warning(f"Could not remove legacy GJC bridge: {exc}")
        return False
    try:
        bridge.parent.rmdir()
    except OSError:
        pass
    print_info("Removed obsolete GJC input bridge; native skills now own ooo routing.")
    return True


def setup_gjc_runtime(
    gjc_path: str,
    *,
    install_runtime_artifacts: Callable[..., bool],
    atomic_write_text: Callable[..., object],
    snapshot_path: Callable[..., object],
    restore_path_snapshot: Callable[..., None],
) -> bool:
    """Configure GJC and roll back every owned path when activation fails."""
    from ouroboros.config.loader import create_default_config, ensure_config_dir
    from ouroboros.gjc import gjc_skills_root
    from ouroboros.runtime_instruction_artifacts import gjc_instruction_path

    config_dir = ensure_config_dir()
    config_path = config_dir / "config.yaml"
    paths = (
        config_path,
        config_dir / "credentials.yaml",
        gjc_skills_root(gjc_agent_dir()),
        gjc_instruction_path().parent,
        gjc_mcp_bridge_config_path().parent,
        gjc_agent_dir() / "extensions" / "ouroboros-ooo-bridge",
    )
    registration_state: dict[str, bool] = {}
    snapshots: tuple[tuple[Path, Any], ...] = ()
    try:
        snapshots = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            create_default_config(config_dir)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            print_error("~/.ouroboros/config.yaml top-level is not a mapping — aborting GJC setup.")
            _restore_gjc_paths(snapshots, restore_path_snapshot)
            return False

        orchestrator = config.get("orchestrator")
        if not isinstance(orchestrator, dict):
            orchestrator = {}
            config["orchestrator"] = orchestrator
        orchestrator.update(runtime_backend="gjc", gjc_cli_path=gjc_path)
        llm = config.get("llm")
        if not isinstance(llm, dict):
            llm = {}
            config["llm"] = llm
        llm["backend"] = "gjc"

        if not install_runtime_artifacts(gjc_path, registration_state=registration_state):
            raise OSError("runtime artifact activation failed")
        atomic_write_text(
            config_path,
            yaml.dump(config, default_flow_style=False, sort_keys=False),
        )
    except (OSError, yaml.YAMLError) as exc:
        _restore_gjc_paths(snapshots, restore_path_snapshot)
        _rollback_new_gjc_mcp_registration(gjc_path, registration_state)
        print_error(f"GJC setup failed; restored the previous state: {exc}")
        return False

    print_success(f"Configured GJC runtime (CLI: {gjc_path})")
    print_info(f"Config saved to: {config_path}")
    return True


def _restore_gjc_paths(
    snapshots: tuple[tuple[Path, Any], ...],
    restore_path_snapshot: Callable[..., None],
) -> None:
    failures: list[str] = []
    for path, snapshot in reversed(snapshots):
        try:
            restore_path_snapshot(path, snapshot, restore_link_targets=False)
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print_warning("GJC setup rollback was incomplete: " + "; ".join(failures))


def _rollback_new_gjc_mcp_registration(gjc_path: str, registration_state: dict[str, bool]) -> None:
    if not registration_state.get("created"):
        return
    if remove_gjc_mcp_server(gjc_path, run_command=subprocess.run):
        registration_state.update(created=False, changed=False)
    else:
        print_warning("GJC setup rollback could not remove the newly registered MCP server.")


def rollback_gjc_files(
    snapshots: tuple[tuple[Path, Any], ...],
    *,
    restore_path_snapshot: Callable[..., None],
) -> None:
    """Restore GJC filesystem artifacts without changing MCP registration state."""
    _restore_gjc_paths(snapshots, restore_path_snapshot)


def rollback_gjc_activation(
    snapshots: tuple[tuple[Path, Any], ...],
    *,
    restore_path_snapshot: Callable[..., None],
    gjc_path: str,
    registration_state: dict[str, bool],
) -> None:
    """Restore owned paths and remove only a registration created by this activation."""
    rollback_gjc_files(snapshots, restore_path_snapshot=restore_path_snapshot)
    _rollback_new_gjc_mcp_registration(gjc_path, registration_state)
