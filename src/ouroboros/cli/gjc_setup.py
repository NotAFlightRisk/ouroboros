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

_GJC_MCP_SHARING = "per-session"
_GJC_MCP_TIMEOUT = 30000
_GJC_MCP_RUNTIME_STATUS = "autoload"


def gjc_mcp_config_path() -> Path:
    """Return GJC's durable user MCP registration file."""
    return gjc_agent_dir() / "mcp.json"


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


def _gjc_mcp_config(entry: object) -> dict[str, object] | None:
    """Return the execution config from either a CLI row or persistent entry."""
    if not isinstance(entry, dict):
        return None
    nested = entry.get("config")
    if isinstance(nested, dict):
        return nested
    return entry


def is_setup_managed_gjc_mcp_entry(entry: object, *, allow_redacted_env: bool = False) -> bool:
    """Return whether *entry* exactly matches setup's execution contract."""
    config = _gjc_mcp_config(entry)
    if config is None:
        return False
    command = config.get("command")
    args = config.get("args")
    env = config.get("env")
    expected_env_values = {str(gjc_mcp_bridge_config_path())}
    if allow_redacted_env:
        expected_env_values.add("<redacted>")
    return (
        config.get("type") == "stdio"
        and isinstance(command, str)
        and isinstance(args, list)
        and _is_exact_launcher_args(command, args)
        and isinstance(env, dict)
        and set(env) == {"OUROBOROS_MCP_CONFIG"}
        and env.get("OUROBOROS_MCP_CONFIG") in expected_env_values
        and config.get("sharing") == _GJC_MCP_SHARING
        and config.get("timeout") == _GJC_MCP_TIMEOUT
    )


def is_active_gjc_mcp_entry(entry: object) -> bool:
    """Return whether GJC reports the registration as session-autoloaded."""
    return isinstance(entry, dict) and entry.get("runtimeStatus") == _GJC_MCP_RUNTIME_STATUS


def persisted_gjc_mcp_entry(path: Path | None = None) -> dict[str, object] | None:
    """Read the durable Ouroboros registration without requiring the GJC launcher."""
    config_path = path or gjc_mcp_config_path()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    entry = servers.get("ouroboros") if isinstance(servers, dict) else None
    return entry if isinstance(entry, dict) else None


def remove_persisted_gjc_mcp_server(path: Path | None = None) -> bool:
    """Remove only an exactly setup-owned durable entry, preserving sibling servers."""
    config_path = path or gjc_mcp_config_path()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    entry = servers.get("ouroboros") if isinstance(servers, dict) else None
    if not is_setup_managed_gjc_mcp_entry(entry):
        return False
    del servers["ouroboros"]
    try:
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


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
    """Register and validate the isolated Ouroboros MCP server through GJC."""
    if registration_state is not None:
        registration_state.update(created=False, changed=False)
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
        persisted = persisted_gjc_mcp_entry()
        if not (
            is_setup_managed_gjc_mcp_entry(existing, allow_redacted_env=True)
            and is_setup_managed_gjc_mcp_entry(persisted)
        ):
            print_error(
                "GJC already has an MCP server named 'ouroboros' that is not the "
                "complete setup-owned registration. Preserved it, but native "
                "Ouroboros activation cannot be verified."
            )
            return False
        if not is_active_gjc_mcp_entry(existing):
            print_error(
                "The existing Ouroboros MCP server is not autoloaded by GJC; "
                "preserved it and kept the legacy route intact."
            )
            return False
        print_info("Ouroboros MCP server in GJC is already up to date and autoloaded.")
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
    add_command = [
        gjc_path,
        "mcp",
        "add",
        "ouroboros",
        "--command",
        command,
        *(f"--arg={arg}" for arg in server_args),
        f"--env=OUROBOROS_MCP_CONFIG={gjc_mcp_bridge_config_path()}",
        "--sharing",
        _GJC_MCP_SHARING,
        "--timeout",
        str(_GJC_MCP_TIMEOUT),
        "--json",
    ]
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
    persisted = persisted_gjc_mcp_entry()
    if (
        not validated_ok
        or not is_setup_managed_gjc_mcp_entry(validated, allow_redacted_env=True)
        or not is_setup_managed_gjc_mcp_entry(persisted)
        or not is_active_gjc_mcp_entry(validated)
    ):
        print_warning("GJC did not retain and autoload the expected Ouroboros MCP registration.")
        if existing is None and (
            is_setup_managed_gjc_mcp_entry(validated, allow_redacted_env=True)
            and is_setup_managed_gjc_mcp_entry(persisted)
            and is_active_gjc_mcp_entry(validated)
        ):
            remove_gjc_mcp_server(gjc_path, run_command=run_command)
            if registration_state is not None:
                registration_state.update(created=False, changed=False)
        return False
    print_success("Registered Ouroboros MCP server in GJC.")
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
    """Configure GJC and roll back only unchanged setup-owned generations."""
    from ouroboros.config.loader import create_default_config, ensure_config_dir
    from ouroboros.gjc import setup_owned_gjc_skill_paths
    from ouroboros.runtime_instruction_artifacts import gjc_instruction_path

    config_dir = ensure_config_dir()
    config_path = config_dir / "config.yaml"
    agent_dir = gjc_agent_dir()
    paths = (
        config_path,
        config_dir / "credentials.yaml",
        *setup_owned_gjc_skill_paths(agent_dir=agent_dir),
        gjc_instruction_path(),
        gjc_mcp_bridge_config_path(),
        agent_dir / "extensions" / "ouroboros-ooo-bridge" / "index.ts",
    )
    registration_state: dict[str, bool] = {}
    snapshots: tuple[tuple[Path, Any], ...] = ()
    expected: tuple[tuple[Path, Any], ...] = ()
    try:
        snapshots = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            create_default_config(config_dir)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected = tuple((path, snapshot_path(path, follow_links=False)) for path in paths)
        config_generation = dict(expected)[config_path]
        if not isinstance(config, dict):
            print_error("~/.ouroboros/config.yaml top-level is not a mapping — aborting GJC setup.")
            _restore_gjc_paths(snapshots, expected, restore_path_snapshot, snapshot_path)
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
        current_after_activation = {path: snapshot_path(path, follow_links=False) for path in paths}
        current_after_activation[config_path] = config_generation
        expected = tuple(current_after_activation.items())
        atomic_write_text(
            config_path,
            yaml.dump(config, default_flow_style=False, sort_keys=False),
            expected_current=config_generation,
        )
    except (OSError, yaml.YAMLError) as exc:
        _restore_gjc_paths(snapshots, expected, restore_path_snapshot, snapshot_path)
        _rollback_new_gjc_mcp_registration(gjc_path, registration_state)
        print_error(f"GJC setup failed; restored the previous state: {exc}")
        return False

    print_success(f"Configured GJC runtime (CLI: {gjc_path})")
    print_info(f"Config saved to: {config_path}")
    return True


def _restore_gjc_paths(
    snapshots: tuple[tuple[Path, Any], ...],
    expected: tuple[tuple[Path, Any], ...],
    restore_path_snapshot: Callable[..., None],
    snapshot_path: Callable[..., object],
) -> None:
    failures: list[str] = []
    expected_by_path = dict(expected)
    for path, snapshot in reversed(snapshots):
        try:
            expected_current = expected_by_path.get(path)
            if (
                expected_current is not None
                and snapshot_path(path, follow_links=False) != expected_current
            ):
                print_warning(f"Preserved concurrently changed GJC setup path: {path}")
                continue
            restore_path_snapshot(path, snapshot, restore_link_targets=False)
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print_warning("GJC setup rollback was incomplete: " + "; ".join(failures))


def _rollback_new_gjc_mcp_registration(gjc_path: str, registration_state: dict[str, bool]) -> None:
    if not registration_state.get("created"):
        return
    listed_ok, current = _listed_gjc_mcp_entry(gjc_path, subprocess.run)
    persisted = persisted_gjc_mcp_entry()
    if not listed_ok or not (
        is_setup_managed_gjc_mcp_entry(current, allow_redacted_env=True)
        and is_setup_managed_gjc_mcp_entry(persisted)
        and is_active_gjc_mcp_entry(current)
    ):
        print_warning(
            "Preserved the GJC MCP registration because it changed after setup created it."
        )
        return
    if remove_gjc_mcp_server(gjc_path, run_command=subprocess.run):
        registration_state.update(created=False, changed=False)
    else:
        print_warning("GJC setup rollback could not remove the newly registered MCP server.")


def rollback_gjc_activation(
    snapshots: tuple[tuple[Path, Any], ...],
    expected: tuple[tuple[Path, Any], ...],
    *,
    restore_path_snapshot: Callable[..., None],
    snapshot_path: Callable[..., object],
    gjc_path: str,
    registration_state: dict[str, bool],
) -> None:
    """Restore unchanged owned generations and remove a registration created here."""
    _restore_gjc_paths(snapshots, expected, restore_path_snapshot, snapshot_path)
    _rollback_new_gjc_mcp_registration(gjc_path, registration_state)
