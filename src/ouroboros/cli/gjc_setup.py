"""GJC runtime setup primitives shared by setup, refresh, and uninstall."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import subprocess

from ouroboros.cli.formatters.panels import print_error, print_info, print_success, print_warning
from ouroboros.runtime_instruction_artifacts import gjc_agent_dir

_GJC_MCP_BRIDGE_CONFIG_CONTENT = "# Managed by ouroboros setup --runtime gjc\nmcp_servers: []\n"


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
        if not is_setup_managed_gjc_mcp_entry(existing):
            print_info("Preserved existing user-managed Ouroboros MCP config in GJC.")
        else:
            print_info("Ouroboros MCP server in GJC is already up to date.")
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
        "per-session",
        "--timeout",
        "30000",
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
    if not validated_ok or not is_setup_managed_gjc_mcp_entry(validated):
        print_warning("GJC did not retain the expected Ouroboros MCP registration.")
        if existing is None:
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
