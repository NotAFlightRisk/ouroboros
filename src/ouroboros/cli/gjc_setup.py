"""GJC runtime setup primitives shared by setup, refresh, and uninstall."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
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
    """Disable redundant upstream MCP fan-in inside GJC-hosted Ouroboros MCP."""
    path = gjc_mcp_bridge_config_path()
    if path.is_symlink():
        print_warning(f"Refusing to replace symlinked GJC MCP bridge config: {path}")
        return False
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


def is_setup_managed_gjc_mcp_entry(entry: object) -> bool:
    """Return whether setup may replace one redacted GJC MCP list entry."""
    if not isinstance(entry, dict):
        return False
    config = entry.get("config")
    if not isinstance(config, dict) or config.get("type") != "stdio":
        return False
    command = config.get("command")
    if not isinstance(command, str) or os.path.basename(command) not in {"uvx", "pipx"}:
        return False
    args = config.get("args")
    return isinstance(args, list) and any(
        isinstance(arg, str) and "ouroboros-ai" in arg for arg in args
    )


def register_gjc_mcp_server(
    gjc_path: str,
    *,
    detect_mcp_entry: Callable[..., dict[str, object] | None],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    detected: dict[str, object] | None = None,
) -> bool:
    """Register the isolated Ouroboros MCP server through GJC's public CLI."""
    detected = detected or detect_mcp_entry(package_spec="ouroboros-ai[mcp]")
    if detected is None:
        print_error(
            "GJC setup requires an isolated MCP 2 launcher. "
            "Install uv/uvx or pipx, then re-run setup."
        )
        return False

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
        return False
    if listed.returncode != 0:
        print_warning(f"Could not inspect GJC MCP registrations: {listed.stderr.strip()}")
        return False
    try:
        payload = json.loads(listed.stdout)
    except json.JSONDecodeError:
        print_warning("GJC MCP list returned malformed JSON; leaving registrations untouched.")
        return False
    servers = payload.get("servers") if isinstance(payload, dict) else None
    existing = (
        next(
            (
                entry
                for entry in servers
                if isinstance(entry, dict) and entry.get("name") == "ouroboros"
            ),
            None,
        )
        if isinstance(servers, list)
        else None
    )
    if existing is not None and not is_setup_managed_gjc_mcp_entry(existing):
        print_info("Preserved existing user-managed Ouroboros MCP config in GJC.")
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
    if existing is not None:
        add_command.append("--force")
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
