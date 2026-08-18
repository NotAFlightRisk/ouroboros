"""Resolve a real POSIX shell for orchestrator-run ``verify_command`` execution.

``_run_ac_verify_gate`` used to hand an AC's ``verify_command`` to
``asyncio.create_subprocess_shell``, which resolves to ``cmd.exe`` on native
Windows. Seed-authored verify commands are POSIX bash by contract (``&&``,
``grep``, single quotes, forward-slash paths), so that default silently ran
them under an interpreter that cannot parse them.

This module follows gajae-code's ``getShellConfig()`` pattern: **resolve a real
POSIX shell binary, never translate the command**. Command authors keep writing
one syntax; the orchestrator guarantees an interpreter that understands it, or
reports that this machine cannot verify the AC at all. The pass/fail signal
still comes from the exit code of the unmodified command — nothing here is
allowed to soften what "verified" means.

Priority mirrors the existing ``get_goose_cli_path`` / ``get_pi_cli_path``
convention: env override -> config -> well-known locations -> PATH.
``OUROBOROS_VERIFY_BASH`` is an executable-path variable, so it is denied from
the untrusted project ``.env`` (see :mod:`ouroboros.config.untrusted_env`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
import shutil

VERIFY_BASH_ENV_VAR = "OUROBOROS_VERIFY_BASH"

# Windows locations Git for Windows installs its bundled bash into. A user who
# has Git for Windows has a real bash even if it is not on PATH.
_GIT_BASH_RELATIVE = PureWindowsPath("Git") / "bin" / "bash.exe"
_GIT_BASH_PROGRAM_FILES_VARS = ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")


@dataclass(frozen=True, slots=True)
class VerifyShellRoute:
    """A concrete interpreter chosen to run ``verify_command`` verbatim.

    ``degraded`` marks a POSIX ``sh`` fallback: it understands the operators and
    quoting seed commands rely on, but not bash-only syntax (``[[ ]]``, arrays).
    It is preferred over refusing to verify, and is reported so an unexplained
    verify failure can be traced back to the interpreter.
    """

    shell_path: str
    source: str
    degraded: bool = False

    def argv(self, command: str) -> tuple[str, ...]:
        """Return the exact argv that runs ``command`` unmodified."""
        return (self.shell_path, "-c", command)


def _running_on_windows() -> bool:
    """Read the platform through one seam.

    Tests cannot monkeypatch ``os.name`` globally: that flips ``pathlib.Path``
    to ``WindowsPath``, which refuses to instantiate on POSIX and breaks every
    unrelated path this call touches. Windows branches are exercised by
    patching this function instead.
    """
    return os.name == "nt"


def _executable(candidate: str | None) -> str | None:
    """Return an executable absolute path for ``candidate``, else ``None``.

    Stale env/config values that no longer point at an executable are treated
    as missing so resolution falls through instead of persisting an unusable
    path — the same tolerance ``get_goose_cli_path`` applies.
    """
    if not candidate:
        return None
    expanded = str(Path(candidate).expanduser())
    return shutil.which(expanded)


def _configured_candidate() -> tuple[str, str] | None:
    env_value = os.environ.get(VERIFY_BASH_ENV_VAR, "").strip()
    resolved = _executable(env_value)
    if resolved:
        return resolved, "env"

    try:
        from ouroboros.config.loader import get_verify_bash_path
    except ImportError:  # pragma: no cover - loader is always importable in-tree
        return None
    resolved = _executable(get_verify_bash_path())
    if resolved:
        return resolved, "config"
    return None


def _windows_candidates() -> tuple[tuple[str, str, bool], ...]:
    candidates: list[tuple[str, str, bool]] = []
    for variable in _GIT_BASH_PROGRAM_FILES_VARS:
        program_files = os.environ.get(variable, "").strip()
        if program_files:
            candidates.append(
                (str(PureWindowsPath(program_files) / _GIT_BASH_RELATIVE), "git_bash", False)
            )
    # Cygwin / MSYS2 installs that put a real bash on PATH. `%SystemRoot%`
    # entries are filtered below: `System32\bash.exe` is the WSL launcher, not
    # a Windows-side shell.
    candidates.append(("bash.exe", "path", False))
    candidates.append(("bash", "path", False))
    return tuple(candidates)


def _is_wsl_launcher(resolved_path: str) -> bool:
    """Recognize ``%SystemRoot%/System32/bash.exe``, which is not a local shell.

    That binary hands the command to a Linux distribution with its own root
    filesystem: the Windows ``cwd`` the gate passes has no meaning inside it,
    and the command would judge a different tree — or fail for reasons that
    have nothing to do with the AC. Both outcomes are worse than saying this
    machine cannot verify, so it is never used as a verify shell.
    """
    system_root = os.environ.get("SYSTEMROOT", "").strip() or "C:\\Windows"
    try:
        candidate = PureWindowsPath(resolved_path)
        return candidate.name.lower() == "bash.exe" and candidate.is_relative_to(
            PureWindowsPath(system_root) / "System32"
        )
    except ValueError:
        return False


def _posix_candidates() -> tuple[tuple[str, str, bool], ...]:
    return (
        ("/bin/bash", "posix_default", False),
        ("/usr/bin/bash", "posix_default", False),
        ("bash", "path", False),
        # Last resort: a POSIX sh still parses the operators, quoting and
        # redirections seed verify commands are built from.
        ("/bin/sh", "posix_sh", True),
        ("sh", "path_sh", True),
    )


def _resolve_uncached() -> VerifyShellRoute | None:
    configured = _configured_candidate()
    if configured is not None:
        shell_path, source = configured
        return VerifyShellRoute(shell_path=shell_path, source=source)

    windows = _running_on_windows()
    candidates = _windows_candidates() if windows else _posix_candidates()
    for candidate, source, degraded in candidates:
        resolved = _executable(candidate)
        if not resolved:
            continue
        if windows and _is_wsl_launcher(resolved):
            continue
        return VerifyShellRoute(shell_path=resolved, source=source, degraded=degraded)
    return None


_ROUTE_CACHE: dict[tuple[str, str, str], VerifyShellRoute] = {}


def _cache_key() -> tuple[str, str, str]:
    """Fingerprint the inputs resolution actually reads.

    Resolution is deterministic in (OS, explicit override, PATH), so one lookup
    per ``ooo run`` is enough; a changed override or PATH invalidates itself
    instead of serving a stale interpreter.
    """
    return (
        "nt" if _running_on_windows() else "posix",
        os.environ.get(VERIFY_BASH_ENV_VAR, "").strip(),
        os.environ.get("PATH", ""),
    )


def resolve_verify_shell() -> VerifyShellRoute | None:
    """Return the interpreter that runs ``verify_command``, or ``None``.

    ``None`` means this machine has no POSIX-compatible interpreter *right
    now*. Callers must treat that as unverifiable — never as a pass.

    Only a successful resolution is cached. A failure is re-resolved on every
    call, because the whole point of retrying an unverifiable AC is that the
    operator can install Git Bash or set ``OUROBOROS_VERIFY_BASH`` while the
    run continues. Git Bash is found through ``%ProgramFiles%``, not ``PATH``,
    so a cached "no shell" keyed on ``PATH`` would never notice the install
    and would defeat the retry it exists to serve.
    """
    key = _cache_key()
    cached = _ROUTE_CACHE.get(key)
    if cached is not None:
        return cached
    route = _resolve_uncached()
    if route is not None:
        _ROUTE_CACHE[key] = route
    return route


def reset_verify_shell_cache() -> None:
    """Drop the resolution cache (tests, and env changes inside one process)."""
    _ROUTE_CACHE.clear()


def verify_shell_unavailable_reason() -> str:
    """Explain the terminal case in a way the operator can act on."""
    if _running_on_windows():
        return (
            "verify_command needs a POSIX shell: no bash found. Install Git for "
            "Windows (which bundles bash) or set "
            f"{VERIFY_BASH_ENV_VAR} to a bash executable."
        )
    return (
        "verify_command needs a POSIX shell: no bash or sh found on this "
        f"machine. Set {VERIFY_BASH_ENV_VAR} to a POSIX shell executable."
    )


# Environment variables that can change a verify_command's verdict without
# changing the workspace or the command. They are stripped from the gate's
# subprocess so the verdict is computed by the tools the orchestrator meant,
# with the configuration the Seed declared.
#
# `PYTHONPATH` and `PYTEST_ADDOPTS` are the sharp ones and are *not* covered by
# the project-`.env` denylist (`config/untrusted_env.py`), which blocks the
# `LD_`/`DYLD_` families and package-manager prefixes but neither of these. A
# repository's own `.env` could therefore set `PYTEST_ADDOPTS="-k nothing"` and
# every pytest-shaped contract in that repo would pass having run no tests.
#
# `PATH` is deliberately NOT stripped: verify commands resolve real tools
# (`uv`, `pytest`, `node`) through it, and a project-local `.venv/bin` entry is
# both common and legitimate. The in-workspace channels this leaves open —
# `conftest.py`, `pytest.ini`, a workspace-owned venv — are not env problems
# and cannot be closed here; they need a differential probe.
VERIFY_ENV_STRIPPED_KEYS = frozenset(
    {
        # Python import-path and startup injection.
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        # pytest argument and plugin injection: `-k`, `-p`, `--exitfirst` and
        # friends can turn any test command into a no-op that exits 0.
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        # Node preload hook, the same class of risk for JS-shaped contracts.
        "NODE_OPTIONS",
    }
)

# Dynamic-loader preload families, stripped by prefix for the same reason the
# untrusted-env policy rejects them by prefix: the member names vary by
# platform and new ones appear.
VERIFY_ENV_STRIPPED_PREFIXES = ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES")


def sanitized_verify_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment a ``verify_command`` may be judged in.

    The gate inherits the orchestrator's environment, which in turn may carry
    values a project `.env` or the operator's shell put there. Anything able to
    bend the verdict from outside the workspace is removed; everything a real
    tool needs to run is kept.
    """
    source = os.environ if base is None else base
    return {
        key: value
        for key, value in source.items()
        if key.upper() not in VERIFY_ENV_STRIPPED_KEYS
        and not key.upper().startswith(VERIFY_ENV_STRIPPED_PREFIXES)
    }
