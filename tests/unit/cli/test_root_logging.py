"""Root CLI logging configuration regressions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ouroboros.cli.main import app, main
from ouroboros.core.errors import ConfigError
from ouroboros.observability.logging import (
    LoggingConfig as RuntimeLoggingConfig,
)
from ouroboros.observability.logging import (
    configure_logging,
    get_current_config,
    get_logger,
    reset_logging,
    set_console_logging,
)


@pytest.fixture(autouse=True)
def reset_logging_state() -> Any:
    """Keep process-global logging configuration isolated between CLI tests."""
    reset_logging()
    set_console_logging(True)
    yield
    reset_logging()
    set_console_logging(True)


def _persisted_config(level: str) -> SimpleNamespace:
    return SimpleNamespace(logging=SimpleNamespace(level=level))


def test_root_callback_filters_info_at_persisted_warning_level(capsys: Any) -> None:
    """The saved config level must govern already-imported application loggers."""
    with patch("ouroboros.cli.main.load_config", return_value=_persisted_config("warning")):
        main()

    config = get_current_config()
    assert config is not None
    assert config.log_level == "WARNING"
    assert config.enable_file_logging is False

    logger = get_logger("root-logging-test")
    logger.info("root.info_should_be_filtered")
    logger.warning("root.warning_should_remain")
    captured = capsys.readouterr()

    assert "root.info_should_be_filtered" not in captured.out + captured.err
    assert "root.warning_should_remain" not in captured.out
    assert "root.warning_should_remain" in captured.err


def test_root_callback_defaults_to_info_when_config_is_missing() -> None:
    """First-run and repair commands must survive a missing or invalid config."""
    with patch(
        "ouroboros.cli.main.load_config",
        side_effect=ConfigError("configuration unavailable"),
    ):
        main()

    config = get_current_config()
    assert config is not None
    assert config.log_level == "INFO"
    assert config.enable_file_logging is False


def test_pm_debug_flag_overrides_persisted_warning_level() -> None:
    """Command-local debug configuration must run after the root callback."""

    def fake_run(coro: object) -> None:
        coro.close()  # type: ignore[union-attr]

    def configure_debug_without_file(config: RuntimeLoggingConfig) -> None:
        configure_logging(
            RuntimeLoggingConfig(
                log_level=config.log_level,
                enable_file_logging=False,
            )
        )

    with (
        patch("ouroboros.cli.main.load_config", return_value=_persisted_config("warning")),
        patch("ouroboros.cli.commands.pm.asyncio.run", side_effect=fake_run),
        patch(
            "ouroboros.cli.commands.pm.configure_logging",
            side_effect=configure_debug_without_file,
        ),
    ):
        result = CliRunner().invoke(app, ["pm", "--debug"])

    assert result.exit_code == 0, result.output
    config = get_current_config()
    assert config is not None
    assert config.log_level == "DEBUG"


def test_cli_subprocess_filters_info_and_keeps_stdout_clean(tmp_path: Path) -> None:
    """A real CLI process must apply warning before application logging begins."""
    home = tmp_path / "home"
    config_dir = home / ".ouroboros"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        "logging:\n  level: warning\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo_root = Path(__file__).parents[3]
    env["PYTHONPATH"] = str(repo_root / "src")
    script = """
from ouroboros.cli.main import app
from ouroboros.observability import get_logger

app(args=["config", "show", "--json"], standalone_mode=False)
logger = get_logger("issue1955-subprocess")
logger.info("subprocess.info_should_be_filtered")
logger.warning("subprocess.warning_should_remain")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    json.loads(result.stdout)
    assert "subprocess.info_should_be_filtered" not in result.stdout + result.stderr
    assert "subprocess.warning_should_remain" not in result.stdout
    assert "subprocess.warning_should_remain" in result.stderr
