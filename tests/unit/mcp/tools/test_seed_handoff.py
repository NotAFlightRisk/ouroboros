"""Opaque Seed handoff and worker-safe rendering regressions."""

from ouroboros.mcp.tools.evaluation_job import (
    resolve_auto_evolve_policy,
    restore_seed_handoff,
)
from ouroboros.mcp.tools.seed_handoff import SeedHandoffRegistry, render_worker_safe_seed


def test_handoff_is_session_bound_and_bounded() -> None:
    registry = SeedHandoffRegistry(max_entries=1)
    first = registry.register(session_id="orch-first", seed_content="goal: first")
    second = registry.register(session_id="orch-second", seed_content="goal: second")

    assert registry.resolve(first, session_id="orch-first") is None
    assert registry.resolve(second, session_id="orch-wrong") is None
    assert registry.resolve("seed_handoff_unknown", session_id="orch-second") is None
    assert registry.resolve(second, session_id="orch-second") == "goal: second"


def test_worker_safe_seed_fails_closed_for_malformed_yaml() -> None:
    raw = "goal: [malformed SECRET_VALUE"

    rendered = render_worker_safe_seed(raw)

    assert "SECRET_VALUE" not in rendered
    assert "Seed omitted: invalid YAML" in rendered


def test_restore_consumes_process_local_handoff_handle() -> None:
    registry = SeedHandoffRegistry()
    handoff_id = registry.register(session_id="orch-restore", seed_content="goal: private")
    original = {"session_id": "orch-restore", "seed_handoff_id": handoff_id}

    restored = restore_seed_handoff(
        original,
        session_id="orch-restore",
        registry=registry,
    )

    assert restored == {"session_id": "orch-restore", "seed_content": "goal: private"}
    assert original["seed_handoff_id"] == handoff_id


def test_disabled_auto_evolve_snapshot_survives_enabled_config_on_reentry() -> None:
    initial, enabled = resolve_auto_evolve_policy({}, configured_enabled=False)
    replayed, replay_enabled = resolve_auto_evolve_policy(initial, configured_enabled=True)

    assert enabled is False
    assert replay_enabled is False
    assert initial["auto_evolve"] is False
    assert replayed["auto_evolve"] is False
    assert "_auto_evolve_max_generations" not in replayed


def test_enabled_auto_evolve_snapshot_survives_disabled_config_on_reentry(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ouroboros.mcp.tools.evaluation_job.get_auto_evolve_max_generations",
        lambda: 4,
    )
    initial, enabled = resolve_auto_evolve_policy({}, configured_enabled=True)
    replayed, replay_enabled = resolve_auto_evolve_policy(initial, configured_enabled=False)

    assert enabled is True
    assert replay_enabled is True
    assert initial["auto_evolve"] is True
    assert replayed["auto_evolve"] is True
    assert replayed["_auto_evolve_max_generations"] == 4
