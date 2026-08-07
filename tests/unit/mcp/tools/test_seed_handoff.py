"""Opaque Seed handoff and worker-safe rendering regressions."""

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
