# Hidden-Checklist Convergence Implementation

> Completed: 2026-08-07
> Branch: detached worktree HEAD

## Summary

`ooo run` now carries an execution through formal evaluation and, on explicit
rejection, into the existing bounded Ralph evolution loop. Harness grading
commands/assertions remain hidden from workers, while retries receive sanitized
facts about missing artifacts, observed commands/files, and verifier output.

The production composition root now owns the entire successor chain:

```text
StartExecuteSeedHandler
  -> StartEvaluateHandler
    -> StartRalphHandler
      -> EvolveStepHandler
        -> EvolutionaryLoop
```

Each arrow is an injected, shared handler instance. Chained execution no longer
constructs bare handlers at runtime, which previously caused the first real
Ralph iteration to terminate with `EvolutionaryLoop not configured` even
though constructor-mocked tests passed.

Passive OpenCode plugin interception now resolves through the same configured
graph rather than the former lightweight definition factory. Other runtimes
retain their existing dispatcher/server ownership model.

When OpenCode passive-plugin dispatch is active, an evaluation with
`auto_evolve: true` remains parent-owned and pollable. The parent runs the
formal evaluator through its in-process path so it receives the verdict and
can seed generation 1 before starting Ralph. Plugin-only delegation remains
the behavior when automatic evolution is disabled.

## Configuration

```yaml
execution:
  auto_evaluate: true
  auto_evolve: true
  auto_evolve_max_generations: 3
```

`auto_evolve` is also a per-call override. The automatic generation budget is
clamped to 1..10, snapshotted when evaluation is enqueued, and bound to the
lineage through a durable policy claim before the Ralph successor starts. The
successor receipt records both `job_id` and the authoritative budget, so later
configuration changes cannot rewrite replay metadata. Auto pipeline run
dispatches explicitly set `auto_evolve: false` because Auto owns a separate
Ralph lineage.

## Testing

- Original feature regression suite: 1,261 passed.
- Corrective convergence suite: 769 passed.
- Mock-free boundary test: rejected Gen 1 evaluation -> real Ralph job -> real
  `EvolveStepHandler` -> real `EvolutionaryLoop` -> completed generation 2;
  the prior passing AC is frozen, the failed AC is active, and Ralph stops on
  `qa passed`.
- Composition test: the registered run/evaluate/Ralph/evolve handlers reuse
  the same configured instances.
- Ruff: all checks passed.
- Mypy: no issues in 513 source files.

## Known Limitations

- The optional model-based coach rewrite remains intentionally deferred; retry
  hints are deterministic.
- Single-AC evaluations without checklist metadata use Ralph's existing
  full-graph focus fallback.
- Plugin Seed handoffs are intentionally process-local until redemption. The
  parent consumes the opaque handle, restores the raw Seed into its private
  evaluation request, and gives a detached evaluation owner no process-local
  handle to resolve. Raw verifier material remains absent from plugin-worker
  prompts and worker-queryable events.
