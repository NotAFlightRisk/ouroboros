# Hidden-Checklist Convergence Requirements

> Generated: 2026-08-07
> Status: Implemented

## Original Request

Implement the design in `/Users/jaegyu.lee/.claude/plans/piped-squishing-ember.md`.

## Clarified Specification

- Hide every harness-owned `verify_command` and `output_assertion` from initial
  and retry worker prompts without a configuration escape hatch.
- Build retry guidance from missing artifacts, sanitized verification output,
  and read-only worker tool/file evidence.
- Formally evaluate completed runs even when AC execution failed, provided an
  evaluable session/artifact exists.
- On explicit evaluation rejection, deterministically project the Seed and
  checklist into generation 1 and delegate convergence to Ralph.
- Preserve fail-open chaining: enqueue/parsing failures never change the run or
  evaluation verdict that preceded them.
- Avoid a second Ralph lineage inside `ooo auto`, which already owns its own
  RALPH_HANDOFF loop.

## Success Criteria

- Hidden contract strings do not appear in worker prompts.
- Run → evaluate → Ralph job IDs are transitively observable.
- Gen1 replay has `lineage.created` plus `lineage.generation.completed`, and the
  next planned generation is 2.
- Approved/unjudged/opted-out evaluations do not enqueue Ralph.
- Static checks and the bounded target test suites pass.
