# Review Conventions

Every PR is reviewed by `ouroboros-agent[bot]` ("ourobot") before it can merge.
Its approval satisfies the required review on `main`.

The bot is strict, and it is strict in a predictable, repeatable way. Most
review rounds are lost to objections that could have been preempted before the
first push. This page exists so you can preempt them.

The rules below were extracted from the bot's actual review bodies on merged
PRs, not from a style guide. Some PRs take a single round; some take seven.
The difference is almost entirely whether the change is a root-cause fix with
proof, or a plausible-looking patch.

## What the review actually contains

The bot posts a fixed structure:

1. **Verdict** — `APPROVE` or `REQUEST_CHANGES`, with the exact HEAD commit it
   checked. If you pushed after it started, the verdict describes the older commit.
2. **What Improved** — what the PR does well.
3. **Issue Requirements** — a table grading your PR against **the linked
   issue's** requirements, each marked *Met*, *Partially met*, or *Not met*.
4. **Prior Findings Status** — every finding from earlier rounds, and whether
   you resolved it. Unaddressed findings carry forward.
5. **Blockers** — a table of `File:Line`, severity, and the finding.

Two consequences worth internalizing:

- **Your PR is graded against the issue, not against your PR description.** A
  vague issue produces a vague grade; a requirement you decided was out of
  scope reads as *Partially met* unless you say in the PR why it is deferred.
  This is why [issue quality](./issue-quality-policy.md) is enforced.
- **The bot reproduces things.** Findings routinely cite a probe it ran —
  *"a focused probe classified such a companion as `decide_later` and made it
  skip-eligible"*, *"A focused runtime probe with `process.wait()` returning
  immediately and both streams remaining open exceeded an outer 0.2-second
  guard"* (#2224, #2239). You are not arguing with a linter. If you claim a
  path is unreachable, expect it to be tested.

## The recurring blockers

Ordered by how often they appear in review bodies.

### 1. Fix the root cause, not the symptom

The most expensive rounds are the ones where a patch changes *which* failure
occurs rather than removing it. Before pushing, ask what the reviewer will ask:
*if this input arrives again through a different path, does the bug come back?*
If the answer is yes, you have moved the symptom.

### 2. Validate untrusted input — never coerce it

`bool(...)` on an external payload is a blocker, not a shortcut:

> "Companion routing fields are coerced with `bool(...)` instead of validated
> as booleans. An untrusted planner payload such as `"decide_later": "false"`
> is interpreted as `True`" — #2224

Anything crossing a trust boundary — a planner payload, an MCP argument, a
config file, `./.env` — gets parsed and validated against a closed set of
accepted values.

### 3. No silent success

A code path that accepts a request, does nothing, and reports success is
always a blocker:

> "The public tool schema accepts `answers`, but plugin mode only examines the
> singular `answer` variable. A request … enters the plugin branch, records no
> rounds, and still returns a successful delegation receipt." — #2224

Corollaries the bot enforces: do not truncate silently, do not drop items from
a batch without saying so, and do not swallow an exception into a default.

### 4. Fail closed

When a check cannot prove the safe condition holds, it must refuse, not
proceed. Recent examples that landed on exactly this principle: making
`is_on_protected_branch` fail closed (#2236), and a gate that errors rather
than passing when it cannot enumerate changed files.

### 5. Keep the public schema and the implementation in agreement

If a declared parameter is not honored on every branch, that is a defect even
when the common path works. See #2224 above.

### 6. Same semantics across every runtime

This project drives many backends. Behavior that changes meaning depending on
which one is active is a blocker:

> "Plugin mode records every normalized pair through
> `InterviewState.record_answer`, bypassing `record_turn_answers`, so the
> documented per-question `[deferred]` and `[decide_later]` controls change
> meaning by runtime." — #2224

When you add a behavior to one adapter, check the others.

### 7. Crash-safety and idempotency on the persistence path

Commit ordering and retry behavior are reviewed explicitly:

> "Batch answer persistence is not replay-safe because interview state is
> committed before the authoritative `pending_batch` metadata is updated." — #2224

> "The new answer path is not idempotent after a successful write.
> `record_turn_answers` unconditionally appends every supplied pair, while the
> handler lock only serializes calls and persists no turn/request identity
> that could recognize a transport retry." — #2224

A lock serializes; it does not make an operation idempotent.

### 8. Bound every wait

Any subprocess wait, network call, or stream drain needs a finite ceiling, and
the timeout must surface through the normal error contract rather than as a
bare exception:

> "Timeout from the changed legacy `communicate()` path is not translated into
> the adapter's `Result.err(ProviderError)` contract or followed by child
> cleanup." — #2239

Bounding the primary wait is not enough if a stream drain sits outside the
deadline (#2239).

### 9. Regression coverage for every new branch

"Add regression coverage for the newly bounded subprocess paths" appears
verbatim as a blocker. A new conditional without a test that exercises it will
be flagged, and the bot names the specific case it wants covered.

### 10. Do not add production contracts for test convenience

Test-only environment variables and test-only public knobs are rejected. Make
the test set up real state instead.

## Preempting a round

Before your first push, walk your own diff and answer:

- Does this remove the cause, or relocate the symptom?
- Every new input: validated, or coerced?
- Every new branch: can it succeed while doing nothing?
- Every new wait: bounded, and does the timeout surface as a typed error?
- Every new behavior: does it mean the same thing on the other runtimes?
- Every new conditional: is there a test that reaches it?
- Does the linked issue list a requirement this PR does not meet? Say so
  explicitly in the PR body, with the reason.

Writing that reasoning into the PR description is not ceremony. The bot reads
it, and a stated, justified scope boundary is treated differently from a
requirement that is simply unmet.

## Mechanics

- **Re-review is triggered by a comment.** Push your fix, then post a comment
  explaining what changed. A push alone is unreliable; empty commits and
  re-requesting a reviewer do not work. Turnaround is typically 7–15 minutes.
- **The review pins a commit.** Check the `HEAD checked` field — a verdict may
  predate your latest push.
- **Prior findings carry forward.** Respond to each one, in the PR
  conversation, even if the answer is "not doing this, because …".
- **Pushing back is legitimate.** The bot is wrong sometimes. Say concretely
  why — with the code path or a reproduction — and it can reverse itself; a
  round of explanation is cheaper than a bad change. What does not work is
  silently ignoring a finding.

## Related

- [CI Gates and Branch Protection](./ci-gates.md) — the automated gates
- [Issue Quality Policy](./issue-quality-policy.md) — why the issue text is graded
- [Verifier Evidence Policy](./verifier-evidence-policy.md) — evidence rules in the verifier
