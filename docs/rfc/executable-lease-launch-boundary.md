# RFC — Executable lease at the process launch boundary

> Status: **Draft — proposed direction**
> Tracks [#1948](https://github.com/Q00/ouroboros/issues/1948).
> Supersedes no behavior from #1945; that PR remains the probe-attestation
> prerequisite rather than the launch-boundary fix.

## Decision

Introduce one `ExecutableLease` authority backed by a small native extension
distributed as exact-versioned companion platform wheels. The extension owns
the operating-system launch primitive and returns a versioned launch receipt.
The shared Codex-family runtime must not call `asyncio.create_subprocess_exec`
for an attested CLI after this contract is activated.

This is intentionally a stacked delivery. The RFC and conformance harness land
first; native platform implementations and script compatibility land before
the shared runtime switches over. Until a platform and entrypoint strategy is
implemented, launch fails closed with a typed compatibility reason. There is no
fallback to the current pathname launch.

## Problem

The runtime currently validates the selected CLI while `_build_command()` is
constructed, then later gives the executable pathname to
`asyncio.create_subprocess_exec()`. The operating system resolves that pathname
again. A same-user process can replace the directory entry in the interval and
cause Ouroboros to execute bytes that supplied none of the accepted
attestation.

Repeating `stat()`, digest, or `--version` immediately before spawn only moves
the interval. It cannot bind evidence to the process image that produces the
effect.

## Threat model

The lease MUST resist an unprivileged same-user process that, between native
attestation acceptance and the first application instruction:

- atomically replaces or mutates the executable;
- changes any symlink in its resolution chain;
- replaces a shebang interpreter or known delegate;
- replaces a Node/Python entrypoint after its interpreter is selected; or
- wins a post-build/pre-spawn scheduling barrier deliberately exposed by a
  regression harness.

The contract does not claim to resist a compromised kernel, administrator/root,
code injection into the Ouroboros process, or out-of-band signal, process-handle,
thread-handle, debugger, ptrace, or injection control of the child. Those powers
can resume or alter a suspended process behind the launch authority's back.

The Python constructor may select a display path before async first-dispatch
attestation. A replacement completed during that interval becomes the candidate
generation presented for fresh native attestation; it does not inherit any
earlier trust. Tampering completed before the native authority accepts that
generation is a supply-chain concern. Mutation after acceptance and before the
first trusted application instruction is in scope. After the child begins, a CLI
may deliberately load unrelated mutable resources; those are outside this launch
contract unless they are part of the entrypoint's declared pre-application load
closure. Neither exclusion makes a pathname recheck authoritative.

## Required invariants

1. **One authority.** `probe_version()` and `spawn()` both delegate to the same
   sole native launch primitive; no public operation creates an attested CLI
   process by another route.
2. **Accepted-evidence binding.** The native authority that accepts the initial
   executable generation retains the opaque capability used by every version
   probe and later launch. A caller cannot reacquire an attacker's replacement
   and attach old evidence to it.
3. **Immutable bytes.** A retained file descriptor or vnode identity alone is
   insufficient. The executable bytes remain enforceably immutable through exec,
   or the backend uses a semantically validated sealed snapshot.
4. **Pre-application closure.** The mapped main image and every loader,
   interpreter, library, delegate, entrypoint, or injected resource that can run
   before trusted application code are frozen or come from an explicit
   system-trust policy. Mutable local members make the strategy unsupported.
5. **No pathname fallback.** Missing native support, an unknown entrypoint, a
   changed object, or receipt failure returns a typed refusal before application
   code runs.
6. **First-instruction gate.** A child cannot run application instructions until
   the native backend has either executed a pinned object directly or verified
   the suspended child's mapped image.
7. **Semantic preservation.** `argv`, cwd, environment, stdio, process groups,
   cancellation, startup/idle timeout, resume, and supported script resource
   semantics remain identical. Otherwise that entrypoint fails closed.
8. **Auditable completion.** Every successful launch produces a closed,
   versioned receipt bound to the lease, command, environment, and mapped image.
9. **Bounded ownership.** File handles, suspended processes, control pipes, and
   temporary artifacts have explicit budgets and are cleaned on every terminal
   path.

## Authority model

```text
selected CLI path
                |
                v
        native authority attest
        - resolve from trusted root
        - freeze executable generation
        - classify entrypoint strategy
        - freeze baseline pre-application load closure
        - retain opaque AttestedExecutable
        - run version probe from that capability
                |
                v
        AttestedExecutable.spawn
        - bind caller argv + exact child env + cwd
        - reject environment that changes the retained closure
        - verify every retained identity
        - platform-native create/exec
        - gate first application instruction
        - verify mapped object(s)
        - emit launch_receipt.v1
                |
                v
        shared runtime process handle
        - existing stdio/event parser
        - existing timeout/cancel/group cleanup
```

The Python layer may select a path and assemble a request, but it cannot turn
validation evidence into launch authority. Before any version probe executes,
the native extension owns initial attestation, entrypoint classification,
baseline pre-application closure freezing, retained capability lifetime, and
spawn. A per-launch environment may only narrow or preserve that closure; it
cannot introduce a newly accepted generation. The capability lives until the
runtime closes; reacquisition requires a new explicit runtime initialization and
cannot inherit the prior attestation.

Before every version probe or task spawn, the native authority independently
rewalks the selected live resolution chain and requires it still to denote the
retained generation. Deletion, upgrade, symlink retargeting, or in-place drift
returns `resolution_changed` or `object_changed`. This live comparison is a
fail-closed drift signal that preserves #1945; it never becomes launch authority
and never causes reacquisition. Recurring version probes required by #1945 run
through the retained capability and the same internal native launch primitive.

## Public contract

The Python package exposes a narrow interface; concrete types may be dataclasses
or extension classes, but callers must not forge their fields.

```python
attested = await lease_provider.attest(
    executable=resolved_cli,
    entrypoint_policy=runtime_entrypoint_policy,
)
version = await attested.probe_version(version_argv)
spawned = await attested.spawn(
    argv=argv,
    cwd=cwd,
    env=child_env,
    stdin=PIPE,
    stdout=PIPE,
    stderr=PIPE,
)
process = spawned.process
receipt = spawned.receipt
```

`ExecutableLeaseUnavailable` is a normal fail-closed outcome with one of a
closed set of reasons:

- `unsupported_platform`
- `unsupported_entrypoint`
- `resolution_changed`
- `object_changed`
- `interpreter_changed`
- `mutable_executable`
- `untrusted_load_dependency`
- `loader_environment_rejected`
- `working_directory_unavailable`
- `launch_verification_failed`
- `launch_verification_timeout`
- `resource_budget_exceeded`
- `native_backend_unavailable`

Callers may render remediation, but may not retry through a pathname launcher.

## Capability ownership and concurrency

The runtime factory creates one lazy, single-flight attestation task per runtime
instance. Parallel first dispatches await that same task; they cannot acquire two
generations. After initialization, the native capability supports concurrent
`spawn()` calls and each returned `LeasedProcess` owns an independent reference
through receipt publication and terminal process cleanup.

Every attested runtime implements idempotent async `aclose()`. The factory or
executor that owns the runtime must call it on runtime replacement, executor
shutdown, and terminal disposal. Closing marks the runtime unavailable to new
probes/spawns, cancels and awaits unfinished attestation or pre-receipt launch
handshakes, and releases the long-lived handles only after those operations have
cleaned up. Already published child processes remain under the existing shared
loop's termination/wait contract; their per-process native references survive
until that cleanup completes.

Close-versus-initialize, close-versus-spawn, caller cancellation, native timeout,
and double-close races have deterministic tests. No cancelled future, failed
attestation, or abandoned first dispatch may retain handles or allow a later
waiter to treat partial state as accepted.

## Launch receipt v1

`launch_receipt.v1` contains at least:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exactly `launch_receipt.v1` |
| `lease_id` | Opaque identifier minted by the native authority |
| `backend_abi` | Native strategy and ABI version |
| `platform` | OS and architecture |
| `entrypoint_kind` | `native`, `node`, `python`, or another explicitly supported kind |
| `original_path` | Operator-selected canonical display path; not authority by itself |
| `resolution_chain_digest` | Digest of the leased chain/object identities |
| `mapped_image_identity` | Platform file identity plus content digest of the executed native image |
| `entrypoint_identities` | Closed list for a supported interpreter/delegate strategy |
| `load_closure_identities` | Loader/library/injection closure plus the system-trust policy used |
| `argv_digest` | Digest of length-delimited argv |
| `environment_digest` | Digest of the exact child environment after normalization |
| `cwd_identity` | Held and spawn-bound directory identity, not only a string path |
| `pid` / `process_group_id` | Spawned process authority returned to the runtime |
| `verified_before_resume` | Must be true for a successful suspended-launch strategy |

The receipt is evidence about a successful launch. It is not accepted from
agent output, ordinary subprocess stdout, or a mutable sidecar file.

## Platform strategies

### Linux

An ordinary retained FD plus `execveat(..., AT_EMPTY_PATH)` defeats rename and
symlink replacement but **does not freeze inode bytes**. It is not sufficient
authority for a mutable executable. A supported native strategy must use an
enforceable write barrier that excludes pre-existing writers and turns every
lease break into refusal, a verified immutable/fs-verity object, or an attested
sealed `memfd` snapshot. A snapshot strategy activates only after tests prove
that `/proc/self/exe`, signing, self-location, adjacent resources, and delegate
semantics remain valid for that installed CLI shape. Otherwise acquisition
returns `mutable_executable` or `unsupported_entrypoint`.

The backend inventories ELF `PT_INTERP`, `DT_NEEDED`, RPATH/RUNPATH and relevant
loader environment. Root-owned system libraries covered by a documented system
policy may be trusted; every mutable local dependency must be frozen and listed
in the receipt. The backend launches from native code, uses a held-directory
operation such as `fchdir` for cwd, and returns pidfd/control-pipe-backed process
authority.

Kernel shebang execution is not an accepted shortcut. A supported script
strategy parses only allowlisted shebang forms from immutable entrypoint bytes,
resolves and leases the complete `/usr/bin/env`/PATH/interpreter chain, FD-execs
the actual interpreter, and supplies the entrypoint only through an immutable
FD or semantically validated snapshot. `FD_CLOEXEC`/fd-derived-name behavior and
descriptor leakage are part of its tests. Every other shebang fails closed.

### macOS

macOS support is gated by a prototype that demonstrates a public, entitlement-
compatible API for immutable source bytes, held-directory cwd semantics, and
main-image plus dyld load-closure verification. A candidate may launch with
`POSIX_SPAWN_START_SUSPENDED`, validate the suspended mappings against held
evidence, then resume, but vnode identity alone is not content immutability.
Failure to prove any primitive keeps macOS typed as `unsupported_platform`.

Because out-of-band `SIGCONT` control is excluded by the threat model, suspension
may be used as the verification gate. A failed or timed-out verification must
kill, wait/reap, and close every handle before returning a refusal.

Native signing, quarantine, and platform policy errors are returned as typed
launch refusals. They are not bypassed by copying the executable.

### Windows

The backend retains source handles that deny write/delete replacement, freezes
or policy-trusts the imported DLL/TLS load closure, starts the child with
`CREATE_SUSPENDED`, compares the main module and pre-application closure with
held evidence, installs held cwd and Proactor-compatible stdio, and preserves the
current per-platform process-group semantics (which create no Windows group),
then resumes it. Any mismatch or timeout terminates and waits for the suspended
child before handles are released. Out-of-band process/thread handle control is
excluded by the threat model.

### Unsupported systems

The Python package remains importable on systems outside the declared dependency
markers, but acquisition returns `unsupported_platform`. On a declared supported
target, a missing mandatory companion wheel is an installation failure. A source
or editable environment that deliberately omits the backend returns
`native_backend_unavailable`. Neither condition silently weakens authority.

## Pre-application load closure and environment

The first-instruction claim covers more than the main file. Each strategy must
inventory the platform loader/interpreter closure that can execute first: ELF
interpreter and shared objects, dyld images, Windows DLLs/TLS callbacks, shebang
interpreters, Node/Python preload hooks, and known delegates. System-owned inputs
may be accepted only by an explicit platform policy; mutable local inputs are
leased/frozen or rejected.

Loader-injection variables such as `LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*`,
`NODE_OPTIONS`, `PYTHONPATH`, and equivalent runtime hooks are rejected unless a
strategy explicitly leases their complete resource closure and records it in
the receipt. The shared child-environment builder performs an early check for
actionable remediation, and the native authority independently repeats the
closed check during every probe/spawn. Python preflight never grants authority.

## Script and delegate compatibility

Native binaries are the first candidate, not automatically supported. Their
loader closure, self-location, adjacent resources, delegates, and executable
immutability must pass the installed-shape inventory. Symlinked launchers bind
the complete symlink chain and terminal object while recording the original
display path separately; executing a terminal FD/snapshot must not change the
CLI's self-location behavior.

Node, Python, and shebang entrypoints require a strategy-specific proof. Merely
leasing the interpreter is insufficient because the interpreter may reopen a
replaced entrypoint or delegate. A supported strategy must prove all of the
following:

- interpreter/delegate and entrypoint objects remain leased through first use;
- `__file__`, `__dirname__`, `import.meta.url`, relative imports,
  `process.argv[0]`, `process.argv0`, `process.argv[1]`, `process.execPath`,
  `require.main.filename`, module-cache keys, package/ESM resolution, source
  maps/stacks, workers, `child_process.fork`, native addons, and resource paths
  retain the CLI's documented behavior;
- lazy delegate selection cannot escape the frozen object set; and
- replacement of either interpreter, entrypoint, package root, or delegate
  cannot create the hostile side-effect marker.

Passing the original mutable entrypoint pathname to Node/Python is forbidden: it
would reopen attacker-controlled bytes. FD and snapshot names are observable, so
support is granted per tested installed CLI shape, never to “Node” in general.

If those properties cannot be met on a platform without a private immutable
package-root snapshot, that entrypoint remains unsupported there. A snapshot is
admissible only when its size is bounded, its complete resource closure is
known, semantic equivalence is tested, and cleanup survives success, failure,
cancellation, and crash recovery.

## Packaging decision

Keep `ouroboros-ai` as a pure Python wheel and publish
`ouroboros-executable-lease` as an exact-versioned `abi3-py312` native extension
package. This requires a new PyO3/maturin, cibuildwheel, auditwheel/delocate,
provenance, and installation-smoke pipeline; the existing standalone TUI Rust
release is useful precedent but is not this pipeline.

The core wheel has a mandatory exact companion dependency on declared supported
OS/architecture markers. Both manylinux and musllinux wheels are required before
enabling the Linux marker because environment markers cannot distinguish glibc
from musl. Missing wheels on a declared target intentionally fail installation.
Unsupported targets receive no dependency and fail closed at runtime.

The initial marker set is Linux
`platform_machine in {"x86_64", "aarch64"}`, macOS
`platform_machine in {"arm64", "x86_64"}`, and Windows
`platform_machine in {"AMD64", "x86_64"}`, each combined with its exact
`platform_system`. The companion publishes wheels only—no sdist that could
silently compile an unaudited local backend during dependency resolution.

Because `ouroboros-ai` uses dynamic hatch-vcs versions, a build metadata hook
generates the exact `Requires-Dist` pin from the resolved build version. Release
checks compare the core version, companion version, backend ABI, and every wheel
filename. Static manual drift is not allowed. Source/editable installs use an
explicit workspace-built backend; a mock backend is allowed only in tests and
never grants production launch authority.

Each native release must publish hashes/provenance, exercise wheel installation
on every target, and run the same adversarial conformance binary. The initial
support matrix is:

| Platform | Architecture | Initial native-binary target |
| --- | --- | --- |
| Linux manylinux | x86_64 | required |
| Linux musllinux | x86_64 | required |
| Linux manylinux | aarch64 | required |
| Linux musllinux | aarch64 | required |
| macOS | arm64 | required |
| macOS | x86_64 | required |
| Windows | x86_64 | required |

The matrix installs one `abi3-py312` wheel under Python 3.12, 3.13, and 3.14 on
each target. Native wheels publish and are verified on PyPI before the core
wheel. A partial native publish never activates core; a core publish may resume
only against the already immutable matching native artifacts. Other targets
remain explicit fail-closed outcomes until their tests and artifacts exist.

## Shared runtime integration

Codex, Copilot, Gemini, Goose, and Grok inherit one launch loop. Integration
replaces its direct `create_subprocess_exec` call with a lease-provider
dependency and keeps the existing reader, event normalization, startup/idle
timeouts, resume handling, cancellation, and process-group cleanup above the
returned process abstraction. Zcode and Antigravity also inherit parts of this
base but are outside #1948 because they do not enable the attestation snapshot;
activating attestation for either requires a separate issue and the same lease
contract.

Current version probes run synchronously in `CodexCliRuntime.__init__`. The
migration removes security attestation from the constructor and performs async
first-dispatch initialization: acquire the long-lived native capability, run the
version probe through it, then retain it for fresh execution and resume. Routing
may cache only the opaque capability and its receipt evidence. Constructor,
resume, probe-failure, and runtime-close tests must migrate together. A runtime
must not probe one path through the lease and later launch another through a raw
subprocess call.

## Leased process protocol

The native backend must adapt its handles to the exact async contract consumed
by the existing shared loop:

- `stdin.write()`, `drain()`, `close()`, and `wait_closed()`;
- async `stdout.read()` and `stderr.read()` with EOF/backpressure behavior;
- `wait()`, a live terminal `returncode`, and a non-reusable process identity;
- `pid`, `terminate()`, and `kill()`;
- POSIX process-group creation and post-exit cleanup; and
- Windows Proactor-compatible pipes and process handles.

Conformance covers each method/state transition, PID reuse resistance,
cancellation during acquire/spawn/verification, signal/group cleanup, pipe
backpressure, early EOF, and child exit before receipt publication.

## Snapshot ownership and recovery

Kernel-owned sealed memory is preferred because process death reclaims it. A
strategy that needs disk snapshots uses one owner-only, no-follow cache root and
an atomic, fsynced ownership manifest containing snapshot digest, byte count,
lease ID, owner process identity resistant to PID reuse, creation time, and
cleanup state. The initial global limits are 16 snapshots, 512 MiB total, 256
MiB per snapshot, and 24 hours maximum age; exceeding any limit fails with
`resource_budget_exceeded` before launch.

Startup and pre-allocation reapers remove only entries whose manifest, owner,
path, and digest validate and whose recorded process generation is no longer
alive. Invalid/ambiguous entries are quarantined without following links. If the
registry cannot be locked, validated, or durably updated, disk-backed snapshot
acquisition fails closed. Tests cover success, launch failure, cancellation,
hard crash, PID reuse, corrupt manifests, symlink substitution, quota exhaustion,
and reaper interruption.

## Conformance harness

Every backend and entrypoint strategy runs deterministic barriers at these
points:

1. after object acquisition;
2. after command construction;
3. immediately before the native create/exec primitive;
4. after child creation but before resume/exec acknowledgement; and
5. during cancellation and launch failure cleanup.

At each applicable barrier, the harness atomically replaces the path, mutates
the original object, swaps a symlink component, replaces an
interpreter/delegate/load dependency, changes cwd ancestry, and injects loader
environment. A passing attack test proves all three:

- the hostile side-effect marker does not exist;
- the mapped image and all strategy identities equal the held lease evidence;
- the receipt names the same authority and exact argv/environment/cwd.

An exception without these negative-effect and identity proofs is not a pass.
Compatibility tests cover native binaries, symlinked launchers, installed
`#!/usr/bin/env node` chains, every supported or explicitly refused shebang,
delegate closure, stdio backpressure, process groups, launch-verification
timeout, cancellation, resume, and durable snapshot cleanup after crashes.

## Delivery stack

1. **RFC and harness contract.** Merge this design, receipt schema, test vectors,
   and a backend-neutral conformance interface. No runtime activation.
2. **Native package and feasibility gates.** Build platform wheels and prove
   immutable bytes, pre-main load closure, cwd binding, public API/privilege
   availability, and native-binary replacement/mutation behavior without
   shared-runtime activation. A platform that fails remains unsupported.
3. **Entrypoint strategies.** Add only the Node/shebang/delegate shapes required
   by installed Codex-family CLIs; unsupported shapes remain typed refusals.
4. **Shared runtime activation.** Switch version probes, fresh execution, and
   resume across Codex, Copilot, Gemini, Goose, and Grok in one contract.
5. **Release and cleanup proof.** Require exact-version wheel availability,
   cross-platform CI, resource budgets, and all terminal cleanup paths before
   closing #1948.

The issue remains open through step 5. No intermediate PR may claim to close it.

## Rejected alternatives

- **A final pathname recheck:** preserves the check-to-exec race.
- **Python `preexec_fn`:** still re-resolves paths, is unsafe with threads, and
  has no Windows contract.
- **Copy only the executable:** changes script/resource/signing behavior and
  leaves delegates unresolved.
- **A child launcher executable:** has the same replacement race unless the
  launch authority is already mapped into the parent; this is why the broker is
  a native extension, not another pathname-launched helper.
- **Best-effort fallback:** converts missing security support into silent
  authority weakening and is forbidden.

## Exit criteria

#1948 may close only when every acceptance criterion in the issue is backed by
current exact-head evidence across the supported platform matrix; Codex,
Copilot, Gemini, Goose, and Grok launch through `ExecutableLease`; hostile
replacement cannot create a side effect; and no raw attested-CLI pathname launch
remains for those five runtimes.
