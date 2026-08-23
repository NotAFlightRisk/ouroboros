# Ouroboros Code Audit Backlog — Round 2

> Generated: 2026-08-23
> Method: 5 parallel analysis agents over `src/ouroboros/` (312k LOC, 597 modules)
> Round 1 findings that are already fixed in PRs #2233–#2241 are **excluded** here.

## Executive summary

| Category | Count | Highest severity |
| :--- | ---: | :--- |
| W. Windows — platform guards & process lifetime | 41 | HIGH (hard crash) |
| X. Windows — paths, encoding, digests | 38 | CRITICAL |
| M. MCP layer & CLI | 33 | HIGH |
| P. Persistence & evaluation | 40 | CRITICAL (silent pass, data loss) |
| D. Documentation ↔ code mapping | 38 | HIGH |
| **Total** | **190** | |

**Headline — why Windows does not work.** Three mechanisms compound so that no
Windows code path is validated by anything:

1. **24 sites guard with `os.name == "nt"`; only 4 use `sys.platform == "win32"`.**
   Only `sys.platform` is narrowed by a type checker.
2. **`[tool.mypy]` globally disables `attr-defined`, `misc`, `arg-type`, `call-arg`,
   `assignment`, `operator`** — so `msvcrt.locking`, `CREATE_NEW_PROCESS_GROUP`,
   and `os.killpg`-on-Windows are all unverifiable *by configuration*.
3. **No Windows CI.** Every job in `test.yml` and `lint.yml` is `ubuntu-latest`,
   and 12 Windows branches carry `# pragma: no cover`.

Good news: **no module-level POSIX-only import exists**, so the package imports
on Windows. It degrades *after* import.

---

## W. Windows — platform guards & process lifetime (41)

### W1–W4. Hard crashes

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| W1 | `hermes/artifacts.py:179` | `ctypes.CDLL(None, use_errno=True)` executes at line 179, **before** the `elif os.name == "nt"` branch at line 200. There is no `LoadLibrary(NULL)` on Windows, so `CDLL(None)` raises and the Windows branch is **dead code**. Every Hermes artifact publish fails. `codex/artifacts.py:403` and `cli/runtime_activation.py:464` get the ordering right — check `nt` first. | HIGH |
| W2 | `orchestrator/heartbeat.py:293-296` | When `fcntl is None` (Windows), `elif path_existed and not is_owned_by_current_process(...): raise OSError` fires **before** the `is_holder_alive()` staleness check at line 298, making it unreachable. On POSIX a dead holder's `flock` succeeds and self-heals. On Windows any leftover `~/.ouroboros/locks/<session>` wedges that session id **forever**. | HIGH |
| W3 | `core/file_lock.py:506-518` | `msvcrt.locking(fd, LK_LOCK, 1)` retries 10× at 1 s then raises `OSError(EDEADLK)`. The `BlockingIOError` translation is gated on `if not blocking`, so the default `blocking=True` leaks a raw `OSError` to callers written for `flock`'s indefinite wait. Any lock held > 10 s becomes a hard error. | HIGH |
| W4 | `core/file_lock.py:39-47, 94-101` | `parent_fd=` raises `ENOTSUP` on Windows. Unreachable today only because the sole caller sits behind an `os.name == "nt"` early return; any new caller crashes on Windows only. | LOW |

### W5–W13. Process lifetime — silent no-ops

Confirmed against CPython: `start_new_session` is **silently ignored** on Windows
(`subprocess.py:1461`, `unused_start_new_session`). It raises nothing.

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| W5 | `dashboard_web/daemon.py:189` | `start_new_session=True  # detach: not killed when the MCP session exits` is a no-op. Without `DETACHED_PROCESS`/`CREATE_NEW_PROCESS_GROUP` the daemon stays in the parent console and dies on Ctrl-C — the opposite of the stated contract. | HIGH |
| W6 | `cli/commands/tui.py:142` | Same no-op for `ooo tui open`. | MED |
| W7 | `cli/commands/tui.py:196-256` | `_dispatch_for_terminal` handles ghostty / iTerm / Terminal / wezterm / vscode, then falls back to gnome-terminal / konsole. **No `wt.exe`, no `cmd /c start`**, so `ooo tui open` never opens a window on Windows. The manual fallback then prints `cd <shlex.quote(path)> && …` — POSIX shell syntax cmd.exe cannot run. | MED |
| W8 | `orchestrator/codex_cli_runtime.py:289` | `_use_process_group = os.name == "posix"` ⇒ launch kwargs `{}`, `_process_group_id()` `None`, `_cleanup_completed_process_group()` returns immediately. Every companion shell / node child of a Codex turn **leaks on Windows**, even though a working Job Object implementation exists 200 lines away. | HIGH |
| W9 | `cli/commands/codex.py:942, 1149-1211` | `start_new_session=os.name == "posix"`, then the non-posix branch only does `proc.terminate()`/`kill()`. Under the shipped `uvx → python` topology the grandchild MCP server survives the probe holding stdio pipes and the SQLite file. | MED |
| W10 | `orchestrator/opencode_runtime.py:627-661` | Windows orphan cleanup shells out to `wmic`, which is **absent by default on Windows 11 24H2 / Server 2025** ⇒ `FileNotFoundError` ⇒ bare `except Exception` ⇒ `_collect_stderr_after_windows_cleanup` then cancels the stderr drain and returns `[]`. Children leak **and** the error output explaining the failure is discarded. | HIGH |
| W11 | `providers/codex_cli_stream.py:293` | `os.getpgid` inside `suppress(ProcessLookupError, PermissionError, OSError)` — `AttributeError` is not suppressed. Safe today only by accident (W8 makes it unreachable). | LOW |
| W12 | `orchestrator/verify_command_runner.py:116-180` | **Reference implementation.** `CREATE_NEW_PROCESS_GROUP \| CREATE_SUSPENDED`, Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, then `ResumeThread`. The only correct Windows containment in the repo — and it is `pragma: no cover`, so nothing verifies it. | FINE |
| W13 | `mcp/detached_jobs.py:290-295` | Correct: `CREATE_NEW_PROCESS_GROUP \| DETACHED_PROCESS`. Both constants exist, so the `getattr(..., 0)` fallback never silently un-detaches. | FINE |

### W14–W23. Permission no-ops — `chmod` only toggles the read-only bit

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| W14 | `config/loader.py:284` | `_set_secure_permissions()` on `credentials.yaml` provides **no access control** on Windows; the file keeps its inherited ACL. **API keys land readable by every local principal.** | HIGH |
| W15 | `config/loader.py:528` | `credentials_file_secure()` compares `(mode & 0o777) == 0o600`; CPython synthesizes `0o666` on Windows, so it can **never** be true — a permanent, unfixable "credentials insecure" verdict. | MED |
| W16 | `cli/commands/config.py:914` | Same `chmod(S_IRUSR\|S_IWUSR)` no-op. | MED |
| W17 | `core/owner_only.py:132-160` | Warns **once per process**, then delegates to `_write_atomic_unscoped`. Atomicity preserved, confidentiality not — yet callers still get `True`. Every Interview/Seed/PM/Auto state file inherits the directory ACL. | MED |
| W18 | `core/owner_only.py:257` | `chmod(path, 0o700)` on `~/.ouroboros` is a no-op. | LOW |
| W19 | `providers/credential_authority.py:94, 113` | Credential store `mode=0o700` / `0o600` ignored. Also `os.link(..., follow_symlinks=False)` at `:137` requires NTFS — fails on FAT/exFAT/network shares. | MED |
| W20 | `mcp/detached_jobs.py:132-134, 148-160` | Detached job request/status files are not private; payloads carry prompts and inherited environment. | MED |
| W21 | `orchestrator/heartbeat.py:88, 194, 284` | Lease and cancellation files world-readable ⇒ **any local user can forge a cancellation record**. | LOW |
| W22 | `cli/runtime_activation.py:376-385` | `_validate_prepared_mode` returns early on `nt`. The comment is correct (the check would be vacuous), but the `requested_mode=0o600` contract is then neither enforced nor verified. | MED |
| W23 | `cli/commands/mcp.py:277` | `_SHELL_ENV_CACHE_FILE.chmod(0o600)` on a cache that holds API keys — no-op. | MED |

### W24–W28. Verification pipeline silently switches off

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| W24 | `harness/deliver_gate.py:730` | `_nofollow_workspace_artifact_matches` opens with `if not nofollow_directory_capabilities_available(): return False`. That helper needs `O_DIRECTORY`, `O_NOFOLLOW`, and `dir_fd` support — **none exist on Windows**. The deliver gate can never confirm a produced artifact and rejects every delivery. | HIGH |
| W25 | `orchestrator/leaf_dispatcher.py:124` | `_bash_soft_fd_limit()` calls `os.sysconf("SC_OPEN_MAX")` → `AttributeError` → `None` ⇒ fd budget `0` ⇒ `workspace_fd_count > total_budget` always true ⇒ `return ()`. **Bash filesystem-effect provenance is entirely disabled**, with no diagnostic. | HIGH |
| W26 | `evaluation/detector.py:518, 525` | `shlex.split(command, posix=(os.name != "nt"))` keeps quote characters in tokens on Windows, and the escape check is `raw_head.startswith(("/", "~"))`. `C:\attacker\pytest` starts with neither ⇒ **an out-of-workspace verify command POSIX refuses is accepted on Windows.** Separately, `posix=False` makes `pytest -k "not slow"` a different token, so the same seed is accepted on Linux and rejected on Windows. Same pattern at `evaluation/languages.py:194`. | HIGH |
| W27 | `evaluation/detector.py:1793-1796` | `_wrapper_invocation_is_runnable` returns `os.name == "nt"` for `.cmd`/`.bat`, but `CreateProcess` **cannot execute batch files** — they need `cmd.exe /c`. The gate admits `gradlew.bat`, then Stage 1 fails to spawn it. | MED |
| W28 | `orchestrator/verify_shell.py:148-163` | Well built (Git Bash discovery, WSL-launcher rejection, `-c` semantics probe) — but with no Git Bash/MSYS2 installed the AC gate can **never** run, so `ooo run` verification is permanently unavailable rather than merely different. | MED |

### W29–W41. MCP lifecycle, PID identity, and the rest

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| W29 | `cli/commands/mcp.py:478` | The orphan watchdog is inert on Windows (`_resolve_client_identity` returns `None`; liveness relies on `os.kill`/`ps`). Worse there than on POSIX: Windows file locking is **mandatory**, so a leaked server actively blocks the next session's writes. | HIGH |
| W30 | `cli/commands/mcp.py:838-846` | No SIGTERM path on Windows, so a client `TerminateProcess` skips the `finally` — and therefore skips the WAL TRUNCATE checkpoint that the comment says prevents unbounded `-wal` growth. | MED |
| W31 | `cli/commands/mcp.py:298-330` | `shell = os.environ.get("SHELL", "/bin/bash")` ⇒ `FileNotFoundError` ⇒ `Warning: shell env load failed` on **every** launch, and no PATH/API-key recovery for a GUI-launched server. `dump_cmd` also hardcodes `python3`. | MED |
| W34 | `orchestrator/heartbeat.py:116-155` | `_get_process_start_time` branches on `Darwin` else assumes Linux `/proc`. On Windows `platform.system()` is `"Windows"` ⇒ takes the Linux branch ⇒ `None`. So the **PID-recycling guard is dead**, while `_is_windows_process_alive` is deliberately fail-open — which is what makes W2 unrecoverable rather than transient. | MED |
| W35 | `core/file_lock.py:512-514` | `msvcrt.locking` has **no shared mode**: `LK_RLCK`≡`LK_LOCK`. `exclusive=False` silently serializes readers, and with W3 a reader can fail outright after 10 s. `checkpoint.py:485` and `interview.py:1234` both expect shared readers. | MED |
| W36 | `core/file_lock.py:143-146` | `stable_parent_authority` yields `None` on Windows ⇒ `_validate_active_lockfile` re-stats **by path** instead of relative to a pinned dirfd — reintroducing the exact TOCTOU the dirfd design eliminates. | MED |
| W37 | `mcp/update_notice.py:77-97` | Structurally correct msvcrt/fcntl split, but inherits W3's 10-second failure. | LOW |
| W38 | `cli/commands/setup.py:2412` | `os.symlink` in the snapshot-restore path needs Developer Mode or `SeCreateSymbolicLinkPrivilege`. **Rollback of a runtime setup fails on a default Windows account.** | MED |
| W39 | `core/seed.py:163-192` | Correct: `PureWindowsPath` + UTF-16 code-unit counting, returns before `os.pathconf`. Note the intentional asymmetry — artifact paths legal on POSIX are rejected on Windows. | FINE |
| W40 | `codex/artifacts.py:403`, `cli/runtime_activation.py:464` | Correct: `os.rename` → `MoveFileEx` without `REPLACE_EXISTING` raises `FileExistsError`, matching `RENAME_NOREPLACE`. Both check `nt` first — the ordering W1 gets wrong. | FINE |
| W41 | `cli/commands/tui.py:329`, `cli/commands/update.py:200`, `cli/opencode_config.py:107`, `cli/streams.py:53`, `cli/windows_codex_mcp.py:29-90`, `evolution/loop.py:254` | All correct. No `WindowsSelectorEventLoopPolicy` is set, which is right — Proactor is what `create_subprocess_exec` needs, and the one `add_signal_handler` is properly guarded. | FINE |

---

## X. Windows — paths, encoding, digests (38)

The dominant real-world breakage is **encoding**, not path separators. 68 of 215
`read_text`/`write_text` calls omit `encoding=`, and 36 `subprocess(text=True)`
calls omit it. On a Korean/Japanese/Chinese Windows box
`locale.getpreferredencoding()` is cp949/cp932/cp936.

### X1–X9. Asymmetric UTF-8 write / locale read

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X1 | `cli/commands/uninstall.py:265, 283` | `claude_md.read_text()` then `write_text(cleaned)`, no encoding — while `runtime_instruction_artifacts.py:80` writes that same file as UTF-8. `ooo uninstall` **raises on a `CLAUDE.md` containing Korean**, or silently re-encodes to cp1252 and permanently mangles the user's instruction file. | HIGH |
| X2 | `cli/commands/setup.py:242, 4088, 4401, 4476, 4519, 4522` | `_atomic_write_text` is exemplary (`encoding="utf-8", newline=""`), but every read-back is bare `read_text()`. `:4088`/`:4401` read VS Code `settings.json`, always UTF-8 and routinely containing `C:\Users\재규\…`. **`ooo setup` aborts for any user with a non-ASCII profile name.** | HIGH |
| X3 | `cli/commands/config.py:1055-1067` | Backup/restore reads and writes both files with the locale codec. The rollback path is exactly where corruption is unrecoverable. | HIGH |
| X4 | `cli/commands/config.py:143` | `yaml.safe_load(config_path.read_text())` — `ooo config` crashes on a Korean project name. | HIGH |
| X5 | `backends/model_catalog.py:404` | `tomllib.loads(read_text())` on Codex `config.toml`. TOML is spec'd UTF-8; decoded as cp949, swallowed by `except Exception` ⇒ **wrong default model, no error.** | MED |
| X6 | `backends/model_catalog.py:392` | Same for `~/.hermes/config.yaml`. | MED |
| X7 | `plugin/agents/registry.py:345` | `path.read_text()` on agent markdown, caller catches bare `except Exception` ⇒ a custom agent with Korean role text is **silently dropped**; user sees "agent not found" with no cause. | MED |
| X8 | `cli/commands/uninstall.py:57, 71, 96, 224, 250, 334, 431, 440` | Eight more pairs over `.mcp.json` and Codex `config.toml` ⇒ uninstall leaves half-rewritten JSON. | MED |
| X9 | `core/worktree.py:427-584` | Six lock/state JSON reads/writes. Safe *today* only because `json.dumps` defaults to `ensure_ascii=True`; `host: socket.gethostname()` on a Korean-named PC is already non-ASCII input. | LOW |

### X10–X15. `subprocess(text=True)` without `encoding=`

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X10 | `core/worktree.py:132` | `_run_git_process` feeds `_resolve_repo_root`. `git rev-parse --show-toplevel` for a repo under `C:\Users\재규\…` returns UTF-8 bytes decoded as cp949 ⇒ `Path(...)` names a directory that does not exist. **Worktree creation fails for every non-ASCII Windows username.** The fix pattern already exists in-file: `_run_git_bytes` at `:149` uses `os.fsdecode`. | CRITICAL |
| X11 | `core/git_workflow.py:132` | Same wrapper; Korean commit messages and stderr become mojibake in error details. | HIGH |
| X12 | `orchestrator/n_version_tournament.py:147, 232, 248, 266, 319` | Five `text=True` git/worktree calls ⇒ path decode crash on non-ASCII roots. | HIGH |
| X13 | `auto/checkpoint_commits.py:158, 177` | Checkpoint commit provenance recorded as mojibake. | MED |
| X14 | `bigbang/brownfield.py:104, 142` | Repo scan mis-decodes Korean filenames. | MED |
| X15 | `core/pm_snapshot.py:67`, `evolution/frugality.py:324`, `mcp/server/adapter.py:2136`, `copilot/model_discovery.py:142`, `cli/commands/mcp.py:326, 385`, `cli/opencode_config.py:69`, `cli/omp_config.py:26, 51`, `codex/runtime_profile.py:62` | Remainder of the 36. (Version-string sites like `update.py:698` are genuinely FINE — ASCII only.) | MED |

### X16–X19. Cross-platform digest mismatch — mode bits, not newlines

Newlines are **not** the problem: every content hash reads binary or hashes
`str.encode("utf-8")`. The mismatch is POSIX mode bits.

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X16 | `plugin/digest.py:70` | `if mode & stat.S_IXUSR: return "0755"`. Windows `os.stat` **never** reports the executable bit, so every file hashes as `"0644"` and `canonical_tree_hash` differs from POSIX for an identical tree. **Plugin trust verification rejects a legitimate plugin on Windows.** | CRITICAL |
| X17 | `plugin/digest.py:84-85` | Git for Windows checks out symlinks as plain files by default ⇒ same entry is a symlink on POSIX, a regular file on Windows ⇒ different mode string *and* content hash. | HIGH |
| X18 | `codex/artifacts.py:449` | `digest.update(str(stat.S_IMODE(mode)))` folds platform-dependent mode into an artifact fingerprint. | HIGH |
| X19 | `core/project_identity.py:578-584` | `sha256(path.read_bytes())` over config files. Ouroboros writes LF (`newline=""`), but a Windows editor or `core.autocrlf=true` produces CRLF ⇒ identity churn. | MED |

### X20–X24. Reserved names & illegal characters — validation exists but is not reused

`core/seed.py:45-91` has the correct, thorough validator (forbidden chars,
reserved stems, trailing space/dot). It is wired into **only**
`expected_artifact_path_error`. Three other sites roll their own.

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X20 | `core/worktree.py:382` | `_lock_path` = `root / f"{durable_id}.json"`. `durable_id`'s only validation is `git check-ref-format`, which **permits `< > \| "`** — all Windows-forbidden — and permits `aux`, `con`, `nul`. `nul.json` is unopenable. | HIGH |
| X21 | `core/worktree.py:648` | Same unvalidated `durable_id` becomes a **directory** name ⇒ `git worktree add` fails. | HIGH |
| X22 | `persistence/checkpoint.py:229` | Second divergent copy of the sanitizer: covers forbidden chars but **not** reserved stems or trailing dot/space. Rescued only accidentally by the `checkpoint_` prefix. | LOW |
| X23 | `auto/adapters.py:46, 1035-1053` | **Third** copy of the reserved-stem list. Correct, but will drift from `seed.py:46`. | LOW |
| X24 | `cli/commands/plugin_cache.py:50-60` | `url_cache_destination` replaces `:` and `/` but not `\ ? * < > \| "`. A repo URL with a query string produces an unopenable cache dir. | MED |

### X25–X27. Long paths — no `\\?\` prefixing anywhere in the repo

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X25 | `core/worktree.py:97-100` | `~/.ouroboros/worktrees/<repo>/<durable_id>/` is 50–70 chars before the checked-out tree begins. Add this project's own depth and **260 is exceeded** ⇒ `git worktree add` fails, and `shutil.rmtree` fails the same way, leaving an unremovable worktree. | HIGH |
| X26 | `core/seed.py:163-172` | The interaction bug: the 260-UTF-16 check is correct, but `workspace` is the deep worktree path from X25, so it **fail-closes on perfectly reasonable artifacts** and blames the artifact rather than the worktree root. | HIGH |
| X27 | `persistence/checkpoint.py:186` | `_MAX_SEED_LEN = 232` bounds the *component* but ignores the 260 *total path* limit. | MED |

### X28–X32. Case sensitivity — `startswith` on `realpath` without `normcase`

`cli/commands/codex.py:818` is the only site that does this correctly.

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X28 | `persistence/checkpoint.py:259` | `startswith(str(base_resolved) + os.sep)` raises `ValueError("Path traversal detected")`. Drive-letter case and 8.3 short-name expansion differ ⇒ **spurious traversal error blocks every checkpoint save.** | HIGH |
| X29 | `verification/verifier.py:1060-1066` | `realpath(f).startswith(real_project + os.sep)` filters which files are verified. A case mismatch silently empties the set ⇒ **verification reports success over zero files.** | HIGH |
| X30 | `evaluation/artifact_collector.py:256, 274, 295` | Same pattern ⇒ artifacts silently dropped from evaluation. | MED |
| X31 | `orchestrator/ac_execution_capsule.py:566` | `realpath(self.workspace) != self.workspace` — `realpath` normalizes case, so a valid absolute workspace differing in case **fails closed**. | HIGH |
| X32 | `orchestrator/ac_runtime_handle_manager.py:381`, `orchestrator/level_context.py:66-68, 150-160` | Same class. | MED |

### X33–X38. Path separators — mostly already correct

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| X33 | `evaluation/languages.py:204` | The "reject absolute paths" control is `head.startswith("/") or head.startswith("~")` — misses `C:\…` and UNC `\\server\share`. A Windows absolute path to an allowlisted basename slips through. | MED |
| X34 | `core/runtime_transition.py:128` | Normalization replaces `/` but not `\` ⇒ two spellings of one key. | LOW |
| X35 | `plugin/manifest.py:511, 523` | FINE — deliberate; drive-qualified paths and backslashes are rejected earlier, and the comment says so. | FINE |
| X36 | `cli/opencode_config.py:170-172` | FINE — explicitly normalizes `\` before splitting; docstring calls out Windows mixed separators. | FINE |
| X37 | `providers/litellm_adapter.py:1444`, `evaluation/detector.py:1290`, `plugin/manifest.py:1080` | FINE — model ids and JSON Pointers, not filesystem paths. | FINE |
| X38 | `core/filesystem_capability.py:134-151` | FINE — guarded by the capability probe and an `anchor != os.sep` check that rejects `C:\`. | FINE |

---

## M. MCP layer & CLI (33)

### M1–M9. Job lifecycle

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| M1 | `mcp/job_manager.py:2885-2901` | The TTL sweep treats `get_snapshot` raising `ValueError` ("no events yet") as **expired** and discards the reservation. A job id allocated but not yet carrying `mcp.job.created` — the state **every detached launch occupies for up to 20 s** — is indistinguishable from a dead id ⇒ `has_unresolved_job_acceptance()` flips to `False` mid-launch and a caller can start a **duplicate job**; the id can also be re-handed-out. | HIGH |
| M2 | `mcp/job_manager.py:2863-2882` | GC only runs from the `get_snapshot` read path, throttled to once/hour. A server that starts detached jobs and never polls **never sweeps**. | MED |
| M3 | `mcp/job_manager.py:2885` | The sweep replays the full event stream for **every** known job id, inline inside one unlucky `ouroboros_job_status` call. | MED |
| M4 | `mcp/job_manager.py:697-710` | `elif job_id in self._monitor_terminalized_jobs:` and the following `else:` have **byte-identical bodies**. Either the branch is dead or the distinction was lost. | LOW |
| M5 | `mcp/detached_jobs.py:396-407` | On acceptance timeout the code deliberately does not kill the worker **and** does not `abandon_reserved_job_id`. If the worker then dies before persisting `mcp.job.created`, no job event ever exists: `get_snapshot` raises forever, `retry_start_allowed=False`, reservation never reconciled ⇒ **a job id that can never reach a terminal status.** | HIGH |
| M6 | `mcp/detached_jobs.py:396-407` | The same timeout `raise` skips `cleanup_worker_artifacts`, which every other exit calls. | MED |
| M7 | `mcp/detached_jobs.py:129-137` | **No GC or TTL for `~/.ouroboros/detached-jobs`.** A SIGKILLed worker leaves `<job>.json` containing full tool arguments — goal text, seed YAML, source excerpts — indefinitely. | MED |
| M8 | `mcp/detached_worker.py:243-262` | The failure path wraps compensation in `except Exception: pass`. If it throws, the worker exits 1 with **no terminal event**, so terminal status is not guaranteed observable through the event stream. | HIGH |
| M9 | `mcp/detached_worker.py:187` | `server._tool_handlers.get(...)` — all durable job execution depends on a private attribute; a rename breaks it at runtime only. | MED |

### M10–M12. Handler size and duplication

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| M10 | `mcp/tools/execution_handlers.py:1178` | `handle()` is **905 lines**, with a 212-line nested `_run_in_background`. Functions ≥200 lines nearby: `job_handlers.py:1376` (498), `authoring_handlers.py:2752` (483), `pm_handler.py:1320` (476), `evaluation_handlers.py:532` (415). Worst in repo: `mcp/server/adapter.py:1495 create_ouroboros_server` at **1114 lines**. | HIGH |
| M11 | `mcp/tools/conductor_handler.py:36-56` vs `synapse_handler.py:32-55` | `_required_text`/`_optional_text` are byte-identical copies. **Six** independent "read an int/string out of `arguments`" implementations exist ⇒ six different error messages for the same client mistake. | MED |
| M12 | `mcp/tools/execution_handlers.py:2085` vs `auto_handler.py:2159` | Divergent cwd policy: `auto_handler` enforces `_require_writable_cwd`, `execution_handlers` does not ⇒ a non-writable cwd is rejected up front by `start_auto` but fails deep inside execution for `execute_seed`. | MED |

### M13–M18. Error taxonomy defined but not enforced

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| M13 | `mcp/server/adapter.py:974-999` | The catch-all returns `MCPToolError(f"Tool execution failed: {e}")` with **no `error_code`, no `details`, `is_retriable=False`**. The entire `errors.py` taxonomy is unreachable for anything a handler raises, and retriable failures are advertised as non-retriable. | HIGH |
| M14 | `mcp/tools/*` | **87 `raise ValueError` vs 1 `raise MCPToolError`** (plus 18 `RuntimeError`). Policy failures — "Mutating conductor actions require engine_ownership_state=closed", "seed_handoff_id is unknown" — all flatten into the same uncoded string via M13. The correct pattern exists elsewhere in the same layer, so it is used *inconsistently*. | HIGH |
| M15 | `mcp/failure_taxonomy.py` | **Not imported by any file under `mcp/tools/`.** Reason codes are derived post-hoc from job status strings, after M13 has already destroyed the original classification. | MED |
| M16 | `mcp/telemetry_boundary.py:255-259` | `raise RuntimeError(str(result.error))` drops `error_code`, `is_retriable`, `details`. The in-process boundary is **weaker than the cross-process one** — `detached_jobs.encode/decode_tool_error_rejection` preserves all of them. | MED |
| M17 | `mcp/types.py:326-358` | `to_input_schema()` omits **`additionalProperties: false`**, and this is exactly what `Draft202012Validator` validates against. Any undeclared key a client sends is accepted and forwarded into `arguments`. | HIGH |
| M18 | `mcp/tools/evaluation_handlers.py:661` | `arguments.get("_force_in_process")` is an **undeclared internal flag**, so per M17 an external client can pass `_force_in_process: true` to `ouroboros_evaluate` and **bypass plugin dispatch**. `auto_handler.py:305-311` explicitly rejects the analogous `_start_auto_lease_token` — that guard exists because this class of bug is real, and it was not applied here. | HIGH |

### M19–M33. CLI

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| M19 | `cli/commands/setup.py:275-545` | `_detect_runtimes` is 271 lines of copy-paste with **inconsistent validation**: `goose`/`kiro`/`copilot`/`pi`/`gjc` verify with `shutil.which`, `codex` checks `is_file() and access(X_OK)`, but `gemini`/`antigravity`/`grok`/`zcode` do **no existence check at all** ⇒ setup reports `✓ gemini → /dead/path`, persists it, and prints "Setup complete!". | HIGH |
| M20 | `cli/commands/setup.py:4770-5100` | `setup()` is a 328-line god-function with 15 `elif` branches. Adding a runtime needs coordinated edits in 4 places. | HIGH |
| M21 | `cli/commands/setup.py:4995-5085` | **Seven `_setup_*` failures are never checked.** `_setup_gemini`, `_setup_pi`, `_setup_gjc`, `_setup_goose`, `_setup_antigravity`, `_setup_grok`, `_setup_zcode` are called bare while every other branch does `if not _setup_X(...): raise typer.Exit(1)`. They return `None` after aborting internally, then the caller prints **"Setup complete!" and exits 0** — scripted installs and CI see success for a runtime that was never configured. | HIGH |
| M22 | `cli/commands/setup.py:3830, 3877, 3919, 4005, 4037, 4602` | `~/.ouroboros/config.yaml` written via `open("w")` + `yaml.dump` — **truncate-then-write with no backup**. A crash or Ctrl-C during dump destroys every unrelated setting. The correct primitive `_atomic_write_text` exists **in the same file** (line 4263) and is used by the Codex/Kiro/Copilot paths but not these. | HIGH |
| M23 | `cli/commands/setup.py:4185-4189` | Same truncation window on the user's `opencode.json` — a file Ouroboros does not own. | HIGH |
| M24 | `cli/commands/setup.py:4171-4183` | JSONC comments silently destroyed; the warning only fires when the suffix is literally `.jsonc`. | MED |
| M25 | `cli/commands/setup.py:4459-4488` | `_cleanup_plugin_artifacts` rmtree's the plugin dir **first**, then edits config under `except: pass` ⇒ OpenCode left pointing at a missing directory, the exact state the function exists to prevent, silently. | MED |
| M26 | `cli/commands/setup.py:2683-2924` | `_setup_codex` has careful snapshot/rollback woven inline — and **none of the other twelve runtimes get any of it**, so rollback quality depends on which runtime the user picked. | MED |
| M27 | `cli/commands/plugin.py:1268-1278` | `_looks_like_url` accepts **`http://`** and `git+http://`, passed straight to `git clone`. `canonical_tree_hash` is computed *after* the clone, so the digest attests to whatever arrived — **trust-on-first-use with no transport integrity**. A network attacker can inject arbitrary executable plugin content. | HIGH |
| M28 | `cli/commands/plugin.py:1296-1312, 1867-1877` | `git clone --depth 1` of default-branch HEAD with **no `--ref`/`--tag`/`--commit` option**. A user cannot install a specific audited commit. | HIGH |
| M29 | `cli/commands/plugin.py:1969-2071` | Multi-plugin install is not transactional **across** plugins: on failure at plugin N, plugins 1..N-1 stay installed and trust-granted, and the success summary never prints — the user is not told which ones landed. | MED |
| M30 | `cli/commands/plugin.py:1897-1949` | A repo becomes a permanently resolvable "known catalog" **before** any install is attempted, even if every plugin in it then fails. | LOW |
| M31 | `cli/main.py:23-50` | Eager import of 26 command modules: **0.49 s and 788 modules for `ooo --help`**. (The optional-extra hypothesis is **disproved** — `mcp`, `textual`, `httpx`, `anthropic` are absent from `sys.modules`, so `--help` does not crash when an extra is missing. The cost is breadth: `qa.py:16` imports `ouroboros.mcp.tools.qa`, `auto.py:16-60` pulls ~15 `ouroboros.auto.*` modules.) | MED |
| M32 | `cli/main.py` | **No top-level exception handler**, and 12 unguarded `asyncio.run` sites (`cancel.py:495,498`, `detect.py:90`, `dispatch.py:239`, `harness.py:301`, `init.py:1143,1222`, `job.py:77`, `qa.py:97`, `resume.py:338`, `status.py:450,508`). They check `Result.is_err` *after* the await but nothing catches an exception *from* it — a locked SQLite store, or any of the 87 `ValueError`s from M14, surfaces as a **raw traceback**. | HIGH |
| M33 | CLI-wide | Exit codes are inconsistent: `status.py` has graded codes, everything else uses flat `1` in two spellings (50 files `typer.Exit(1)`, 80 `typer.Exit(code=1)`), and `plugin_dispatch.py` uses raw `SystemExit`, bypassing typer's rendering. | MED |

---

## P. Persistence & evaluation (40)

Two premises from the brief are **wrong** and corrected here:
`REWARD_HACKING_VETO_THRESHOLD` is **not** duplicated — `pipeline.py:23` imports
it from `models.py:32`. And `checkpoint.py` has **no compression at all**, so
there was nothing to review there.

### P1–P4. Durability

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P1 | `persistence/sqlite_connection.py:19` | **`synchronous=NORMAL` under WAL: commits are not fsynced.** COMMIT returns success without flushing the WAL, so on OS crash or power loss the most recent committed transactions are lost. Every "durable" contract — `append_durable`, terminal-session CAS, `_run_to_settlement` — is durable only against *process* death. A session can replay as if its terminal event never happened. | HIGH |
| P2 | `persistence/sqlite_connection.py:14-18` | WAL enablement failure is **swallowed and never read back**. The connection silently keeps rollback-journal mode, taking a whole-database lock; connections in one pool can be in different journal modes; nothing is logged — the concurrency design degrades under exactly the contention it was added for. | HIGH |
| P3 | `persistence/event_store.py:1519-1545` | `"database is locked"` can surface **during COMMIT**, when the insert may already be durable. The retry re-inserts the same `event.id`, hits the PK constraint, which is not a "locked" string, and raises `PersistenceError` ⇒ **the caller compensates for an event that is actually in the store.** | MED |
| P4 | `persistence/event_store.py:3011-3042` | WAL checkpointing is best-effort and swallowed, invoked from one place. Under sustained contention the 2 s `busy_timeout` fails and `-wal` grows unbounded; a surviving `-wal` holds committed data that per P1 is not fsynced. | MED |

### P5–P6. Append-only invariant

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P5 | `persistence/schema.py:41-72` | **Append-only is a convention, not an invariant.** No `BEFORE UPDATE`/`BEFORE DELETE` trigger, no read-only view — `events` is a plain writable table. No `UPDATE`/`DELETE` exists today, but `migrations/runner.py:86` executes arbitrary SQL from `scripts/*.sql`, which is a supported path to rewriting history. | HIGH |
| P6 | `persistence/event_store.py:781-820` | One-winner guards are enforced at the Python API boundary only. A second binary writing the same DB with raw SQL can insert a second terminal event, and the CAS tables would then disagree with the stream. | MED |

### P7–P9. Schema migration

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P7 | `persistence/schema.py:232-246` | **No schema version at all** — no `PRAGMA user_version`, no version row, no compatibility check. `create_all` creates *missing tables* and never alters existing ones. An older binary opening a newer DB fails with `OperationalError: no such column` **deep inside a replay**, not with a clean "database is newer" error. For a tool distributed over PyPI with users on mixed versions, this is the most likely real corruption report. | HIGH |
| P8 | `persistence/migrations/runner.py:54` | **The migration runner is not wired into the event store** — it is called only from `brownfield.py:198`. The `events` table and all six guard tables have **no migration path whatsoever**. | HIGH |
| P9 | `persistence/migrations/runner.py:86` | `sql_content.split(";")` shreds any migration containing a semicolon inside a string literal, trigger body, or `BEGIN…END`. Guarantees the first non-trivial migration fails. | MED |

### P10–P16. Indexes, ordering, growth

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P10 | `persistence/schema.py:66` | `ix_events_aggregate_type` is a redundant prefix of two other indexes — pure write amplification on the hottest table. | LOW |
| P11 | `persistence/schema.py:69-71` | **No `(event_type, timestamp)` composite index.** `get_all_sessions()` had to hardcode `INDEXED BY` in raw SQL (the comment cites 30 s+ queries); `query_latest_events_per_aggregate` has the same shape with **no hint**, so `ooo status`/picker latency grows with total store size. | HIGH |
| P12 | `persistence/event_store.py:1949-1960` | The `INDEXED BY` fallback catches `OperationalError` too broadly — a locked or corrupt DB silently re-runs the full unindexed scan, **doubling the lock window** during the contention that caused the first failure. | MED |
| P13 | `persistence/event_store.py:1713` | **Replay orders by `(timestamp, id)` where `id` is a random UUID.** Two events appended in causal order that share a stored timestamp replay in UUID order — possibly reversed. SQLite `rowid` is the true insertion order, is already used for cursors, and is not used for replay. Aggregate reconstruction is not guaranteed to match what happened. | HIGH |
| P14 | `persistence/schema.py:56-63` | **Two timestamp encodings in one text column**: the Python default writes microseconds + `+00:00`, `server_default=CURRENT_TIMESTAMP` writes `YYYY-MM-DD HH:MM:SS`. `ORDER BY timestamp` is lexical, so they do not interleave correctly. | MED |
| P15 | `persistence/event_store.py` | **Nothing ever prunes.** No `DELETE`, no retention window, no `VACUUM`, no archival anywhere. Combined with P11 the symptom is that `ooo status` gets progressively slower and never recovers. | HIGH |
| P16 | `persistence/event_store.py:1684-1727` | `replay()` is unbounded — every event for an aggregate into a Python list, no limit, no streaming. | MED |

### P17–P23. Checkpoints

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P17 | `persistence/checkpoint.py:322` | No compression, `indent=2`, no size cap on `state`. ~2–3× the necessary bytes plus an fsync every 300 s × 4 levels per seed. | LOW |
| P18 | `persistence/checkpoint.py:682-686` | **Total corruption is laundered into "starting fresh".** When all four levels fail integrity, `load()` returns a message that `RecoveryManager.recover` **substring-matches**, returning `Result.ok(None)` — the same value as a first run. The caller starts from zero and the next `save()` rotates the still-recoverable chain out. A corrupt-checkpoint incident is indistinguishable from a cold start. | CRITICAL |
| P19 | `persistence/checkpoint.py:441, 446, 635, 685, 689, 693` | Corruption is reported with **`print()`** — six sites. In the MCP stdio server stdout **is** the JSON-RPC channel, so this risks corrupting the protocol stream, and the evidence bypasses structured logging entirely. | HIGH |
| P20 | `persistence/checkpoint.py:313-325` | The staged file is fsynced but **the parent directory never is**, so the rename can be lost on power loss. Together with P1 the "durable checkpoint" story is weaker than the surrounding comments claim. | MED |
| P21 | `persistence/checkpoint.py:521-540` | `_rotate_checkpoints` is **dead production code that deletes data** — called only from tests. It `unlink()`s without staging, the exact behavior `_publish_staged_checkpoint` replaced. | MED |
| P22 | `persistence/checkpoint.py:628-635` | `PeriodicCheckpointer` swallows every failure forever: no counter, no backoff, no telemetry. A permanently failing checkpoint is invisible until recovery finds nothing. | MED |
| P23 | `persistence/checkpoint.py:161-166` | Rotation caps at 4 files **per seed_id**, but seed_ids are unbounded ⇒ the checkpoint directory accumulates forever. | LOW |

### P24–P31. Evaluation Stage 1/2 — silent passes

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P24 | `evaluation/mechanical.py:300-307, 245` | **A skipped check counts as a pass, and an all-skipped Stage 1 reports `passed=True`.** An unconfigured check returns `CheckResult(passed=True, details={"skipped": True})`, and `all_passed = all(...)` over five of those is `True`. `MechanicalResult` carries **no "was anything executed" bit**, so `stage1_passed=True` cannot distinguish "lint+build+test green" from "nothing ran". The tool description tells the user Stage 1 skips — the machine-readable verdict says verified. | CRITICAL |
| P25 | `evaluation/languages.py:208, 247` | A **rejected** command (not on the allowlist, absolute path) returns `None`, indistinguishable from "not configured" ⇒ a `mechanical.toml` the user authored yields a **green Stage 1** with only a `log.warning`. | HIGH |
| P26 | `evaluation/mechanical.py:238-256` | Coverage threshold is unenforced whenever the output format is unrecognized: only two hardcoded pytest-cov patterns are parsed, so jest / tarpaulin / go / JaCoCo yield `None` and the `coverage_score is not None` guard **skips the NFR9 gate** — while still reporting "coverage passed". | MED |
| P27 | `evaluation/mechanical.py:128` | `return_code=process.returncode or 0` maps `None` to 0, i.e. to pass. | LOW |
| P28 | `evaluation/pipeline.py:246-256` | **A pipeline with no stages enabled approves unconditionally.** `final_approved = True` is the initial value, narrowed only if `stage2_result` exists; the reward-hacking veto also requires `stage2_result is not None`, so it cannot fire. | HIGH |
| P29 | `evaluation/semantic.py:252-253` | **The anti-gaming signal fails open.** `reward_hacking_risk` is absent from the schema's `required` list and defaults to `0.0`, so the `>= 0.7` veto can **never** fire for any model that omits the optional field. | HIGH |
| P30 | `evaluation/semantic.py:190-199` vs `266-276` | The prompt states empty `questions_used`/`evidence` "is treated as a verification failure"; the parser accepts empty tuples for backward compatibility and **nothing downstream inspects them**. An evaluator that shows no work is approved exactly like one that does. | HIGH |
| P31 | `evaluation/semantic.py:212-249` | **FINE** — malformed output correctly fails closed. Noted so the contrast with P24/P25/P29/P30 is visible. | FINE |

### P32–P34. Reproducibility

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P32 | `evaluation/semantic.py:83`, `consensus.py:139, 603` | **The scoring path is nondeterministic by construction**: Stage 2 at `temperature=0.2`, Stage 3 votes at `0.3`, **no seed parameter**. The same artifact can land on either side of `SEMANTIC_APPROVAL_SCORE = 0.8` and of the 0.7 veto. Answering the question directly: verdicts are **not** reproducible, and the nondeterminism is in the *gate inputs*, not just the prose. | HIGH |
| P33 | `evaluation/consensus.py:365-380, 505-509` | Errored voters are dropped from `votes`, then `majority_ratio = approving / len(votes)` ⇒ **the denominator depends on which voters happened to error.** The same artifact is approved 2/2 on one run and rejected 2/3 on the next. | MED |
| P34 | `verification/verifier.py:1047-1072` | `glob.glob(...)` truncated to `MAX_FILES_PER_HINT = 100`. Glob order is filesystem-dependent, so **which** 100 files are scanned — and therefore `VERIFIED` vs `UNVERIFIABLE` — differs between machines. | MED |

### P35–P38. Stage 3 consensus

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P35 | `evaluation/consensus.py:375-380, 432-439` | **The quorum silently shrinks from 3 to 2.** Docstrings claim "minimum 3 models", the guard is `len(votes) < 2`. With one voter erroring, two approvals give `ratio = 1.0` and Stage 3 approves on two votes; the error list is not surfaced in `ConsensusResult`. | HIGH |
| P36 | `evaluation/consensus.py:141` | `majority_threshold = 0.66` approximates 2/3, so for counts not divisible by 3 the gate is **looser than documented**. | LOW |
| P37 | `evaluation/consensus.py:395-449` | **Single-model fallback is treated as consensus.** Without OpenRouter credentials the *same* model answers three perspective prompts and 2-of-3 approves. `is_single_model=True` is recorded honestly, but `pipeline._build_result` uses `stage3_result.approved` with **no downgrade** — "multi-model verification" silently becomes self-review. | MED |
| P38 | `evaluation/trigger.py:120-165` | `reward_hacking_risk` is **not** a trigger condition, so the strongest gaming suspicion never buys a second opinion. Conversely, pipeline returns early when `ac_compliance` is false, so `STAGE2_UNCERTAINTY` can only fire on the **pass** path. | MED |

### P39–P40. Thresholds and the `verification/` vs `evaluation/` split

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| P39 | `evaluation/semantic.py:85` | `satisfaction_threshold: float = 0.8` is **dead** — nothing reads it; the real gate is `models.SEMANTIC_APPROVAL_SCORE`. A caller who sets it gets silence instead of an effect. This, not the veto constant, is the live drift hazard. Related: `mechanical.py:70` and `languages.py:288` are two independent defaults for one coverage knob. | MED |
| P40 | `verification/models.py:139-142` | `verified_pass` returns `agent_reported_pass` when a report has **zero** assertions — an AC the extractor could not parse passes on the agent's own word. Contained only because `strict` defaults to True; a single caller with `strict=False` converts every unextractable AC into a free pass. | MED |

**On `verification/` vs `evaluation/`** — both live, not duplicates.
`verification/` scans source ("does the code actually contain what the AC
claims") and is the **fail-closed** side. `evaluation/` runs the LLM pipeline
("does a model think the artifact satisfies the AC"). The smell is naming:
four different notions of "verified" exist, and per P24 the `evaluation/` notion
can be true with nothing executed while `verification/` reports `SKIPPED` for the
same run — two approval computations that can disagree with no reconciliation.

---

## D. Documentation ↔ code mapping (38)

**Verified clean — record so future passes skip these:** broken skill→tool
references (**0**), documented-default vs code-default mismatches (**0 of 91**),
documented-but-removed events (**0**), stale internal markdown links (**0 of ~200**),
README numeric claims (**9 of 9 confirmed**). The defect mass is *undocumented
surface*, not wrong documentation.

| # | Location | Issue | Sev |
| ---: | :--- | :--- | :--- |
| D1 | audit tooling | The 9 "missing tools" a naive `grep name="..."` reports are **all false positives**: `ouroboros_python` is a shell helper defined inside the skills, `ouroboros_mcp` is a local variable, `ouroboros_ouroboros__*` is the plugin-prefixed call form, and `ouroboros_brownfield`/`_session_signal`/`_record_conductor_decision` register via a `_TOOL_NAME` constant. Fix the script or it keeps producing phantoms. | INFO |
| D2 | `docs/api/mcp.md:258` | The "Ouroboros MCP Tools" section documents **7 of ~32** registered tools, and no file under `docs/` enumerates the full catalog. | HIGH |
| D3 | `mcp/tools/query_handlers.py:619` | `ouroboros_ac_dashboard` is registered but appears in **0** docs and **0** skills — completely undiscoverable. | MED |
| D4 | `mcp/tools/job_handlers.py:627` | `ouroboros_cancel_execution` documented nowhere, and `skills/cancel/SKILL.md` references only `ouroboros_cancel_job` ⇒ two cancellation surfaces, one reachable. | MED |
| D5 | `docs/config-reference.md:143-158` | The Top-Level Sections table omits **2 of 16** real sections: `telemetry` and `seed`. | HIGH |
| D6 | `config/models.py:794` | **`seed.verify_command_gate` is undocumented and run-blocking.** Set to `block` it refuses the run for any AC lacking `verify_command`. Its own docstring says the default "moves toward `block` over time" — a future default flip in an undocumented key. | HIGH |
| D7 | `config/models.py:774` | `telemetry.enabled` (default **True**) is absent from the reference that claims to list "All `config.yaml` options". | MED |
| D8 | `docs/config-reference.md:183` | `orchestrator.runtime_backend` lists 13 values; the code `Literal` has 14 — **`goose` is omitted**, yet the same file accepts `goose` for `llm.backend` at `:217`. | MED |
| D9 | `docs/config-reference.md:181-196` | The `orchestrator` table omits **7 CLI-path keys** (`hermes_cli_path`, `gemini_cli_path`, `kiro_cli_path`, `goose_cli_path`, `gjc_cli_path`, `antigravity_cli_path`, `grok_cli_path`), all fully docstring'd in code. | MED |
| D10 | `config/models.py:724` | `orchestrator.verify_bash_path` undocumented — and its own comment notes it is "an executable path fed straight into a subprocess". | HIGH |
| D11 | `config/models.py:728-731` | **The whole worktree family is undocumented**: `use_worktrees` (default **True** — mutating workflows run in a different directory than the user's checkout), `worktree_root`, `worktree_cleanup` (default `prune-merged`, **deletes worktrees and `ooo/*` branches**), `worktree_lock_stale_after_minutes`. Default-on and destructive-adjacent. | HIGH |
| D12 | `config/models.py:732-733` | `pm_snapshot_worktrees` (default **True**) causes PM exploration to **fetch + `git reset --hard`** against `origin/HEAD` on every PM interview start. No documented off switch. | MED |
| D13 | `config/models.py:674` | `orchestrator.runtime_profile` — the per-stage routing surface with startup validation — has **0 hits** in `config-reference.md`. | MED |
| D14 | `config/models.py:696` | `orchestrator.reasoning_effort` undocumented as a config key (the matrix documents per-runtime *support*, not the key). | MED |
| D15 | `config/models.py:700` | `orchestrator.opencode_mode` undocumented — it gates `EXTERNAL_HOST_BRIDGE` subagent orchestration. | MED |
| D16 | `config/models.py:705, 727` | `opencode_stdout_idle_timeout_seconds` and `usage_limit_pause_hours` (default **5.0** — how long Ouroboros pauses on provider quota) undocumented. | LOW |
| D17 | `docs/config-reference.md:370-379` | The `execution` table omits **8 live fields**, including `run_verify_commands` (default **True**) and `ac_retry_attempts` (default 2) — both of which change pass/fail semantics of every run — plus `verify_command_timeout_seconds`, `cross_harness_redispatch`, `n_version_tournament`, `decomposition_mode`, `context_pack`, `tui_autolaunch`. | HIGH |
| D18 | `docs/config-reference.md:378` | Documents `project_guidance` default as `[]`; the field is `tuple[str, ...] = ()`. Cosmetic — `config show` will render a tuple. | LOW |
| D19 | `docs/config-reference.md:319` | **CONFIRMED accurate**: tier `cost_factor` 1 / 10 / 30 matches `models.py:840-877`. | FINE |
| D20 | `docs/config-reference.md:488` | **CONFIRMED accurate**: `majority_threshold = 0.66`. The "inert field" annotations throughout the evaluation tables are accurate and unusually well maintained. | FINE |
| D21 | `docs/events.md` | Documents **21 of 36** canonical event types, while presenting itself as the stability contract consumers "can rely on". Undocumented: **all 13 `lineage.*`** (the entire read model behind `ooo status`, `resume-session`, and the TUI), **all 8 `evaluation.*`**, three `interview.*`, two `conductor.*`, three `ontology.*`, and `control.session.signal.*`. | HIGH |
| D22 | `docs/events.md` | **No documented-but-nonexistent events.** The 5 apparent misses are the slash-compressed heading form and all expand to real types. | FINE |
| D23 | `docs/README.md` | `docs/events.md` is **not linked from the docs index** — a stability contract nothing points to. | MED |
| D24 | `docs/runtime-capability-matrix.md:75-87` | **All 12 spot-checked cells are accurate** (Gemini, Kiro, Copilot, Pi, GJC capability rows; Copilot/Pi/Kiro/Hermes/OpenCode/Codex/Claude parameter rows). | FINE |
| D25 | `docs/runtime-capability-matrix.md:44, 73` | **Goose has no column in either main table**, despite a full declared contract and presence in `VALID_RUNTIME_BACKENDS`. It appears in the parameter table at `:81`, so it is the only backend present in one table and absent from the other two — and it is **not** covered by the file's own "newer backends" disclaimer, which names only Antigravity/Grok/Zcode. | HIGH |
| D26 | `docs/runtime-capability-matrix.md:35-41` | The tables cover **9 of 14** valid backends. Missing: `goose`, `antigravity`, `grok`, `zcode`, `claude_mcp`. The disclaimer excuses 3; `goose` and `claude_mcp` are unexcused — and `claude_mcp` appears as a first-class choice in the same file's config snippet. | MED |
| D27 | `docs/runtime-capability-matrix.md:81` | The parameter table omits GJC, which declares `permission_mode_support=IGNORED` — a genuinely lossy, user-visible behavior with no cell. Antigravity and Grok declare `TRANSLATED` support, also uncelled. | MED |
| D28 | `orchestrator/adapter.py:965` | **`RuntimeCapabilities.session_signals` (Ouroboros Synapse) has no row anywhere in the matrix** — 0 occurrences of "Synapse" or "session_signal". Four runtimes declare full support; a user on Codex/Hermes/Gemini/Kiro gets silently **no signal delivery** with no table to consult, while `skills/auto/SKILL.md:186` routes work through it. | MED |
| D29 | `orchestrator/adapter.py:958-962` | `subagent_orchestration` has no matrix row either; OpenCode's `EXTERNAL_HOST_BRIDGE` is only covered in a separate guide. | MED |
| D30 | `docs/auto-runtime-semantics.md:26` | Links `../src/ouroboros/auto/operational_task.py` — **file does not exist**, and no `OperationalTask` symbol exists anywhere. It is cited as the authoritative location of path selection. | MED |
| D31 | `docs/rfc/verify-gate-evidential-force.md:101` | Cites `src/ouroboros/core/verify_command_plan.py:338` with a line number — **file does not exist**, so the RFC's evidence chain is unverifiable. | MED |
| D32 | `docs/contributing/findings-registry.md:814` | References `src/ouroboros/orchestrator/evolution/loop.py`; the real path is `src/ouroboros/evolution/loop.py`. | LOW |
| D33 | `HANDOFF.md:66` | References `src/ouroboros/bigbang/ontology.py` — does not exist. Ontology handling lives in `core/lineage.py` and `events/ontology.py`. | LOW |
| D34 | `docs/history/master-roadmap-2026-07.md` | 5 nonexistent paths, all inside already-executed `git rm` / "Create …" instructions. **Expected** — flagged only so a path-checker whitelists `docs/history/`. | FINE |
| D35 | `README.md` | **All 9 checkable numeric claims confirmed**: 30 generations (`convergence.py:56`), 0.95 similarity (`:53`), drift 50/30/20 and 0.3 (`drift.py:51-53`), tiers 1/10/30, the `0.5·name + 0.3·type + 0.2·exact` formula verbatim at `lineage.py:371`, stagnation window 3, 70% question overlap (`convergence.py:600`), ambiguity 0.2, and the ambiguity weights 40/30/30 & 35/25/25/15. | FINE |
| D36 | `README.md:463` | "LiteLLM adapter (100+ models)" has **no in-repo constant** — an upstream property, unfalsifiable here. Not a defect. | INFO |
| D37 | `docs/` link graph | **0 broken relative links** out of ~200 resolved. | FINE |
| D38 | `docs/README.md` | Omits **17 existing docs**, including `events.md` and **8 English runtime guides** (`goose`, `gemini`, `copilot`, `kiro`, `pi`, `gjc`, `antigravity`, `grok`). Note the asymmetry at `:27-29`: Copilot, Kiro and Goose are linked **only in their Korean translations** — the English originals exist but are unlinked. Compounds D25: Goose is missing from both the matrix and the English index. | MED |

---

## Proposed PR decomposition

| PR | Scope | Items | Risk |
| :--- | :--- | :--- | :--- |
| 1 | Windows hard crashes: CDLL ordering + stale-lease wedge | W1, W2, W34 | low |
| 2 | Windows encoding: git subprocess + asymmetric `read_text` | X1–X4, X10, X11 | low |
| 3 | Windows digest determinism + case-insensitive containment | X16, X17, X28, X29, X31 | med |
| 4 | Evaluation silent passes | P24, P25, P28, P29, P30 | med |
| 5 | Checkpoint corruption detection + stdout leak | P18, P19 | low |
| 6 | Persistence durability + schema version | P1, P2, P7, P8 | high |
| 7 | Replay determinism (`rowid` ordering) | P13, P14 | high |
| 8 | Consensus quorum honesty | P35, P37, P33 | med |
| 9 | `setup.py` atomic config writes + unchecked failures | M21, M22, M23 | med |
| 10 | MCP schema hardening (`additionalProperties`, `_force_in_process`) | M17, M18 | low |
| 11 | Job reservation TTL race | M1, M5, M8 | high |
| 12 | Plugin supply chain: reject `http://`, add `--ref` | M27, M28 | med |
| 13 | CLI top-level error handling | M32, M33 | low |
| 14 | Windows process detachment | W5, W6, W7, W8, W10 | med |
| 15 | Config reference: document 24 missing fields | D5–D17 | low |
| 16 | `docs/events.md`: document 30 missing event types | D21, D23 | low |
| 17 | Runtime matrix: add Goose + Synapse rows | D25–D29 | low |
| 18 | Fix stale doc code references | D30–D33, D38 | low |
| 19 | Windows CI job + `sys.platform` guards + narrow mypy | W-root-cause | high |
