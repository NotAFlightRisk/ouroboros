# The Development Loop

You cloned this repo to change something. This page is about the shortest path
from an edit to seeing that edit actually run.

Read it before the architecture docs. Knowing where a module lives does not
help if the code you are running is not the code you edited — which, by
default in this repository, it is not.

## The trap: by default you are not running your own code

The checked-in `.mcp.json` points an MCP client at the **published PyPI
package**, not at your working tree:

```json
{ "command": "uvx",
  "args": ["--isolated", "--python", ">=3.12", "--from", "ouroboros-ai[mcp]",
           "ouroboros", "mcp", "serve", "..."] }
```

So if you clone the repo, edit a handler, and open your agent client in the
project directory, the server that answers is the last release — your change
has no effect and nothing warns you. Same for `uvx ouroboros ...` on the
command line.

Point the tooling at your working tree instead. There are two surfaces, and
which one you need depends on what you changed.

### Surface 1 — the CLI

The package installs three console scripts (`pyproject.toml:88`):

| Script | Entry point |
|---|---|
| `ooo` | `ouroboros.cli.main:app` |
| `ouroboros` | `ouroboros.cli.main:app` |
| `ozo` | `ouroboros.cli.commands.zcode:app` |

Inside the repo, `uv run` already resolves to your working tree — no install
step, no staleness:

```bash
uv run ouroboros --version
uv run ooo status
```

To make your working tree the `ooo` on your `PATH` everywhere (useful when a
client spawns the binary for you):

```bash
uv tool install --force --editable . --python '>=3.12'
```

The `--python '>=3.12'` matters. `uvx`/`uv tool` otherwise resolve against the
machine's default interpreter, and on a 3.11 box the MCP server dies before it
can answer `initialize`. Any launcher you generate must carry the same floor.

### Surface 2 — the MCP server

Most of this project's behavior reaches a user through MCP, so this is the
surface you will usually need. Run the server straight from your working tree:

```bash
uv run --directory /path/to/your/clone ouroboros mcp serve
```

(`serve` is defined at `src/ouroboros/cli/commands/mcp.py:1115`.)

To make a client use it, replace the `ouroboros` entry in your client's MCP
config — `~/.claude/mcp.json` for Claude Code, or the project `.mcp.json` —
with the local form, and **preserve every other server entry in the file**:

```json
"ouroboros": {
  "command": "uv",
  "args": ["run", "--directory", "/path/to/your/clone", "ouroboros", "mcp", "serve"],
  "timeout": 600
}
```

Keep connection settings and runtime selection separate: the MCP config
carries `command` / `args` / `timeout` only. Which model or runtime is used is
config, not connection — see below. Back up the file before you edit it, and
restore it when you are done testing so you do not silently keep running a
stale branch weeks later.

**Restart the client after changing MCP config.** Nothing hot-reloads.

> If the server fails to start, suspect a *different* server first. One broken
> entry in the client's MCP config can take the whole startup down. Run the
> serve command by hand and read the first ~40 lines of output.

## Runtime selection lives in config, not in the connection

Config file: `~/.ouroboros/config.yaml`.

```yaml
llm:
  backend: claude_code      # claude_code | codex | litellm | opencode
orchestrator:
  runtime_backend: claude   # which agent runtime executes work
```

Resolution order (`src/ouroboros/config/loader.py:1742`):

1. environment variable `OUROBOROS_LLM_BACKEND` — a per-shell override
2. `~/.ouroboros/config.yaml`
3. the built-in default (`claude_code`)

A CLI flag on `mcp serve` can be overridden by config in some paths, so if a
runtime selection appears to be ignored, check `config.yaml` before assuming
the flag is broken.

## Where state and output go

| What | Where |
|---|---|
| Config | `~/.ouroboros/config.yaml` |
| Event database | `~/.ouroboros/ouroboros.db` (`persistence/event_store.py:552`) |
| Logs | `~/.ouroboros/logs/ouroboros.log` (`observability/logging.py:163`) |
| Worktrees created by runs | `~/.ouroboros/worktrees/` |

This state accumulates and it is not small — the event DB and its WAL grow
across runs, and abandoned run worktrees are the usual cause of a full disk.
Clean up with the built-in command, which checks locks and dirty trees:

```bash
uv run ouroboros cleanup --dry-run   # report only
uv run ouroboros cleanup --force
```

Never `rm -rf ~/.ouroboros/worktrees` by hand — a live run may hold one.

## Fastest verification per change type

| You changed | Minimum to see it work | Client restart? |
|---|---|---|
| Pure Python (no MCP surface) | `uv run pytest tests/unit/<area>` | no |
| CLI command or flag | `uv run ooo <command>` | no |
| MCP tool handler | point the client at local source, then call the tool | **yes** |
| `SKILL.md` / markdown | depends on dev-mode vs installed-plugin resolution | usually yes |

Scope your test runs while iterating:

```bash
uv run pytest tests/unit/<area> -q
```

The full suite runs in parallel — a large win, and not enabled by default:

```bash
uv run pytest tests/ --ignore=tests/unit/mcp --ignore=tests/integration/mcp \
  --ignore=tests/e2e -n auto --dist worksteal
```

`tests/unit/mcp` is excluded there deliberately. That suite has leaked to real
state — writing to the real event DB and creating real worktrees. CI runs it
in a clean container, which is where it belongs; running it repeatedly against
your own `$HOME` inside a shared checkout is how people lose work. See
[Testing Guide](./testing-guide.md).

## Before you open the PR

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ --fix
uv run mypy src/ouroboros
uv run pytest
```

Then read [Review Conventions](./review-conventions.md) — the reviewer is
strict and predictable, and most rounds are lost to objections you can
preempt. Gate details are in [CI Gates](./ci-gates.md).
