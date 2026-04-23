# CLI reference

ctxrot ships a [Typer](https://typer.tiangolo.com/)-based CLI. Every command reads from (or writes to) a SQLite database created by `CtxRotCallback` — default `ctxrot.db` in the current directory.

!!! tip
    All commands accept `--db, -d` to point at a different database. Commands that read a single session default to the latest one unless you pass `--session, -s`.

## `ctxrot` / `ctxrot dashboard`

Launch the Textual TUI dashboard. Both forms are equivalent.

```bash
ctxrot --db ctxrot.db
ctxrot dashboard --db ctxrot.db --session 7a3f9e2c1d0b
```

| Flag | Short | Default | Meaning |
|------|-------|---------|---------|
| `--db` | `-d` | `ctxrot.db` | Path to the SQLite database |
| `--session` | `-s` | *(latest)* | Open the dashboard directly on this session |

Press `q` to quit.

## `ctxrot analyze`

Compute repetition + efficiency metrics on a single session using only local signals — no LLM calls. Reads the database read-only.

```bash
ctxrot analyze --db ctxrot.db --session 7a3f9e2c1d0b
ctxrot analyze --json > analysis.json
```

| Flag | Short | Default | Meaning |
|------|-------|---------|---------|
| `--db` | `-d` | `ctxrot.db` | Database path |
| `--session` | `-s` | *(latest)* | Session ID to analyze |
| `--json` | — | `false` | Output the full result dict as JSON |

Human output prints per-iteration `ngram_jaccard / sequence_similarity / cumulative_max` scores, flags the onset iteration if any exceeds `0.4`, and lists per-iteration efficiency ratios. The summary (including `initial_efficiency` / `final_efficiency`) is available via `--json`. See [Concepts](concepts.md) for what the numbers mean.

## `ctxrot export`

Emit one session per line as JSONL in the [opentraces](https://www.opentraces.ai/schema/latest) `TraceRecord` v0.3.0 shape (default) or a native ctxrot format.

```bash
ctxrot export --db ctxrot.db --all -o all.jsonl
```

See the dedicated [Export](export.md) page for the full reference — filter flags, output formats, and the privacy note.

## `ctxrot deep-analyze`

RLM-powered semantic analysis. Produces a structured markdown report with sections for session overview, context growth, efficiency trends, repetition analysis, tool impact, rot diagnosis, and recommendations.

```bash
ctxrot deep-analyze --db ctxrot.db --session 7a3f9e2c1d0b
```

!!! warning "Requires Deno + API key"
    `deep-analyze` uses `dspy.RLM`, which runs a sandboxed Python interpreter via [Deno](https://deno.land). Install Deno first, and provide an API key via `--api-key`, `OPENAI_API_KEY`, or a `.env` file.

See [Deep analysis](deep-analysis.md) for the full flag list, cost estimates, and credential resolution order.

## `ctxrot reset`

Truncate all tables — sessions, LM calls, tool calls — in the database. Destructive, no confirmation prompt.

```bash
ctxrot reset --db ctxrot.db
```

## Commands marked *coming soon*

`ctxrot tail` (stream LM calls in real-time) and `ctxrot summary` (one-shot session stats) currently print `Coming soon` and exit. They're tracked as future work.
