# Export

`ctxrot export` emits one session per line as JSONL. The default format matches the [opentraces](https://www.opentraces.ai/schema/latest) `TraceRecord` schema v0.3.0, so ctxrot sessions can be shared, archived, or handed off to opentraces for publishing to the Hugging Face Hub without re-mapping.

## Privacy

!!! warning "Content is exported raw"
    `export` emits **raw LM messages, completions, and tool I/O** whenever they were captured at run time (i.e. `CtxRotCallback(store_content=True)`). ctxrot prints a warning once at the start of every export; reviewing the output for secrets and PII before sharing it is your responsibility.

If content was *not* captured, those fields pass through as `null` — ctxrot does not retroactively reconstruct them. Redaction is on the roadmap but not yet available.

## Selecting sessions

Filters compose with **AND**; explicit `--session` bypasses everything else.

| Flags | What you get |
|-------|--------------|
| *(none)* | Latest session (same default as `analyze` / `deep-analyze`) |
| `--all` | Every session in the DB |
| `-s ID` (repeatable) | Explicit session IDs |
| `--since DT` / `--until DT` | ISO-datetime range on session start time |
| `--only-errored` / `--only-completed` | Terminal-state filter (mutually exclusive) |

## Flags

```text
Usage: ctxrot export [OPTIONS]

Options:
  --db              -d   TEXT     SQLite database path            [default: ctxrot.db]
  --session         -s   TEXT     Session ID (repeatable for multiple IDs)
  --all                           Export every session in the DB
  --since                TEXT     Sessions started at/after this ISO datetime
  --until                TEXT     Sessions started at/before this ISO datetime
  --only-errored                  Only sessions with terminal_state='errored'
  --only-completed                Only sessions with terminal_state='completed'
  --format          -f   TEXT     "opentraces" or "ctxrot"         [default: opentraces]
  --output          -o   TEXT     Output file path (stdout if omitted)
```

## Examples

```bash
# Latest session to a file
ctxrot export -o latest.jsonl

# Everything in the DB
ctxrot export --all -o all.jsonl

# A few specific sessions
ctxrot export -s 7a3f9e2c1d0b -s 9b1c2d3e4f5a -o picked.jsonl

# All failed sessions on or after April 6, 2026
ctxrot export --since 2026-04-06 --only-errored -o failures.jsonl

# Native ctxrot format (debug / roundtrip)
ctxrot export --all --format ctxrot -o all-native.jsonl
```

## Record shape (opentraces v0.3.0)

One JSONL line per session. Abridged example:

```json
{
  "schema_version": "0.3.0",
  "trace_id": "7a3f9e2c1d0b",
  "session_id": "7a3f9e2c1d0b",
  "timestamp_start": "2026-04-22T14:37:26.875+00:00",
  "timestamp_end":   "2026-04-22T14:37:29.375+00:00",
  "agent":    { "name": "rlm", "model": "openai/gpt-4o-mini" },
  "outcome":  { "success": true, "terminal_state": "goal_reached" },
  "lifecycle": "final",
  "metrics": {
    "total_steps": 2,
    "total_input_tokens": 203,
    "total_output_tokens": 65,
    "total_cache_read_tokens": 60,
    "total_cache_creation_tokens": 10,
    "total_duration_s": 1.7,
    "cache_hit_rate": 0.2956,
    "estimated_cost_usd": 0.003
  },
  "steps": [
    {
      "step_index": 1,
      "role": "assistant",
      "model": "openai/gpt-4o-mini",
      "content": "thinking...",
      "timestamp": "2026-04-22T14:37:26.875+00:00",
      "call_type": "action",
      "token_usage": { "input_tokens": 123, "output_tokens": 45,
                       "cache_read_tokens": 20, "cache_write_tokens": 10 },
      "tool_calls":    [ { "tool_call_id": "t1", "tool_name": "web_search",
                           "input": { "q": "..." }, "duration_ms": 279 } ],
      "observations":  [ { "source_call_id": "t1", "content": "results: ..." } ]
    }
  ],
  "metadata": { "ctxrot_version": "0.1.0", "source": "dspy-callback",
                "framework": "dspy", "mode": "rlm", "max_prompt_tokens": 123 }
}
```

A few things worth knowing:

- **`agent.name`** is the DSPy top-level module class name (lower-cased) — e.g. `"rlm"`, `"react"`, `"chainofthought"` — falling back to `"dspy-agent"` if the mode wasn't captured.
- **Tool I/O is split** across `steps[].tool_calls[]` (invocation: `tool_call_id`, `tool_name`, `input`, `duration_ms`) and `steps[].observations[]` (result: `source_call_id`, `content`, `error`), keyed together by the tool call id.
- **RLM reasoning tree.** For `rlm` sessions, each step carries `call_type` (`"action"` or `"sub_query"`). `sub_query` steps additionally carry `parent_step`, pointing to the `step_index` of the `action` that triggered them — this is how to reconstruct the reasoning tree from an exported record. Non-RLM sessions omit both fields.
- **`outcome.terminal_state`** uses the schema's enum, not ctxrot's internal labels. The mapping:

    | ctxrot (SQLite `sessions.terminal_state`) | Exported `outcome` |
    |---|---|
    | `"completed"` | `{ "success": true,  "terminal_state": "goal_reached" }` |
    | `"errored"`   | `{ "success": false, "terminal_state": "error" }` |
    | *null* (session never finished) | `outcome` omitted; `lifecycle: "provisional"` |

- **`metrics.cache_hit_rate`** is a fraction in `[0, 1]` (per the schema's rate convention), not a percentage.
- **`metrics.total_duration_s`** is in seconds (the SQLite layer stores ms; the mapper converts).
- **Dropped from the v0.3.0 export but kept in the native format:** per-step `cost`, `error`, `duration_ms`, raw `messages`, and RLM `iteration` — none of these have a home in `TraceRecord.Step`. If you need them, use `--format ctxrot`.

