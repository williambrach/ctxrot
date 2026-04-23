# Deep analysis

`ctxrot deep-analyze` uses DSPy's [`RLM`](https://dspy.ai) (Reasoning Language Model) to perform **semantic** analysis on a recorded session — the kind of reasoning that string metrics can't do on their own.

The RLM receives session metadata, growth curves, and pre-computed rot metrics up front, and can pull full prompt/completion text on demand via tools. The output is a structured markdown report: session overview, context growth pattern, efficiency trends, repetition analysis, tool impact, rot diagnosis (severity + onset iteration), and recommendations.

!!! warning "Deno required"
    `deep-analyze` runs a sandboxed Python interpreter via [Deno](https://deno.land).
    Install with `curl -fsSL https://deno.land/install.sh | sh` or see [the Deno install guide](https://deno.land/#installation).

!!! warning "Work in progress"
    Deep analysis is still early and may produce misaligned output. Prompts, tool surface, and report structure are subject to change.

## Quickstart

```bash
ctxrot deep-analyze --db ctxrot.db --session 7a3f9e2c1d0b
```

If `--session` is omitted, the latest session is used. Credentials are resolved in this order:

1. Explicit `--api-key` / `--api-base` flags
2. `OPENAI_API_KEY` / `OPENAI_API_BASE` environment variables
3. `API_KEY` / `API_BASE` environment variables
4. Variables loaded from `--env-file` (default `.env`)

## How it works

```
                  ┌───────────────────┐
 session data ──► │  RLM (main LM)    │ ──► markdown report
 growth curves    │  REPL, sandboxed  │     (7 sections)
 pre-computed     │  via Deno         │
 rot metrics      └─────────┬─────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │ tools the RLM can call:       │
            │  compute_repetition_score     │
            │  compute_all_repetition_scores│
            │  get_completion_text(seq)     │
            │  get_messages_json(seq)       │
            │  get_tool_output(id)          │
            └───────────────────────────────┘
```

The RLM writes small Python snippets that run inside the sandbox. Those snippets inspect the session, call the ctxrot-provided tools, and occasionally query a cheaper **sub-LM** (`--sub-model`) for semantic questions — e.g. "is this repetition structural DSPy format vs substantive looping?". The budget on sub-LM calls keeps costs bounded.

## Flags

```text
Usage: ctxrot deep-analyze [OPTIONS]

Options:
  --db         -d  TEXT     SQLite database path              [default: ctxrot.db]
  --session    -s  TEXT     Session ID (latest if omitted)
  --query      -q  TEXT     Focus area or question            [default: "Perform a comprehensive context rot analysis."]
  --model      -m  TEXT     Main LM for RLM reasoning         [default: openai/gpt-5.4]
  --sub-model      TEXT     Sub LM for semantic analysis      [default: openai/gpt-5.4-mini]
  --max-iters      INT      Max RLM REPL iterations           [default: 15]
  --max-calls      INT      Max sub-LLM calls                 [default: 30]
  --api-key        TEXT     API key (or OPENAI_API_KEY / API_KEY in .env)
  --api-base       TEXT     API base URL (or OPENAI_API_BASE / API_BASE in .env)
  --env-file       TEXT     Path to .env file                 [default: .env]
  --json                    Output full result as JSON
  --verbose    -v           Show RLM reasoning steps
  --yes        -y           Skip cost warning confirmation
```

## Cost

Running cost depends on session size and how many sub-LLM calls the RLM actually makes. For a typical 10–20 iteration ReAct session with `gpt-5.4` as the main model and `gpt-5.4-mini` as the sub-model, expect **~$0.10 – $2.00 per run**. `deep-analyze` prints a cost estimate and asks for confirmation unless you pass `--yes`.

## Programmatic use

`deep-analyze` is a thin wrapper around [`run_deep_analysis`](api.md#run_deep_analysis). Call it directly from Python if you want the report + trajectory returned as a dict:

```python
from ctxrot import CtxRotStore, run_deep_analysis

store = CtxRotStore("ctxrot.db", read_only=True)
result = run_deep_analysis(
    store,
    session_id="7a3f9e2c1d0b",
    query="Focus on why the prompt tokens plateau after iteration 8.",
)

print(result["report"])
print(f"RLM used {len(result['trajectory'])} REPL iterations")
```
