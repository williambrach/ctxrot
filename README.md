<div align="center">

![ctxrot](assets/logo.png)

# ctxrot


*Understand your ReAct agent's context window and fight context rot.*


</div>

> **Note:** ctxrot currently supports only [DSPy>=3.1.3](https://dspy.ai) and may produce mis-aligned output. Please [report any issues](https://github.com/williambrach/ctxrot/issues) you encounter — the API may change.

## Install

```bash
uv add ctxrot
```

## Quick start

### 1. Attach the callback

```python
import dspy
from ctxrot import CtxRotCallback

callback = CtxRotCallback(db_path="ctxrot.db", store_content=True)

dspy.configure(
    lm=dspy.LM("openai/gpt-5.4-mini"),
    callbacks=[callback],
)
```

- A new session is created automatically each time a DSPy module starts. Every LM call and tool call is recorded to SQLite.
- Set `store_content=True` to also store full prompts and completions (needed for repetition analysis).

### 2. Run your agent as usual

```python
react = dspy.ReAct(MySignature, tools=[tool_a, tool_b])
result = react(question="What is the capital of France?")
```

### 3. View the dashboard

```bash
ctxrot --db ctxrot.db
```

### More examples

In [examples](examples/), see:
- [`cot_simple.py`](examples/cot_simple.py) — simplest "hello world" with ChainOfThought
- [`react_multihop.py`](examples/react_multihop.py) — ReAct multi-hop QA with Wikipedia search
- [`custom_module.py`](examples/custom_module.py) — custom DSPy module pipeline
- [`rlm_reasoning.py`](examples/rlm_reasoning.py) — RLM iterative reasoning loop (requires Deno)

## CLI commands

```bash
# Launch TUI dashboard
ctxrot --db ctxrot.db

# Reset database
ctxrot reset --db ctxrot.db

# WIP : using RLM for ctx analysis (requires API key / API base)
ctxrot deep-analyze --db ctxrot.db --session <session_id>

```

## How it works

A SQLite database is created at `db_path`. The callback hooks into DSPy's `BaseCallback` and populates tables at runtime — a session row on `on_module_start`, an LM call row on `on_lm_end`, and a tool call row on `on_tool_end`.

```
Your DSPy agent  →  CtxRotCallback  →  SQLite  →  TUI dashboard / analysis
   (unchanged)       (just listens)     (local)
```

### What it tracks

- **Per LM call** — prompt tokens, completion tokens, cache read/write tokens, cost, duration
- **Per tool call** — tool name, duration, estimated output tokens
- **Per session** — model, start time, mode (react, chainofthought, etc.)
- **Optionally** — full prompt messages and completion text (`store_content=True`)

### Context rot detection

As context grows, agents start repeating themselves and producing less useful output. ctxrot detects this with two local signals (no LLM calls, token counting via [tokie](https://github.com/chonkie-inc/tokie)):

> requires `store_content=True`
 
>DSPy structural markers (`[[ ## ... ## ]]`) are stripped before comparison.

**Repetition** (per-iteration) :

| Metric | What it measures | How |
|--------|-----------------|-----|
| `ngram_jaccard` | Word-level overlap vs previous completion | Jaccard similarity of word 3-gram sets. `>0.4` = looping. |
| `sequence_similarity` | Character-level similarity vs previous completion | `rapidfuzz.fuzz.ratio / 100`. Catches paraphrased repetition. |
| `cumulative_max` | Max overlap vs *any* prior completion | Max `ngram_jaccard` across all earlier iterations. Catches non-consecutive loops. |



**Efficiency** (per-iteration): 

A declining ratio across iterations means the model generates less output relative to its input a sign the context window is saturated.
> `efficiency_ratio = completion_tokens / prompt_tokens`

**LLM analysis**

Uses RLM to perform semantic analysis on a recorded session. The RLM receives session metadata, growth curves, and pre-computed rot metrics, and can pull full prompt/completion text on demand via tools (`get_completion_text`, `get_messages_json`, `get_tool_output`, `compute_repetition_score`). Output is a structured markdown report: session overview, context growth pattern, efficiency trends, repetition analysis, tool impact, rot diagnosis (severity + onset iteration), and recommendations.

> Deno required

```bash
Usage: ctxrot deep-analyze [OPTIONS]

Options:
  --db         -d  TEXT     SQLite database path              [default: ctxrot.db]
  --session    -s  TEXT     Session ID (latest if omitted)
  --query      -q  TEXT     Focus area or question            [default: comprehensive analysis]
  --model      -m  TEXT     Main LM for RLM reasoning         [default: openai/gpt-5.4]
  --sub-model      TEXT     Sub LM for semantic analysis      [default: openai/gpt-5.4-mini]
  --max-iters      INT     Max RLM REPL iterations            [default: 15]
  --max-calls      INT     Max sub-LLM calls                  [default: 30]
  --api-key        TEXT     API key (or OPENAI_API_KEY env)
  --api-base       TEXT     API base URL (or OPENAI_API_BASE env)
  --env-file       TEXT     Path to .env file                  [default: .env]
  --json                    Output full result as JSON
  --verbose    -v           Show RLM reasoning steps
  --yes        -y           Skip cost warning confirmation
```