# Concepts

## What context rot is

As context grows, LLM agents start repeating themselves and producing less useful output. The model is still generating tokens — it just isn't *saying* anything new. ctxrot makes this visible with two families of signals that are cheap to compute and require no LLM calls of their own:

1. **Repetition** — how much each new completion overlaps with earlier ones
2. **Efficiency** — how much the model outputs relative to the input it receives

## How the callback works

A SQLite database is created at `db_path`. [`CtxRotCallback`](api.md#ctxrotcallback) hooks into DSPy's `BaseCallback` and populates three tables at runtime — a session row on `on_module_start`, an LM call row on `on_lm_end`, and a tool call row on `on_tool_end`.

```
Your DSPy agent  →  CtxRotCallback  →  SQLite  →  TUI dashboard / analysis
   (unchanged)       (just listens)     (local)
```

Sessions close automatically when the top-level DSPy module returns. The terminal state is recorded as `errored` if the module raised and `completed` otherwise; both `analyze` and `deep-analyze` surface it.

Session state lives in a `ContextVar`, so `asyncify`/`streamify` worker threads each see an isolated session — concurrent agent calls don't stomp on each other.

### What gets tracked

| Per LM call | Per tool call | Per session |
|---|---|---|
| Prompt tokens, completion tokens | Tool name, duration | Model, mode (`react`, `chainofthought`, …) |
| Cache read / write tokens | Estimated output tokens | Start time, end time |
| Cost, duration | — | Terminal state (`completed` / `errored`) |
| *(opt)* full prompt messages + completion text | *(opt)* full input JSON + output text | — |

The "opt" rows only populate if you passed `store_content=True` when constructing the callback.

## Context rot detection

Local signals only. No LLM calls. Token counting uses [tokie](https://github.com/chonkie-inc/tokie).

!!! warning "Requires content capture"
    Repetition analysis needs `store_content=True` when you construct `CtxRotCallback`. DSPy structural markers (`[[ ## ... ## ]]`) are stripped before comparison so they don't inflate overlap scores.

### Repetition — per-iteration

| Metric | What it measures | How |
|--------|-----------------|-----|
| `ngram_jaccard` | Word-level overlap vs previous completion | Jaccard similarity of word 3-gram sets. `> 0.4` = looping. |
| `sequence_similarity` | Character-level similarity vs previous completion | `rapidfuzz.fuzz.ratio / 100`. Catches paraphrased repetition that n-grams miss. |
| `cumulative_max` | Max overlap vs *any* prior completion | Max `ngram_jaccard` across every earlier iteration. Catches non-consecutive loops. |

`analyze` flags the **onset iteration** as the first iteration whose `ngram_jaccard` exceeds `0.4`.

### Efficiency — per-iteration

A declining ratio across iterations means the model generates less output relative to its input — a sign the context window is saturated.

```python
efficiency_ratio = completion_tokens / prompt_tokens
```

`analyze` also reports the initial and final efficiency so you can see drift at a glance.

## What the metrics are *not*

- **Not a hallucination detector.** A high `ngram_jaccard` means the agent is repeating itself, not that the repeated content is wrong.
- **Not a universal cost budget.** `efficiency_ratio` is a *shape* metric; declining ratios can be normal for certain prompts (e.g., classification) and still be healthy.
- **Not a replacement for manual review.** They're triage signals — [`deep-analyze`](deep-analysis.md) uses them as one input among several when producing its report.
