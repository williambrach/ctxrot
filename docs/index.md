---
title: ctxrot
description: Context analytics for DSPy agents — track LM calls, detect context rot, inspect sessions in a TUI.
hide:
  - navigation
  - toc
---

# ctxrot { #hero-title }

!!! note "Alpha"
    ctxrot currently supports only [DSPy>=3.1.3](https://dspy.ai) and may produce mis-aligned output.
    Please [report any issues](https://github.com/williambrach/ctxrot/issues) you encounter — the API may change.

## Install

```bash
uv add ctxrot
```

## What it does

- **Records** every LM call and tool call from your DSPy agent into a local SQLite database via a drop-in `CtxRotCallback`.
- **Detects** repetition and efficiency degradation — the two signals of context rot — without making any LLM calls of its own.
- **Visualizes** sessions in a Textual TUI dashboard with growth curves, per-iteration metrics, and an RLM tree view.
- **Exports** sessions to JSONL in the [opentraces](https://github.com/JayFarei/opentraces) `TraceRecord` shape (or a native format), ready to share or archive.
- **Deep-analyzes** a session with an RLM agent that produces a structured rot report — optional, requires Deno + an API key.

## Next steps

- :material-rocket-launch-outline: &nbsp; **[Quickstart](quickstart.md)** — attach the callback, run your agent, open the dashboard.
- :material-book-open-page-variant-outline: &nbsp; **[Concepts](concepts.md)** — what context rot is and the metrics ctxrot uses to detect it.
- :material-console: &nbsp; **[CLI reference](cli.md)** — `dashboard`, `analyze`, `export`, `deep-analyze`, `reset`.
- :material-api: &nbsp; **[Python API](api.md)** — `CtxRotCallback`, `CtxRotStore`, `analyze_session`, `run_deep_analysis`.
