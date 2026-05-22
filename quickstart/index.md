# Quickstart

A minimal walk-through: attach the callback, run your DSPy agent, open the dashboard.

## 1. Install

```bash
uv add ctxrot
```

ctxrot requires Python 3.12+ and [DSPy ≥ 3.1.3](https://dspy.ai).

## 2. Attach the callback

```python
import dspy
from ctxrot import CtxRotCallback

callback = CtxRotCallback(db_path="ctxrot.db", store_content=True)

dspy.configure(
    lm=dspy.LM("openai/gpt-5.4-mini"),
    callbacks=[callback],
)
```

- A new session is created automatically each time a top-level DSPy module starts. Every LM call and tool call is recorded to SQLite.
- Set `store_content=True` to also store full prompt messages and completion text — required for repetition detection.

## 3. Run your agent as usual

```python
react = dspy.ReAct("question -> answer", tools=[tool_a, tool_b])
result = react(question="What is the capital of France?")
```

No changes to your agent code — the callback just listens.

## 4. View the dashboard

```bash
ctxrot --db ctxrot.db
```

The TUI opens on the **Feed** — a list of sessions with LM-call and tool-call feeds.

## 5. Run a local analysis

Without leaving the terminal, you can compute repetition and efficiency metrics for the latest session:

```bash
ctxrot analyze --db ctxrot.db
```

See [Concepts](concepts.md) for what the numbers mean, and the [CLI reference](cli.md) for every command and flag.

## Next

- **Understand the metrics** → [Concepts](concepts.md)
- **Dig into a session with an LLM** → [Deep analysis](deep-analysis.md)
- **Share a session with a teammate** → [Export](export.md)
