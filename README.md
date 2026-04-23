# ctxrot

*Understand your ReAct agent's context window and fight context rot.*


> **Note:** ctxrot currently supports only [DSPy>=3.1.3](https://dspy.ai) and may produce mis-aligned output. Please [report any issues](https://github.com/williambrach/ctxrot/issues) you encounter — the API may change.

## Install

```bash
uv add ctxrot
```

## Quick start

```python
import dspy
from ctxrot import CtxRotCallback

callback = CtxRotCallback(db_path="ctxrot.db", store_content=True)

dspy.configure(
    lm=dspy.LM("openai/gpt-5.4-mini"),
    callbacks=[callback],
)

react = dspy.ReAct("question -> answer", tools=[tool_a, tool_b])
result = react(question="What is the capital of France?")
```

Then open the TUI dashboard:

```bash
ctxrot --db ctxrot.db
```

## Learn more

[Documentation](https://williambrach.github.io/ctxrot/)

The site also publishes an [`llms.txt`](https://williambrach.github.io/ctxrot/llms.txt) / [`llms-full.txt`](https://williambrach.github.io/ctxrot/llms-full.txt) for LLM agents that want to ingest the docs directly.

## License

MIT
