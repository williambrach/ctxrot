
# Runnable examples

Starter scripts live in [`examples/`](https://github.com/williambrach/ctxrot/tree/main/examples) in the repo. Each one writes to its own SQLite file so you can keep them side-by-side and compare context growth patterns.

### Setup

```bash
uv add ctxrot python-dotenv
cp .env.example .env  # then fill in your credentials
```

For `rlm_reasoning.py`: also install [Deno](https://deno.land/#installation).

All examples load from a `.env` file in the project root:

```env
MODEL=openai/gpt-4.1-mini   # any litellm model string
API_KEY=sk-...              # optional if OPENAI_API_KEY is already set
API_BASE=https://...        # optional, for custom endpoints
```

### The examples

| File | DSPy module | DB file | What it shows |
|------|-------------|---------|---------------|
| [`cot_simple.py`](https://github.com/williambrach/ctxrot/blob/main/examples/cot_simple.py) | `ChainOfThought` | `ctxrot-cot.db` | Hello world — single-call tracking |
| [`react_multihop.py`](https://github.com/williambrach/ctxrot/blob/main/examples/react_multihop.py) | `ReAct` | `ctxrot-react.db` | Multi-hop QA with Wikipedia search — context growth across iterations |
| [`custom_module.py`](https://github.com/williambrach/ctxrot/blob/main/examples/custom_module.py) | custom `Module` | `ctxrot-custom.db` | Pipeline of 3 LM calls with growing prompts |
| [`rlm_reasoning.py`](https://github.com/williambrach/ctxrot/blob/main/examples/rlm_reasoning.py) | `RLM` | `ctxrot-rlm.db` | REPL reasoning loop — most dramatic context growth |
| [`optimizer_bootstrap.py`](https://github.com/williambrach/ctxrot/blob/main/examples/optimizer_bootstrap.py) | `BootstrapFewShot` + `ChainOfThought` | `ctxrot-optimizer.db` | Compile-time LM-call volume — one session per trainset example |

### Run one

```bash
# Pick any example
python examples/react_multihop.py

# Then view the dashboard
ctxrot --db ctxrot-react.db
```

Once you have a session to look at, try:

- **Local rot metrics** — `ctxrot analyze --db ctxrot-react.db`
- **Export to JSONL** — `ctxrot export --db ctxrot-react.db --all -o react.jsonl`
- **Deep analysis** — `ctxrot deep-analyze --db ctxrot-rlm.db` (the RLM example tends to surface the most interesting rot patterns)

## Next

- **Understand the metrics** → [Concepts](concepts.md)
- **Dig into a session with an LLM** → [Deep analysis](deep-analysis.md)
- **Share a session with a teammate** → [Export](export.md)
