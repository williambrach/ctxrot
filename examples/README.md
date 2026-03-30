# ctxrot examples

## Prerequisites

- `uv add ctxrot python-dotenv`
- Copy `.env.example` to `.env` and fill in your credentials
- For `rlm_reasoning.py`: install [Deno](https://deno.land/#installation)

## Configuration

All examples load from a `.env` file in the project root:

```env
MODEL=openai/gpt-4.1-mini   # any litellm model string
API_KEY=sk-...              # optional if already in OPENAI_API_KEY
API_BASE=https://...        # optional, for custom endpoints
```

## Examples

| File | DSPy module | DB file | What it shows |
|------|-------------|---------|---------------|
| `cot_simple.py` | `ChainOfThought` | `ctxrot-cot.db` | Hello world — single call tracking |
| `react_multihop.py` | `ReAct` | `ctxrot-react.db` | Multi-hop QA with search — context growth across iterations |
| `custom_module.py` | Custom `Module` | `ctxrot-custom.db` | Pipeline of 3 LM calls with growing prompts |
| `rlm_reasoning.py` | `RLM` | `ctxrot-rlm.db` | REPL reasoning loop — most dramatic context growth |

## Run

```bash
# Pick any example
python examples/react_multihop.py

# View the dashboard
ctxrot --db ctxrot.db
