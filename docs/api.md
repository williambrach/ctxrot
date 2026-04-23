# Python API

The four objects re-exported from the top-level `ctxrot` module.

```python
from ctxrot import (
    CtxRotCallback,     # attach to dspy.configure(callbacks=[...])
    CtxRotStore,        # read/write the SQLite database directly
    analyze_session,    # compute local rot metrics
    run_deep_analysis,  # RLM-powered semantic analysis
)
```

---

## CtxRotCallback

The DSPy `BaseCallback` you attach to your `dspy.configure(...)` call. Pass `store_content=True` if you want the full prompt/completion text captured — it's required for repetition analysis.

::: ctxrot.CtxRotCallback
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - session_id

---

## CtxRotStore

A thin SQLite wrapper. `CtxRotCallback` uses it internally, but you can also instantiate it yourself (with `read_only=True` for safety) when you want to query sessions programmatically without going through the CLI.

::: ctxrot.CtxRotStore
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - get_latest_session_id
        - get_session_ids
        - get_session
        - get_session_summary
        - get_growth_data
        - get_tool_impact
        - get_lm_call_content
        - get_tool_call_content
        - truncate_all
        - close

---

## analyze_session

Compute local repetition + efficiency metrics for one session. Returns a plain dict that's safe to `json.dumps`.

::: ctxrot.analyze_session
    options:
      show_root_heading: false
      heading_level: 3

---

## run_deep_analysis

Kick off an RLM-powered deep analysis. Requires [Deno](https://deno.land) and an API key. See [Deep analysis](deep-analysis.md) for the full workflow and the CLI wrapper.

::: ctxrot.run_deep_analysis
    options:
      show_root_heading: false
      heading_level: 3
