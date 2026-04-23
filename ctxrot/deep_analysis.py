import json
import shutil
from collections.abc import Callable
from typing import Any

import dspy
from rapidfuzz import fuzz

from ctxrot.analysis import (
    _clean_completion,
    _compute_repetition,
    _jaccard,
    _ngram_set,
    analyze_session,
)
from ctxrot.storage import CtxRotStore

_SIGNATURE_INSTRUCTIONS = """\
You are an expert analyst diagnosing "context rot" — the degradation of LLM \
performance as context windows grow during multi-turn agent sessions.

## Inputs

`session_data` is a Python dict with these top-level keys:
- "session": {{id, started_at, ended_at, model, mode, terminal_state}} — \
session metadata. terminal_state is one of "completed", "errored", or null \
(null = legacy session, outcome unknown).
- "summary": {{total_calls, total_prompt, total_completion, total_cache_read, \
cache_hit_pct, total_cost, total_duration_ms, max_prompt_tokens}} — aggregates
- "growth": list of per-iteration dicts with {{seq, prompt_tokens, \
completion_tokens, cache_read_tokens, cache_write_tokens, cost, model}}
- "tool_impact": list of {{tool_name, call_count, avg_tokens}}
- "tools_by_iteration": dict keyed by seq (string), values are lists of \
{{tool_name, output_tokens_est, duration_ms}}
- "content": list of {{seq, prompt_char_count, completion_char_count}} — \
metadata only (use get_completion_text(seq) and get_messages_json(seq) to \
retrieve full text on demand). EMPTY if content was not stored.
- "tool_content": list of {{tool_call_id, output_char_count}} — metadata \
only (use get_tool_output(tool_call_id) to retrieve full text on demand). \
EMPTY if content was not stored.
- "rot_analysis": pre-computed metrics with {{has_content, repetition, \
efficiency, summary}}. Use rot_analysis["summary"] for a quick overview \
(repetition_detected, onset_iteration, max_repetition, initial_efficiency, \
final_efficiency). rot_analysis["efficiency"] has per-iteration ratios. \
rot_analysis["repetition"] has per-iteration ngram/sequence scores (None \
if no content was stored).

`analysis_query` is the user's specific question or focus area. If it \
requests something specific, prioritize that in your analysis and address \
it directly. Otherwise perform comprehensive analysis.

## Analysis Strategy

Phase 1 — Explore: Programmatically examine session_data. Print summary, \
growth trends, content availability. Compute derived metrics: prompt growth \
deltas, efficiency trend slope, cost per iteration. Scale exploration depth \
to session complexity (more iterations for sessions with 20+ calls, fewer \
for small sessions).

Phase 2 — Analyze: If content is available, use \
compute_all_repetition_scores() for a full overview, then retrieve specific \
completions with get_completion_text(seq) to inspect interesting iterations. \
Use compute_repetition_score(text_a, text_b) for ad-hoc comparisons. \
Reserve llm_query() (budget: {max_llm_calls} calls total) for semantic \
questions that string metrics cannot answer — e.g., classifying whether \
repeated text is structural (DSPy format) vs substantive (actual looping). \
Identify the rot onset: where efficiency drops below 50% of initial or \
repetition exceeds 0.4.

Phase 3 — Draft Report: Compose the markdown report.

Phase 4 — Self-Critique: Before submitting, review your draft report. Check:
  - Does every claim have a supporting number from your analysis?
  - Did you miss any anomalies in the data you explored?
  - Is the severity rating consistent with the evidence?
  - Are the recommendations specific and actionable (not generic)?
If you find gaps, fix them. Then SUBMIT.

## Key Signals to Detect

- Prompt growth pattern: linear (healthy), superlinear (problematic), \
plateau (compression active)
- Efficiency degradation: completion_tokens/prompt_tokens declining over \
iterations
- Completion length collapse: completions getting shorter in later \
iterations (model "giving up")
- Repetition onset: high ngram_jaccard or sequence_similarity between \
consecutive or non-consecutive completions
- Tool context bloat: specific tools injecting disproportionate tokens \
(compare tool_impact avg_tokens with prompt growth deltas)
- Cache behavior: high cache_hit_pct = stable prefix; low = volatile context
- Cost acceleration: cost per iteration increasing faster than linearly
- Terminal state: session["terminal_state"] == "errored" is load-bearing \
evidence for severe rot; factor it into the Rot Diagnosis severity rating.

## Report Structure

Produce a markdown report with these sections:
1. **Session Overview** — model, mode, duration, total calls, total cost, \
total tokens
2. **Context Growth** — growth pattern classification with numbers, \
prompt token deltas per iteration
3. **Efficiency Trends** — efficiency ratio curve, whether degradation is \
detected, marginal returns
4. **Repetition Analysis** (skip if no content) — scores, onset point, \
structural vs substantive repetition
5. **Tool Impact** — which tools contribute most to context growth, \
patterns across iterations
6. **Rot Diagnosis** — severity (none / mild / moderate / severe), onset \
iteration, primary factors
7. **Recommendations** — specific actionable steps (e.g., summarize tool \
outputs, reduce max iterations, compress history, prune system prompts)

If content was not stored, skip section 4 and note the limitation in \
section 6.
"""


class _AnalysisSignature(dspy.Signature):
    """session_data, analysis_query -> analysis_report"""

    session_data: str = dspy.InputField()
    analysis_query: str = dspy.InputField()
    analysis_report: str = dspy.OutputField()


def _build_signature(max_llm_calls: int = 30) -> type[dspy.Signature]:
    instructions = _SIGNATURE_INSTRUCTIONS.format(max_llm_calls=max_llm_calls)
    return _AnalysisSignature.with_instructions(instructions)


def prepare_session_data(
    store: CtxRotStore, session_id: str
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Load all session data for analysis.

    Returns a tuple of:
      - rlm_data: lightweight dict for the RLM (bulk text stripped)
      - full_content: complete LM call content rows (for tool closures)
      - full_tool_content: complete tool call content rows (for tool closures)
    """
    session = store.get_session(session_id)
    if not session:
        msg = f"Session {session_id} not found"
        raise ValueError(msg)

    growth = store.get_growth_data(session_id)
    summary = store.get_session_summary(session_id)
    tool_impact = store.get_tool_impact(session_id)
    tools_by_iter = store.get_tools_by_iteration(session_id)
    full_content = store.get_lm_call_content(session_id)
    full_tool_content = store.get_tool_call_content(session_id)
    rot_analysis = analyze_session(store, session_id)

    # Convert tools_by_iter keys to strings for JSON serialization
    tools_by_iter_str = {str(k): v for k, v in tools_by_iter.items()}

    # Lightweight metadata — bulk text is accessible via tools on demand
    content_meta = [
        {
            "seq": c["seq"],
            "prompt_char_count": c.get("prompt_char_count"),
            "completion_char_count": c.get("completion_char_count"),
        }
        for c in full_content
    ]
    tool_content_meta = [
        {
            "tool_call_id": t["tool_call_id"],
            "output_char_count": t.get("output_char_count"),
        }
        for t in full_tool_content
    ]

    # Order keys so the most actionable info appears in the 500-char
    # preview that the RLM sees before writing exploration code.
    rlm_data = {
        "rot_analysis": rot_analysis,
        "summary": summary,
        "session": session,
        "growth": growth,
        "tool_impact": tool_impact,
        "tools_by_iteration": tools_by_iter_str,
        "content": content_meta,
        "tool_content": tool_content_meta,
    }

    return rlm_data, full_content, full_tool_content


def _make_tools(
    full_content: list[dict], full_tool_content: list[dict]
) -> list[Callable]:
    """Build custom tools for the RLM sandbox."""

    def compute_repetition_score(text_a: str, text_b: str) -> str:
        """Compute repetition metrics between two text completions.

        Returns JSON with ngram_jaccard (0-1) and sequence_similarity (0-1).
        Higher values mean more repetition.
        """
        clean_a = _clean_completion(text_a)
        clean_b = _clean_completion(text_b)
        ngrams_a = _ngram_set(clean_a)
        ngrams_b = _ngram_set(clean_b)
        jaccard_score = _jaccard(ngrams_a, ngrams_b)
        fuzzy_ratio_pct = fuzz.ratio(clean_a, clean_b)
        raw_similarity = fuzzy_ratio_pct / 100.0
        result = {
            "ngram_jaccard": round(jaccard_score, 3),
            "sequence_similarity": round(raw_similarity, 3),
        }
        return json.dumps(result)

    def compute_all_repetition_scores() -> str:
        """Compute repetition scores for ALL consecutive completion pairs at once.

        Returns JSON list of {seq, ngram_jaccard, sequence_similarity,
        cumulative_max}. Much faster than calling compute_repetition_score
        on each pair individually. Returns empty list if no content was stored.
        """
        if not full_content:
            return json.dumps([])
        return json.dumps(_compute_repetition(full_content))

    def get_completion_text(seq: int, max_chars: int = 0) -> str:
        """Get the completion text for a specific iteration number.

        Args:
            seq: The iteration sequence number.
            max_chars: If > 0, truncate result to this many characters.

        Returns empty string if content was not stored.
        """
        for row in full_content:
            if row["seq"] == seq:
                text = row.get("completion") or ""
                if max_chars > 0 and len(text) > max_chars:
                    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"
                return text
        return ""

    def get_messages_json(seq: int, max_chars: int = 0) -> str:
        """Get the prompt messages JSON for a specific iteration.

        Args:
            seq: The iteration sequence number.
            max_chars: If > 0, truncate result to this many characters.

        Returns empty string if content was not stored.
        """
        for row in full_content:
            if row["seq"] == seq:
                text = row.get("messages_json") or ""
                if max_chars > 0 and len(text) > max_chars:
                    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"
                return text
        return ""

    def get_tool_output(tool_call_id: str, max_chars: int = 0) -> str:
        """Get the output text of a specific tool call by its ID.

        Args:
            tool_call_id: The tool call identifier.
            max_chars: If > 0, truncate result to this many characters.

        Returns empty string if content was not stored.
        """
        for row in full_tool_content:
            if row["tool_call_id"] == tool_call_id:
                text = row.get("output_text") or ""
                if max_chars > 0 and len(text) > max_chars:
                    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"
                return text
        return ""

    return [
        compute_repetition_score,
        compute_all_repetition_scores,
        get_completion_text,
        get_messages_json,
        get_tool_output,
    ]


def check_deno_available() -> bool:
    """Check if Deno runtime is available (required by RLM)."""
    return shutil.which("deno") is not None


def _load_env_file(env_path: str = ".env") -> None:
    """Load variables from a .env file into os.environ (if file exists)."""
    import os
    from pathlib import Path

    path = Path(env_path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_REQUIRED_SECTIONS = [
    "Session Overview",
    "Context Growth",
    "Efficiency Trends",
    "Repetition Analysis",
    "Tool Impact",
    "Rot Diagnosis",
    "Recommendations",
]


def _validate_report(report: str, has_content: bool) -> list[str]:
    """Return list of missing required section names."""
    missing = []
    report_lower = report.lower()
    for section in _REQUIRED_SECTIONS:
        if section == "Repetition Analysis" and not has_content:
            continue  # Expected to be skipped when no content
        if section.lower() not in report_lower:
            missing.append(section)
    return missing


def run_deep_analysis(
    store: CtxRotStore,
    session_id: str,
    query: str = "Perform a comprehensive context rot analysis.",
    main_model: str = "openai/gpt-5.4",
    sub_model: str = "openai/gpt-5.4-mini",
    max_iterations: int = 15,
    max_llm_calls: int = 30,
    verbose: bool = False,
    api_key: str | None = None,
    api_base: str | None = None,
    env_file: str | None = ".env",
) -> dict[str, Any]:
    """Run RLM-powered deep analysis on a session.

    API credentials are resolved in order:
      1. Explicit api_key / api_base parameters
      2. Environment variables (OPENAI_API_KEY, OPENAI_API_BASE)
      3. Variables loaded from env_file (default: .env)

    Returns a dict with:
      - "report": markdown analysis report
      - "trajectory": RLM's REPL interaction history
      - "session_id": the analyzed session ID
      - "missing_sections": list of required report sections not found
    """
    if not check_deno_available():
        msg = (
            "Deno runtime not found. "
            "RLM requires Deno for its sandboxed Python interpreter.\n"
            "Install: https://deno.land/#installation\n"
            "(e.g., curl -fsSL https://deno.land/install.sh | sh)"
        )
        raise RuntimeError(msg)

    import os

    # Load .env before creating LM instances
    if env_file:
        _load_env_file(env_file)

    rlm_data, full_content, full_tool_content = prepare_session_data(
        store, session_id
    )

    # Resolve credentials: explicit param > OPENAI_* env > API_* env
    resolved_key = (
        api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    )
    resolved_base = (
        api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("API_BASE")
    )

    lm_kwargs: dict[str, Any] = {}
    if resolved_key:
        lm_kwargs["api_key"] = resolved_key
    if resolved_base:
        lm_kwargs["api_base"] = resolved_base

    main_lm = dspy.LM(main_model, cache=False, **lm_kwargs)
    sub_lm = dspy.LM(sub_model, cache=False, **lm_kwargs)

    tools = _make_tools(full_content, full_tool_content)
    sig_cls = _build_signature(max_llm_calls=max_llm_calls)
    assert isinstance(sig_cls, type) and issubclass(sig_cls, dspy.Signature)

    with dspy.context(lm=main_lm):
        rlm = dspy.RLM(
            signature=sig_cls,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            tools=tools,
            sub_lm=sub_lm,
            verbose=verbose,
        )

        result = rlm(
            session_data=rlm_data,
            analysis_query=query,
        )

    report = result.analysis_report
    trajectory = getattr(result, "trajectory", [])
    has_content = bool(full_content)
    missing = _validate_report(report, has_content)

    return {
        "report": report,
        "trajectory": trajectory,
        "session_id": session_id,
        "missing_sections": missing,
    }
