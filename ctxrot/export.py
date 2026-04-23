import json
from collections.abc import Callable, Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import TextIO

from ctxrot.storage import CtxRotStore

OPENTRACES_SCHEMA_VERSION = "0.3.0"

try:
    _CTXROT_VERSION = version("ctxrot")
except PackageNotFoundError:
    _CTXROT_VERSION = "unknown"

# ctxrot's internal terminal_state values (written by the callback) mapped onto
# the opentraces 0.3.0 enum. `success` is the schema's primary outcome signal;
# `terminal_state` narrows the reason. None means the session didn't finish
# cleanly and we leave both unset.
_TERMINAL_STATE_MAP: dict[str, tuple[bool, str]] = {
    "completed": (True, "goal_reached"),
    "errored": (False, "error"),
}


def resolve_session_ids(
    store: CtxRotStore,
    *,
    sessions: list[str],
    all_: bool,
    since: str | None,
    until: str | None,
    only_errored: bool,
    only_completed: bool,
) -> Iterator[str]:
    """Translate export CLI flags into a stream of session IDs.

    Precedence:
      1. Explicit `sessions` list bypasses all filters (each ID is validated).
      2. Any of `all_`, `since`, `until`, `only_errored`, `only_completed`
         runs the filter query (they compose with AND at the SQL level).
      3. Otherwise, yield just the latest session ID (matches `analyze` /
         `deep-analyze` default behavior).
    """
    if sessions:
        for sid in sessions:
            if store.get_session(sid) is None:
                msg = f"Session not found: {sid}"
                raise ValueError(msg)
            yield sid
        return

    terminal_state: str | None = None
    if only_errored:
        terminal_state = "errored"
    elif only_completed:
        terminal_state = "completed"

    has_filter = bool(all_ or since or until or only_errored or only_completed)
    if has_filter:
        yield from store.get_session_ids(
            since=since, until=until, terminal_state=terminal_state
        )
        return

    latest = store.get_latest_session_id()
    if latest is not None:
        yield latest


def build_opentraces_record(
    store: CtxRotStore,
    session_id: str,
) -> dict:
    """Build a dict matching opentraces TraceRecord v0.3.0 for one session."""
    session = store.get_session(session_id)
    if session is None:
        msg = f"Session not found: {session_id}"
        raise ValueError(msg)

    summary = store.get_session_summary(session_id)
    lm_calls = store.get_lm_calls_full(session_id)
    tool_calls = store.get_tool_calls_full(session_id)

    tools_by_seq: dict[int, list[dict]] = {}
    for tc in tool_calls:
        tools_by_seq.setdefault(tc["after_seq"], []).append(tc)

    steps = [_build_step(lm, tools_by_seq.get(lm["seq"], [])) for lm in lm_calls]

    total_cache_write = sum(lm["cache_write_tokens"] or 0 for lm in lm_calls)
    total_prompt = summary["total_prompt"] or 0
    cache_hit_rate = (
        round(summary["total_cache_read"] / total_prompt, 4)
        if total_prompt > 0
        else 0.0
    )
    total_duration_ms = summary["total_duration_ms"] or 0

    record: dict = {
        "schema_version": OPENTRACES_SCHEMA_VERSION,
        "trace_id": session_id,
        "session_id": session_id,
        "timestamp_start": session["started_at"],
        "agent": {
            "name": session["mode"] or "dspy-agent",
            "model": session["model"],
        },
        "steps": steps,
        "metrics": {
            "total_steps": summary["total_calls"],
            "total_input_tokens": summary["total_prompt"],
            "total_output_tokens": summary["total_completion"],
            "total_cache_read_tokens": summary["total_cache_read"],
            "total_cache_creation_tokens": total_cache_write,
            "total_duration_s": round(total_duration_ms / 1000.0, 3),
            "cache_hit_rate": cache_hit_rate,
            "estimated_cost_usd": summary["total_cost"],
        },
        "metadata": {
            "ctxrot_version": _CTXROT_VERSION,
            "source": "dspy-callback",
            "framework": "dspy",
            "mode": session["mode"],
            "max_prompt_tokens": summary["max_prompt_tokens"],
        },
        "lifecycle": "final" if session["terminal_state"] else "provisional",
    }

    if session["ended_at"]:
        record["timestamp_end"] = session["ended_at"]

    outcome = _build_outcome(session["terminal_state"])
    if outcome:
        record["outcome"] = outcome

    return record


def _build_outcome(terminal_state: str | None) -> dict | None:
    if terminal_state is None:
        return None
    mapped = _TERMINAL_STATE_MAP.get(terminal_state)
    if mapped is None:
        # Unknown ctxrot state — surface it as-is under terminal_state; the
        # schema permits null there, and a raw string is more honest than
        # silently dropping the signal.
        return {"terminal_state": terminal_state}
    success, mapped_state = mapped
    return {"success": success, "terminal_state": mapped_state}


def _build_step(lm: dict, tool_call_rows: list[dict]) -> dict:
    """Build one opentraces Step.

    Fields without a schema home (per-step cost/error/duration/messages/
    iteration) are intentionally dropped — keeping them would make the record
    non-compliant. Use the `ctxrot` native format if you need that detail.
    """
    tool_calls: list[dict] = []
    observations: list[dict] = []
    for tc in tool_call_rows:
        tc_entry: dict = {
            "tool_call_id": tc["id"],
            "tool_name": tc["tool_name"],
        }
        tool_input = _parse_tool_input(tc["input_json"])
        if tool_input is not None:
            tc_entry["input"] = tool_input
        if tc["duration_ms"] is not None:
            tc_entry["duration_ms"] = tc["duration_ms"]
        tool_calls.append(tc_entry)

        obs: dict = {"source_call_id": tc["id"]}
        if tc["output_text"] is not None:
            obs["content"] = tc["output_text"]
        if tc["error"]:
            obs["error"] = tc["error"]
        observations.append(obs)

    step: dict = {
        "step_index": lm["seq"],
        "role": "assistant",
        "token_usage": {
            "input_tokens": lm["prompt_tokens"],
            "output_tokens": lm["completion_tokens"],
            "cache_read_tokens": lm["cache_read_tokens"],
            "cache_write_tokens": lm["cache_write_tokens"],
        },
    }
    if lm["model"]:
        step["model"] = lm["model"]
    if lm["completion"]:
        step["content"] = lm["completion"]
    if lm["started_at"]:
        step["timestamp"] = lm["started_at"]
    if lm["call_type"]:
        step["call_type"] = lm["call_type"]
    if lm["parent_seq"] is not None:
        step["parent_step"] = lm["parent_seq"]
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observations:
        step["observations"] = observations
    return step


def _parse_tool_input(input_json: str | None) -> dict | None:
    """Return the stored tool input as a dict, or None if unavailable/invalid.

    Storage keeps the input as a JSON string; the 0.3.0 schema types
    ToolCall.input as a dict, so non-dict or malformed payloads are dropped.
    """
    if not input_json:
        return None
    try:
        parsed = json.loads(input_json)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_native_record(
    store: CtxRotStore,
    session_id: str,
) -> dict:
    """Build a dict matching ctxrot's native SQLite shape for one session.

    Useful for debugging the export mapper and for roundtripping a session
    without schema translation. Not a stable public format — the shape may
    change whenever the SQLite schema does.
    """
    session = store.get_session(session_id)
    if session is None:
        msg = f"Session not found: {session_id}"
        raise ValueError(msg)

    summary = store.get_session_summary(session_id)
    lm_calls = store.get_lm_calls_full(session_id)
    tool_calls = store.get_tool_calls_full(session_id)

    return {
        "ctxrot_version": _CTXROT_VERSION,
        "session": session,
        "summary": summary,
        "lm_calls": lm_calls,
        "tool_calls": tool_calls,
    }


def stream_export(
    store: CtxRotStore,
    ids: Iterator[str],
    output: TextIO,
    *,
    fmt: str = "opentraces",
    progress: Callable[[int], None] | None = None,
    progress_every: int = 10,
) -> int:
    """Write one JSONL line per session. Returns the number of records.

    Builds each record on demand — the full list is never held in memory,
    so `--all` on a large DB streams flat.
    """
    if fmt == "opentraces":
        build = build_opentraces_record
    elif fmt == "ctxrot":
        build = build_native_record
    else:
        msg = f"Unknown format: {fmt!r}. Use 'opentraces' or 'ctxrot'."
        raise ValueError(msg)

    count = 0
    for sid in ids:
        record = build(store, sid)
        output.write(json.dumps(record, ensure_ascii=False, default=str))
        output.write("\n")
        count += 1
        if progress is not None and count % progress_every == 0:
            progress(count)
    if progress is not None and count > 0 and count % progress_every != 0:
        progress(count)
    return count
