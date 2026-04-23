import atexit
import json
import logging
import time
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from dspy.utils.callback import BaseCallback

from ctxrot import pricing
from ctxrot.storage import CtxRotStore
from ctxrot.tokenizer import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class _SessionState:
    """Per-call mutable state, stored in a ContextVar for thread isolation.

    One instance lives for the duration of one top-level DSPy module run —
    created in on_module_start when no state exists yet, cleared in
    on_module_end when module_depth returns to 0.
    """

    session_id: str
    # Sequence number for LM calls within this session. Increments on each
    # on_lm_end so we can order calls (and link tool calls to the LM call
    # that triggered them via tool_calls.after_seq).
    seq: int = 0
    # How deeply nested we are in DSPy modules. The outermost module enters
    # at depth 1; we close out the session when depth returns to 0.
    module_depth: int = 0
    # Per-call-id staging for LM/tool start data so on_*_end can match its
    # corresponding on_*_start (and compute duration, recover prompt messages, …).
    lm_starts: dict[str, tuple] = field(default_factory=dict)
    tool_starts: dict[str, tuple] = field(default_factory=dict)
    # ── RLM tree tracking ──────────────────────────────────────────────────
    # For non-RLM sessions all of the below stay at their defaults.
    # For RLM, we classify each LM call as either an "action" (the Predict
    # reasoning step that drives one iteration) or a "sub_query" (any LM
    # call made downstream while still inside that Predict). This lets the
    # TUI / analysis layer reconstruct the reasoning tree.
    mode: str | None = None  # set from the top-level module's class name (e.g. "rlm", "react")
    rlm_iteration: int = 0  # current action number; only bumps on action calls
    rlm_last_action_seq: int = 0  # seq of the most recent action — sub_queries link here as parent
    rlm_in_predict: bool = False  # True while we are inside the RLM's Predict sub-module
    rlm_predict_call_id: str | None = None  # call_id of the active Predict, used to detect its exit


# Per-thread session state. ContextVar (rather than a plain attribute) means
# asyncify/streamify worker threads each see isolated state — concurrent agent
# calls don't stomp on each other's session.
_session_ctx: ContextVar[_SessionState | None] = ContextVar(
    "ctxrot_session", default=None
)


def _close_store_at_exit(store_ref: "weakref.ref[CtxRotStore]") -> None:
    store = store_ref()
    if store is None:
        return
    try:
        store.close()
    except Exception as e:
        logger.debug("Failed to close store at exit: %s", e)


class CtxRotCallback(BaseCallback):
    """Captures LM and tool call data into SQLite.

    Auto-creates a new session each time a top-level DSPy module run starts.

    Concurrency-safe: each ``asyncify``/``streamify`` worker thread gets its own
    ``_SessionState`` via a ``ContextVar``, so multiple concurrent agent calls
    do not interfere with each other.

    Args:
        db_path: Path to SQLite database file.
        store_content: If True, also store full prompt messages and completion text
                       for context rot analysis.
    """

    def __init__(self, db_path: str = "ctxrot.db", store_content: bool = False) -> None:
        self._store = CtxRotStore(db_path)
        self._store_content = store_content
        # Cached so the .session_id property keeps working *after* a run finishes.
        # The ContextVar is cleared on module_end, but callers often want to
        # inspect the session id post-run (e.g. to pass it to `analyze`).
        self._last_session_id: str | None = None
        # Ensure the store is closed on interpreter shutdown so SQLite gets a
        # chance to truncate-checkpoint and remove the -wal/-shm sidecars.
        # weakref.ref lets us avoid keeping the callback alive just for atexit.
        store_ref = weakref.ref(self._store)
        atexit.register(_close_store_at_exit, store_ref)

    @staticmethod
    def _safe_json_dumps(obj: dict) -> str:
        # LM payloads can contain non-serializable objects (pydantic models,
        # bytes, custom classes). default=str coerces what it can; the except
        # catches the rest so a serialization failure never breaks the user's call.
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    @staticmethod
    def _extract_completion_text(entry: dict[str, Any]) -> str | None:
        # DSPy's history entries come in two shapes depending on the adapter:
        #   1. A LiteLLM response object with .choices[0].message.content
        #      (the common case for chat completion APIs).
        #   2. A list of "outputs" — strings or dicts — used by older /
        #      non-chat adapters.
        # Try the LiteLLM shape first, then fall back to outputs.
        response = entry.get("response")
        if response is not None and hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            msg = first_choice.message
            content = getattr(msg, "content", None)
            if content:
                return content
        outputs = entry.get("outputs")
        if outputs:
            first_output = outputs[0]
            if isinstance(first_output, str):
                return first_output
            if isinstance(first_output, dict):
                return first_output.get("text", "")
        return None

    @property
    def session_id(self) -> str | None:
        # Prefer the live session if a run is in progress; otherwise return the
        # last completed session so callers can inspect it after the run ends.
        state = _session_ctx.get(None)
        if state is not None:
            return state.session_id
        return self._last_session_id

    # ── Module callbacks (auto-session) ─────────────────────────────────

    def on_module_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        # Auto-create a session when any top-level DSPy module starts.
        # Nested modules (e.g. a Predict inside an RLM) just bump module_depth
        # — they share the outer session.
        type_name = type(instance).__name__
        state = _session_ctx.get(None)
        if state is None:
            # Top-level entry — open a fresh session row in the DB.
            lm = getattr(instance, "lm", None)
            model = getattr(lm, "model", None)
            session_id = uuid4().hex[:12]
            state = _SessionState(session_id=session_id, mode=type_name.lower())
            _session_ctx.set(state)
            self._last_session_id = session_id
            self._store.insert_session(
                session_id, started_at=time.time(), model=model, mode=type_name.lower()
            )
        state.module_depth += 1

        # RLM-specific: when the RLM's inner Predict module starts, mark that
        # the *next* LM call will be an "action" call. depth==2 specifically
        # picks the Predict that the RLM itself owns:
        #   depth 1 = RLM (the outer module)
        #   depth 2 = its Predict
        # Anything deeper (e.g. nested Predicts called by tools) is a sub_query.
        if (
            state.mode == "rlm"
            and type_name == "Predict"
            and state.module_depth == 2
        ):
            state.rlm_in_predict = True
            state.rlm_predict_call_id = call_id

    def on_module_end(
        self,
        call_id: str,
        outputs: object | None,
        exception: Exception | None = None,
    ) -> None:
        state = _session_ctx.get(None)
        if state is not None:
            # Mirror of on_module_start: when the same Predict that flipped
            # rlm_in_predict on closes, flip it back off.
            if state.rlm_predict_call_id == call_id:
                state.rlm_in_predict = False
                state.rlm_predict_call_id = None
            state.module_depth -= 1
            # Outermost module exiting — finalize the session row.
            if state.module_depth == 0:
                terminal_state = "errored" if exception is not None else "completed"
                try:
                    self._store.update_session_end(
                        state.session_id,
                        time.time(),
                        terminal_state,
                    )
                except Exception as e:
                    logger.warning("Failed to update session end: %s", e)
                # Clear the ContextVar so the next top-level run starts a fresh session.
                _session_ctx.set(None)

    # ── LM callbacks ────────────────────────────────────────────────────

    def on_lm_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        state = _session_ctx.get(None)
        if state is None:
            # No active session — caller is using the LM outside a tracked
            # DSPy module. Nothing to record.
            return
        # Stash the LM instance + start time + (optionally) the prompt messages
        # keyed by call_id. on_lm_end pops this back out to compute duration
        # and emit one row per call.
        messages = inputs.get("messages") if self._store_content else None
        state.lm_starts[call_id] = (instance, time.time(), messages)

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        state = _session_ctx.get(None)
        if state is None:
            return
        # Pair this end with its on_lm_start data; if there's no pairing
        # (start was missed, e.g. callback added mid-run), skip it.
        start_data = state.lm_starts.pop(call_id, None)
        if not start_data:
            return

        instance, started_at, messages = start_data
        state.seq += 1
        ended_at = time.time()

        # ── RLM call classification ────────────────────────────────────────
        # In RLM mode, every LM call is one of:
        #   - "action": the Predict step that drives one reasoning iteration
        #   - "sub_query": any LM call made downstream while inside that Predict
        # Non-RLM modes leave call_type / iteration / parent_seq as None.
        call_type = None
        iteration = None
        parent_seq = None
        if state.mode == "rlm":
            if state.rlm_in_predict:
                # We're entering the action call itself — bump the iteration
                # counter and remember our seq so any sub_queries that follow
                # can point back to us as their parent.
                state.rlm_iteration += 1
                call_type = "action"
                iteration = state.rlm_iteration
                state.rlm_last_action_seq = state.seq
            else:
                # Sub-query within the current iteration's reasoning.
                call_type = "sub_query"
                iteration = state.rlm_iteration
                parent_seq = state.rlm_last_action_seq

        # Defaults used if usage extraction fails — we still record a row
        # with whatever we have, marked with `error` if applicable.
        prompt_tokens = 0
        completion_tokens = 0
        cache_read = 0
        cache_write = 0
        cost = None
        model = getattr(instance, "model", None) or "unknown"
        entry = None

        # ── Usage extraction ───────────────────────────────────────────────
        # Wrapped in a broad try because LM history shapes vary across providers
        # and DSPy versions; we never want a parsing miss to break the LM call.
        try:
            response = None
            history = getattr(instance, "history", None)
            if history:
                # The most recent history entry corresponds to the call that
                # just ended. (history is append-only within a single LM run.)
                entry = history[-1]
                response = entry.get("response")
                cost = entry.get("cost")
                # Model resolution: only override the configured model if it's
                # missing. Otherwise honor whatever the user set on the LM
                # instance (the API can echo back a longer/canonical name).
                if model == "unknown":
                    response_model = entry.get("response_model")
                    history_model = entry.get("model")
                    api_model = getattr(response, "model", None) if response else None
                    model = response_model or api_model or history_model or model

            usage = getattr(response, "usage", None) if response else None
            if usage is not None:
                # `or 0` guards against the attribute existing but being None,
                # which getattr's default value doesn't catch.
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                if cost is None:
                    cost = getattr(usage, "cost", None)

                # Cache-token reporting differs between providers:
                #   - OpenAI exposes prompt_tokens_details.cached_tokens
                #     (and .cache_creation_tokens for prompt caching)
                #   - Anthropic via LiteLLM uses the underscored attrs
                #     `_cache_read_input_tokens` / `_cache_creation_input_tokens`
                # Try the standard nested shape first, fall back to the flat one.
                prompt_token_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_token_details is not None:
                    cache_read = getattr(prompt_token_details, "cached_tokens", 0) or 0
                    cache_write = (
                        getattr(prompt_token_details, "cache_creation_tokens", 0) or 0
                    )
                else:
                    cache_read = getattr(usage, "_cache_read_input_tokens", 0) or 0
                    cache_write = getattr(usage, "_cache_creation_input_tokens", 0) or 0

            # Only fall back to our own pricing table if the provider didn't
            # supply cost. Provider-reported cost is authoritative when present.
            if cost is None:
                cost = pricing.calculate_cost(
                    model, prompt_tokens, completion_tokens, cache_read, cache_write
                )
        except Exception as e:
            logger.debug("Failed to extract usage: %s", e)

        # ── Content extraction (only if store_content=True) ────────────────
        # Optional, off-by-default capture of full prompt + completion text.
        # The analysis layer uses this for repetition / context-rot detection.
        # Skipped by default because it can multiply DB size and may contain PII.
        messages_json = None
        completion_text = None
        prompt_char_count = None
        completion_char_count = None

        if self._store_content:
            try:
                # Prefer messages from on_lm_start (cleaner — exactly what was
                # sent); fall back to history if for some reason they weren't
                # captured at start time.
                if messages is None and entry is not None:
                    messages = entry.get("messages")

                if entry is not None:
                    completion_text = self._extract_completion_text(entry)

                if messages is not None:
                    messages_json = self._safe_json_dumps(messages)

                # Cheap proxy for prompt size; the authoritative count is in
                # prompt_tokens. char counts are useful when tokens are missing.
                prompt_char_count = 0
                if messages and isinstance(messages, list):
                    prompt_char_count = sum(
                        len(str(m.get("content", "")))
                        for m in messages
                        if isinstance(m, dict)
                    )

                completion_char_count = len(completion_text) if completion_text else 0
            except Exception as e:
                logger.warning("Failed to extract LM call content: %s", e)

        # Persist one row per LM call. Wrapped in try so a DB hiccup degrades
        # gracefully — the user's LM result still flows through unaffected.
        try:
            self._store.insert_lm_call(
                id=call_id,
                session_id=state.session_id,
                seq=state.seq,
                model=model,
                started_at=started_at,
                ended_at=ended_at,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                cost=cost,
                error=str(exception) if exception else None,
                messages_json=messages_json,
                completion=completion_text,
                prompt_char_count=prompt_char_count,
                completion_char_count=completion_char_count,
                call_type=call_type,
                iteration=iteration,
                parent_seq=parent_seq,
            )
        except Exception as e:
            logger.warning("Failed to insert LM call: %s", e)

    # ── Tool callbacks ──────────────────────────────────────────────────

    def on_tool_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        state = _session_ctx.get(None)
        if state is None:
            return
        # DSPy tools may set `.name` explicitly (e.g. via the Tool() wrapper);
        # if not, fall back to the underlying function/class name. Final fallback
        # is the literal "unknown" so we never crash on naming.
        display_name = getattr(instance, "name", None)
        name = display_name or getattr(instance, "__name__", "unknown")
        input_data = inputs if self._store_content else None
        state.tool_starts[call_id] = (name, time.time(), input_data)

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        state = _session_ctx.get(None)
        if state is None:
            return
        start_data = state.tool_starts.pop(call_id, None)
        if not start_data:
            return

        name, started_at, input_data = start_data
        # Tool outputs can be any type; serialize to string and estimate tokens
        # so the analysis layer can attribute context-window growth to specific
        # tools (a tool that returns 10k-token blobs is a common rot driver).
        raw_output = outputs or ""
        output_str = str(raw_output)
        output_tokens_est = count_tokens(output_str)

        # ── Optional content capture ───────────────────────────────────────
        input_json = None
        content_output_text = None
        content_output_char_count = None

        if self._store_content:
            try:
                if input_data is not None:
                    input_json = self._safe_json_dumps(input_data)
                content_output_text = output_str
                content_output_char_count = len(output_str)
            except Exception as e:
                logger.warning("Failed to extract tool call content: %s", e)

        try:
            self._store.insert_tool_call(
                id=call_id,
                session_id=state.session_id,
                # after_seq pins this tool to the LM call that just ended.
                # That linkage is what lets the analysis layer attribute
                # context growth to the specific reasoning step that triggered it.
                after_seq=state.seq,
                tool_name=name,
                started_at=started_at,
                ended_at=time.time(),
                output_tokens_est=output_tokens_est,
                error=str(exception) if exception else None,
                input_json=input_json,
                output_text=content_output_text,
                output_char_count=content_output_char_count,
            )
        except Exception as e:
            logger.warning("Failed to insert tool call: %s", e)
