import json
import logging
import time
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
    """Per-call mutable state, stored in a ContextVar for thread isolation."""

    session_id: str
    seq: int = 0
    module_depth: int = 0
    lm_starts: dict[str, tuple] = field(default_factory=dict)
    tool_starts: dict[str, tuple] = field(default_factory=dict)
    # RLM tree tracking
    mode: str | None = None
    rlm_iteration: int = 0
    rlm_last_action_seq: int = 0
    rlm_in_predict: bool = False
    rlm_predict_call_id: str | None = None


_session_ctx: ContextVar[_SessionState | None] = ContextVar(
    "ctxrot_session", default=None
)


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
        self._last_session_id: str | None = None

    @staticmethod
    def _safe_json_dumps(obj: dict) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    @staticmethod
    def _extract_completion_text(entry: dict[str, Any]) -> str | None:
        response = entry.get("response")
        if response is not None and hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            msg = first_choice.message
            content = getattr(msg, "content", None)
            if content:
                return content
        outputs = entry.get("outputs")
        if outputs:
            first = outputs[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return first.get("text", "")
        return None

    @property
    def session_id(self) -> str | None:
        state = _session_ctx.get(None)
        if state is not None:
            return state.session_id
        return self._last_session_id

    # ── Module callbacks (auto-session) ─────────────────────────────────

    def on_module_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        # Auto-create session when any top-level DSPy module starts
        type_name = type(instance).__name__
        state = _session_ctx.get(None)
        if state is None:
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

        # Track Predict sub-module entry for RLM call classification
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
            # Track Predict sub-module exit for RLM
            if state.rlm_predict_call_id == call_id:
                state.rlm_in_predict = False
                state.rlm_predict_call_id = None
            state.module_depth -= 1
            if state.module_depth == 0:
                _session_ctx.set(None)

    # ── LM callbacks ────────────────────────────────────────────────────

    def on_lm_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        state = _session_ctx.get(None)
        if state is None:
            return
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
        start_data = state.lm_starts.pop(call_id, None)
        if not start_data:
            return

        instance, started_at, messages = start_data
        state.seq += 1
        ended_at = time.time()

        # RLM call classification
        call_type = None
        iteration = None
        parent_seq = None
        if state.mode == "rlm":
            if state.rlm_in_predict:
                state.rlm_iteration += 1
                call_type = "action"
                iteration = state.rlm_iteration
                state.rlm_last_action_seq = state.seq
            else:
                call_type = "sub_query"
                iteration = state.rlm_iteration
                parent_seq = state.rlm_last_action_seq

        prompt_tokens = 0
        completion_tokens = 0
        cache_read = 0
        cache_write = 0
        cost = None
        model = getattr(instance, "model", None) or "unknown"
        entry = None

        try:
            response = None
            has_history = hasattr(instance, "history") and instance.history
            if has_history:
                entry = instance.history[-1]
                response = entry.get("response")
                cost = entry.get("cost")
                # Only fall back to API/response model if the configured model is unknown
                if model == "unknown":
                    response_model = entry.get("response_model")
                    history_model = entry.get("model")
                    api_model = getattr(response, "model", None) if response else None
                    model = response_model or api_model or history_model or model

            usage = getattr(response, "usage", None) if response else None
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                if cost is None:
                    cost = getattr(usage, "cost", None)

                prompt_token_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_token_details is not None:
                    cache_read = getattr(prompt_token_details, "cached_tokens", 0) or 0
                    cache_write = (
                        getattr(prompt_token_details, "cache_creation_tokens", 0) or 0
                    )
                else:
                    cache_read = getattr(usage, "_cache_read_input_tokens", 0) or 0
                    cache_write = getattr(usage, "_cache_creation_input_tokens", 0) or 0

            if cost is None:
                cost = pricing.calculate_cost(
                    model, prompt_tokens, completion_tokens, cache_read, cache_write
                )
        except Exception as e:
            logger.debug("Failed to extract usage: %s", e)

        # Extract content if enabled
        messages_json = None
        completion_text = None
        prompt_char_count = None
        completion_char_count = None

        if self._store_content:
            try:
                # Get messages — prefer from on_lm_start inputs, fallback to history
                if messages is None and entry is not None:
                    messages = entry.get("messages")

                # Extract completion text from history
                if entry is not None:
                    completion_text = self._extract_completion_text(entry)

                if messages is not None:
                    messages_json = self._safe_json_dumps(messages)

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
        raw_output = outputs or ""
        output_str = str(raw_output)
        output_tokens_est = count_tokens(output_str)

        # Extract content if enabled
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
