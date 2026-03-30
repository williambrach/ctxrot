"""Feed view — reverse-chronological history of LM calls."""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.widgets import DataTable, Static


class FeedView(Container):
    """DataTable showing LM call history. Enter on a row opens Growth for that session."""

    class OpenSession(Message):
        """Posted when user presses Enter on a row."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    DEFAULT_CSS = """
    FeedView {
        layout: vertical;
        height: 1fr;
    }
    FeedView .feed-label {
        height: 1;
        padding: 0 1;
        text-style: bold;
    }
    FeedView DataTable {
        height: 1fr;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        markup: bool = True,
    ) -> None:
        super().__init__(
            name=name, id=id, classes=classes, disabled=disabled, markup=markup
        )
        self._last_lm_count = -1
        self._row_session_ids: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("Sessions", classes="feed-label")
        yield DataTable(id="feed-lm-table", cursor_type="row")

    def on_mount(self) -> None:
        lm_table = self.query_one("#feed-lm-table", DataTable)
        lm_table.add_columns(
            "Time",
            "Session",
            "Model",
            "Calls",
            "In",
            "Out",
            "Cache%",
            "Cost",
            "Duration",
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_idx = event.cursor_row
        session_id = self._row_session_ids.get(row_idx)
        if session_id:
            self.post_message(self.OpenSession(session_id))

    def update_from_data(self, session_rows: list[dict]) -> None:
        if len(session_rows) != self._last_lm_count:
            self._last_lm_count = len(session_rows)
            self._row_session_ids.clear()
            lm_table = self.query_one("#feed-lm-table", DataTable)
            lm_table.clear()
            for i, r in enumerate(session_rows):
                self._row_session_ids[i] = r["session_id"]
                prompt = r["prompt_tokens"] or 0
                cache_read = r["cache_read_tokens"] or 0
                cache_pct = (
                    f"{cache_read / prompt * 100:.1f}%" if prompt > 0 else "0.0%"
                )
                lm_table.add_row(
                    _fmt_time(r["started_at"]),
                    _fmt_session(r["session_id"]),
                    r["model"] or "",
                    str(r["call_count"]),
                    f"{prompt:,}",
                    f"{r['completion_tokens'] or 0:,}",
                    cache_pct,
                    _fmt_cost(r["cost"]),
                    _fmt_duration(r["duration_ms"]),
                )


def _fmt_time(ts: datetime | None) -> str:
    if ts is None:
        return "--"
    if isinstance(ts, datetime):
        return ts.strftime("%H:%M:%S")
    return str(ts)


def _fmt_session(sid: str) -> str:
    return sid[:4] if sid else "--"


def _fmt_cost(c: float | None) -> str:
    if c is None:
        return "--"
    return f"${c:.4f}"


def _fmt_duration(ms: int | None) -> str:
    if ms is None:
        return "--"
    return f"{ms / 1000:.1f}s"
