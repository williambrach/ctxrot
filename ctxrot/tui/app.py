"""Textual App shell for ctxrot TUI."""

from __future__ import annotations

import logging
import os

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header
from textual.worker import Worker, WorkerState

from ctxrot.storage import CtxRotStore
from ctxrot.tui.feed import FeedView
from ctxrot.tui.growth import GrowthView


class CtxRotApp(App):
    """ctxrot TUI — LLM context analytics dashboard."""

    TITLE = "ctxrot"
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("t", "toggle_stats", "Toggle stats"),
        Binding("c", "toggle_content", "Toggle content"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    Screen {
        layout: vertical;
    }
    #growth-view {
        display: none;
    }
    """

    def __init__(
        self, db_path: str = "ctxrot.db", session_id: str | None = None
    ) -> None:
        super().__init__()
        self._db_path = db_path
        self._session_id = session_id
        self._current_view = "feed"
        self._store: CtxRotStore | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield FeedView(id="feed-view")
        yield GrowthView(id="growth-view")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "all sessions"
        self.set_interval(2.0, self._poll_db)
        self._poll_db()
        self.query_one("#feed-lm-table", DataTable).focus()

    def _get_store(self) -> CtxRotStore | None:
        if not os.path.exists(self._db_path):
            return None
        if self._store is None:
            self._store = CtxRotStore(self._db_path, read_only=True)
        return self._store

    def _fetch_data(self) -> dict | None:
        """Run all DB queries in a worker thread. No widget access here."""
        store = self._get_store()
        if store is None:
            return None

        view = self._current_view
        session_id = self._session_id

        if view == "feed":
            return {
                "view": "feed",
                "session_rows": store.get_feed_sessions(),
            }
        elif view == "growth":
            sid = session_id or store.get_latest_session_id()
            if not sid:
                return None

            growth = store.get_growth_data(sid)
            summary = store.get_session_summary(sid)
            session = store.get_session(sid)
            session_model = (session or {}).get("model")
            first_call_model = growth[0]["model"] if growth else None
            model = first_call_model or session_model or "unknown"

            # Determine which panel is visible — snapshotted before worker launch
            visible = getattr(self, "_pending_visible_panel", "chart")

            # Rot analysis (lightweight, computed on the fly)
            from ctxrot.analysis import analyze_session

            rot = analyze_session(store, sid)

            session_mode = (session or {}).get("mode")

            result: dict = {
                "view": "growth",
                "session_id": sid,
                "session_mode": session_mode,
                "growth": growth,
                "summary": summary,
                "model": model,
                "tools_by_iter": {},
                "tool_impact": [],
                "content_rows": [],
                "rot": rot,
            }

            if visible in ("chart", "stats", "tree"):
                result["tools_by_iter"] = store.get_tools_by_iteration(sid)
            if session_mode == "rlm" and visible in ("chart", "tree"):
                result["tree_data"] = store.get_rlm_tree_data(sid)
            if visible == "stats":
                result["tool_impact"] = store.get_tool_impact(sid)
            if visible == "content":
                result["content_rows"] = store.get_lm_call_content(sid)

            return result

        return None

    def _poll_db(self) -> None:
        """Launch a background worker to fetch data without blocking the UI."""
        # Snapshot the visible panel so the worker thread knows what to fetch
        if self._current_view == "growth":
            try:
                self._pending_visible_panel = self.query_one(
                    "#growth-view", GrowthView
                )._visible_panel()
            except Exception:
                self._pending_visible_panel = "chart"

        self.run_worker(self._fetch_data, thread=True, exclusive=True, group="poll")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Dispatch fetched data to views on the main thread."""
        if event.worker.group != "poll":
            return
        if event.state != WorkerState.SUCCESS:
            if event.state == WorkerState.ERROR:
                logging.getLogger(__name__).debug(
                    "Poll worker failed: %s", event.worker.error
                )
                self._store = None
            return

        data = event.worker.result
        if data is None:
            self.sub_title = "waiting for data..."
            return

        try:
            if data["view"] == "feed":
                feed_view = self.query_one("#feed-view", FeedView)
                feed_view.update_from_data(data["session_rows"])
            elif data["view"] == "growth":
                self.sub_title = f"session: {data['session_id']}"
                growth_view = self.query_one("#growth-view", GrowthView)
                growth_view.update_from_data(data)
        except Exception as e:
            logging.getLogger(__name__).debug("Dispatch failed: %s", e)

    def on_feed_view_open_session(self, event: FeedView.OpenSession) -> None:
        """Enter on a Feed row — drill into Growth for that session."""
        self._session_id = event.session_id
        self._current_view = "growth"
        self.query_one("#feed-view").display = False
        self.query_one("#growth-view").display = True
        self._poll_db()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Hide growth-only bindings when not in growth view."""
        if (
            action in ("toggle_stats", "toggle_content")
            and self._current_view != "growth"
        ):
            return None
        return True

    def action_toggle_stats(self) -> None:
        """Toggle between chart and scrollable stats in Growth view."""
        self.query_one("#growth-view", GrowthView).action_toggle_stats()

    def action_toggle_content(self) -> None:
        """Toggle between current view and scrollable content in Growth view."""
        self.query_one("#growth-view", GrowthView).action_toggle_content()

    def on_unmount(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    async def action_back(self) -> None:
        """Escape returns to Feed from Growth."""
        if self._current_view == "growth":
            self._current_view = "feed"
            self._session_id = None
            self.sub_title = "all sessions"
            self.query_one("#growth-view").display = False
            self.query_one("#feed-view").display = True
            self._poll_db()
            self.query_one("#feed-lm-table", DataTable).focus()
