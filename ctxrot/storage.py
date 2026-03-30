import logging
import sqlite3
import threading
from datetime import UTC, datetime

_log = logging.getLogger(__name__)


class CtxRotStore:
    """Persists LM call, tool call, and session data in SQLite."""

    def __init__(
        self,
        db_path: str = "ctxrot.db",
        read_only: bool = False,
    ) -> None:
        self._db_path = db_path
        self._read_only = read_only
        self._lock = threading.Lock()
        self._con: sqlite3.Connection

        if read_only:
            read_only_uri = f"file:{db_path}?mode=ro"
            self._con = sqlite3.connect(
                read_only_uri, uri=True, check_same_thread=False
            )
        else:
            self._con = sqlite3.connect(db_path, check_same_thread=False)
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA busy_timeout=5000")
            self._ensure_schema()

    # ── Schema ──────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        con = self._con
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                model      TEXT,
                mode       TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS lm_calls (
                id                 TEXT PRIMARY KEY,
                session_id         TEXT NOT NULL,
                seq                INTEGER NOT NULL,
                model              TEXT NOT NULL,
                started_at         TEXT NOT NULL,
                ended_at           TEXT,
                duration_ms        INTEGER,
                prompt_tokens      INTEGER DEFAULT 0,
                completion_tokens  INTEGER DEFAULT 0,
                cache_read_tokens  INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                cost               REAL,
                error              TEXT,
                messages_json         TEXT,
                completion            TEXT,
                prompt_char_count     INTEGER,
                completion_char_count INTEGER,
                call_type          TEXT,
                iteration          INTEGER,
                parent_seq         INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id                TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL,
                after_seq         INTEGER NOT NULL DEFAULT 0,
                tool_name         TEXT NOT NULL,
                started_at        TEXT NOT NULL,
                ended_at          TEXT,
                duration_ms       INTEGER,
                output_tokens_est INTEGER DEFAULT 0,
                error             TEXT,
                input_json        TEXT,
                output_text       TEXT,
                output_char_count INTEGER
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_lm_session ON lm_calls (session_id)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_lm_session_seq ON lm_calls (session_id, seq)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_tc_session ON tool_calls (session_id)"
        )
        con.commit()

    # ── Write ───────────────────────────────────────────────────────────

    def _write(self, sql: str, params: list) -> None:
        with self._lock:
            self._con.execute(sql, params)
            self._con.commit()

    def insert_session(
        self,
        session_id: str,
        started_at: float,
        model: str | None = None,
        mode: str | None = None,
    ) -> None:
        self._write(
            "INSERT INTO sessions (id, started_at, model, mode) VALUES (?, ?, ?, ?)",
            [session_id, _ts(started_at), model, mode],
        )

    def insert_lm_call(
        self,
        *,
        id: str,
        session_id: str,
        seq: int,
        model: str,
        started_at: float,
        ended_at: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost: float | None = None,
        error: str | None = None,
        messages_json: str | None = None,
        completion: str | None = None,
        prompt_char_count: int | None = None,
        completion_char_count: int | None = None,
        call_type: str | None = None,
        iteration: int | None = None,
        parent_seq: int | None = None,
    ) -> None:
        duration_ms = int((ended_at - started_at) * 1000)
        self._write(
            """INSERT INTO lm_calls
               (id, session_id, seq, model, started_at, ended_at, duration_ms,
                prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens,
                cost, error,
                messages_json, completion, prompt_char_count, completion_char_count,
                call_type, iteration, parent_seq)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                id,
                session_id,
                seq,
                model,
                _ts(started_at),
                _ts(ended_at),
                duration_ms,
                prompt_tokens,
                completion_tokens,
                cache_read_tokens,
                cache_write_tokens,
                cost,
                error,
                messages_json,
                completion,
                prompt_char_count,
                completion_char_count,
                call_type,
                iteration,
                parent_seq,
            ],
        )

    def insert_tool_call(
        self,
        *,
        id: str,
        session_id: str,
        after_seq: int,
        tool_name: str,
        started_at: float,
        ended_at: float,
        output_tokens_est: int = 0,
        error: str | None = None,
        input_json: str | None = None,
        output_text: str | None = None,
        output_char_count: int | None = None,
    ) -> None:
        duration_ms = int((ended_at - started_at) * 1000)
        self._write(
            """INSERT INTO tool_calls
               (id, session_id, after_seq, tool_name, started_at, ended_at, duration_ms,
                output_tokens_est, error,
                input_json, output_text, output_char_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                id,
                session_id,
                after_seq,
                tool_name,
                _ts(started_at),
                _ts(ended_at),
                duration_ms,
                output_tokens_est,
                error,
                input_json,
                output_text,
                output_char_count,
            ],
        )

    # ── Read ────────────────────────────────────────────────────────────

    def get_latest_session_id(self) -> str | None:
        row = self._con.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def get_session(self, session_id: str) -> dict | None:
        row = self._con.execute(
            "SELECT id, started_at, model, mode FROM sessions WHERE id = ?",
            [session_id],
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "started_at": row[1], "model": row[2], "mode": row[3]}

    def get_growth_data(self, session_id: str) -> list[dict]:
        rows = self._con.execute(
            """SELECT seq, prompt_tokens, completion_tokens,
                      cache_read_tokens, cache_write_tokens, cost, model
               FROM lm_calls WHERE session_id = ? ORDER BY seq""",
            [session_id],
        ).fetchall()
        return [
            {
                "seq": r[0],
                "prompt_tokens": r[1],
                "completion_tokens": r[2],
                "cache_read_tokens": r[3],
                "cache_write_tokens": r[4],
                "cost": r[5],
                "model": r[6],
            }
            for r in rows
        ]

    def get_session_summary(self, session_id: str) -> dict:
        row = self._con.execute(
            """SELECT
                 COUNT(*) as total_calls,
                 COALESCE(SUM(prompt_tokens), 0) as total_prompt,
                 COALESCE(SUM(completion_tokens), 0) as total_completion,
                 COALESCE(SUM(cache_read_tokens), 0) as total_cache_read,
                 COALESCE(SUM(cost), 0) as total_cost,
                 COALESCE(SUM(duration_ms), 0) as total_duration_ms,
                 MAX(prompt_tokens) as max_prompt_tokens
               FROM lm_calls WHERE session_id = ?""",
            [session_id],
        ).fetchone()
        total_calls = row[0]
        total_prompt = row[1]
        total_completion = row[2]
        total_cache_read = row[3]
        total_cost = row[4]
        total_duration_ms = row[5]
        max_prompt_tokens = row[6] or 0
        cache_hit_pct = (
            (total_cache_read / total_prompt * 100) if total_prompt > 0 else 0.0
        )
        return {
            "total_calls": total_calls,
            "total_prompt": total_prompt,
            "total_completion": total_completion,
            "total_cache_read": total_cache_read,
            "cache_hit_pct": round(cache_hit_pct, 1),
            "total_cost": total_cost,
            "total_duration_ms": total_duration_ms,
            "max_prompt_tokens": max_prompt_tokens,
        }

    def get_tool_impact(self, session_id: str) -> list[dict]:
        rows = self._con.execute(
            """SELECT tool_name, COUNT(*) as call_count,
                      CAST(AVG(output_tokens_est) AS INTEGER) as avg_tokens
               FROM tool_calls WHERE session_id = ?
               GROUP BY tool_name ORDER BY avg_tokens DESC""",
            [session_id],
        ).fetchall()
        return [
            {"tool_name": r[0], "call_count": r[1], "avg_tokens": r[2]} for r in rows
        ]

    def get_tools_by_iteration(self, session_id: str) -> dict[int, list[dict]]:
        """Return tool calls grouped by the iteration (after_seq) they were called after."""
        rows = self._con.execute(
            """SELECT after_seq, tool_name, output_tokens_est, duration_ms
               FROM tool_calls WHERE session_id = ?
               ORDER BY after_seq, started_at""",
            [session_id],
        ).fetchall()
        result: dict[int, list[dict]] = {}
        for r in rows:
            seq = r[0]
            result.setdefault(seq, []).append(
                {
                    "tool_name": r[1],
                    "output_tokens_est": r[2],
                    "duration_ms": r[3],
                }
            )
        return result

    # ── RLM Tree ──────────────────────────────────────────────────────────

    def get_rlm_tree_data(self, session_id: str) -> list[dict]:
        """Return LM calls for an RLM session with tree metadata."""
        rows = self._con.execute(
            """SELECT seq, model, prompt_tokens, completion_tokens,
                      cost, duration_ms, call_type, iteration, parent_seq,
                      completion
               FROM lm_calls WHERE session_id = ?
               ORDER BY seq""",
            [session_id],
        ).fetchall()
        result = [
            {
                "seq": r[0],
                "model": r[1],
                "prompt_tokens": r[2],
                "completion_tokens": r[3],
                "cost": r[4],
                "duration_ms": r[5],
                "call_type": r[6],
                "iteration": r[7],
                "parent_seq": r[8],
                "completion": r[9],
            }
            for r in rows
        ]
        # Detect extract: last action whose prompt_tokens dropped
        actions = [r for r in result if r["call_type"] == "action"]
        if len(actions) >= 2:
            last = actions[-1]
            prev = actions[-2]
            if last["prompt_tokens"] < prev["prompt_tokens"]:
                last["call_type"] = "extract"
        return result

    # ── Feed ─────────────────────────────────────────────────────────────

    def get_feed_sessions(self, limit: int = 200) -> list[dict]:
        rows = self._con.execute(
            """SELECT
                   min(started_at) AS started_at,
                   session_id,
                   (SELECT model FROM lm_calls c2
                    WHERE c2.session_id = lm_calls.session_id
                    ORDER BY c2.seq DESC LIMIT 1) AS model,
                   sum(prompt_tokens) AS prompt_tokens,
                   sum(completion_tokens) AS completion_tokens,
                   sum(cache_read_tokens) AS cache_read_tokens,
                   sum(cost) AS cost,
                   sum(duration_ms) AS duration_ms,
                   count(*) AS call_count,
                   count(error) AS error_count
               FROM lm_calls
               GROUP BY session_id
               ORDER BY min(started_at) DESC
               LIMIT ?""",
            [limit],
        ).fetchall()
        return [
            {
                "started_at": r[0],
                "session_id": r[1],
                "model": r[2],
                "prompt_tokens": r[3],
                "completion_tokens": r[4],
                "cache_read_tokens": r[5],
                "cost": r[6],
                "duration_ms": r[7],
                "call_count": r[8],
                "error_count": r[9],
            }
            for r in rows
        ]

    def get_feed_lm_calls(
        self, session_id: str | None = None, limit: int = 200
    ) -> list[dict]:
        if session_id:
            rows = self._con.execute(
                """SELECT started_at, session_id, model, prompt_tokens, completion_tokens,
                          cache_read_tokens, cost, duration_ms, error
                   FROM lm_calls WHERE session_id = ?
                   ORDER BY started_at DESC LIMIT ?""",
                [session_id, limit],
            ).fetchall()
        else:
            rows = self._con.execute(
                """SELECT started_at, session_id, model, prompt_tokens, completion_tokens,
                          cache_read_tokens, cost, duration_ms, error
                   FROM lm_calls ORDER BY started_at DESC LIMIT ?""",
                [limit],
            ).fetchall()
        return [
            {
                "started_at": r[0],
                "session_id": r[1],
                "model": r[2],
                "prompt_tokens": r[3],
                "completion_tokens": r[4],
                "cache_read_tokens": r[5],
                "cost": r[6],
                "duration_ms": r[7],
                "error": r[8],
            }
            for r in rows
        ]

    def get_feed_tool_calls(
        self, session_id: str | None = None, limit: int = 200
    ) -> list[dict]:
        if session_id:
            rows = self._con.execute(
                """SELECT started_at, session_id, tool_name, duration_ms, output_tokens_est, error
                   FROM tool_calls WHERE session_id = ?
                   ORDER BY started_at DESC LIMIT ?""",
                [session_id, limit],
            ).fetchall()
        else:
            rows = self._con.execute(
                """SELECT started_at, session_id, tool_name, duration_ms, output_tokens_est, error
                   FROM tool_calls ORDER BY started_at DESC LIMIT ?""",
                [limit],
            ).fetchall()
        return [
            {
                "started_at": r[0],
                "session_id": r[1],
                "tool_name": r[2],
                "duration_ms": r[3],
                "output_tokens_est": r[4],
                "error": r[5],
            }
            for r in rows
        ]

    # ── Content ───────────────────────────────────────────────────────────

    def get_lm_call_content(self, session_id: str) -> list[dict]:
        """Return stored LM call content for a session, ordered by seq."""
        try:
            rows = self._con.execute(
                """SELECT seq, messages_json, completion,
                          prompt_char_count, completion_char_count
                   FROM lm_calls
                   WHERE session_id = ? AND messages_json IS NOT NULL
                   ORDER BY seq""",
                [session_id],
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "seq": r[0],
                "messages_json": r[1],
                "completion": r[2],
                "prompt_char_count": r[3],
                "completion_char_count": r[4],
            }
            for r in rows
        ]

    def get_tool_call_content(self, session_id: str) -> list[dict]:
        """Return stored tool call content for a session."""
        try:
            rows = self._con.execute(
                """SELECT id, input_json, output_text, output_char_count
                   FROM tool_calls
                   WHERE session_id = ? AND input_json IS NOT NULL
                   ORDER BY id""",
                [session_id],
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "tool_call_id": r[0],
                "input_json": r[1],
                "output_text": r[2],
                "output_char_count": r[3],
            }
            for r in rows
        ]

    # ── Admin ───────────────────────────────────────────────────────────

    def truncate_all(self) -> None:
        self._con.execute("DELETE FROM lm_calls")
        self._con.execute("DELETE FROM tool_calls")
        self._con.execute("DELETE FROM sessions")
        self._con.commit()

    def close(self) -> None:
        con = self._con
        del self._con
        con.close()


# ── Helpers ─────────────────────────────────────────────────────────────


def _ts(epoch: float) -> str:
    """Convert epoch float to ISO-8601 string."""
    utc_dt = datetime.fromtimestamp(epoch, tz=UTC)
    return utc_dt.isoformat()
