import json
import logging
import threading
import time
from typing import Any, Dict, Optional
from config import settings

logger = logging.getLogger("oanda_agent.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_sentiment_cache (
    cache_key   TEXT PRIMARY KEY,
    instrument  TEXT NOT NULL,
    model       TEXT NOT NULL,
    result      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS llm_sentiment_cache_created_at_idx ON llm_sentiment_cache (created_at);
CREATE TABLE IF NOT EXISTS signal_journal (
    instrument      TEXT NOT NULL,
    entry_tf        TEXT NOT NULL,
    bar_time        BIGINT NOT NULL,
    direction       TEXT NOT NULL,
    tech_reason     TEXT,
    price           DOUBLE PRECISION,
    atr             DOUBLE PRECISION,
    sentiment_score DOUBLE PRECISION,
    high_impact     BOOLEAN,
    decision        TEXT NOT NULL,
    details         TEXT,
    sl_atr          DOUBLE PRECISION,
    tp_atr          DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instrument, entry_tf, bar_time)
);
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    token_hash  TEXT PRIMARY KEY,
    expires_at  DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_state (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

RECONNECT_INTERVAL_SECONDS = 60
RETENTION_DAYS = 7


class _PostgresBacked:
    """Lazy, self-healing Postgres connection shared by the optional DB-backed stores."""

    def __init__(self, disabled_msg: str):
        self._conn = None
        self._lock = threading.Lock()  # Agent cycle and manual scans run in separate threads
        self._last_connect_attempt = 0.0
        if not settings.DATABASE_URL:
            logger.info(disabled_msg)

    def _connection(self):
        if self._conn is not None and not self._conn.closed:
            return self._conn
        if not settings.DATABASE_URL or time.time() - self._last_connect_attempt < RECONNECT_INTERVAL_SECONDS:
            return None
        self._last_connect_attempt = time.time()
        try:
            import psycopg

            self._conn = psycopg.connect(settings.DATABASE_URL, autocommit=True, connect_timeout=5)
            self._conn.execute(SCHEMA_SQL)
            logger.info(f"Connected to Postgres ({type(self).__name__}).")
            return self._conn
        except Exception as e:
            logger.warning(f"Postgres unavailable for {type(self).__name__} (retry in {RECONNECT_INTERVAL_SECONDS}s): {e}")
            self._conn = None
            return None


class LLMResultCache(_PostgresBacked):
    """
    Postgres-backed cache of LLM sentiment results, keyed by a hash of the prompt inputs.
    Optional: without DATABASE_URL (or with the DB down) every lookup misses and the agent calls the LLM as before.
    """

    def __init__(self):
        super().__init__("DATABASE_URL not set; LLM result caching disabled.")

    def get(self, cache_key: str, max_age_minutes: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connection()
            if conn is None:
                return None
            try:
                row = conn.execute(
                    "SELECT result FROM llm_sentiment_cache "
                    "WHERE cache_key = %s AND created_at > now() - make_interval(mins => %s)",
                    (cache_key, max_age_minutes),
                ).fetchone()
                return row[0] if row else None
            except Exception as e:
                logger.warning(f"LLM cache read failed: {e}")
                self._conn = None
                return None

    def set(self, cache_key: str, instrument: str, model: str, result: Dict[str, Any]):
        with self._lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                conn.execute(
                    "INSERT INTO llm_sentiment_cache (cache_key, instrument, model, result) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (cache_key) DO UPDATE SET result = EXCLUDED.result, created_at = now()",
                    (cache_key, instrument, model, json.dumps(result)),
                )
                conn.execute(
                    "DELETE FROM llm_sentiment_cache WHERE created_at < now() - make_interval(days => %s)",
                    (RETENTION_DAYS,),
                )
            except Exception as e:
                logger.warning(f"LLM cache write failed: {e}")
                self._conn = None


class AgentStateStore(_PostgresBacked):
    """
    Small key/value store so agent state (run/pause switch, daily loss baseline) survives container restarts.
    Optional: without DATABASE_URL (or with the DB down) reads return the default and writes are dropped.
    """

    def __init__(self):
        super().__init__("DATABASE_URL not set; agent state will not persist across restarts.")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            conn = self._connection()
            if conn is None:
                return default
            try:
                row = conn.execute("SELECT value FROM agent_state WHERE key = %s", (key,)).fetchone()
                return row[0] if row else default
            except Exception as e:
                logger.warning(f"Agent state read failed ({key}): {e}")
                self._conn = None
                return default

    def set(self, key: str, value: Any):
        with self._lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                conn.execute(
                    "INSERT INTO agent_state (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                    (key, json.dumps(value)),
                )
            except Exception as e:
                logger.warning(f"Agent state write failed ({key}): {e}")
                self._conn = None


class SignalJournal(_PostgresBacked):
    """
    One row per technical BUY/SELL signal per entry bar, with the sentiment score and what the agent decided.
    Lets us measure later whether the LLM filter improves outcomes (see `python backtest.py journal`).
    A later decision on the same bar overwrites an earlier one, except an executed trade is never overwritten.
    """

    def __init__(self):
        super().__init__("DATABASE_URL not set; signal journal disabled.")

    def record(self, row: Dict[str, Any]):
        with self._lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                conn.execute(
                    "INSERT INTO signal_journal (instrument, entry_tf, bar_time, direction, tech_reason, price, atr, "
                    "sentiment_score, high_impact, decision, details, sl_atr, tp_atr, confidence) "
                    "VALUES (%(instrument)s, %(entry_tf)s, %(bar_time)s, %(direction)s, %(tech_reason)s, %(price)s, "
                    "%(atr)s, %(sentiment_score)s, %(high_impact)s, %(decision)s, %(details)s, "
                    "%(sl_atr)s, %(tp_atr)s, %(confidence)s) "
                    "ON CONFLICT (instrument, entry_tf, bar_time) DO UPDATE SET "
                    "sentiment_score = EXCLUDED.sentiment_score, high_impact = EXCLUDED.high_impact, "
                    "decision = EXCLUDED.decision, details = EXCLUDED.details, recorded_at = now() "
                    "WHERE signal_journal.decision NOT LIKE 'EXECUTED%%'",
                    row,
                )
            except Exception as e:
                logger.warning(f"Signal journal write failed: {e}")
                self._conn = None

    def fetch_all(self) -> list:
        with self._lock:
            conn = self._connection()
            if conn is None:
                return []
            try:
                cur = conn.execute("SELECT * FROM signal_journal ORDER BY bar_time")
                cols = [c.name for c in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception as e:
                logger.warning(f"Signal journal read failed: {e}")
                self._conn = None
                return []


class SessionStore(_PostgresBacked):
    """Persists dashboard login sessions (as SHA-256 hashes of the tokens) so a restart doesn't log you out."""

    def __init__(self):
        super().__init__("DATABASE_URL not set; dashboard sessions will not survive restarts.")

    def save(self, token_hash: str, expires_at: float):
        with self._lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                conn.execute(
                    "INSERT INTO dashboard_sessions (token_hash, expires_at) VALUES (%s, %s) "
                    "ON CONFLICT (token_hash) DO UPDATE SET expires_at = EXCLUDED.expires_at",
                    (token_hash, expires_at),
                )
                conn.execute("DELETE FROM dashboard_sessions WHERE expires_at < %s", (time.time(),))
            except Exception as e:
                logger.warning(f"Session save failed: {e}")
                self._conn = None

    def lookup(self, token_hash: str) -> Optional[float]:
        with self._lock:
            conn = self._connection()
            if conn is None:
                return None
            try:
                row = conn.execute(
                    "SELECT expires_at FROM dashboard_sessions WHERE token_hash = %s", (token_hash,)
                ).fetchone()
                return float(row[0]) if row else None
            except Exception as e:
                logger.warning(f"Session lookup failed: {e}")
                self._conn = None
                return None

    def delete(self, token_hash: str):
        with self._lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                conn.execute("DELETE FROM dashboard_sessions WHERE token_hash = %s", (token_hash,))
            except Exception as e:
                logger.warning(f"Session delete failed: {e}")
                self._conn = None
