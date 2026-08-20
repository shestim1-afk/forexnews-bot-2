import sqlite3
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from . import config


@dataclass
class NewsItem:
    source: str            # e.g. "forexlive", "twitter:elonmusk", "federal_reserve"
    category: str          # forex_macro | geopolitics_war | energy | central_banks | twitter
    title: str
    body: str
    url: str
    published: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def dedup_hash(self) -> str:
        raw = f"{self.source}|{self.url or self.title}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _connect():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_items (
            hash TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            seen_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT,
            direction TEXT,
            confidence REAL,
            reason TEXT,
            source_title TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            action TEXT,
            entry REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            confidence REAL,
            details TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS telegram_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    return conn


def count_seen() -> int:
    """Used to detect whether this is the bot's very first run ever (empty
    DB) so main.py can skip classifying old backlog news on first launch."""
    conn = _connect()
    try:
        cur = conn.execute("SELECT COUNT(*) FROM seen_items")
        return cur.fetchone()[0]
    finally:
        conn.close()


def is_new(item: NewsItem) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("SELECT 1 FROM seen_items WHERE hash = ?", (item.dedup_hash,))
        return cur.fetchone() is None
    finally:
        conn.close()


def mark_seen(item: NewsItem) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO seen_items (hash, source, title, seen_at) VALUES (?, ?, ?, ?)",
            (item.dedup_hash, item.source, item.title, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def save_classification(asset: str, direction: str, confidence: float, reason: str, source_title: str) -> None:
    """Persists every meaningful (asset, direction, confidence) impact from
    news classification -- not just ones that crossed the alert threshold --
    so other tools (like the BTC scalp bot) can factor recent news sentiment
    into their own scoring, and so we build a history for later analysis."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO classifications (asset, direction, confidence, reason, source_title, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (asset, direction, confidence, reason, source_title, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_classifications(asset_keyword: str, hours: int = 2) -> list[dict]:
    """Returns classification rows where the asset name contains the given
    keyword (case-insensitive), from the last N hours."""
    conn = _connect()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = conn.execute(
            """SELECT asset, direction, confidence, reason, source_title, created_at
               FROM classifications
               WHERE asset LIKE ? AND created_at >= ?
               ORDER BY created_at DESC""",
            (f"%{asset_keyword}%", cutoff),
        )
        cols = ["asset", "direction", "confidence", "reason", "source_title", "created_at"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def save_scalp_signal(symbol: str, action: str, entry: float | None, sl: float | None,
                       tp1: float | None, tp2: float | None, confidence: float, details: str) -> None:
    """Logs every scalp signal generated (including NO TRADE calls) with a
    timestamp, so that once enough signals accumulate, we can go back and
    check what price actually did afterward -- the only way to find out if
    the confidence score means anything real."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO scalp_signals (symbol, action, entry, sl, tp1, tp2, confidence, details, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, action, entry, sl, tp1, tp2, confidence, details, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
        
    def get_unevaluated_signals(min_age_hours: int = 4, max_age_hours: int = 48) -> list[dict]:
    """Returns actionable (LONG/SHORT) signals old enough to have likely
    resolved, that don't have an outcome recorded yet. max_age_hours caps how
    far back we look, so a huge backlog can't overwhelm one run."""
    conn = _connect()
    try:
        cutoff_min = (datetime.now(timezone.utc) - timedelta(hours=min_age_hours)).isoformat()
        cutoff_max = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        cur = conn.execute(
            """SELECT id, symbol, action, entry, sl, tp1, tp2, confidence, created_at
               FROM scalp_signals
               WHERE action != 'NO TRADE'
                 AND created_at <= ?
                 AND created_at >= ?
                 AND id NOT IN (SELECT signal_id FROM signal_outcomes)
               ORDER BY created_at ASC
               LIMIT 30""",
            (cutoff_min, cutoff_max),
        )
        cols = ["id", "symbol", "action", "entry", "sl", "tp1", "tp2", "confidence", "created_at"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def save_signal_outcome(signal_id: int, outcome: str, exit_price: float, r_multiple: float) -> None:
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO signal_outcomes (signal_id, outcome, exit_price, r_multiple, evaluated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (signal_id, outcome, exit_price, r_multiple, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_outcome_stats() -> list[dict]:
    """Aggregate win/loss stats per symbol, across all evaluated signals."""
    conn = _connect()
    try:
        cur = conn.execute(
            """SELECT s.symbol, o.outcome, o.r_multiple
               FROM signal_outcomes o
               JOIN scalp_signals s ON s.id = o.signal_id"""
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    by_symbol: dict[str, dict] = {}
    for symbol, outcome, r in rows:
        b = by_symbol.setdefault(symbol, {"wins": 0, "losses": 0, "expired": 0, "r_sum": 0.0, "resolved": 0})
        if outcome == "WIN":
            b["wins"] += 1
            b["r_sum"] += r
            b["resolved"] += 1
        elif outcome == "LOSS":
            b["losses"] += 1
            b["r_sum"] += r
            b["resolved"] += 1
        else:
            b["expired"] += 1

    result = []
    for symbol, b in by_symbol.items():
        win_rate = b["wins"] / b["resolved"] if b["resolved"] else None
        avg_r = b["r_sum"] / b["resolved"] if b["resolved"] else None
        result.append({
            "symbol": symbol, "wins": b["wins"], "losses": b["losses"], "expired": b["expired"],
            "win_rate": win_rate, "avg_r": avg_r,
        })
    return result

