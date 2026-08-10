import sqlite3
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
