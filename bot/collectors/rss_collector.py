"""Generic RSS collector. Works for any standard RSS/Atom feed, which covers
forex news outlets, geopolitics/energy sites, and central bank press feeds --
no scraping involved, just parsing feeds the sites publish on purpose."""

import logging
from datetime import datetime, timezone
import feedparser

from ..db import NewsItem
from .. import config

logger = logging.getLogger(__name__)


def _parse_entry(entry, source: str, category: str) -> NewsItem:
    title = getattr(entry, "title", "").strip()
    body = getattr(entry, "summary", "") or getattr(entry, "description", "")
    url = getattr(entry, "link", "")

    published = datetime.now(timezone.utc)
    if getattr(entry, "published_parsed", None):
        try:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass

    return NewsItem(
        source=source,
        category=category,
        title=title,
        body=body,
        url=url,
        published=published,
    )


def fetch_all() -> list[NewsItem]:
    """Poll every feed in config.RSS_FEEDS and return parsed NewsItems.
    Failures on individual feeds are logged and skipped, not fatal."""
    items: list[NewsItem] = []

    for category, urls in config.RSS_FEEDS.items():
        for url in urls:
            source = url.split("/")[2] if "//" in url else url
            try:
                parsed = feedparser.parse(url)
                if parsed.bozo and not parsed.entries:
                    logger.warning("Feed unreachable or malformed: %s (%s)", url, parsed.bozo_exception)
                    continue
                for entry in parsed.entries:
                    items.append(_parse_entry(entry, source, category))
            except Exception as e:
                logger.warning("Failed to fetch feed %s: %s", url, e)
                continue

    return items
