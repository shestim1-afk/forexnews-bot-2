"""Free-tier Twitter/X collector.

X's free API tier does not support reading tweets at usable volume, so this
collector uses free alternatives instead:

  1. Nitter public instances (unofficial, RSS-based, free) -- unreliable,
     instances rotate/die often, so we try a list and skip silently if all
     are down.
  2. RSS-Bridge (self-hosted, free) -- more stable if you run your own
     instance (e.g. via `docker run -p 3000:80 rssbridge/rss-bridge`).
     Set RSSBRIDGE_URL in config/.env to use this instead of/alongside
     Nitter.

Nothing here scrapes x.com directly -- both paths consume RSS feeds that
those tools generate on your behalf, which is far less likely to get an IP
banned than direct scraping.
"""

import logging
import feedparser

from ..db import NewsItem
from .. import config

logger = logging.getLogger(__name__)

RSSBRIDGE_URL = getattr(config, "RSSBRIDGE_URL", "")  # e.g. http://localhost:3000


def _try_nitter(username: str) -> list[NewsItem]:
    items = []
    for instance in config.NITTER_INSTANCES:
        url = f"{instance}/{username}/rss"
        try:
            parsed = feedparser.parse(url)
            if parsed.bozo and not parsed.entries:
                continue  # try next instance
            for entry in parsed.entries:
                items.append(
                    NewsItem(
                        source=f"twitter:{username}",
                        category="twitter",
                        title=getattr(entry, "title", ""),
                        body=getattr(entry, "summary", ""),
                        url=getattr(entry, "link", ""),
                    )
                )
            if items:
                return items  # this instance worked, no need to try others
        except Exception as e:
            logger.debug("Nitter instance %s failed for %s: %s", instance, username, e)
            continue
    return items


def _try_rssbridge(username: str) -> list[NewsItem]:
    if not RSSBRIDGE_URL:
        return []
    url = f"{RSSBRIDGE_URL}/?action=display&bridge=Twitter&context=By+username&u={username}&format=Atom"
    items = []
    try:
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            items.append(
                NewsItem(
                    source=f"twitter:{username}",
                    category="twitter",
                    title=getattr(entry, "title", ""),
                    body=getattr(entry, "summary", ""),
                    url=getattr(entry, "link", ""),
                )
            )
    except Exception as e:
        logger.debug("RSS-Bridge failed for %s: %s", username, e)
    return items


def fetch_all() -> list[NewsItem]:
    """Try RSS-Bridge first (more stable if self-hosted), fall back to
    Nitter public instances. If both fail for a given account, log a
    warning and move on -- Twitter being unreachable should never crash
    the bot."""
    all_items: list[NewsItem] = []

    for username in config.TWITTER_WATCHLIST:
        items = _try_rssbridge(username) or _try_nitter(username)
        if not items:
            logger.warning(
                "No Twitter data for @%s -- all free sources unavailable this cycle", username
            )
        all_items.extend(items)

    return all_items
