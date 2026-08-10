"""Free collector for Trump's Truth Social posts via trumpstruth.org, a
public, actively-maintained RSS archive (see config.TRUTH_SOCIAL_RSS).

This is more reliable than Nitter for this specific account since Trump
posts primarily on Truth Social rather than X during his second term."""

import logging
import feedparser

from ..db import NewsItem
from .. import config

logger = logging.getLogger(__name__)


def fetch_all() -> list[NewsItem]:
    url = getattr(config, "TRUTH_SOCIAL_RSS", "")
    if not url:
        return []

    items = []
    try:
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            logger.warning("Truth Social archive feed unreachable: %s", url)
            return []
        for entry in parsed.entries:
            items.append(
                NewsItem(
                    source="truthsocial:realDonaldTrump",
                    category="twitter",  # kept in same category bucket as other social posts
                    title=getattr(entry, "title", "")[:200],
                    body=getattr(entry, "summary", ""),
                    url=getattr(entry, "link", ""),
                )
            )
    except Exception as e:
        logger.warning("Truth Social collector failed: %s", e)

    return items
