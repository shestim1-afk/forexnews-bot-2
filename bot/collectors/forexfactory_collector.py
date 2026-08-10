"""Free collector for Forex Factory's economic calendar, via the official
public JSON feed that Forex Factory itself publishes for third-party tools
(the same one countless MetaTrader news indicators use). No login, no
scraping -- this is data Forex Factory intends for this kind of use.

Docs/community references confirm this feed is rate-limited to roughly 2
requests per 5 minutes -- since this bot polls once per 5-minute cycle, a
single request per cycle stays comfortably within that limit.

The feed returns the *entire current week's* calendar every time, so this
collector filters down to only events whose scheduled time falls within the
last RECENT_WINDOW_MINUTES -- i.e. things that just happened, which is what
actually matters for a reactive news bot (a rate cut decision, a CPI print,
etc. landing right now vs. something 4 days away)."""

import logging
from datetime import datetime, timezone, timedelta
import requests

from ..db import NewsItem
from .. import config

logger = logging.getLogger(__name__)

RECENT_WINDOW_MINUTES = 20  # only alert on events that just printed
MIN_IMPACT = {"High", "Medium"}  # skip Low-impact/noise events entirely


def fetch_all() -> list[NewsItem]:
    url = getattr(config, "FF_CALENDAR_URL", "")
    if not url:
        return []

    items = []
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        logger.warning("Forex Factory calendar fetch failed: %s", e)
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=RECENT_WINDOW_MINUTES)

    for e in events:
        impact = e.get("impact", "")
        if impact not in MIN_IMPACT:
            continue

        try:
            event_time = datetime.fromisoformat(e["date"])
        except Exception:
            continue

        if not (cutoff <= event_time <= now):
            continue

        title = e.get("title", "")
        country = e.get("country", "")
        forecast = e.get("forecast", "")
        previous = e.get("previous", "")
        actual = e.get("actual", "")

        body = f"{country} {title} -- impact: {impact}."
        if actual:
            body += f" Actual: {actual}, forecast: {forecast}, previous: {previous}."
        else:
            body += f" Forecast: {forecast}, previous: {previous} (actual not yet posted)."

        dedup_key = f"{country}-{title}-{e['date']}"

        items.append(
            NewsItem(
                source="forexfactory_calendar",
                category="forex_macro",
                title=f"{country} {impact} impact: {title}",
                body=body,
                url=f"https://www.forexfactory.com/calendar#{dedup_key}",
                published=event_time,
            )
        )

    return items
