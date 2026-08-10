"""Pulls the latest print for a handful of key macro/rate series from FRED
(Federal Reserve Economic Data - free official API). This gives the bot hard
numbers (Fed funds rate, CPI, unemployment) to complement news sentiment."""

import logging
import requests
from datetime import datetime, timezone

from ..db import NewsItem
from .. import config

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> human label
SERIES = {
    "FEDFUNDS": "Effective Federal Funds Rate",
    "CPIAUCSL": "US CPI (all items)",
    "UNRATE": "US Unemployment Rate",
    "DGS10": "10-Year Treasury Yield",
}


def fetch_all() -> list[NewsItem]:
    if not config.FRED_API_KEY:
        logger.info("FRED_API_KEY not set, skipping FRED collector.")
        return []

    items: list[NewsItem] = []
    for series_id, label in SERIES.items():
        try:
            resp = requests.get(
                FRED_BASE,
                params={
                    "series_id": series_id,
                    "api_key": config.FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observations", [])
            if not obs:
                continue
            latest = obs[0]
            title = f"{label}: {latest['value']} (as of {latest['date']})"
            items.append(
                NewsItem(
                    source="fred",
                    category="central_banks",
                    title=title,
                    body=f"Latest official print for {label} from FRED.",
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    published=datetime.now(timezone.utc),
                )
            )
        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", series_id, e)
            continue

    return items
