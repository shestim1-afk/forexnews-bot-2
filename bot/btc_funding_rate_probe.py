"""BTC Perpetual Funding Rate DATA-AVAILABILITY FEASIBILITY PROBE.

Explicitly NOT a strategy, NOT a backtest, does NOT touch the production
XAU bot. Tests whether Binance's PUBLIC futures API (no key required)
provides sufficient historical funding-rate data to investigate a
market-neutral funding-rate-harvesting hypothesis -- a genuinely
different mechanism class from every BTC test run so far in this
project (all of which were directional: trend, momentum, mean-reversion,
S/R bounce). This would be long spot + short perpetual (or vice versa),
hedging out price risk and collecting the periodic funding payment --
distinct from any directional bet.

IMPORTANT CAVEAT, stated honestly: this uses a genuinely NEW data
source (Binance's public API) that no other module in this project has
ever called. Unlike every other probe here, actual network
connectivity to this endpoint could not be verified in advance -- it
requires the real GitHub Actions run to confirm. This module is built
and its parsing/classification logic tested against realistic mocked
responses, but the live connection itself is unverified until it runs
for real.

Binance's public funding-rate endpoint requires no authentication (it
is public market data), so this does not depend on any new secret or
API key.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import requests

from . import telegram_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("btc_funding_rate_probe")

BINANCE_FUNDING_RATE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOL = "BTCUSDT"
REQUIRED_START = "2025-01-01"
REQUIRED_END = "2025-12-31"


def fetch_funding_rate_page(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict] | None:
    """One page of Binance's public funding-rate history. Returns None
    on any failure -- never raises, so the caller can classify DATA
    INSUFFICIENT cleanly rather than crash."""
    try:
        r = requests.get(
            BINANCE_FUNDING_RATE_URL,
            params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit},
            timeout=20,
        )
        if r.status_code != 200:
            logger.warning("Binance funding-rate request failed: HTTP %d, %s", r.status_code, r.text[:300])
            return None
        data = r.json()
        if not isinstance(data, list):
            logger.warning("Unexpected response shape from Binance: %s", str(data)[:300])
            return None
        return data
    except Exception as e:
        logger.error("Binance funding-rate request raised an exception: %s", e)
        return None


def fetch_full_range(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """Paginates through Binance's 1000-record-per-call limit. Funding
    events occur every 8 hours (3/day) -- a full year is ~1095 records,
    fitting in ~2 pages."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    all_records = []
    current_start = start_ms
    max_pages = 10
    for _ in range(max_pages):
        page = fetch_funding_rate_page(symbol, current_start, end_ms)
        if page is None:
            break
        if not page:
            break
        all_records.extend(page)
        last_time = page[-1]["fundingTime"]
        if last_time >= end_ms or len(page) < 1000:
            break
        current_start = last_time + 1
    return all_records


async def run():
    lines = [
        "*BTC Perpetual Funding Rate Data-Availability Probe (feasibility check ONLY)*",
        "No strategy built, no backtest run, production bot untouched. Tests a genuinely NEW data source "
        "(Binance's public futures API) for a market-neutral hypothesis distinct from every directional BTC "
        "test run so far in this project.\n",
    ]

    logger.info("Requesting a small sample from Binance's public funding-rate endpoint...")
    sample = fetch_funding_rate_page(SYMBOL, int(datetime.now().timestamp() * 1000) - 7 * 86400000, int(datetime.now().timestamp() * 1000), limit=5)

    if sample is None or len(sample) == 0:
        lines.append("*STEP 1 -- existence check: FAILED*")
        lines.append("Could not reach or parse Binance's public funding-rate endpoint. This may be a real "
                      "connectivity block (e.g. GitHub Actions runner region restrictions) or an API change.")
        lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT*")
        await telegram_bot.send_text("\n".join(lines))
        logger.info("Sent funding-rate probe report -- existence check failed")
        return

    lines.append(f"*STEP 1 -- existence check: PASSED* ({len(sample)} recent records retrieved)")
    lines.append(f"Sample funding rates: {[r.get('fundingRate') for r in sample[:3]]}")
    lines.append("")

    logger.info("Fetching full %s to %s funding-rate history...", REQUIRED_START, REQUIRED_END)
    full_history = fetch_full_range(SYMBOL, REQUIRED_START, REQUIRED_END)

    if not full_history:
        lines.append("*STEP 2 -- historical depth check: FAILED*")
        lines.append("Existence check passed, but the full historical range request returned nothing usable.")
        lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT*")
        await telegram_bot.send_text("\n".join(lines))
        logger.info("Sent funding-rate probe report -- depth check failed")
        return

    timestamps = [datetime.fromtimestamp(r["fundingTime"] / 1000) for r in full_history]
    actual_start, actual_end = min(timestamps), max(timestamps)
    covers_dev = actual_start <= datetime.strptime("2025-01-01", "%Y-%m-%d")
    covers_oos = actual_end >= datetime.strptime("2025-12-30", "%Y-%m-%d")

    # Sanity check: genuine 8H funding should produce close to 3 records/day
    days_span = max((actual_end - actual_start).days, 1)
    expected_records = days_span * 3
    completeness_pct = 100 * len(full_history) / expected_records if expected_records > 0 else None

    lines.append(f"*STEP 2 -- historical depth check*")
    lines.append(f"{len(full_history)} funding events retrieved: {actual_start} to {actual_end}")
    lines.append(f"Covers full 2025 dev+OOS window: {'YES' if (covers_dev and covers_oos) else 'NO'}")
    if completeness_pct is not None:
        lines.append(f"Completeness vs expected 3/day 8H cadence: {completeness_pct:.0f}%")

    lines.append("")
    if covers_dev and covers_oos and completeness_pct and completeness_pct > 90:
        lines.append("*CLASSIFICATION: DATA SUFFICIENT -- a genuinely new hypothesis (funding-rate harvesting) may be proposed as Phase 2.*")
        lines.append(
            "This clears the data-foundation bar only. It does NOT mean funding-rate harvesting is profitable -- "
            "still requires a full frozen specification (position construction, funding thresholds, hedge "
            "execution assumptions, realistic costs on BOTH legs, dev/OOS split) before any backtest, same "
            "discipline as everything else in this project."
        )
    else:
        lines.append("*CLASSIFICATION: DATA INSUFFICIENT -- coverage or completeness does not meet the required standard.*")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent funding-rate probe report")


if __name__ == "__main__":
    asyncio.run(run())
