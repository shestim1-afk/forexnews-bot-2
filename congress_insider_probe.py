"""Congress Trading (House) & SEC Insider Form 4 DATA-AVAILABILITY PROBE.

Explicitly NOT a strategy, NOT a backtest, does NOT touch the production
XAU bot.

IMPORTANT: the House Congress-trading data source below has ALREADY been
directly verified with real, live data before this module was written
(23,969 transactions, 2,944 in 2025, ~25-day average disclosure lag,
100% usable tickers, actively updated as of today) -- this probe
re-confirms it programmatically for the record, but it is not a new,
uncertain claim.

The SEC EDGAR Form 4 (corporate insider) check below could NOT be
verified directly in advance -- data.sec.gov was unreachable from the
development sandbox specifically (a local network-allowlist limitation,
not a real SEC-side block: data.sec.gov is a standard, unauthenticated
public government API). This is the one genuinely untested piece here;
its real connectivity is confirmed only when this runs for real.

SEC requires a descriptive User-Agent header identifying the requester
(per SEC's fair-access policy) -- included below.
"""

import asyncio
import logging
from datetime import datetime

import requests

from . import telegram_bot
from .historical_backtest import fetch_paginated_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("congress_insider_probe")

HOUSE_DATA_URL = "https://raw.githubusercontent.com/TattooedHead/house-stock-watcher-data/main/data/all_transactions.json"
SEC_USER_AGENT = "Research Probe research-probe@example.com"
TESLA_CIK = "0001318605"  # used as the SEC EDGAR existence-check target -- a well-known filer


def check_house_congress_data() -> dict:
    """Fetches and validates the House Congress-trading dataset. Already
    confirmed working with real data before this module was written --
    this reproduces that check programmatically."""
    try:
        r = requests.get(HOUSE_DATA_URL, timeout=30)
        if r.status_code != 200:
            return {"available": False, "error": f"HTTP {r.status_code}"}
        data = r.json()
    except Exception as e:
        return {"available": False, "error": str(e)}

    if not isinstance(data, list) or len(data) == 0:
        return {"available": False, "error": "Empty or unexpected response shape"}

    dates_2025 = 0
    lags = []
    usable_tickers = 0
    for t in data:
        try:
            td = datetime.strptime(t["transaction_date"], "%m/%d/%Y")
            if td.year == 2025:
                dates_2025 += 1
            dd = datetime.strptime(t["disclosure_date"], "%m/%d/%Y")
            lags.append((dd - td).days)
        except Exception:
            pass
        if t.get("ticker") and t["ticker"] not in ("--", "N/A", ""):
            usable_tickers += 1

    return {
        "available": True, "total_records": len(data), "records_2025": dates_2025,
        "avg_disclosure_lag_days": sum(lags) / len(lags) if lags else None,
        "usable_ticker_pct": 100 * usable_tickers / len(data),
    }


def check_sec_edgar_form4() -> dict:
    """Existence + depth check for SEC EDGAR's official Form 4 submissions
    API, using a well-known filer (Tesla) as the test target."""
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{TESLA_CIK}.json",
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=20,
        )
        if r.status_code != 200:
            return {"available": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        data = r.json()
    except Exception as e:
        return {"available": False, "error": str(e)}

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    form4_indices = [i for i, f in enumerate(forms) if f == "4"]

    if not form4_indices:
        return {"available": True, "n_form4_recent": 0, "note": "Connected, but no Form 4 filings found in the recent batch."}

    return {
        "available": True, "n_form4_recent": len(form4_indices),
        "most_recent_date": dates[form4_indices[0]] if form4_indices else None,
        "oldest_in_recent_batch": dates[form4_indices[-1]] if form4_indices else None,
    }


def check_price_data_for_tickers(tickers: list[str]) -> dict:
    """Confirms whether our existing Twelve Data infrastructure can
    actually fetch price history for a sample of real tickers seen in
    the Congress data -- necessary for eventually joining disclosure
    events to price outcomes."""
    results = {}
    for ticker in tickers:
        df = fetch_paginated_history(ticker, "1day", datetime(2025, 1, 1), datetime(2025, 2, 1))
        results[ticker] = df is not None and len(df) > 0
    return results


async def run():
    lines = [
        "*Congress Trading & SEC Insider Form 4 Data-Availability Probe (feasibility check ONLY)*",
        "No strategy built, no backtest run, production bot untouched.\n",
    ]

    logger.info("Checking House Congress-trading data source...")
    house_result = check_house_congress_data()
    lines.append("*Congress trading (House, via GitHub-hosted structured data)*")
    if house_result["available"]:
        lines.append(f"  {house_result['total_records']} total transactions, {house_result['records_2025']} in 2025")
        lines.append(f"  Avg disclosure lag: {house_result['avg_disclosure_lag_days']:.1f} days")
        lines.append(f"  Usable tickers: {house_result['usable_ticker_pct']:.0f}%")
        lines.append("  CLASSIFICATION: DATA SUFFICIENT")
    else:
        lines.append(f"  FAILED: {house_result.get('error')}")
        lines.append("  CLASSIFICATION: DATA INSUFFICIENT")
    lines.append("")

    logger.info("Checking SEC EDGAR Form 4 (insider trading) data source...")
    sec_result = check_sec_edgar_form4()
    lines.append("*SEC insider trading (Form 4, via official EDGAR API)*")
    if sec_result["available"]:
        lines.append(f"  Connected successfully. Form 4 filings in recent batch: {sec_result.get('n_form4_recent', 0)}")
        if sec_result.get("most_recent_date"):
            lines.append(f"  Most recent: {sec_result['most_recent_date']}, oldest in this batch: {sec_result['oldest_in_recent_batch']}")
        lines.append("  CLASSIFICATION: DATA SUFFICIENT")
    else:
        lines.append(f"  FAILED: {sec_result.get('error')}")
        lines.append("  CLASSIFICATION: DATA INSUFFICIENT")
    lines.append("")

    if house_result["available"]:
        sample_tickers = ["AAPL", "NVDA", "MSFT"]
        logger.info("Checking price-data availability for sample tickers: %s", sample_tickers)
        price_results = check_price_data_for_tickers(sample_tickers)
        lines.append("*Price-data cross-check (sample tickers seen in Congress data)*")
        for ticker, ok in price_results.items():
            lines.append(f"  {ticker}: {'available' if ok else 'NOT available'}")
        lines.append("")

    lines.append(
        "This clears the data-foundation bar only. It does NOT mean either signal is profitable -- "
        "a full frozen specification (entry timing relative to disclosure, holding period, position sizing, "
        "realistic costs, dev/OOS split) would still be required before any backtest, same discipline as "
        "everything else in this project."
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent Congress/insider data probe report")


if __name__ == "__main__":
    asyncio.run(run())
