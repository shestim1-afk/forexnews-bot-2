"""SEC Form 4 Insider Cluster-Buying -- frozen specification (Phase 2).

HYPOTHESIS: when 3+ distinct insiders at the same company file open-market
PURCHASE transactions (SEC Form 4, transaction code 'P' -- NOT option
exercises, grants, or scheduled 10b5-1 plan trades, which the cited
academic literature identifies as "routine" with no predictive power)
within a rolling 5-day window, the stock may show excess forward returns
over the following month. This is the frozen, pre-declared specification
approved before any code was written.

IMPORTANT, stated honestly: SEC's data.sec.gov and www.sec.gov hosts were
both unreachable from the development sandbox (a local network-allowlist
limitation, not a real SEC-side block -- both are standard,
unauthenticated public government endpoints). The Form 4 XML parsing
logic below is built from the well-documented, stable, long-standing SEC
schema, but could not be verified against a live filing in advance. Its
real behavior is confirmed only when this runs for real in GitHub
Actions -- exactly the same caveat already given for the Binance funding
-rate probe and the SEC EDGAR check in congress_insider_probe.py.

UNIVERSE (a real, frozen specification choice, not an incidental detail):
restricted to a curated list of ~50 large, liquid US equities across
multiple sectors -- NOT the full market. A result here speaks only to
large-cap insider clustering, not the broader market. This scoping was
agreed BEFORE any code was written, for genuine infrastructure/rate-limit
reasons (scanning the full US equity market's Form 4 filings is not
tractable in a single run), not to rescue a disappointing result.

FROZEN PARAMETERS:
- Cluster: >=3 distinct insiders (by unique reporting-owner CIK, not by
  transaction count), each filing an open-market PURCHASE (transaction
  code 'P'), at the same issuer, within a rolling 5-day window.
- Entry: the next trading day after the cluster-completing filing's own
  filing date (not the transaction date -- the filing date is what is
  actually, publicly knowable at that point; using transaction date would
  be lookahead, since Form 4 can be filed up to 2 business days after the
  trade).
- Holding period: fixed 21 trading days.
- Position sizing: equal-weighted per signal.
- Costs: 0.10% round-trip (spread + commission), a conservative equity
  assumption -- open to revision if real broker data suggests otherwise.
- Dev: 2025-01-01 to 2025-08-31. OOS: 2025-09-01 to 2025-12-31, untouched
  until development is frozen.
- Falsification: net expectancy must beat a matched random-stock
  benchmark (same methodology as the ML study's random-filter control),
  adequate sample (>=30 signals), not concentration-driven, survives OOS
  unchanged.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

from . import telegram_bot
from .historical_backtest import fetch_paginated_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("insider_cluster")

SEC_USER_AGENT = "Research Probe research-probe@example.com"
SEC_REQUEST_DELAY_SECONDS = 0.15

CLUSTER_MIN_INSIDERS = 3
CLUSTER_WINDOW_DAYS = 5
HOLDING_PERIOD_TRADING_DAYS = 21
SPREAD_PCT = 0.10

DEV_START, DEV_END = "2025-01-01", "2025-09-01"
OOS_START, OOS_END = "2025-09-01", "2026-01-01"

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP",
    "XOM", "CVX", "COP", "SLB",
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "RTX",
    "WMT", "PG", "KO", "PEP", "MCD", "NKE", "HD", "COST", "DIS",
    "T", "VZ", "CMCSA", "NFLX",
    "TSLA",
]


def fetch_ticker_to_cik_map() -> dict[str, str]:
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": SEC_USER_AGENT}, timeout=20,
        )
        if r.status_code != 200:
            logger.error("Failed to fetch ticker->CIK map: HTTP %d", r.status_code)
            return {}
        data = r.json()
        return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    except Exception as e:
        logger.error("Exception fetching ticker->CIK map: %s", e)
        return {}


def fetch_form4_accession_numbers(cik: str, start_date: str, end_date: str) -> list[dict]:
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers={"User-Agent": SEC_USER_AGENT}, timeout=20,
        )
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception as e:
        logger.warning("Failed to fetch submissions for CIK %s: %s", cik, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    buffered_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=CLUSTER_WINDOW_DAYS)).strftime("%Y-%m-%d")
    results = []
    for i, form in enumerate(forms):
        if form == "4" and buffered_start <= dates[i] < end_date:
            results.append({"filing_date": dates[i], "accession": accessions[i]})
    return results


def fetch_form4_transaction_detail(cik: str, accession: str) -> dict | None:
    accession_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{accession}.txt"
    try:
        r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
        if r.status_code != 200:
            return None
        content = r.text
    except Exception as e:
        logger.warning("Failed to fetch Form 4 detail for %s/%s: %s", cik, accession, e)
        return None

    owner_cik_match = re.search(r"<rptOwnerCik>(\d+)</rptOwnerCik>", content)
    owner_name_match = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", content)
    txn_code_match = re.search(r"<transactionCode>([A-Z])</transactionCode>", content)
    txn_date_match = re.search(r"<transactionDate>\s*<value>([\d-]+)</value>", content)
    shares_match = re.search(r"<transactionShares>\s*<value>([\d.]+)</value>", content)

    if not (owner_cik_match and txn_code_match and txn_date_match):
        return None

    return {
        "owner_cik": owner_cik_match.group(1),
        "owner_name": owner_name_match.group(1) if owner_name_match else "unknown",
        "transaction_code": txn_code_match.group(1),
        "transaction_date": txn_date_match.group(1),
        "shares": float(shares_match.group(1)) if shares_match else None,
    }


def detect_clusters(transactions: list[dict], ticker: str) -> list[dict]:
    """Given all parsed PURCHASE transactions for one company (already
    filtered to transaction_code == 'P' by the caller), finds every
    distinct group of >=3 unique insiders (by owner_cik) filing within a
    rolling CLUSTER_WINDOW_DAYS window. Returns one signal per cluster,
    dated at the LATEST (cluster-completing) filing_date in that group --
    never an earlier one, which would be lookahead."""
    if len(transactions) < CLUSTER_MIN_INSIDERS:
        return []

    sorted_txns = sorted(transactions, key=lambda t: t["filing_date"])
    signals = []
    used_filing_dates = set()

    for i in range(len(sorted_txns)):
        anchor_date = datetime.strptime(sorted_txns[i]["filing_date"], "%Y-%m-%d")
        window_end = anchor_date + timedelta(days=CLUSTER_WINDOW_DAYS)

        window_txns = [
            t for t in sorted_txns
            if anchor_date <= datetime.strptime(t["filing_date"], "%Y-%m-%d") <= window_end
        ]
        distinct_insiders = set(t["owner_cik"] for t in window_txns)

        if len(distinct_insiders) >= CLUSTER_MIN_INSIDERS:
            completing_date = max(t["filing_date"] for t in window_txns)
            if completing_date in used_filing_dates:
                continue  # already emitted a signal ending on this exact date
            signals.append({
                "ticker": ticker, "cluster_filing_date": completing_date,
                "n_insiders": len(distinct_insiders),
            })
            used_filing_dates.add(completing_date)

    return signals


def compute_signal_return(df_daily: pd.DataFrame, cluster_filing_date: str) -> dict | None:
    """Entry = the next TRADING day's close after cluster_filing_date
    (using the actual fetched trading-day index, not calendar days, so
    weekends/holidays are correctly skipped). Exit = HOLDING_PERIOD_
    TRADING_DAYS trading days later. Returns None if there isn't enough
    forward data yet (e.g. too close to the end of the available series)."""
    filing_dt = datetime.strptime(cluster_filing_date, "%Y-%m-%d")
    after_filing = df_daily[df_daily["datetime"] > filing_dt]
    if len(after_filing) == 0:
        return None
    entry_idx = after_filing.index[0]

    exit_idx = entry_idx + HOLDING_PERIOD_TRADING_DAYS
    if exit_idx >= len(df_daily):
        return None

    entry_price = df_daily["close"].iloc[entry_idx]
    exit_price = df_daily["close"].iloc[exit_idx]
    if entry_price == 0:
        return None

    gross_return_pct = 100 * (exit_price - entry_price) / entry_price
    net_return_pct = gross_return_pct - SPREAD_PCT
    return {
        "entry_date": df_daily["datetime"].iloc[entry_idx], "exit_date": df_daily["datetime"].iloc[exit_idx],
        "entry_price": entry_price, "exit_price": exit_price,
        "gross_return_pct": gross_return_pct, "net_return_pct": net_return_pct,
    }


def random_benchmark(all_returns_pool: list[float], n_signals: int, n_repeats: int = 200, seed: int = 42) -> dict:
    """Matches the ML study's random-filter-benchmark methodology: repeatedly
    samples n_signals returns at random from the full pool of all
    computed 21-day holding-period returns (across the whole universe,
    not just cluster-flagged ones), to see whether the cluster signal
    beats what picking randomly would have done."""
    import random
    rng = random.Random(seed)
    if n_signals <= 0 or n_signals > len(all_returns_pool):
        return {"mean_net_return_pct": None}
    means = []
    for _ in range(n_repeats):
        sample = rng.sample(all_returns_pool, n_signals)
        means.append(sum(sample) / len(sample))
    return {"mean_net_return_pct": sum(means) / len(means), "n_repeats": n_repeats}


async def run_period(period: str = "dev"):
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)

    lines = [
        f"*SEC Form 4 Insider Cluster-Buying ({period.upper()}): {start_date} to {end_date}*",
        f"Curated universe: {len(UNIVERSE)} large-cap tickers (not the full market -- a frozen scoping choice). "
        "Cluster: >=3 distinct insiders, open-market PURCHASES only, within a 5-day window. "
        "21-trading-day hold, 0.10% round-trip cost.\n",
    ]

    logger.info("Fetching ticker->CIK map...")
    ticker_to_cik = fetch_ticker_to_cik_map()
    if not ticker_to_cik:
        lines.append("*CLASSIFICATION: DATA INSUFFICIENT -- could not fetch the SEC ticker->CIK mapping.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    all_signal_returns = []
    all_pool_returns = []  # every computed 21-day return, cluster or not -- the random-benchmark pool
    n_tickers_processed = 0
    n_clusters_detected = 0

    for ticker in UNIVERSE:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            logger.warning("No CIK found for %s -- skipping", ticker)
            continue

        accessions = fetch_form4_accession_numbers(cik, start_date, end_date)
        time.sleep(SEC_REQUEST_DELAY_SECONDS)

        purchase_txns = []
        for a in accessions:
            detail = fetch_form4_transaction_detail(cik, a["accession"])
            time.sleep(SEC_REQUEST_DELAY_SECONDS)
            if detail and detail["transaction_code"] == "P":
                purchase_txns.append({"owner_cik": detail["owner_cik"], "filing_date": a["filing_date"]})

        clusters = detect_clusters(purchase_txns, ticker)
        n_clusters_detected += len(clusters)

        if not clusters:
            n_tickers_processed += 1
            continue

        df_daily = fetch_paginated_history(ticker, "1day",
                                            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30),
                                            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=45))
        if df_daily is None or len(df_daily) == 0:
            logger.warning("No price data for %s -- skipping its %d cluster(s)", ticker, len(clusters))
            n_tickers_processed += 1
            continue

        for cluster in clusters:
            result = compute_signal_return(df_daily, cluster["cluster_filing_date"])
            if result:
                all_signal_returns.append(result["net_return_pct"])
                all_pool_returns.append(result["net_return_pct"])

        n_tickers_processed += 1

    n = len(all_signal_returns)
    lines.append(f"Tickers processed: {n_tickers_processed}/{len(UNIVERSE)}, clusters detected: {n_clusters_detected}, resolved signals: {n}")

    if n == 0:
        lines.append("\n*CLASSIFICATION: INCONCLUSIVE -- no resolved cluster signals in this period.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    avg_net_return = sum(all_signal_returns) / n
    bench = random_benchmark(all_pool_returns, n) if len(all_pool_returns) >= n else {"mean_net_return_pct": None}

    lines.append(f"\nCluster-signal avg net return ({HOLDING_PERIOD_TRADING_DAYS}-day hold): {avg_net_return:+.3f}%")
    if bench["mean_net_return_pct"] is not None:
        lines.append(f"Random-benchmark avg net return (same universe, same period): {bench['mean_net_return_pct']:+.3f}%")

    adequate_sample = n >= 30
    beats_benchmark = bench["mean_net_return_pct"] is not None and avg_net_return > bench["mean_net_return_pct"]

    if not adequate_sample:
        classification = "INCONCLUSIVE"
        reason = f"Only {n} resolved signals -- below the 30-signal minimum."
    elif avg_net_return > 0 and beats_benchmark:
        classification = "DEVELOPMENT PROMISING" if period == "dev" else "OOS PROMISING"
        reason = "Beats both zero and the random-stock benchmark with an adequate sample."
    else:
        classification = "FAILED"
        reason = "Does not beat the random-stock benchmark and/or is not net positive."

    lines.append(f"\n*CLASSIFICATION: {classification}*")
    lines.append(reason)
    if period == "dev" and classification != "DEVELOPMENT PROMISING":
        lines.append("Per protocol: does not justify proceeding to OOS.")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent insider cluster-buying report (%s), classification=%s", period, classification)


if __name__ == "__main__":
    asyncio.run(run_period("dev"))
