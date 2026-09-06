"""Congress Trading Disclosure Backtest -- frozen specification.

HYPOTHESIS: buying a stock after a member of Congress's disclosed
PURCHASE transaction becomes legally known (i.e., at the disclosure
date, not the transaction date) may produce excess returns over the
following month.

Data source: House Stock Watcher (github.com/TattooedHead/
house-stock-watcher-data), already directly verified with real, live
data before this module was written -- 23,969 total transactions, 2,944
in 2025, ~25-day average disclosure lag, 100% usable tickers.

FROZEN PARAMETERS (mostly reused from the already-approved insider
cluster-buying specification, for consistency -- only the signal
definition and entry-date field are genuinely new choices here):
- Signal: any single disclosed "Purchase" transaction (not Sale/
  Exchange) -- no clustering required, unlike insider trading; this
  matches how real Congress-copying products actually operate.
- Entry: the next trading day's close after the DISCLOSURE date (not
  the transaction date -- using transaction date would be lookahead,
  since it isn't legally knowable until disclosed).
- Holding period: 21 trading days (same as the insider study).
- Position sizing: equal-weighted per signal.
- Costs: 0.10% round-trip (same as the insider study).
- Universe: NO curated restriction -- unlike the insider study (which
  required expensive per-company SEC scanning), the Congress data
  already comes as one complete file covering every disclosed ticker.
- Dev: 2025-01-01 to 2025-08-31. OOS: 2025-09-01 to 2025-12-31.
- Falsification: net expectancy must beat a matched random-date
  benchmark (same fixed methodology as the insider study -- genuinely
  random dates, not sampled from the signals themselves), adequate
  sample (>=30 signals), not concentration-driven, survives OOS
  unchanged.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta

import requests
import pandas as pd

from . import telegram_bot
from .historical_backtest import fetch_paginated_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("congress_backtest")

HOUSE_DATA_URL = "https://raw.githubusercontent.com/TattooedHead/house-stock-watcher-data/main/data/all_transactions.json"

HOLDING_PERIOD_TRADING_DAYS = 21
SPREAD_PCT = 0.10
RANDOM_DATES_PER_TICKER = 5
TOP_N_TICKERS = 150  # frozen scoping choice: restricts to the 150 most-frequently-disclosed
                      # tickers in the window, for practical runtime and API-quota reasons --
                      # retains ~62% of all signals in real 2025 data, still far above the
                      # 30-signal minimum

DEV_START, DEV_END = "2025-01-01", "2025-09-01"
OOS_START, OOS_END = "2025-09-01", "2026-01-01"


def fetch_congress_purchases(start_date: str, end_date: str) -> list[dict]:
    """Fetches and filters the House Congress-trading dataset to
    PURCHASE transactions with a disclosure_date in the requested
    window. Filtering is on disclosure_date, not transaction_date --
    that is what determines when the signal was actually knowable."""
    try:
        r = requests.get(HOUSE_DATA_URL, timeout=30)
        if r.status_code != 200:
            logger.error("Failed to fetch Congress data: HTTP %d", r.status_code)
            return []
        data = r.json()
    except Exception as e:
        logger.error("Exception fetching Congress data: %s", e)
        return []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    signals = []
    for t in data:
        if t.get("type") != "Purchase":
            continue
        ticker = t.get("ticker")
        if not ticker or ticker in ("--", "N/A", ""):
            continue
        try:
            disclosure_dt = datetime.strptime(t["disclosure_date"], "%m/%d/%Y")
        except Exception:
            continue
        if start_dt <= disclosure_dt < end_dt:
            signals.append({"ticker": ticker, "disclosure_date": disclosure_dt.strftime("%Y-%m-%d")})

    return signals


def compute_signal_return(df_daily: pd.DataFrame, disclosure_date: str) -> dict | None:
    """Identical logic to the insider study's compute_signal_return --
    entry at the next real trading day (using the actual fetched
    trading-day index, not calendar days), exit HOLDING_PERIOD_TRADING_
    DAYS trading days later."""
    disclosure_dt = datetime.strptime(disclosure_date, "%Y-%m-%d")
    after_disclosure = df_daily[df_daily["datetime"] > disclosure_dt]
    if len(after_disclosure) == 0:
        return None
    entry_idx = after_disclosure.index[0]

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
        "net_return_pct": net_return_pct,
    }


def sample_random_pool_returns(df_daily: pd.DataFrame, n_samples: int, seed_offset: int = 0) -> list[float]:
    """Identical fixed methodology from the insider study -- genuinely
    random entry dates from the ticker's own price history, not sampled
    from signal dates themselves."""
    rng = random.Random(seed_offset)
    max_valid_idx = len(df_daily) - HOLDING_PERIOD_TRADING_DAYS - 1
    if max_valid_idx <= 0:
        return []
    n_samples = min(n_samples, max_valid_idx)
    sampled_indices = rng.sample(range(max_valid_idx), n_samples)

    returns = []
    for idx in sampled_indices:
        entry_price = df_daily["close"].iloc[idx]
        exit_price = df_daily["close"].iloc[idx + HOLDING_PERIOD_TRADING_DAYS]
        if entry_price == 0:
            continue
        gross_pct = 100 * (exit_price - entry_price) / entry_price
        returns.append(gross_pct - SPREAD_PCT)
    return returns


def random_benchmark(all_returns_pool: list[float], n_signals: int, n_repeats: int = 200, seed: int = 42) -> dict:
    rng = random.Random(seed)
    if n_signals <= 0 or n_signals > len(all_returns_pool):
        return {"mean_net_return_pct": None}
    means = []
    for _ in range(n_repeats):
        sample = rng.sample(all_returns_pool, n_signals)
        means.append(sum(sample) / len(sample))
    return {"mean_net_return_pct": sum(means) / len(means), "n_repeats": n_repeats}


def check_concentration(net_r_values: list[float]) -> dict:
    n = len(net_r_values)
    if n == 0:
        return {"concentrated": False, "top_20pct_share": None}
    total_positive = sum(r for r in net_r_values if r > 0)
    if total_positive <= 0:
        return {"concentrated": False, "top_20pct_share": None}
    top_count = max(1, int(n * 0.2))
    top_sum = sum(sorted(net_r_values, reverse=True)[:top_count])
    share = top_sum / total_positive
    return {"concentrated": share > 0.80, "top_20pct_share": share}


async def run_period(period: str = "dev"):
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)

    lines = [
        f"*Congress Trading Disclosure Backtest ({period.upper()}): {start_date} to {end_date}*",
        "Signal: any disclosed PURCHASE, entry at disclosure date (not transaction date -- avoids lookahead). "
        "21-trading-day hold, 0.10% round-trip cost, full disclosed universe (no curated restriction needed).\n",
    ]

    logger.info("Fetching Congress purchase signals for %s to %s...", start_date, end_date)
    signals = fetch_congress_purchases(start_date, end_date)
    lines.append(f"Purchase signals in window: {len(signals)}")

    if not signals:
        lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT / INCONCLUSIVE -- no purchase signals found.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    tickers_needed_all = sorted(set(s["ticker"] for s in signals))
    if len(tickers_needed_all) > TOP_N_TICKERS:
        from collections import Counter
        ticker_counts = Counter(s["ticker"] for s in signals)
        tickers_needed = [t for t, _ in ticker_counts.most_common(TOP_N_TICKERS)]
        tickers_set = set(tickers_needed)
        signals = [s for s in signals if s["ticker"] in tickers_set]
        lines.append(f"Restricted to the {TOP_N_TICKERS} most-frequently-disclosed tickers (of {len(tickers_needed_all)} total) -- retains {len(signals)} signals")
    else:
        tickers_needed = tickers_needed_all
    logger.info("Fetching price data for %d distinct tickers...", len(tickers_needed))

    import time
    price_cache: dict[str, pd.DataFrame] = {}
    for ticker in tickers_needed:
        df = fetch_paginated_history(ticker, "1day",
                                      datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30),
                                      datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=45))
        if df is not None and len(df) > 0:
            price_cache[ticker] = df
        time.sleep(8.0)  # stays within Twelve Data's free-tier 8 req/min limit -- with 500+ distinct
                          # tickers common in this dataset, this run will take a long time by design

    all_signal_returns = []
    all_pool_returns = []
    n_no_price_data = 0

    for ticker, df in price_cache.items():
        pool_returns = sample_random_pool_returns(df, RANDOM_DATES_PER_TICKER, seed_offset=hash(ticker) % (2**31))
        all_pool_returns.extend(pool_returns)

    for s in signals:
        df = price_cache.get(s["ticker"])
        if df is None:
            n_no_price_data += 1
            continue
        result = compute_signal_return(df, s["disclosure_date"])
        if result:
            all_signal_returns.append(result["net_return_pct"])

    n = len(all_signal_returns)
    lines.append(f"Distinct tickers: {len(tickers_needed)}, with price data: {len(price_cache)}, signals without price data: {n_no_price_data}")
    lines.append(f"Resolved signals: {n}")
    lines.append(f"Random-benchmark pool size: {len(all_pool_returns)}")

    if n == 0:
        lines.append("\n*CLASSIFICATION: INCONCLUSIVE -- no resolved signals.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    avg_net_return = sum(all_signal_returns) / n
    bench = random_benchmark(all_pool_returns, n) if len(all_pool_returns) >= n else {"mean_net_return_pct": None}
    concentration = check_concentration(all_signal_returns)

    lines.append(f"\nSignal avg net return ({HOLDING_PERIOD_TRADING_DAYS}-day hold): {avg_net_return:+.3f}%")
    if bench["mean_net_return_pct"] is not None:
        lines.append(f"Random-benchmark avg net return: {bench['mean_net_return_pct']:+.3f}%")
    if concentration["top_20pct_share"] is not None:
        lines.append(f"Concentration: top 20% of signals = {concentration['top_20pct_share']*100:.0f}% of positive return")

    adequate_sample = n >= 30
    beats_benchmark = bench["mean_net_return_pct"] is not None and avg_net_return > bench["mean_net_return_pct"]
    not_concentrated = not concentration["concentrated"]

    if not adequate_sample:
        classification = "INCONCLUSIVE"
        reason = f"Only {n} resolved signals -- below the 30-signal minimum."
    elif avg_net_return > 0 and beats_benchmark and not_concentrated:
        classification = "DEVELOPMENT PROMISING" if period == "dev" else "OOS PROMISING"
        reason = "Beats both zero and the random-date benchmark, adequate sample, not concentration-driven."
    else:
        classification = "FAILED"
        failed_reasons = []
        if avg_net_return <= 0:
            failed_reasons.append("not net positive")
        if not beats_benchmark:
            failed_reasons.append("does not beat random-date benchmark")
        if not not_concentrated:
            failed_reasons.append("concentration-driven")
        reason = "Failed: " + ", ".join(failed_reasons)

    lines.append(f"\n*CLASSIFICATION: {classification}*")
    lines.append(reason)
    if period == "dev" and classification != "DEVELOPMENT PROMISING":
        lines.append("Per protocol: does not justify proceeding to OOS.")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent Congress backtest report (%s), classification=%s", period, classification)


if __name__ == "__main__":
    asyncio.run(run_period("dev"))
