"""SC 13D Individual-Filer Stake Disclosure Backtest -- frozen specification.

HYPOTHESIS: buying a stock after an INDIVIDUAL (not institutional) files
an initial Schedule 13D (active/activist-intent 5%+ stake disclosure)
may produce excess returns over the following month -- the mechanism
behind the Markiplier/GoPro case, tested at scale rather than as a
single anecdote.

WEAKER evidentiary grounding than the insider-cluster or Congress
studies -- based on one dramatic anecdote, not established academic
literature.

FROZEN PARAMETERS:
- Signal: INITIAL "SCHEDULE 13D" filings only -- excludes "SCHEDULE
  13D/A" amendments, and does not include 13G (passive holders, mostly
  institutions).
- Individual-filer classification: the FILED BY name (fetched from each
  filing's real header, not the daily index -- confirmed via direct
  diagnostic that the daily index inconsistently lists either the filer
  or subject company) does NOT contain any institutional keyword.
- Entry: the next trading day's close after the filing date.
- Holding period: 21 trading days. Costs: 0.10% round-trip.
- Processing cap: a frozen maximum number of filings per run, reported
  explicitly if hit.
- Dev: 2025-01-01 to 2025-08-31. OOS: 2025-09-01 to 2025-12-31.
- Falsification: net expectancy must beat a matched random-date
  benchmark, adequate sample (>=30 signals), not concentration-driven,
  survives OOS unchanged.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

from . import config
from .historical_backtest import fetch_paginated_history
from .stake_disclosure_probe import check_daily_index, parse_daily_index_line, fetch_filing_header_detail, sanitize_for_telegram, SEC_USER_AGENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sc13d_backtest")

HOLDING_PERIOD_TRADING_DAYS = 21
SPREAD_PCT = 0.10
RANDOM_DATES_PER_TICKER = 5
MAX_FILINGS_PER_RUN = 2000
SEC_REQUEST_DELAY_SECONDS = 0.15

DEV_START, DEV_END = "2025-01-01", "2025-09-01"
OOS_START, OOS_END = "2025-09-01", "2026-01-01"

INSTITUTIONAL_KEYWORDS = [
    "LLC", "L.L.C", "LP", "L.P", "INC", "CORP", "FUND", "CAPITAL", "MANAGEMENT",
    "PARTNERS", "TRUST", "GROUP", "ADVISORS", "ADVISERS", "HOLDINGS", "LTD",
    "CO.", "COMPANY", "ASSOCIATES", "VENTURES", "INVESTMENTS", "SECURITIES",
    "ASSET", "WEALTH", "FINANCIAL",
]


def is_individual_filer(name: str) -> bool:
    if not name:
        return False
    name_upper = name.upper()
    return not any(kw in name_upper for kw in INSTITUTIONAL_KEYWORDS)


_ticker_map_cache: dict[int, str] | None = None


def fetch_ticker_from_cik(cik: str) -> str | None:
    """FIX: this was re-downloading SEC's entire ticker mapping file
    (a multi-thousand-entry JSON) from scratch on EVERY call -- if 50+
    unique companies needed a ticker lookup, that meant 50+ redundant
    full-file downloads. Now fetched exactly once per run and cached at
    module level."""
    global _ticker_map_cache
    if _ticker_map_cache is None:
        try:
            r = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": SEC_USER_AGENT}, timeout=20,
            )
            if r.status_code != 200:
                _ticker_map_cache = {}
            else:
                data = r.json()
                _ticker_map_cache = {v["cik_str"]: v["ticker"].upper() for v in data.values()}
        except Exception as e:
            logger.warning("Failed to fetch SEC ticker map: %s", e)
            _ticker_map_cache = {}

    try:
        return _ticker_map_cache.get(int(cik))
    except (ValueError, TypeError):
        return None


def compute_signal_return(df_daily: pd.DataFrame, filing_date: str) -> dict | None:
    filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
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
    return {"net_return_pct": gross_return_pct - SPREAD_PCT}


def sample_random_pool_returns(df_daily: pd.DataFrame, n_samples: int, seed_offset: int = 0) -> list[float]:
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
        returns.append(100 * (exit_price - entry_price) / entry_price - SPREAD_PCT)
    return returns


def random_benchmark(pool: list[float], n_signals: int, n_repeats: int = 200, seed: int = 42) -> dict:
    rng = random.Random(seed)
    if n_signals <= 0 or n_signals > len(pool):
        return {"mean_net_return_pct": None}
    means = [sum(rng.sample(pool, n_signals)) / n_signals for _ in range(n_repeats)]
    return {"mean_net_return_pct": sum(means) / len(means)}


def check_concentration(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"concentrated": False, "top_20pct_share": None}
    total_positive = sum(v for v in values if v > 0)
    if total_positive <= 0:
        return {"concentrated": False, "top_20pct_share": None}
    top_count = max(1, int(n * 0.2))
    top_sum = sum(sorted(values, reverse=True)[:top_count])
    share = top_sum / total_positive
    return {"concentrated": share > 0.80, "top_20pct_share": share}


def _send_telegram_direct(text: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Cannot send: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=20,
        )
        if r.status_code == 200 and r.json().get("ok") is True:
            return True
        logger.error("Telegram send failed: HTTP %d, response: %s", r.status_code, r.text[:300])
        return False
    except Exception as e:
        logger.error("Telegram send raised an exception: %s", e)
        return False


async def run_period(period: str = "dev"):
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)
    start_dt, end_dt = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")

    lines = [
        f"*SC 13D Individual-Filer Backtest ({period.upper()}): {start_date} to {end_date}*",
        "Signal: INITIAL Schedule 13D only (excludes 13D/A amendments and 13G). Individual filers only "
        "(name-keyword heuristic, checked against each filing's real header). 21-day hold, 0.10% cost.\n",
    ]

    logger.info("Scanning daily index for SC 13D filings across the period...")
    all_13d_lines = []
    day_count = 0
    current = start_dt
    while current < end_dt:
        try:
            quarter = (current.month - 1) // 3 + 1
            url = f"https://www.sec.gov/Archives/edgar/daily-index/{current.year}/QTR{quarter}/form.{current.strftime('%Y%m%d')}.idx"
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
            if r.status_code == 200:
                day_count += 1
                for l in r.text.splitlines():
                    stripped = l.strip()
                    if stripped.startswith("SCHEDULE 13D") and not stripped.startswith("SCHEDULE 13D/A"):
                        all_13d_lines.append((l, current.strftime("%Y-%m-%d")))
        except Exception as e:
            logger.warning("Failed to scan %s: %s", current, e)
        time.sleep(SEC_REQUEST_DELAY_SECONDS)
        current += timedelta(days=1)

    lines.append(f"Filing days scanned: {day_count}, initial SC 13D filings found: {len(all_13d_lines)}")

    capped = False
    if len(all_13d_lines) > MAX_FILINGS_PER_RUN:
        all_13d_lines = all_13d_lines[:MAX_FILINGS_PER_RUN]
        capped = True
        lines.append(f"PROCESSING CAP HIT -- limited to the first {MAX_FILINGS_PER_RUN} filings (frozen cap, reported explicitly).")

    individual_signals = []
    n_processed = 0
    n_fetch_failed = 0
    n_no_filed_by = 0
    n_institutional = 0
    n_individual_no_subject_cik = 0
    n_individual = 0
    sample_institutional_names = []
    sample_no_subject_cik_names = []
    raw_subject_section_sample = None

    for line, filing_date in all_13d_lines:
        parsed = parse_daily_index_line(line)
        if not parsed:
            continue
        detail = fetch_filing_header_detail(parsed["cik"], parsed["accession"])
        time.sleep(SEC_REQUEST_DELAY_SECONDS)
        n_processed += 1

        if not detail:
            n_fetch_failed += 1
            continue
        if not detail.get("filed_by"):
            n_no_filed_by += 1
            continue

        # FIX: previously "individual AND has subject_cik" was one
        # combined condition -- a correctly-identified individual filer
        # whose subject_cik simply failed to extract was silently lumped
        # into the SAME bucket as genuine institutions, confirmed by real
        # names ("Carter Denise P.", "Jeldi Arun", "Myers Michael") that
        # is_individual_filer() itself classifies correctly in isolation
        # showing up in the "institutional" sample. These are now tracked
        # as two genuinely separate, honestly labeled outcomes.
        if not is_individual_filer(detail["filed_by"]):
            n_institutional += 1
            if len(sample_institutional_names) < 5:
                sample_institutional_names.append(detail["filed_by"])
        elif not detail.get("subject_cik"):
            n_individual_no_subject_cik += 1
            if len(sample_no_subject_cik_names) < 5:
                sample_no_subject_cik_names.append(detail["filed_by"])
            if raw_subject_section_sample is None and detail.get("subject_section_raw"):
                raw_subject_section_sample = detail["subject_section_raw"]
        else:
            n_individual += 1
            individual_signals.append({"subject_cik": detail["subject_cik"], "filing_date": filing_date})

    lines.append(
        f"Filings processed: {n_processed} -- fetch failed: {n_fetch_failed}, no filer name extracted: {n_no_filed_by}, "
        f"classified institutional: {n_institutional}, individual but missing subject company CIK: {n_individual_no_subject_cik}, "
        f"classified individual (usable signal): {n_individual}"
    )
    if sample_institutional_names:
        lines.append(f"Sample classified institutional: {sanitize_for_telegram(str(sample_institutional_names))}")
    if sample_no_subject_cik_names:
        lines.append(f"Sample individual but missing subject CIK: {sanitize_for_telegram(str(sample_no_subject_cik_names))}")
    if raw_subject_section_sample:
        lines.append(f"Raw SUBJECT COMPANY section (real content, for direct inspection): {sanitize_for_telegram(raw_subject_section_sample)}")

    if n_individual == 0:
        lines.append("\n*CLASSIFICATION: INCONCLUSIVE -- no individual-filer signals found.*")
        _send_telegram_direct("\n".join(lines))
        return

    cik_to_ticker: dict[str, str | None] = {}
    all_signal_returns = []
    all_pool_returns = []
    n_no_ticker = 0
    n_no_price = 0

    price_cache: dict[str, pd.DataFrame | None] = {}

    for sig in individual_signals:
        cik = sig["subject_cik"]
        if cik not in cik_to_ticker:
            cik_to_ticker[cik] = fetch_ticker_from_cik(cik)
            time.sleep(SEC_REQUEST_DELAY_SECONDS)
        ticker = cik_to_ticker[cik]
        if not ticker:
            n_no_ticker += 1
            continue

        # FIX: price data was previously re-fetched for every individual
        # signal, even when the same ticker appeared multiple times
        # across the year (common) -- for tickers with no Twelve Data
        # coverage, this meant repeating a full 3-retry failure cycle
        # (confirmed in a real run: ~30-90 seconds wasted per repeat
        # occurrence) every single time that ticker showed up again.
        # Now fetched exactly once per unique ticker, cached (including
        # the "no data" case, so a known-bad ticker never triggers a
        # second retry cycle either).
        if ticker not in price_cache:
            price_cache[ticker] = fetch_paginated_history(
                ticker, "1day",
                start_dt - timedelta(days=10),
                end_dt + timedelta(days=60),
            )
            # FIX: this loop had NO rate-limiting delay between different
            # tickers' price fetches at all -- confirmed by a real run
            # where EVERY fetch failed with 429 Too Many Requests (not
            # 404 Not Found), meaning Twelve Data's documented free-tier
            # limit (8 requests/minute) was being hit almost immediately
            # once more than a handful of unique tickers were involved.
            # Worse than just slow: a persistent 429 was being silently
            # treated the same as a persistent 404 ("no data, real depth
            # limit") by the shared retry logic, meaning legitimate
            # tickers with real data may have been wrongly excluded.
            # 8 seconds between NEW (uncached) ticker fetches keeps this
            # loop within the documented rate limit.
            time.sleep(8.0)
        df = price_cache[ticker]
        if df is None or len(df) == 0:
            n_no_price += 1
            continue

        result = compute_signal_return(df, sig["filing_date"])
        if result:
            all_signal_returns.append(result["net_return_pct"])
        all_pool_returns.extend(sample_random_pool_returns(df, RANDOM_DATES_PER_TICKER, seed_offset=hash(ticker) % (2**31)))

    n = len(all_signal_returns)
    lines.append(f"No ticker found: {n_no_ticker}, no price data: {n_no_price}, resolved signals: {n}")

    if n == 0:
        lines.append("\n*CLASSIFICATION: INCONCLUSIVE -- no resolved signals with price data.*")
        _send_telegram_direct("\n".join(lines))
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
        failed = []
        if avg_net_return <= 0:
            failed.append("not net positive")
        if not beats_benchmark:
            failed.append("does not beat random-date benchmark")
        if not not_concentrated:
            failed.append("concentration-driven")
        classification = "FAILED"
        reason = "Failed: " + ", ".join(failed)

    lines.append(f"\n*CLASSIFICATION: {classification}*")
    lines.append(reason)
    if capped:
        lines.append("NOTE: the processing cap was hit -- this result covers a partial sample of the true period, not the full one.")

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent SC 13D backtest report (%s), classification=%s", period, classification)


if __name__ == "__main__":
    asyncio.run(run_period("dev"))
