"""Broad Risk-Regime (Equity Index) -> XAU DATA-AVAILABILITY FEASIBILITY PROBE.

This is explicitly NOT a strategy, NOT a backtest, and does NOT touch the
production XAU bot. It answers one question only: is a broad equity/risk
instrument available with sufficient historical granularity to even
attempt testing whether a risk-on/risk-off regime signal provides
information genuinely independent of the existing XAU signal.

Method: identical to the DXY probe -- cheap existence checks first (a
handful of candles, minimal API cost) across several plausible symbol
conventions Twelve Data might use for a broad US equity index,
discovered empirically, not assumed. Only if a symbol proves to exist
does this probe attempt a deeper historical-depth check matching the
actual required dev+OOS window, at the same 4H granularity the XAU
system runs on.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from . import telegram_bot
from .historical_backtest import fetch_paginated_history
from . import scalp_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("risk_index_probe")

# Candidate symbol conventions -- discovered empirically, not assumed
CANDIDATE_SYMBOLS = ["SPX", "SPY", "US500", "GSPC", "SP500"]

REQUIRED_START = "2025-01-01"
REQUIRED_END = "2025-12-31"


def probe_symbol_exists(symbol: str) -> dict:
    """Cheapest possible check: request a handful of recent daily
    candles, exactly mirroring the DXY probe's method."""
    import requests
    if not scalp_analysis.TWELVEDATA_API_KEY:
        return {"symbol": symbol, "exists": False, "error": "No API key configured"}
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": symbol, "interval": "1day", "outputsize": 5, "apikey": scalp_analysis.TWELVEDATA_API_KEY},
            timeout=20,
        )
        data = r.json()
        if r.status_code == 200 and data.get("status") != "error" and "values" in data and data["values"]:
            return {
                "symbol": symbol, "exists": True,
                "sample_dates": [v["datetime"] for v in data["values"]],
                "http_status": r.status_code,
            }
        else:
            return {
                "symbol": symbol, "exists": False,
                "http_status": r.status_code,
                "error": data.get("message", data.get("code", "unknown error")),
            }
    except Exception as e:
        return {"symbol": symbol, "exists": False, "error": str(e)}


async def run():
    lines = [
        "*🔍 Broad Risk-Index Data-Availability Probe (feasibility check ONLY)*",
        "No strategy built, no backtest run, production bot untouched. This answers only: do we have usable data?\n",
    ]

    logger.info("Checking %d candidate equity-index symbol conventions...", len(CANDIDATE_SYMBOLS))
    existence_results = []
    for symbol in CANDIDATE_SYMBOLS:
        result = probe_symbol_exists(symbol)
        existence_results.append(result)
        logger.info("Symbol '%s': exists=%s (%s)", symbol, result["exists"], result.get("error", "OK"))

    lines.append("*Step 1 -- symbol existence check (cheap, 5-candle daily request per candidate)*")
    working_symbols = []
    for r in existence_results:
        if r["exists"]:
            lines.append(f"  YES `{r['symbol']}`: EXISTS, sample dates {r['sample_dates'][:2]}")
            working_symbols.append(r["symbol"])
        else:
            lines.append(f"  NO `{r['symbol']}`: {r.get('error', 'not found')}")

    if not working_symbols:
        lines.append("")
        lines.append("*CLASSIFICATION: DATA INSUFFICIENT*")
        lines.append(
            "None of the tested symbol conventions resolved on our current Twelve Data plan. "
            "This does not necessarily mean a broad equity index is unavailable at Twelve Data entirely -- it may "
            "use a symbol convention not tested here, or may require a higher plan tier (same pattern as XAG/USD). "
            "A defensible alternative would need to be explicitly proposed and documented as a separate hypothesis, "
            "not silently substituted here."
        )
        await telegram_bot.send_text("\n".join(lines))
        logger.info("Sent risk-index probe report -- no working symbol found")
        return

    test_symbol = working_symbols[0]
    lines.append("")
    lines.append(f"*Step 2 -- historical depth check ({REQUIRED_START} to {REQUIRED_END}, 4H granularity)*")
    lines.append(f"Testing symbol: `{test_symbol}`")

    target_start = datetime.strptime(REQUIRED_START, "%Y-%m-%d")
    target_end = datetime.strptime(REQUIRED_END, "%Y-%m-%d") + timedelta(days=1)
    df_4h = fetch_paginated_history(test_symbol, "4h", target_start, target_end)

    if df_4h is None or len(df_4h) == 0:
        lines.append(f"  NO Symbol exists for recent daily data, but 4H historical fetch back to {REQUIRED_START} failed or returned nothing.")
        lines.append("")
        lines.append("*CLASSIFICATION: DATA INSUFFICIENT*")
        lines.append("Symbol resolves, but does not appear to support the intraday granularity/historical depth this research needs.")
        await telegram_bot.send_text("\n".join(lines))
        logger.info("Sent risk-index probe report -- symbol exists but insufficient depth")
        return

    actual_start = df_4h["datetime"].min()
    actual_end = df_4h["datetime"].max()
    covers_dev = actual_start <= datetime.strptime("2025-01-01", "%Y-%m-%d")
    covers_oos = actual_end >= datetime.strptime("2025-12-30", "%Y-%m-%d")

    # Session-gap check: a genuine equity index has real trading-hours
    # gaps (weekends, market close) -- unlike XAU/BTC/GBPJPY's near-24/5
    # coverage. Report the actual candle count vs a naive continuous
    # expectation to make this visible rather than assumed.
    days_span = max((actual_end - actual_start).days, 1)
    naive_continuous_expected = days_span * 6  # 6 four-hour bars/day if fully continuous
    coverage_ratio = 100 * len(df_4h) / naive_continuous_expected if naive_continuous_expected > 0 else None

    lines.append(f"  {len(df_4h)} 4H candles achieved: {actual_start} to {actual_end}")
    lines.append(f"  Covers full dev period (2025-01-01 to 2025-08-31): {'YES' if covers_dev else 'NO'}")
    lines.append(f"  Covers full OOS period (2025-09-01 to 2025-12-31): {'YES' if covers_oos else 'NO'}")
    if coverage_ratio is not None:
        lines.append(f"  Candle density vs a fully-continuous 24h feed: {coverage_ratio:.0f}% -- a genuine equity index SHOULD show real gaps here (weekends, market-close hours), unlike XAU/BTC's near-continuous coverage")

    lines.append("")
    lines.append("*Answers to the required checks*")
    lines.append(f"1-2. Symbol/source: Twelve Data, `{test_symbol}` (empirically confirmed working)")
    lines.append(f"3-4. Timeframe: 4H confirmed working; daily also confirmed via the Step 1 existence check")
    lines.append(f"5-7. Historical coverage: {actual_start} to {actual_end}, {len(df_4h)} candles")
    lines.append(f"8-9. Dev+OOS coverage: {'both confirmed sufficient' if (covers_dev and covers_oos) else 'INSUFFICIENT -- see above'}")
    lines.append("10-11. Timestamp convention/timezone: same Twelve Data `time_series` format used throughout this project (UTC-based)")
    lines.append(f"12-14. Session gaps/missing candles: candle density {coverage_ratio:.0f}% of a fully-continuous feed -- {'consistent with genuine equity trading-hours gaps' if coverage_ratio and coverage_ratio < 80 else 'unexpectedly close to continuous -- verify this is genuinely an index/equity feed, not a CFD/synthetic proxy with different session behavior'}" if coverage_ratio else "12-14. Could not assess gap pattern")
    lines.append(f"15. Instrument type: `{test_symbol}` -- exact classification (index vs ETF vs CFD) not independently confirmed by this probe; would need verification before Phase 2")
    lines.append("16. Liquidity/reliability: not independently assessed by this probe -- would need real broker verification for Phase 2")
    lines.append(f"17-19. Development/OOS/alignment feasibility: {'appears feasible based on coverage' if (covers_dev and covers_oos) else 'not currently feasible -- see coverage gap above'}")
    lines.append("20. Sufficient history for the intended design: contingent on the above")

    lines.append("")
    lines.append("*Independence assessment (structural, not yet a profitability claim)*")
    lines.append(
        "A broad equity/risk index is mechanistically distinct from XAU 4H trend-following -- it reflects general "
        "risk-on/risk-off sentiment across equities, not gold-specific price action. This is a structurally different "
        "input class from every other candidate tested so far, INCLUDING the already-failed DXY/EUR-USD proxy work "
        "(those were specifically USD-strength signals; this would be a general risk-sentiment signal). "
        "This assessment does NOT claim the mechanism is profitable -- only that it is genuinely distinct, which is "
        "the specific question this phase was asked to answer."
    )

    lines.append("")
    if covers_dev and covers_oos:
        lines.append("*CLASSIFICATION: DATA SUFFICIENT -- PHASE 2 MAY BE PROPOSED*")
        lines.append(
            f"`{test_symbol}` is empirically confirmed to exist, fetch reliably, and cover the full required dev+OOS "
            f"window at 4H granularity via our existing free-tier infrastructure, with no new cost. This clears the "
            f"data-foundation bar -- it does NOT mean a risk-regime signal will improve or diversify the XAU strategy, "
            f"only that the question can now be legitimately tested."
        )
    else:
        lines.append("*CLASSIFICATION: DATA INSUFFICIENT -- CANDIDATE C CLOSED*")
        lines.append("Symbol exists and fetches, but does not cover the full required historical window -- see the coverage check above.")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent risk-index probe report")


if __name__ == "__main__":
    asyncio.run(run())
