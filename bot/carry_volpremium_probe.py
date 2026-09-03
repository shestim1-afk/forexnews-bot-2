"""Carry & Volatility Risk Premium DATA-AVAILABILITY FEASIBILITY PROBES.

Explicitly NOT strategies, NOT backtests, does NOT touch the production
XAU bot. These were the two candidates from the original research map
flagged as "likely blocked" but never actually empirically verified --
this probe closes that gap the same way the DXY and risk-index probes
did: cheap existence checks first, real data if a symbol resolves,
DATA INSUFFICIENT if not, no assumptions either way.

CARRY: requires genuine interest-rate/bond-yield data. Spot FX price
data is NOT a substitute and is not used as one here -- if no rate/yield
symbol resolves, this closes as DATA INSUFFICIENT, not approximated.

VOLATILITY RISK PREMIUM: requires genuine implied-volatility data (VIX
or a VIX-tracking instrument). Spot price volatility is NOT the same
thing and is not substituted here either.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from . import telegram_bot
from .historical_backtest import fetch_paginated_history
from . import scalp_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("carry_volpremium_probe")

CARRY_CANDIDATE_SYMBOLS = ["US10Y", "US2Y", "DGS10", "TNX", "US10YT"]
VOL_PREMIUM_CANDIDATE_SYMBOLS = ["VIX", "VXX", "UVXY", "VIXY"]

REQUIRED_START = "2025-01-01"
REQUIRED_END = "2025-12-31"


def probe_symbol_exists(symbol: str) -> dict:
    """Identical cheap existence check used by the DXY and risk-index probes."""
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
            return {"symbol": symbol, "exists": True, "sample_dates": [v["datetime"] for v in data["values"]], "http_status": r.status_code}
        return {"symbol": symbol, "exists": False, "http_status": r.status_code, "error": data.get("message", data.get("code", "unknown error"))}
    except Exception as e:
        return {"symbol": symbol, "exists": False, "error": str(e)}


def check_depth(symbol: str) -> dict:
    target_start = datetime.strptime(REQUIRED_START, "%Y-%m-%d")
    target_end = datetime.strptime(REQUIRED_END, "%Y-%m-%d") + timedelta(days=1)
    df_4h = fetch_paginated_history(symbol, "4h", target_start, target_end)
    if df_4h is None or len(df_4h) == 0:
        return {"sufficient": False, "n_candles": 0}
    actual_start, actual_end = df_4h["datetime"].min(), df_4h["datetime"].max()
    covers_dev = actual_start <= datetime.strptime("2025-01-01", "%Y-%m-%d")
    covers_oos = actual_end >= datetime.strptime("2025-12-30", "%Y-%m-%d")
    return {"sufficient": covers_dev and covers_oos, "n_candles": len(df_4h), "start": actual_start, "end": actual_end}


def _run_candidate_group(group_name: str, candidates: list[str]) -> list[str]:
    lines = [f"*Step -- {group_name} symbol existence check*"]
    working = []
    for symbol in candidates:
        result = probe_symbol_exists(symbol)
        logger.info("%s symbol '%s': exists=%s (%s)", group_name, symbol, result["exists"], result.get("error", "OK"))
        if result["exists"]:
            lines.append(f"  YES `{symbol}`: EXISTS, sample dates {result['sample_dates'][:2]}")
            working.append(symbol)
        else:
            lines.append(f"  NO `{symbol}`: {result.get('error', 'not found')}")
    return lines, working


async def run():
    lines = [
        "*Carry & Volatility Risk Premium Data-Availability Probe (feasibility check ONLY)*",
        "No strategy built, no backtest run, production bot untouched. This closes the two remaining unverified "
        "items from the original research map -- both were previously flagged as likely blocked but never "
        "empirically confirmed.\n",
    ]

    # --- Carry ---
    carry_lines, carry_working = _run_candidate_group("Carry (interest-rate/bond-yield)", CARRY_CANDIDATE_SYMBOLS)
    lines += carry_lines
    lines.append("")
    if not carry_working:
        lines.append("*CARRY CLASSIFICATION: DATA INSUFFICIENT*")
        lines.append(
            "None of the tested rate/yield symbol conventions resolved on our current Twelve Data plan. Consistent "
            "with the original expectation that a spot-price FX/crypto API does not include genuine interest-rate "
            "data. Spot FX price is NOT used as a substitute. Closed -- not approximated."
        )
    else:
        test_symbol = carry_working[0]
        depth = check_depth(test_symbol)
        lines.append(f"`{test_symbol}` resolved -- checking historical depth ({depth.get('n_candles', 0)} candles, "
                      f"{depth.get('start', 'N/A')} to {depth.get('end', 'N/A')})")
        if depth["sufficient"]:
            lines.append("*CARRY CLASSIFICATION: DATA SUFFICIENT -- worth a proper Phase 1 investigation as a new, separate hypothesis.*")
        else:
            lines.append("*CARRY CLASSIFICATION: DATA INSUFFICIENT -- symbol exists but historical depth/coverage does not meet the required window.*")
    lines.append("")

    # --- Volatility Risk Premium ---
    vol_lines, vol_working = _run_candidate_group("Volatility Risk Premium (VIX/implied-vol)", VOL_PREMIUM_CANDIDATE_SYMBOLS)
    lines += vol_lines
    lines.append("")
    if not vol_working:
        lines.append("*VOL PREMIUM CLASSIFICATION: DATA INSUFFICIENT*")
        lines.append(
            "None of the tested VIX/implied-volatility symbol conventions resolved. Consistent with the original "
            "expectation that implied-volatility/options data is not available through this data tier. Spot price "
            "volatility is NOT used as a substitute. Closed -- not approximated."
        )
    else:
        test_symbol = vol_working[0]
        depth = check_depth(test_symbol)
        lines.append(f"`{test_symbol}` resolved -- checking historical depth ({depth.get('n_candles', 0)} candles, "
                      f"{depth.get('start', 'N/A')} to {depth.get('end', 'N/A')})")
        if depth["sufficient"]:
            lines.append("*VOL PREMIUM CLASSIFICATION: DATA SUFFICIENT -- worth a proper Phase 1 investigation as a new, separate hypothesis.*")
            lines.append(
                "Note: this would need its OWN economic/statistical foundation test before any strategy design -- "
                "same discipline as every other candidate tested so far. A resolving symbol here is a necessary, "
                "not sufficient, condition."
            )
        else:
            lines.append("*VOL PREMIUM CLASSIFICATION: DATA INSUFFICIENT -- symbol exists but historical depth/coverage does not meet the required window.*")

    lines.append(
        "\nThis probe deliberately covers both remaining unverified research-map items in one run, per the "
        "explicit no-parameter-search discipline -- no alternate symbol conventions will be tried beyond what is "
        "listed here if both close as insufficient."
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent carry/vol-premium probe report")


if __name__ == "__main__":
    asyncio.run(run())
