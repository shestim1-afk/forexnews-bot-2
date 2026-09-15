"""XAU/USD 10-Year Historical Depth Audit -- feasibility check ONLY.

Checks whether Twelve Data's free tier actually has ~10 years of XAU/USD
4H data available, and estimates the request budget needed to pull it,
BEFORE committing to a full extended backtest. No strategy changes, no
new hypothesis -- this is purely a data-availability check for re-running
the SAME already-validated strategy (4h-ATR trend, both the 0.667R
baseline and 1.0R candidate) over a longer period, not a search for a
different "best" strategy.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from . import config
from .historical_backtest import fetch_paginated_history

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("xau_depth_audit")

TARGET_YEARS_BACK = 10


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


async def run():
    lines = [
        f"*XAU/USD {TARGET_YEARS_BACK}-Year Historical Depth Audit (feasibility check ONLY)*",
        "No strategy changes. Checks whether enough real 4H XAU data exists to re-test the SAME "
        "already-validated baseline/candidate strategy over a longer period -- not a search for a new one.\n",
    ]

    target_start = datetime.now() - timedelta(days=365 * TARGET_YEARS_BACK)
    logger.info("Requesting XAU/USD 4h data back to %s...", target_start.date())

    df = fetch_paginated_history("XAU/USD", "4h", target_start, datetime.now())

    if df is None or len(df) == 0:
        lines.append("FAILED -- could not retrieve any XAU/USD 4h data.")
        lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT*")
        _send_telegram_direct("\n".join(lines))
        return

    actual_start = df["datetime"].min()
    actual_end = df["datetime"].max()
    years_covered = (actual_end - actual_start).days / 365.25
    n_candles = len(df)

    lines.append(f"Requested back to: {target_start.date()}")
    lines.append(f"Actual data received: {actual_start.date()} to {actual_end.date()} ({years_covered:.1f} years, {n_candles} candles)")

    covers_target = years_covered >= (TARGET_YEARS_BACK - 0.5)
    lines.append(f"Covers ~{TARGET_YEARS_BACK} years: {'YES' if covers_target else 'NO -- shorter history than requested'}")

    if covers_target:
        lines.append(
            "\n*CLASSIFICATION: DATA SUFFICIENT -- a full re-run of the existing baseline/candidate strategy "
            "over this extended period can proceed as a robustness check, unchanged specification.*"
        )
    else:
        lines.append(
            f"\n*CLASSIFICATION: PARTIAL -- only {years_covered:.1f} years available, not the full "
            f"{TARGET_YEARS_BACK}. A re-run is still possible over the available window, just shorter than hoped.*"
        )

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent XAU depth audit report")


if __name__ == "__main__":
    asyncio.run(run())
