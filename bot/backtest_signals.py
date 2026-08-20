"""Checks past LONG/SHORT scalp signals against what price actually did
afterward, using real 15-minute candle data -- did SL or TP1 get hit first?

This is the honesty check for the whole scalp system: a confidence score is
just a guess until it's been measured against real outcomes. Runs once a
day, evaluates any signal old enough to have likely resolved (4+ hours),
and sends a running win-rate summary to Telegram.

Methodology, stated plainly:
- A signal is a WIN if price touches TP1 before touching SL, a LOSS if SL is
  touched first. If a single 15-minute candle's range covers BOTH levels (a
  fast, volatile move), it's conservatively scored as a LOSS -- we can't know
  which was actually touched first within that candle, and assuming the
  worse outcome avoids overstating performance.
- If neither level is touched within 48 hours, it's marked EXPIRED and
  excluded from the win rate (not a resolved trade either way), though its
  approximate result is still recorded.
- Only tracks the TP1 target, not TP2 or the separate range-trade setups --
  keeping the methodology simple and unambiguous matters more right now than
  covering every scenario.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

import requests

from . import db
from . import scalp_analysis
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest_signals")

SYMBOL_LOOKUP = {s["display"]: s["api"] for s in scalp_analysis.SYMBOLS}
LOOKAHEAD_HOURS = 48


def fetch_price_window(api_symbol: str, start_dt: datetime, end_dt: datetime) -> list[dict] | None:
    if not scalp_analysis.TWELVEDATA_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": api_symbol,
                "interval": "15min",
                "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "apikey": scalp_analysis.TWELVEDATA_API_KEY,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            logger.warning("Twelve Data error fetching window for %s: %s", api_symbol, data.get("message", data))
            return None
        values = data["values"]
        values.reverse()
        return [{"high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])} for c in values]
    except Exception as e:
        logger.warning("Failed to fetch price window for %s: %s", api_symbol, e)
        return None


def evaluate_signal(signal: dict, candles: list[dict]) -> tuple[str, float, float]:
    entry, sl, tp1, action = signal["entry"], signal["sl"], signal["tp1"], signal["action"]
    risk = abs(entry - sl)

    for c in candles:
        if action == "LONG":
            hit_sl = c["low"] <= sl
            hit_tp1 = c["high"] >= tp1
        else:
            hit_sl = c["high"] >= sl
            hit_tp1 = c["low"] <= tp1

        if hit_sl:
            return "LOSS", sl, -1.0
        if hit_tp1:
            profit = (tp1 - entry) if action == "LONG" else (entry - tp1)
            r = profit / risk if risk else 0.0
            return "WIN", tp1, r

    last_close = candles[-1]["close"] if candles else entry
    profit = (last_close - entry) if action == "LONG" else (entry - last_close)
    r = profit / risk if risk else 0.0
    return "EXPIRED", last_close, r


async def run():
    signals = db.get_unevaluated_signals(min_age_hours=4, max_age_hours=LOOKAHEAD_HOURS)
    if not signals:
        logger.info("No signals eligible for evaluation right now")
        return

    evaluated_count = 0
    for signal in signals:
        api_symbol = SYMBOL_LOOKUP.get(signal["symbol"])
        if not api_symbol:
            continue

        created_at = datetime.fromisoformat(signal["created_at"])
        window_end = min(created_at + timedelta(hours=LOOKAHEAD_HOURS), datetime.now(timezone.utc))

        candles = fetch_price_window(api_symbol, created_at, window_end)
        if candles is None:
            continue

        outcome, exit_price, r_multiple = evaluate_signal(signal, candles)

        window_fully_elapsed = (datetime.now(timezone.utc) - created_at) >= timedelta(hours=LOOKAHEAD_HOURS)
        if outcome == "EXPIRED" and not window_fully_elapsed:
            continue

        db.save_signal_outcome(signal["id"], outcome, exit_price, r_multiple)
        evaluated_count += 1
        logger.info("Signal #%d (%s %s): %s, R=%.2f", signal["id"], signal["symbol"], signal["action"], outcome, r_multiple)

    if evaluated_count == 0:
        logger.info("No new outcomes finalized this run")
        return

    stats = db.get_outcome_stats()
    lines = [f"*📊 Backtest Update* ({evaluated_count} newly evaluated)\n"]
    for s in stats:
        if s["win_rate"] is not None:
            lines.append(
                f"*{s['symbol']}*: {s['wins']}W / {s['losses']}L "
                f"({s['win_rate']*100:.0f}% win rate), avg R: {s['avg_r']:+.2f}"
                + (f", {s['expired']} expired" if s["expired"] else "")
            )
        else:
            lines.append(f"*{s['symbol']}*: no resolved signals yet" + (f" ({s['expired']} expired)" if s["expired"] else ""))

    lines.append("")
    lines.append("_TP1 vs SL, whichever hit first in 15m candles. Small early sample -- treat with appropriate skepticism until more signals accumulate._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent backtest summary")


if __name__ == "__main__":
    asyncio.run(run())
