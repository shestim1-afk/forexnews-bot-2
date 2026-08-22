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

from . import config
from . import db
from . import risk_controller
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
    """Fetches 15-minute candles between two timestamps via Twelve Data.
    Assumes Twelve Data's returned timestamps are UTC, matching how this
    project stores signal creation times -- if that assumption is ever
    wrong for a given symbol, evaluations could be mistimed. Worth
    revisiting if outcomes look implausible."""
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
        values.reverse()  # oldest first
        return [{"high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])} for c in values]
    except Exception as e:
        logger.warning("Failed to fetch price window for %s: %s", api_symbol, e)
        return None


def evaluate_signal(signal: dict, candles: list[dict]) -> tuple[str, float, float]:
    """Returns (outcome, exit_price, r_multiple)."""
    entry, sl, tp1, action = signal["entry"], signal["sl"], signal["tp1"], signal["action"]
    risk = abs(entry - sl)

    for c in candles:
        if action == "LONG":
            hit_sl = c["low"] <= sl
            hit_tp1 = c["high"] >= tp1
        else:  # SHORT
            hit_sl = c["high"] >= sl
            hit_tp1 = c["low"] <= tp1

        if hit_sl:  # checked first -- conservative on same-bar ambiguity
            r = -1.0
            return "LOSS", sl, r
        if hit_tp1:
            profit = (tp1 - entry) if action == "LONG" else (entry - tp1)
            r = profit / risk if risk else 0.0
            return "WIN", tp1, r

    # Neither hit within the window
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
            continue  # try again next run

        outcome, exit_price, r_multiple = evaluate_signal(signal, candles)

        # Only finalize EXPIRED once the full lookahead window has actually
        # elapsed -- otherwise leave it unevaluated so a later run can still
        # catch a legitimate late TP/SL hit.
        window_fully_elapsed = (datetime.now(timezone.utc) - created_at) >= timedelta(hours=LOOKAHEAD_HOURS)
        if outcome == "EXPIRED" and not window_fully_elapsed:
            continue

        db.save_signal_outcome(signal["id"], outcome, exit_price, r_multiple)
        evaluated_count += 1
        logger.info("Signal #%d (%s %s): %s, R=%.2f", signal["id"], signal["symbol"], signal["action"], outcome, r_multiple)

        # Only feed this outcome into the daily risk controller if it was
        # actually "taken" under the risk framework at generation time --
        # a signal that was SKIPPED for exceeding a daily limit shouldn't
        # count against that day's simulated P&L or loss streak.
        if signal.get("risk_allowed"):
            signal_date_key = created_at.strftime("%Y-%m-%d")
            risk_usd = config.ACCOUNT_SIZE_USD * (config.RISK_PCT_PER_TRADE / 100)
            risk_controller.record_outcome(r_multiple, risk_usd, date_key=signal_date_key)

    if evaluated_count == 0:
        logger.info("No new outcomes finalized this run")
        return

    await telegram_bot.send_text(format_full_stats_message(evaluated_count))
    logger.info("Sent backtest summary")


def format_full_stats_message(newly_evaluated: int | None = None) -> str:
    """Shared formatter used both by this daily backtest job and the
    on-demand /stats Telegram command, so both show identical numbers."""
    stats = db.get_outcome_stats()
    strategy_stats = db.get_strategy_type_stats()

    if not stats:
        return "No signals have been evaluated yet. Signals need to be at least 4 hours old before they're checked -- check back once some have had time to play out."

    header = "*📊 Backtest Update*" if newly_evaluated is not None else "*📊 Win/Loss Record So Far*"
    if newly_evaluated:
        header += f" ({newly_evaluated} newly evaluated)"
    lines = [header + "\n"]

    lines.append("_By symbol:_")
    for s in stats:
        if s["win_rate"] is not None:
            pf = "∞" if s["profit_factor"] is None and s["wins"] > 0 else (f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "N/A")
            lines.append(
                f"*{s['symbol']}*: {s['wins']}W / {s['losses']}L "
                f"({s['win_rate']*100:.0f}% win rate), avg R: {s['avg_r']:+.2f}, "
                f"profit factor: {pf}, max drawdown: {s['max_drawdown_r']:.2f}R"
                + (f", {s['expired']} expired" if s["expired"] else "")
            )
        else:
            lines.append(f"*{s['symbol']}*: no resolved signals yet" + (f" ({s['expired']} expired)" if s["expired"] else ""))

    if strategy_stats:
        lines.append("")
        lines.append("_By strategy type:_")
        for s in strategy_stats:
            if s["win_rate"] is not None:
                lines.append(f"*{s['strategy_type']}*: {s['wins']}W / {s['losses']}L ({s['win_rate']*100:.0f}%), avg R: {s['avg_r']:+.2f}")

    lines.append("")
    lines.append("_TP1 vs SL, whichever hit first in 15m candles. Treat with appropriate skepticism until many signals have accumulated._")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(run())
