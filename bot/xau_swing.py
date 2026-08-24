"""Live forward test of the ONE validated, cost-adjusted-positive
configuration found in this whole project: gold trend-following, with
SL/TP sized off the 4-hour ATR instead of the tight 5-minute default.

CRITICAL: the entry decision logic here is IDENTICAL to what the backtest
validated (score_and_decide, same 4H/1H trend alignment + 15M/5M momentum
confirmation) -- only the SL/TP width changed for the backtest, and only
the SL/TP width changes here. Changing the entry logic now would mean
this is no longer a genuine forward test of what was actually validated.

This has its OWN isolated risk tracking (namespaced "xau_swing:" in
risk_controller), separate from every other symbol/strategy in this
project, so its forward-test P&L is never mixed with unvalidated signals.
Starts at config.XAU_SWING_RISK_PCT (0.25% by default) -- not the general
risk setting -- per the explicit forward-test plan: confirm live behavior
matches the backtest before scaling risk up.

Outcome evaluation reuses historical_backtest.find_outcome_detailed
directly, so live results are judged by EXACTLY the same logic the
backtest was validated with -- no risk of subtle inconsistency between
"what we backtested" and "what we're now measuring live".
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from . import db
from . import risk_controller
from . import scalp_analysis
from . import telegram_bot
from .historical_backtest import find_outcome_detailed, fetch_full_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xau_swing")

API_SYMBOL = "XAU/USD"
SYMBOL_TAG = "Gold (XAU/USD) [4h-swing]"  # distinct from the live 5m-ATR "Gold (XAU/USD)" tag
STRATEGY_TYPE = "trend_4h_swing"
RISK_NAMESPACE_PREFIX = "xau_swing"


def _today_namespaced_key() -> str:
    return f"{RISK_NAMESPACE_PREFIX}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


async def analyze_and_signal():
    """Runs once per cycle (intended cadence: every 4 hours, matching the
    timeframe the SL/TP is sized on -- more frequent checks wouldn't add
    genuinely new information between 4H candle closes). Generates a
    signal using the SAME entry logic as the validated backtest, sized
    with 4H ATR, checked against this forward test's own isolated risk
    budget."""
    tf_data, dfs = {}, {}
    for label, interval in scalp_analysis.TIMEFRAMES.items():
        df = scalp_analysis.fetch_klines(API_SYMBOL, interval)
        if df is None or len(df) < 205:
            logger.warning("Insufficient data for XAU/USD 4h-swing check (%s)", label)
            return
        tf_data[label] = scalp_analysis.compute_indicators(df)
        dfs[label] = df
        time.sleep(scalp_analysis.REQUEST_DELAY_SECONDS)

    divergence = scalp_analysis.detect_rsi_divergence(dfs["15m"])
    news_direction, news_summary = scalp_analysis.get_news_bias(["Gold", "XAU"])
    decision = scalp_analysis.score_and_decide(tf_data, news_direction)

    # Log every cycle, including NO TRADE, for the same "is filtering
    # helping or hurting" analysis the rest of this project relies on
    db.save_evaluation(
        source="live", symbol=SYMBOL_TAG, strategy_type=STRATEGY_TYPE, action=decision["action"],
        confidence=decision["confidence"], entry=None, sl=None, tp1=None, tp2=None,
        details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction}",
    )

    if decision["action"] == "NO TRADE":
        logger.info("XAU 4h-swing: no signal this cycle")
        return

    entry_price = tf_data["5m"]["close"]
    atr_4h = tf_data[config.XAU_SWING_ATR_TIMEFRAME]["atr"]
    sl = entry_price - 1.5 * atr_4h if decision["action"] == "LONG" else entry_price + 1.5 * atr_4h
    tp1 = entry_price + 1.0 * atr_4h if decision["action"] == "LONG" else entry_price - 1.0 * atr_4h
    tp2 = entry_price + 2.0 * atr_4h if decision["action"] == "LONG" else entry_price - 2.0 * atr_4h

    open_signal = db.get_open_signal(SYMBOL_TAG, STRATEGY_TYPE)
    if open_signal is not None and open_signal["action"] == decision["action"]:
        logger.info("XAU 4h-swing: still the same open trade from before, not a new entry")
        return

    risk_usd = config.ACCOUNT_SIZE_USD * (config.XAU_SWING_RISK_PCT / 100)
    allowed, reason = risk_controller.check_and_reserve(risk_usd, date_key=_today_namespaced_key())

    db.save_scalp_signal(
        symbol=SYMBOL_TAG, action=decision["action"], entry=entry_price, sl=sl, tp1=tp1, tp2=tp2,
        confidence=decision["confidence"],
        details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction} divergence={divergence}",
        strategy_type=STRATEGY_TYPE, risk_allowed=allowed,
    )

    status_emoji = "✅" if allowed else "⛔"
    status_text = "taken (forward test)" if allowed else f"SKIPPED -- {reason}"
    # NOTE: deliberately NOT using scalp_analysis.calculate_position_size()
    # here -- that function is hardcoded to config.RISK_PCT_PER_TRADE (the
    # general setting), which would show a position size inconsistent with
    # this forward test's actual isolated 0.25% risk tracking.
    price_risk_per_unit = abs(entry_price - sl)
    units = risk_usd / price_risk_per_unit if price_risk_per_unit else 0
    size_str = f"{units:.2f} oz (risking ${risk_usd:.2f})" if price_risk_per_unit else None

    message = (
        f"*🥇 XAU/USD 4H-Swing Forward Test*\n\n"
        f"{status_emoji} *{decision['action']}* ({decision['confidence']}%) -- {status_text}\n"
        f"Entry {entry_price:.2f} | SL {sl:.2f} | TP1 {tp1:.2f} | TP2 {tp2:.2f}\n"
    )
    if size_str:
        message += f"Size: {size_str}\n"
    message += (
        f"\n4H={tf_data['4h']['trend']} 1H={tf_data['1h']['trend']} 15M={tf_data['15m']['trend']} 5M={tf_data['5m']['trend']}\n"
        f"News: {news_summary}\n\n"
        f"_This is the validated 4H-ATR configuration, at {config.XAU_SWING_RISK_PCT:.2f}% risk. "
        f"Forward test only -- backtest validation, not a guarantee._"
    )
    await telegram_bot.send_text(message)
    logger.info("Sent XAU 4h-swing signal: %s", decision["action"])


async def evaluate_outcomes():
    """Checks any unresolved signal against real subsequent price, reusing
    find_outcome_detailed from the validated backtest so live results are
    judged by EXACTLY the same logic. Uses a 7-day lookahead (not the
    4-48h window the rest of the live system uses), since this configuration's
    stops are wide enough that trades genuinely take days to resolve."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT id, action, entry, sl, tp1, tp2, created_at
               FROM scalp_signals
               WHERE symbol = ? AND strategy_type = ? AND action != 'NO TRADE'
                 AND id NOT IN (SELECT signal_id FROM signal_outcomes)
               ORDER BY created_at ASC""",
            (SYMBOL_TAG, STRATEGY_TYPE),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("No unresolved XAU 4h-swing signals to evaluate")
        return

    for signal_id, action, entry, sl, tp1, tp2, created_at in rows:
        created_dt = datetime.fromisoformat(created_at)
        age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
        window_fully_elapsed = age_hours >= config.XAU_SWING_LOOKAHEAD_HOURS

        # Fetch 5m candles covering the full lookahead window since entry --
        # use the backtest's own fetcher (parses datetime, pulls up to 5000
        # candles / ~17 days), not the live scan's fetch_klines (only 250
        # candles, nowhere near enough for a 7-day lookahead).
        df_5m = fetch_full_history(API_SYMBOL, "5min")
        if df_5m is None:
            continue

        detail = find_outcome_detailed(df_5m, created_dt, entry, sl, tp1, tp2, action, lookahead_hours=config.XAU_SWING_LOOKAHEAD_HOURS)

        if detail["outcome"] == "EXPIRED" and not window_fully_elapsed:
            continue  # leave unresolved -- might still hit TP/SL before the full window elapses

        db.save_signal_outcome(
            signal_id, detail["outcome"], detail["exit_price"], detail["r_multiple"],
            mae_r=detail["mae_before_tp1_r"], mfe_r=detail["mfe_before_tp1_r"],
        )
        logger.info("XAU 4h-swing signal #%d: %s, R=%.2f", signal_id, detail["outcome"], detail["r_multiple"])

        risk_usd = config.ACCOUNT_SIZE_USD * (config.XAU_SWING_RISK_PCT / 100)
        signal_date_key = f"{RISK_NAMESPACE_PREFIX}:{created_dt.strftime('%Y-%m-%d')}"
        risk_controller.record_outcome(detail["r_multiple"], risk_usd, date_key=signal_date_key)


def get_forward_test_summary() -> dict:
    """Aggregates the isolated xau_swing risk-tracking history into a
    running equity curve and max drawdown across the whole forward-test
    period -- not just a single day's state."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT date, cumulative_usd, trades_taken FROM daily_risk_state
               WHERE date LIKE ? ORDER BY date ASC""",
            (f"{RISK_NAMESPACE_PREFIX}:%",),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"days": 0, "total_pnl_usd": 0.0, "max_drawdown_usd": 0.0, "trades_taken": 0}

    cum, peak, max_dd = 0.0, 0.0, 0.0
    total_trades = 0
    for date_key, daily_pnl, trades_taken in rows:
        cum += daily_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        total_trades += trades_taken

    return {
        "days": len(rows), "total_pnl_usd": cum, "max_drawdown_usd": max_dd,
        "trades_taken": total_trades, "first_day": rows[0][0].split(":")[1], "last_day": rows[-1][0].split(":")[1],
    }


async def send_forward_test_status():
    summary = get_forward_test_summary()
    if summary["days"] == 0:
        await telegram_bot.send_text("*🥇 XAU 4H-Swing Forward Test Status*\n\nNo data yet -- the forward test hasn't recorded any resolved days.")
        return

    pnl_emoji = "🟢" if summary["total_pnl_usd"] > 0 else ("🔴" if summary["total_pnl_usd"] < 0 else "⚪")
    lines = [
        "*🥇 XAU 4H-Swing Forward Test Status*",
        f"_Period: {summary['first_day']} to {summary['last_day']} ({summary['days']} day(s) with activity)_\n",
        f"{pnl_emoji} Cumulative P&L: ${summary['total_pnl_usd']:+.2f}",
        f"Total trades taken: {summary['trades_taken']}",
        f"Max drawdown so far: ${summary['max_drawdown_usd']:.2f}",
        f"Risk per trade: {config.XAU_SWING_RISK_PCT:.2f}% of ${config.ACCOUNT_SIZE_USD:.0f} account",
        "\n_This tracks ONLY the isolated 4H-swing forward test, not the rest of the bot's signals._",
    ]
    await telegram_bot.send_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(analyze_and_signal())
