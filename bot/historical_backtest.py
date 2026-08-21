"""Historical backtest: replays the SAME decision logic used by the live
scalp bot against past price data, to see what it would have decided at
each point in time -- without letting it "see" any future candles.

Honest limitation, stated upfront: Twelve Data's free tier caps each
request at 5,000 candles. For BTC's 5-minute data (which the live system
also needs for entry timing), that's only about 17 days of history --
the real bottleneck, since 4H/1H/15M data could go back much further on
their own. This gives a genuine first sample, not a multi-month backtest.

How look-ahead bias is avoided: at each replay timestamp, every timeframe's
data is sliced to only include candles up to and including that exact
timestamp -- never anything after it. This mirrors exactly what the live
bot would have seen if it had actually been running at that moment.

Every evaluated setup gets logged via db.save_evaluation() (source='backtest'),
including NO TRADE ones -- not just the actionable signals -- so we can
later ask whether the filtering is actually improving quality. Actionable
signals get their outcome (WIN/LOSS/EXPIRED) determined by scanning forward
within the SAME already-fetched historical data (no extra API calls needed
for outcome-checking, since it's self-contained historical data). Also
tracks MAE/MFE per trade (Maximum Adverse/Favorable Excursion), computed
only from price action after entry and only up to the actual exit point.

sl_mult/tp1_mult/tp2_mult let you A/B test different risk:reward ratios
against identical historical data -- results are tagged distinctly in the
database so a variant run never mixes with the baseline numbers for the
same symbol.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import db
from . import scalp_analysis
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("historical_backtest")

REPLAY_STEP_MINUTES = 30  # matches the live bot's actual scan cadence
WARMUP_BARS = 250  # matches CANDLE_LIMIT used by the live indicators
LOOKAHEAD_HOURS_FOR_OUTCOME = 24  # how far forward to check TP/SL within the historical data


def fetch_full_history(api_symbol: str, interval: str, outputsize: int = 5000) -> pd.DataFrame | None:
    """Like scalp_analysis.fetch_klines, but requests the maximum history
    Twelve Data's free tier allows in one call, and keeps a proper parsed
    'datetime' column for time-based slicing."""
    if not scalp_analysis.TWELVEDATA_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": api_symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": scalp_analysis.TWELVEDATA_API_KEY,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            logger.warning("Twelve Data error for %s %s: %s", api_symbol, interval, data.get("message", data))
            return None
        values = data["values"]
        values.reverse()
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
        return df
    except Exception as e:
        logger.warning("Failed to fetch full history for %s %s: %s", api_symbol, interval, e)
        return None


def slice_up_to(df: pd.DataFrame, cutoff_dt, window: int = WARMUP_BARS) -> pd.DataFrame | None:
    """Returns the most recent `window` candles at or before cutoff_dt --
    never anything after it. Returns None if there isn't enough history yet
    (still in the warmup period)."""
    sliced = df[df["datetime"] <= cutoff_dt]
    if len(sliced) < window:
        return None
    return sliced.iloc[-window:].reset_index(drop=True)


def compute_trade_levels_variant(action: str, entry: float, atr: float, sl_mult: float, tp1_mult: float, tp2_mult: float) -> dict:
    """Like scalp_analysis.compute_trade_levels, but with configurable
    SL/TP multipliers -- lets us A/B test different risk:reward ratios
    against the same historical data without touching the live bot's
    actual behavior (which stays fixed at 1.5/1.0/2.0 unless you decide
    to adopt a different ratio permanently)."""
    if action == "LONG":
        return {"entry": entry, "sl": entry - sl_mult * atr, "tp1": entry + tp1_mult * atr, "tp2": entry + tp2_mult * atr}
    elif action == "SHORT":
        return {"entry": entry, "sl": entry + sl_mult * atr, "tp1": entry - tp1_mult * atr, "tp2": entry - tp2_mult * atr}
    return {"entry": None, "sl": None, "tp1": None, "tp2": None}


def evaluate_at(cutoff_dt, dfs_full: dict, sl_mult: float = 1.5, tp1_mult: float = 1.0, tp2_mult: float = 2.0) -> tuple[dict, dict, dict | None, str | None, dict | None, dict | None] | None:
    """Runs the exact same indicator/decision logic as the live bot, but on
    data sliced to only what would have been known at cutoff_dt. Returns
    None if any timeframe doesn't have enough warmup data yet at this point.
    sl_mult/tp1_mult/tp2_mult default to the live bot's actual ratios --
    pass different values to test a variant."""
    tf_data, dfs = {}, {}
    for label in ["4h", "1h", "15m", "5m"]:
        sliced = slice_up_to(dfs_full[label], cutoff_dt)
        if sliced is None:
            return None
        tf_data[label] = scalp_analysis.compute_indicators(sliced)
        dfs[label] = sliced

    divergence = scalp_analysis.detect_rsi_divergence(dfs["15m"])
    decision = scalp_analysis.score_and_decide(tf_data, "neutral")  # no live news context available historically
    range_setup = scalp_analysis.detect_range_setup(tf_data, dfs, divergence)
    sweep = scalp_analysis.detect_liquidity_sweep(dfs["15m"], tf_data["15m"]["atr"])
    breakout_retest = scalp_analysis.detect_breakout_retest(dfs["15m"], tf_data["15m"]["atr"])

    entry_price = tf_data["5m"]["close"]
    atr_5m = tf_data["5m"]["atr"]
    levels = compute_trade_levels_variant(decision["action"], entry_price, atr_5m, sl_mult, tp1_mult, tp2_mult)

    return tf_data, decision, levels, divergence, range_setup, sweep, breakout_retest


def find_outcome(df_5m_full: pd.DataFrame, entry_time, entry: float, sl: float, tp1: float, action: str) -> tuple[str, float, float, float, float]:
    """Same WIN/LOSS/EXPIRED logic as the live backtester, but scanning
    forward within already-fetched historical 5-minute data instead of
    making a new API call. Also tracks Maximum Adverse Excursion (MAE) and
    Maximum Favorable Excursion (MFE), in R-multiples -- how far price moved
    against/in favor of the position before it closed. Calculated ONLY from
    price action after entry, and only up to the actual exit point (not
    beyond it), so it reflects what this specific trade actually
    experienced, not unrelated later price action.

    Returns (outcome, exit_price, r_multiple, mae_r, mfe_r)."""
    window_end = entry_time + timedelta(hours=LOOKAHEAD_HOURS_FOR_OUTCOME)
    forward = df_5m_full[(df_5m_full["datetime"] > entry_time) & (df_5m_full["datetime"] <= window_end)]
    risk = abs(entry - sl)
    max_adverse, max_favorable = 0.0, 0.0

    for _, c in forward.iterrows():
        if action == "LONG":
            adverse = entry - c["low"]
            favorable = c["high"] - entry
            hit_sl, hit_tp1 = c["low"] <= sl, c["high"] >= tp1
        else:
            adverse = c["high"] - entry
            favorable = entry - c["low"]
            hit_sl, hit_tp1 = c["high"] >= sl, c["low"] <= tp1

        max_adverse = max(max_adverse, adverse)
        max_favorable = max(max_favorable, favorable)
        mae_r = max_adverse / risk if risk else 0.0
        mfe_r = max_favorable / risk if risk else 0.0

        if hit_sl:
            return "LOSS", sl, -1.0, mae_r, mfe_r
        if hit_tp1:
            profit = (tp1 - entry) if action == "LONG" else (entry - tp1)
            return "WIN", tp1, (profit / risk if risk else 0.0), mae_r, mfe_r

    last_close = forward.iloc[-1]["close"] if len(forward) else entry
    profit = (last_close - entry) if action == "LONG" else (entry - last_close)
    mae_r = max_adverse / risk if risk else 0.0
    mfe_r = max_favorable / risk if risk else 0.0
    return "EXPIRED", last_close, (profit / risk if risk else 0.0), mae_r, mfe_r


async def run(api_symbol: str = "BTC/USD", display_symbol: str = "BTC/USD",
              sl_mult: float = 1.5, tp1_mult: float = 1.0, tp2_mult: float = 2.0):
    is_variant = (sl_mult, tp1_mult, tp2_mult) != (1.5, 1.0, 2.0)
    result_symbol = f"{display_symbol} (R:R {tp1_mult:.1f}:{sl_mult:.1f})" if is_variant else display_symbol

    logger.info("Fetching historical data for %s...", display_symbol)
    dfs_full = {}
    for label, interval in scalp_analysis.TIMEFRAMES.items():
        df = fetch_full_history(api_symbol, interval)
        if df is None or len(df) < WARMUP_BARS:
            logger.error("Not enough historical data for %s on %s -- aborting backtest", display_symbol, label)
            return
        dfs_full[label] = df
        logger.info("%s: %d candles, from %s to %s", label, len(df), df["datetime"].min(), df["datetime"].max())

    earliest_start = max(df["datetime"].min() for df in dfs_full.values())
    latest_end = min(df["datetime"].max() for df in dfs_full.values())
    replay_start = earliest_start + timedelta(minutes=WARMUP_BARS * 5)
    replay_end = latest_end - timedelta(hours=LOOKAHEAD_HOURS_FOR_OUTCOME)

    if replay_start >= replay_end:
        logger.error("Not enough historical range to replay after accounting for warmup and outcome lookahead")
        return

    logger.info("Replaying from %s to %s in %d-minute steps (SL=%.1fx TP1=%.1fx TP2=%.1fx ATR)",
                replay_start, replay_end, REPLAY_STEP_MINUTES, sl_mult, tp1_mult, tp2_mult)

    step = timedelta(minutes=REPLAY_STEP_MINUTES)
    t = replay_start
    evaluated_count = 0
    actionable_count = 0

    while t <= replay_end:
        result = evaluate_at(t, dfs_full, sl_mult, tp1_mult, tp2_mult)
        if result is None:
            t += step
            continue

        tf_data, decision, levels, divergence, range_setup, sweep, breakout_retest = result

        eval_id = db.save_evaluation(
            source="backtest", symbol=result_symbol, strategy_type="trend", action=decision["action"],
            confidence=decision["confidence"], entry=levels["entry"], sl=levels["sl"],
            tp1=levels["tp1"], tp2=levels["tp2"],
            details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']}",
            evaluated_at=t.isoformat(),
        )
        evaluated_count += 1

        if decision["action"] != "NO TRADE":
            actionable_count += 1
            outcome, exit_price, r_multiple, mae_r, mfe_r = find_outcome(
                dfs_full["5m"], t, levels["entry"], levels["sl"], levels["tp1"], decision["action"]
            )
            db.save_backtest_outcome(eval_id, outcome, exit_price, r_multiple, mae_r, mfe_r)

        t += step

    logger.info("Backtest complete: %d evaluated, %d actionable", evaluated_count, actionable_count)

    summary = db.get_backtest_summary(result_symbol)
    lines = [f"*📈 Historical Backtest: {result_symbol}*\n"]
    lines.append(f"Period: {replay_start.strftime('%Y-%m-%d')} to {replay_end.strftime('%Y-%m-%d')} ({REPLAY_STEP_MINUTES}-min steps)")
    lines.append(f"Total evaluated: {summary['total_evaluated']} ({summary['no_trade_count']} NO TRADE, {summary['actionable_count']} actionable)")
    if summary["win_rate"] is not None:
        pf = f"{summary['profit_factor']:.2f}" if summary["profit_factor"] is not None else "N/A"
        lines.append(
            f"Resolved: {summary['wins']}W / {summary['losses']}L ({summary['win_rate']*100:.0f}% win rate), "
            f"avg R: {summary['avg_r']:+.2f}, profit factor: {pf}, max drawdown: {summary['max_drawdown_r']:.2f}R"
            + (f", {summary['expired']} expired" if summary["expired"] else "")
        )
    else:
        lines.append("No actionable signals resolved in this window.")
    lines.append("")
    lines.append("_Limited to ~17 days of 5-minute history (free-tier cap). Small sample -- a starting point, not proof of an edge._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent backtest summary")


if __name__ == "__main__":
    asyncio.run(run())
