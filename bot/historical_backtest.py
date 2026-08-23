"""Historical backtest: replays the SAME decision logic used by the live
scalp bot against past price data, to see what it would have decided at
each point in time -- without letting it "see" any future candles.

NEW: atr_timeframe lets you compute SL/TP width from a HIGHER timeframe's
ATR (1h, 4h) instead of the tight 5-minute one the live bot uses by
default. This exists because we found, with real data, that 5m-ATR-based
stops are so tight that even a modest spread cost erases 30-70% of the
entire risk unit -- every strategy type came back net negative after
costs. Using a wider timeframe's ATR for the stop distance mechanically
shrinks that cost's share of risk, at the cost of holding trades longer
and getting fewer of them per day (which was explicitly requested: aiming
for ~5 trades/day, not dozens).

lookahead_hours is now configurable (default 24) since wider, swing-style
stops can genuinely take days to resolve, not hours.
"""

import asyncio
import logging
import time
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

REPLAY_STEP_MINUTES = 30
WARMUP_BARS = 250
LOOKAHEAD_HOURS_FOR_OUTCOME_DEFAULT = 24


def fetch_full_history(api_symbol: str, interval: str, outputsize: int = 5000) -> pd.DataFrame | None:
    if not scalp_analysis.TWELVEDATA_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": api_symbol, "interval": interval, "outputsize": outputsize, "apikey": scalp_analysis.TWELVEDATA_API_KEY},
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
    sliced = df[df["datetime"] <= cutoff_dt]
    if len(sliced) < window:
        return None
    return sliced.iloc[-window:].reset_index(drop=True)


def fetch_paginated_history(api_symbol: str, interval: str, target_start: datetime, target_end: datetime,
                             chunk_size: int = 5000, request_delay: float = 8.0, max_retries: int = 2) -> pd.DataFrame | None:
    if not scalp_analysis.TWELVEDATA_API_KEY:
        return None
    all_chunks = []
    current_end = target_end
    max_iterations = 60

    for _ in range(max_iterations):
        if current_end <= target_start:
            break

        data = None
        for attempt in range(max_retries + 1):
            try:
                r = requests.get(
                    "https://api.twelvedata.com/time_series",
                    params={
                        "symbol": api_symbol, "interval": interval, "outputsize": chunk_size,
                        "end_date": current_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "apikey": scalp_analysis.TWELVEDATA_API_KEY,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning("Failed to fetch chunk for %s %s ending %s (attempt %d/%d): %s",
                                api_symbol, interval, current_end, attempt + 1, max_retries + 1, e)
                data = None

            if data is not None and data.get("status") != "error" and "values" in data and data["values"]:
                break

            if attempt < max_retries:
                logger.info(
                    "Empty/error response for %s %s ending %s -- retrying (attempt %d/%d)",
                    api_symbol, interval, current_end, attempt + 2, max_retries + 1,
                )
                time.sleep(request_delay * 2)

        if data is None or data.get("status") == "error" or "values" not in data or not data["values"]:
            logger.info(
                "No data for %s %s before %s after %d attempts -- treating as the real historical depth limit",
                api_symbol, interval, current_end, max_retries + 1,
            )
            break

        values = data["values"]
        values.reverse()
        chunk = pd.DataFrame(values)
        chunk["datetime"] = pd.to_datetime(chunk["datetime"])
        for col in ["open", "high", "low", "close"]:
            chunk[col] = chunk[col].astype(float)
        chunk["volume"] = chunk["volume"].astype(float) if "volume" in chunk.columns else 0.0
        all_chunks.append(chunk)

        earliest = chunk["datetime"].min()
        if earliest >= current_end:
            break
        current_end = earliest - timedelta(seconds=1)
        time.sleep(request_delay)

    if not all_chunks:
        return None
    combined = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return combined[combined["datetime"] >= target_start].reset_index(drop=True)


def compute_trade_levels_variant(action: str, entry: float, atr: float, sl_mult: float, tp1_mult: float, tp2_mult: float) -> dict:
    if action == "LONG":
        return {"entry": entry, "sl": entry - sl_mult * atr, "tp1": entry + tp1_mult * atr, "tp2": entry + tp2_mult * atr}
    elif action == "SHORT":
        return {"entry": entry, "sl": entry + sl_mult * atr, "tp1": entry - tp1_mult * atr, "tp2": entry - tp2_mult * atr}
    return {"entry": None, "sl": None, "tp1": None, "tp2": None}


def evaluate_at(cutoff_dt, dfs_full: dict, sl_mult: float = 1.5, tp1_mult: float = 1.0, tp2_mult: float = 2.0,
                 atr_timeframe: str = "5m") -> tuple[dict, dict, dict | None, str | None, dict | None, dict | None] | None:
    """atr_timeframe selects which timeframe's ATR sets the SL/TP distance
    -- '5m' (the live bot's default, tight/scalp-style), '15m', '1h', or
    '4h' (wider, swing-style, meant to survive real spread costs better).
    Entry price is always the current 5m close regardless of which ATR is
    used for sizing the stop -- only the WIDTH of SL/TP changes."""
    tf_data, dfs = {}, {}
    for label in ["4h", "1h", "15m", "5m"]:
        sliced = slice_up_to(dfs_full[label], cutoff_dt)
        if sliced is None:
            return None
        tf_data[label] = scalp_analysis.compute_indicators(sliced)
        dfs[label] = sliced

    divergence = scalp_analysis.detect_rsi_divergence(dfs["15m"])
    decision = scalp_analysis.score_and_decide(tf_data, "neutral")
    range_setup = scalp_analysis.detect_range_setup(tf_data, dfs, divergence)
    sweep = scalp_analysis.detect_liquidity_sweep(dfs["15m"], tf_data["15m"]["atr"])
    breakout_retest = scalp_analysis.detect_breakout_retest(dfs["15m"], tf_data["15m"]["atr"])

    entry_price = tf_data["5m"]["close"]
    atr_for_sizing = tf_data[atr_timeframe]["atr"]
    levels = compute_trade_levels_variant(decision["action"], entry_price, atr_for_sizing, sl_mult, tp1_mult, tp2_mult)

    return tf_data, decision, levels, divergence, range_setup, sweep, breakout_retest


def find_outcome_detailed(df_5m_full: pd.DataFrame, entry_time, entry: float, sl: float,
                           tp1: float, tp2: float | None, action: str,
                           lookahead_hours: float = LOOKAHEAD_HOURS_FOR_OUTCOME_DEFAULT) -> dict:
    """Same WIN/LOSS/EXPIRED/MAE/MFE logic as before. lookahead_hours is now
    a parameter (not a fixed global) since wider, swing-style stops can
    genuinely take days to resolve, not hours -- a 24-hour window would
    incorrectly mark many genuinely-still-open swing trades as EXPIRED."""
    window_end = entry_time + timedelta(hours=lookahead_hours)
    forward = df_5m_full[(df_5m_full["datetime"] > entry_time) & (df_5m_full["datetime"] <= window_end)].reset_index(drop=True)
    risk = abs(entry - sl)

    max_adverse_before, max_favorable_before = 0.0, 0.0
    tp1_hit, tp1_hit_time, tp1_row_idx = False, None, None
    outcome, exit_price, r_multiple, exit_time = None, None, None, None

    for i in range(len(forward)):
        c = forward.iloc[i]
        if action == "LONG":
            adverse, favorable = entry - c["low"], c["high"] - entry
            hit_sl, hit_tp1_now = c["low"] <= sl, c["high"] >= tp1
        else:
            adverse, favorable = c["high"] - entry, entry - c["low"]
            hit_sl, hit_tp1_now = c["high"] >= sl, c["low"] <= tp1

        if not tp1_hit:
            max_adverse_before = max(max_adverse_before, adverse)
            max_favorable_before = max(max_favorable_before, favorable)

            if hit_sl:
                outcome, exit_price, r_multiple = "LOSS", sl, -1.0
                exit_time = c["datetime"]
                break
            if hit_tp1_now:
                tp1_hit, tp1_hit_time, tp1_row_idx = True, c["datetime"], i
                profit = (tp1 - entry) if action == "LONG" else (entry - tp1)
                outcome, exit_price, r_multiple = "WIN", tp1, (profit / risk if risk else 0.0)
                exit_time = c["datetime"]

    if outcome is None:
        last_row = forward.iloc[-1] if len(forward) else None
        last_close = last_row["close"] if last_row is not None else entry
        exit_time = last_row["datetime"] if last_row is not None else window_end
        profit = (last_close - entry) if action == "LONG" else (entry - last_close)
        outcome, exit_price, r_multiple = "EXPIRED", last_close, (profit / risk if risk else 0.0)

    result = {
        "outcome": outcome, "exit_price": exit_price, "r_multiple": r_multiple, "exit_time": exit_time,
        "mae_before_tp1_r": max_adverse_before / risk if risk else 0.0,
        "mfe_before_tp1_r": max_favorable_before / risk if risk else 0.0,
        "tp1_hit": tp1_hit,
        "tp2_hit": None, "time_to_tp1_minutes": None, "time_to_tp2_minutes": None,
        "mfe_after_tp1_r": None, "max_giveback_after_tp1_r": None,
        "returned_to_entry_after_tp1": None, "time_to_exit_minutes": None,
    }

    if tp1_hit:
        result["time_to_tp1_minutes"] = (tp1_hit_time - entry_time).total_seconds() / 60

        peak_favorable_after, tp2_hit, tp2_time, returned_to_entry = 0.0, False, None, False
        last_time, last_close_after = tp1_hit_time, exit_price

        for j in range(tp1_row_idx, len(forward)):
            c = forward.iloc[j]
            if action == "LONG":
                favorable_now = c["high"] - entry
                if not tp2_hit and tp2 is not None and c["high"] >= tp2:
                    tp2_hit, tp2_time = True, c["datetime"]
                if c["low"] <= entry:
                    returned_to_entry = True
            else:
                favorable_now = entry - c["low"]
                if not tp2_hit and tp2 is not None and c["low"] <= tp2:
                    tp2_hit, tp2_time = True, c["datetime"]
                if c["high"] >= entry:
                    returned_to_entry = True
            peak_favorable_after = max(peak_favorable_after, favorable_now)
            last_time, last_close_after = c["datetime"], c["close"]

        final_favorable = (last_close_after - entry) if action == "LONG" else (entry - last_close_after)
        giveback_r = max(0.0, (peak_favorable_after - final_favorable) / risk) if risk else 0.0

        result["mfe_after_tp1_r"] = peak_favorable_after / risk if risk else 0.0
        result["max_giveback_after_tp1_r"] = giveback_r
        result["tp2_hit"] = tp2_hit
        result["time_to_tp2_minutes"] = (tp2_time - entry_time).total_seconds() / 60 if tp2_time else None
        result["returned_to_entry_after_tp1"] = returned_to_entry
        result["time_to_exit_minutes"] = (last_time - entry_time).total_seconds() / 60

    return result


def process_signal_type(strategy_type: str, action: str | None, entry: float | None, sl: float | None,
                         tp1: float | None, tp2: float | None, confidence: float, details: str,
                         t: datetime, dfs_full: dict, tracker: dict, result_symbol: str,
                         log_no_signal: bool = False, lookahead_hours: float = LOOKAHEAD_HOURS_FOR_OUTCOME_DEFAULT) -> str:
    if action is None:
        if log_no_signal:
            db.save_evaluation(
                source="backtest", symbol=result_symbol, strategy_type=strategy_type, action="NO TRADE",
                confidence=confidence, entry=None, sl=None, tp1=None, tp2=None,
                details=details, evaluated_at=t.isoformat(),
            )
        tracker["direction"], tracker["exit_time"] = None, None
        return "no_trade" if log_no_signal else "absent"

    is_continuation = (
        tracker["direction"] == action
        and tracker["exit_time"] is not None
        and t <= tracker["exit_time"]
    )
    if is_continuation:
        return "continuation"

    eval_id = db.save_evaluation(
        source="backtest", symbol=result_symbol, strategy_type=strategy_type, action=action,
        confidence=confidence, entry=entry, sl=sl, tp1=tp1, tp2=tp2,
        details=details, evaluated_at=t.isoformat(),
    )
    detail = find_outcome_detailed(dfs_full["5m"], t, entry, sl, tp1, tp2, action, lookahead_hours=lookahead_hours)
    db.save_backtest_outcome(
        eval_id, detail["outcome"], detail["exit_price"], detail["r_multiple"],
        mae_r=detail["mae_before_tp1_r"], mfe_r=detail["mfe_before_tp1_r"],
        mae_before_tp1_r=detail["mae_before_tp1_r"], mfe_before_tp1_r=detail["mfe_before_tp1_r"],
        tp1_hit=detail["tp1_hit"], tp2_hit=detail["tp2_hit"],
        time_to_tp1_minutes=detail["time_to_tp1_minutes"], time_to_tp2_minutes=detail["time_to_tp2_minutes"],
        mfe_after_tp1_r=detail["mfe_after_tp1_r"], max_giveback_after_tp1_r=detail["max_giveback_after_tp1_r"],
        returned_to_entry_after_tp1=detail["returned_to_entry_after_tp1"], time_to_exit_minutes=detail["time_to_exit_minutes"],
    )
    tracker["direction"] = action
    tracker["exit_time"] = detail["exit_time"]
    return "new"


async def run(api_symbol: str = "BTC/USD", display_symbol: str = "BTC/USD",
              sl_mult: float = 1.5, tp1_mult: float = 1.0, tp2_mult: float = 2.0,
              deep_start_date: str | None = None, deep_end_date: str | None = None,
              atr_timeframe: str = "5m", lookahead_hours: float | None = None):
    """atr_timeframe: '5m' (live bot default) / '15m' / '1h' / '4h' -- which
    timeframe's ATR sets the SL/TP width. lookahead_hours: how long to wait
    for TP/SL before marking EXPIRED -- defaults to 24h for 5m/15m, but
    auto-extends to 168h (7 days) for 1h/4h unless you override it, since
    wider stops genuinely take longer to resolve."""
    if lookahead_hours is None:
        lookahead_hours = 168.0 if atr_timeframe in ("1h", "4h") else LOOKAHEAD_HOURS_FOR_OUTCOME_DEFAULT

    is_variant = (sl_mult, tp1_mult, tp2_mult) != (1.5, 1.0, 2.0)
    result_symbol = f"{display_symbol} (R:R {tp1_mult:.1f}:{sl_mult:.1f})" if is_variant else display_symbol
    if atr_timeframe != "5m":
        result_symbol = f"{result_symbol} [{atr_timeframe}-ATR]"
    if deep_start_date and deep_end_date:
        result_symbol = f"{result_symbol} [{deep_start_date} to {deep_end_date}]"

    cleared = db.clear_backtest_data(result_symbol)
    if cleared:
        logger.info("Cleared %d prior evaluation(s) for %s before starting this fresh run", cleared, result_symbol)

    logger.info("Fetching historical data for %s...", display_symbol)
    dfs_full = {}
    if deep_start_date and deep_end_date:
        target_start = datetime.strptime(deep_start_date, "%Y-%m-%d").replace(tzinfo=None)
        target_end = datetime.strptime(deep_end_date, "%Y-%m-%d").replace(tzinfo=None) + timedelta(days=1)
        for label, interval in scalp_analysis.TIMEFRAMES.items():
            df = fetch_paginated_history(api_symbol, interval, target_start, target_end)
            if df is None or len(df) < WARMUP_BARS:
                logger.error("Not enough historical data for %s on %s -- aborting backtest", display_symbol, label)
                return
            dfs_full[label] = df
            logger.info("%s: %d candles, from %s to %s", label, len(df), df["datetime"].min(), df["datetime"].max())
    else:
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
    replay_end = latest_end - timedelta(hours=lookahead_hours)

    if replay_start >= replay_end:
        logger.error("Not enough historical range to replay after accounting for warmup and outcome lookahead")
        return

    logger.info("Replaying from %s to %s in %d-minute steps (SL=%.1fx TP1=%.1fx TP2=%.1fx %s-ATR, lookahead=%.0fh)",
                replay_start, replay_end, REPLAY_STEP_MINUTES, sl_mult, tp1_mult, tp2_mult, atr_timeframe, lookahead_hours)

    step = timedelta(minutes=REPLAY_STEP_MINUTES)
    t = replay_start
    totals = {"evaluated": 0, "actionable": 0, "continuations": 0}
    STRATEGY_TYPES = ["trend", "range", "liquidity_sweep", "breakout_retest"]
    open_trades = {st: {"direction": None, "exit_time": None} for st in STRATEGY_TYPES}

    while t <= replay_end:
        result = evaluate_at(t, dfs_full, sl_mult, tp1_mult, tp2_mult, atr_timeframe=atr_timeframe)
        if result is None:
            t += step
            continue

        tf_data, decision, levels, divergence, range_setup, sweep, breakout_retest = result
        htf_details = f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']}"

        trend_action = decision["action"] if decision["action"] != "NO TRADE" else None
        status = process_signal_type(
            "trend", trend_action, levels.get("entry"), levels.get("sl"), levels.get("tp1"), levels.get("tp2"),
            decision["confidence"], htf_details, t, dfs_full, open_trades["trend"], result_symbol,
            log_no_signal=True, lookahead_hours=lookahead_hours,
        )
        if status in ("new", "no_trade"):
            totals["evaluated"] += 1
        if status == "new":
            totals["actionable"] += 1
        elif status == "continuation":
            totals["continuations"] += 1

        # Range/sweep/breakout still use their own SL logic (level-anchored,
        # not the chosen atr_timeframe) -- multi-timeframe stop testing is
        # scoped to trend here, since that's the one strategy with any
        # real evidence behind it worth testing further
        if range_setup:
            r_action, r_entry, r_sl, r_tp1, r_details = (
                range_setup["direction"], range_setup["entry"], range_setup["sl"], range_setup["tp"], range_setup["reason"],
            )
        else:
            r_action = r_entry = r_sl = r_tp1 = None
            r_details = ""
        status = process_signal_type(
            "range", r_action, r_entry, r_sl, r_tp1, None, 60, r_details,
            t, dfs_full, open_trades["range"], result_symbol, lookahead_hours=lookahead_hours,
        )
        if status == "new":
            totals["evaluated"] += 1
            totals["actionable"] += 1
        elif status == "continuation":
            totals["continuations"] += 1

        if sweep:
            atr15 = tf_data["15m"]["atr"]
            sweep_entry = tf_data["15m"]["close"]
            sw_action = "LONG" if sweep["direction"] == "bullish" else "SHORT"
            sw_levels = scalp_analysis.compute_sweep_breakout_levels(sweep["direction"], sweep["swept_level"], sweep_entry, atr15)
            sw_sl, sw_tp1 = sw_levels["sl"], sw_levels["tp1"]
            sw_details = f"swept {sweep['swept_level']:.5f}"
        else:
            sw_action = sweep_entry = sw_sl = sw_tp1 = None
            sw_details = ""
        status = process_signal_type(
            "liquidity_sweep", sw_action, sweep_entry, sw_sl, sw_tp1, None, 60, sw_details,
            t, dfs_full, open_trades["liquidity_sweep"], result_symbol, lookahead_hours=lookahead_hours,
        )
        if status == "new":
            totals["evaluated"] += 1
            totals["actionable"] += 1
        elif status == "continuation":
            totals["continuations"] += 1

        if breakout_retest:
            atr15 = tf_data["15m"]["atr"]
            br_entry = tf_data["15m"]["close"]
            br_action = "LONG" if breakout_retest["direction"] == "bullish" else "SHORT"
            br_levels = scalp_analysis.compute_sweep_breakout_levels(breakout_retest["direction"], breakout_retest["level"], br_entry, atr15)
            br_sl, br_tp1 = br_levels["sl"], br_levels["tp1"]
            br_details = f"retested {breakout_retest['level']:.5f}"
        else:
            br_action = br_entry = br_sl = br_tp1 = None
            br_details = ""
        status = process_signal_type(
            "breakout_retest", br_action, br_entry, br_sl, br_tp1, None, 60, br_details,
            t, dfs_full, open_trades["breakout_retest"], result_symbol, lookahead_hours=lookahead_hours,
        )
        if status == "new":
            totals["evaluated"] += 1
            totals["actionable"] += 1
        elif status == "continuation":
            totals["continuations"] += 1

        t += step

    logger.info("Backtest complete: %d evaluated, %d actionable, %d continuations skipped",
                totals["evaluated"], totals["actionable"], totals["continuations"])

    days_in_period = max((replay_end - replay_start).total_seconds() / 86400, 1e-9)

    summary = db.get_backtest_summary(result_symbol)
    strategy_stats = db.get_backtest_strategy_type_stats(result_symbol)

    lines = [f"*📈 Historical Backtest: {result_symbol}*\n"]
    lines.append(f"Period: {replay_start.strftime('%Y-%m-%d')} to {replay_end.strftime('%Y-%m-%d')} ({REPLAY_STEP_MINUTES}-min steps, {atr_timeframe}-ATR sizing)")
    lines.append(f"Total evaluated: {summary['total_evaluated']} ({summary['no_trade_count']} NO TRADE, {summary['actionable_count']} actionable)")
    if summary["win_rate"] is not None:
        pf = f"{summary['profit_factor']:.2f}" if summary["profit_factor"] is not None else "N/A"
        lines.append(
            f"Overall: {summary['wins']}W / {summary['losses']}L ({summary['win_rate']*100:.0f}% win rate), "
            f"avg R: {summary['avg_r']:+.2f}, profit factor: {pf}, max drawdown: {summary['max_drawdown_r']:.2f}R"
            + (f", {summary['expired']} expired" if summary["expired"] else "")
        )
    else:
        lines.append("No actionable signals resolved in this window.")

    if strategy_stats:
        lines.append("")
        lines.append("_By strategy type (trades/day shown for the trend row, since only that one uses the wider stop):_")
        for s in strategy_stats:
            if s["win_rate"] is not None:
                pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "N/A"
                trades_per_day_str = f", {s['trades']/days_in_period:.1f} trades/day" if s["strategy_type"] == "trend" else ""
                lines.append(
                    f"*{s['strategy_type']}* (n={s['trades']}{trades_per_day_str}): {s['wins']}W/{s['losses']}L ({s['win_rate']*100:.0f}%), "
                    f"avg R {s['avg_r']:+.2f}, PF {pf}, max DD {s['max_drawdown_r']:.2f}R"
                    + (f", {s['expired']} expired" if s["expired"] else "")
                )
            else:
                lines.append(f"*{s['strategy_type']}*: no resolved trades yet")

    lines.append("")
    if deep_start_date and deep_end_date:
        lines.append(
            f"_Deep backtest requested {deep_start_date} to {deep_end_date}. Actual achieved range shown above._"
        )
    else:
        lines.append("_Limited to ~17 days of 5-minute history (free-tier cap)._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent backtest summary")


if __name__ == "__main__":
    asyncio.run(run())
