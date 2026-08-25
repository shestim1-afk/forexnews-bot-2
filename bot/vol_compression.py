"""Volatility Compression -> Expansion backtest.

HYPOTHESIS (frozen before testing, per protocol): an objectively defined
period of unusually compressed volatility creates a subsequent expansion
regime with positive directional expectancy after realistic transaction
costs. This is a genuinely different mechanism from trend-following,
range mean-reversion, or the previously-tested breakout+retest strategy
-- the entry trigger here is explicitly a VOLATILITY REGIME TRANSITION,
not a price level break on its own (a breakout with no preceding
compression does NOT qualify as a signal here).

ALL PARAMETERS BELOW ARE FROZEN BEFORE ANY BACKTEST WAS RUN. None were
tuned against results. Per protocol, if this hypothesis fails, it is
reported as a failed hypothesis -- not re-parameterized until it works.

Frozen definitions:
- Compression: ATR(14) percentile rank, over a trailing 180-candle
  window, in the bottom 15%.
- Compression QUALIFIES once this holds for >=6 consecutive 4H candles
  (~1 day). That period's price range (high/low) and average ATR become
  the fixed reference for everything that follows.
- Expansion trigger: within the 10 candles following qualification, the
  FIRST candle whose true range exceeds 1.5x the compression period's
  average ATR. If no such candle exists in the window, the event is
  logged as NO EXPANSION -- a genuine null result the falsification
  analysis depends on, not a discarded data point.
- Direction (a SEPARATE assumption from the volatility hypothesis
  itself, documented as such, not part of the core mechanism being
  tested): LONG if the expansion candle closes above the compression
  period's high; SHORT if it closes below the low. If it closes inside
  the range, the event is logged as EXPANSION WITHOUT DIRECTION.
- Stop: beyond the compression range, buffered by 0.25x the compression
  period's average ATR.
- Exit: a FIXED 2.0R target -- explicitly a TEMPORARY RESEARCH CONTROL,
  not a claimed final design. The hypothesis under test is about entry
  timing (does compression predict a real subsequent move), not exit
  optimization; a trailing exit would need a materially different
  backtest evaluation loop, deferred as a separate question if this
  entry hypothesis itself survives.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import ta

from . import db
from . import telegram_bot
from .historical_backtest import fetch_full_history, fetch_paginated_history, find_outcome_detailed, WARMUP_BARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vol_compression")

STRATEGY_TYPE = "vol_compression"

# Frozen parameters -- see module docstring for justification. Do not
# adjust these based on backtest results.
ATR_PERIOD = 14
PERCENTILE_LOOKBACK = 180
COMPRESSION_PERCENTILE_THRESHOLD = 15
COMPRESSION_MIN_DURATION = 6
EXPANSION_WATCH_WINDOW = 10
EXPANSION_TRUE_RANGE_MULT = 1.5
STOP_BUFFER_ATR_MULT = 0.25
TP_R_MULT = 2.0  # temporary research control, see module docstring
LOOKAHEAD_HOURS_FOR_OUTCOME = 168  # 7 days, consistent with the validated 4H swing config


def _true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def detect_compression_events(df_4h: pd.DataFrame) -> list[dict]:
    """Scans the 4H series for compression events per the frozen
    definitions above. Returns one dict per qualifying compression
    period, whether or not it produced an expansion or a tradeable
    direction -- every qualifying compression is a data point for the
    falsification analysis, not just the ones that became trades."""
    atr_series = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"], window=ATR_PERIOD).average_true_range()
    tr_series = [_true_range(df_4h["high"].iloc[i], df_4h["low"].iloc[i], df_4h["close"].iloc[i - 1] if i > 0 else df_4h["close"].iloc[0]) for i in range(len(df_4h))]

    n = len(df_4h)
    events = []
    streak = 0
    i = PERCENTILE_LOOKBACK  # need a full lookback window before percentiles are meaningful

    while i < n:
        atr_val = atr_series.iloc[i]
        if pd.isna(atr_val):
            i += 1
            continue

        window_vals = sorted(atr_series.iloc[i - PERCENTILE_LOOKBACK:i].dropna().tolist())
        if not window_vals:
            i += 1
            continue
        rank = sum(1 for v in window_vals if v <= atr_val) / len(window_vals) * 100

        if rank <= COMPRESSION_PERCENTILE_THRESHOLD:
            streak += 1
        else:
            streak = 0

        if streak == COMPRESSION_MIN_DURATION:
            # Qualifying compression period just completed -- its
            # reference range/ATR come from these COMPRESSION_MIN_DURATION candles
            comp_start = i - COMPRESSION_MIN_DURATION + 1
            comp_slice = df_4h.iloc[comp_start:i + 1]
            compression_high = comp_slice["high"].max()
            compression_low = comp_slice["low"].min()
            compression_avg_atr = atr_series.iloc[comp_start:i + 1].mean()
            qualify_time = df_4h["datetime"].iloc[i]

            event = {
                "qualify_time": qualify_time, "compression_high": compression_high,
                "compression_low": compression_low, "compression_avg_atr": compression_avg_atr,
                "expansion_found": False, "expansion_time": None, "expansion_true_range_r": None,
                "direction": None, "entry": None, "sl": None, "tp": None,
            }

            # Watch the next EXPANSION_WATCH_WINDOW candles for the expansion trigger
            for j in range(i + 1, min(i + 1 + EXPANSION_WATCH_WINDOW, n)):
                if tr_series[j] > EXPANSION_TRUE_RANGE_MULT * compression_avg_atr:
                    event["expansion_found"] = True
                    event["expansion_time"] = df_4h["datetime"].iloc[j]
                    event["expansion_true_range_r"] = tr_series[j] / compression_avg_atr if compression_avg_atr else None
                    close_j = df_4h["close"].iloc[j]

                    if close_j > compression_high:
                        event["direction"] = "LONG"
                        entry = close_j
                        sl = compression_low - STOP_BUFFER_ATR_MULT * compression_avg_atr
                        risk = entry - sl
                        event.update({"entry": entry, "sl": sl, "tp": entry + TP_R_MULT * risk})
                    elif close_j < compression_low:
                        event["direction"] = "SHORT"
                        entry = close_j
                        sl = compression_high + STOP_BUFFER_ATR_MULT * compression_avg_atr
                        risk = sl - entry
                        event.update({"entry": entry, "sl": sl, "tp": entry - TP_R_MULT * risk})
                    # else: expansion without direction -- event stays direction=None
                    break

            events.append(event)
            streak = 0  # skip forward past this event before scanning for the next one
            i += 1 + EXPANSION_WATCH_WINDOW
            continue

        i += 1

    return events


async def run(api_symbol: str = "XAU/USD", display_symbol: str = "XAU/USD",
              deep_start_date: str | None = None, deep_end_date: str | None = None):
    """Fetches 4H data for compression detection and 5m data for outcome
    evaluation (reusing find_outcome_detailed directly, same logic as the
    validated backtest), tags results distinctly as strategy_type
    'vol_compression' so all existing analytics tooling (cost-adjusted,
    monthly, cost distribution, MAE/MFE diagnostics) works on this
    automatically without new report code."""
    result_symbol = f"{display_symbol} [vol-compression-4h]"
    if deep_start_date and deep_end_date:
        result_symbol = f"{result_symbol} [{deep_start_date} to {deep_end_date}]"

    cleared = db.clear_backtest_data(result_symbol)
    if cleared:
        logger.info("Cleared %d prior evaluation(s) for %s", cleared, result_symbol)

    logger.info("Fetching 4H and 5m data for %s...", display_symbol)
    if deep_start_date and deep_end_date:
        target_start = datetime.strptime(deep_start_date, "%Y-%m-%d")
        target_end = datetime.strptime(deep_end_date, "%Y-%m-%d") + timedelta(days=1)
        df_4h = fetch_paginated_history(api_symbol, "4h", target_start, target_end)
        df_5m = fetch_paginated_history(api_symbol, "5min", target_start, target_end)
    else:
        df_4h = fetch_full_history(api_symbol, "4h")
        df_5m = fetch_full_history(api_symbol, "5min")

    if df_4h is None or len(df_4h) < PERCENTILE_LOOKBACK + COMPRESSION_MIN_DURATION:
        logger.error("Not enough 4H data for %s -- aborting", display_symbol)
        return
    if df_5m is None or len(df_5m) < WARMUP_BARS:
        logger.error("Not enough 5m data for %s -- aborting", display_symbol)
        return

    logger.info("4H: %d candles (%s to %s), 5m: %d candles (%s to %s)",
                len(df_4h), df_4h["datetime"].min(), df_4h["datetime"].max(),
                len(df_5m), df_5m["datetime"].min(), df_5m["datetime"].max())

    events = detect_compression_events(df_4h)
    logger.info("Detected %d compression events", len(events))

    # Event-level stats (independent of whether a trade was taken)
    n_events = len(events)
    n_expansion = sum(1 for e in events if e["expansion_found"])
    n_directional = sum(1 for e in events if e["direction"] is not None)
    n_no_expansion = n_events - n_expansion
    n_expansion_no_direction = n_expansion - n_directional

    trades_evaluated = 0
    for event in events:
        if event["direction"] is None:
            continue
        # Only evaluate trades whose entry time leaves room for the full
        # lookahead window within the fetched 5m data
        if event["expansion_time"] + timedelta(hours=LOOKAHEAD_HOURS_FOR_OUTCOME) > df_5m["datetime"].max():
            continue

        eval_id = db.save_evaluation(
            source="backtest", symbol=result_symbol, strategy_type=STRATEGY_TYPE, action=event["direction"],
            confidence=60, entry=event["entry"], sl=event["sl"], tp1=event["tp"], tp2=None,
            details=f"compression_high={event['compression_high']:.5f} compression_low={event['compression_low']:.5f} "
                    f"expansion_tr_r={event['expansion_true_range_r']:.2f}",
            evaluated_at=event["expansion_time"].isoformat(),
        )
        detail = find_outcome_detailed(
            df_5m, event["expansion_time"], event["entry"], event["sl"], event["tp"], None,
            event["direction"], lookahead_hours=LOOKAHEAD_HOURS_FOR_OUTCOME,
        )
        db.save_backtest_outcome(
            eval_id, detail["outcome"], detail["exit_price"], detail["r_multiple"],
            mae_r=detail["mae_before_tp1_r"], mfe_r=detail["mfe_before_tp1_r"],
            mae_before_tp1_r=detail["mae_before_tp1_r"], mfe_before_tp1_r=detail["mfe_before_tp1_r"],
            tp1_hit=detail["tp1_hit"], tp2_hit=detail["tp2_hit"],
            time_to_tp1_minutes=detail["time_to_tp1_minutes"], time_to_tp2_minutes=detail["time_to_tp2_minutes"],
            mfe_after_tp1_r=detail["mfe_after_tp1_r"], max_giveback_after_tp1_r=detail["max_giveback_after_tp1_r"],
            returned_to_entry_after_tp1=detail["returned_to_entry_after_tp1"], time_to_exit_minutes=detail["time_to_exit_minutes"],
        )
        trades_evaluated += 1

    logger.info("Backtest complete: %d compression events, %d trades evaluated", n_events, trades_evaluated)

    lines = [f"*🌊 Volatility Compression -> Expansion: {result_symbol}*\n"]
    lines.append(f"4H data: {df_4h['datetime'].min().strftime('%Y-%m-%d')} to {df_4h['datetime'].max().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("*Event-level statistics (the hypothesis test itself, independent of whether a trade resulted):*")
    lines.append(f"  Compression events detected: {n_events}")
    if n_events > 0:
        lines.append(f"  Produced expansion (TR > {EXPANSION_TRUE_RANGE_MULT}x compression ATR within {EXPANSION_WATCH_WINDOW} candles): {n_expansion} ({100*n_expansion/n_events:.1f}%)")
        lines.append(f"  No expansion within window: {n_no_expansion} ({100*n_no_expansion/n_events:.1f}%)")
        lines.append(f"  Expansion WITH clear direction (traded): {n_directional} ({100*n_directional/n_events:.1f}%)")
        lines.append(f"  Expansion WITHOUT clear direction (not traded): {n_expansion_no_direction} ({100*n_expansion_no_direction/n_events:.1f}%)")
    lines.append(f"  Trades evaluated: {trades_evaluated}")
    lines.append("")
    lines.append(
        "_Frozen parameters: 15th percentile ATR compression, 6-candle minimum duration, "
        f"{EXPANSION_TRUE_RANGE_MULT}x TR expansion trigger within {EXPANSION_WATCH_WINDOW} candles, "
        f"{TP_R_MULT}R fixed exit (temporary research control). Run the standard Cost-Adjusted Analysis, "
        "Temporal Consistency, and MAE/MFE Diagnostic reports against the symbol/strategy tags below "
        "for full expectancy analysis using existing tooling._"
    )
    lines.append(f"Symbol tag: `{result_symbol}`")
    lines.append(f"Strategy type: `{STRATEGY_TYPE}`")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent volatility compression report")


if __name__ == "__main__":
    asyncio.run(run())
