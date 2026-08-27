"""BTC Large-Displacement Momentum Continuation.

HYPOTHESIS: an unusually large, fast BTC price displacement MAY represent
leveraged forced-liquidation flow. If so, that forced-flow mechanism may
create short-term momentum continuation after the initial displacement.

CRITICAL FRAMING, enforced throughout this module: a large-displacement
candle is only a PROXY. We have no direct liquidation data. Nothing here
should describe a detected event as a confirmed liquidation cascade --
only as a "large-displacement event."

ALL PARAMETERS BELOW ARE FROZEN PER THE APPROVED SPECIFICATION. None are
to be adjusted after seeing development-period results. If the hypothesis
fails, it is reported as failed -- not re-parameterized.

Frozen definitions:
- Event: 1H BTC candle whose |close_t - close_(t-1)| exceeds 4.0x ATR(14),
  where ATR is computed on the preceding 100 1H candles EXCLUDING the
  event candle itself (to avoid the event contaminating its own reference).
- Entry: at the OPEN of the next 1H candle after the event.
- Direction: matches the event candle's own direction (close>open -> LONG,
  close<open -> SHORT). The event's own qualification IS the confirmation
  -- no secondary filter.
- Stop: 2.0x ATR(14) on the 4H timeframe, as of the event candle's close.
- Exit: a FIXED 3.0R target -- a temporary research control, not a final
  design (per the same reasoning as the volatility compression module).
- Cost: 0.25% round-trip base assumption (0.15% realistic BTC exchange
  spread + 0.10% slippage buffer specific to entries right after
  high-volatility events, when order books are more likely thin).
- Development period: 2025-01-01 to 2025-08-31. Out-of-sample period:
  2025-09-01 to 2025-12-31 -- NOT touched until development results are
  reviewed and explicitly approved for OOS testing.

TWO SEPARATE VERDICTS are always reported, per explicit instruction:
- SCIENTIFIC verdict: does the event predict continuation at all (vs the
  unconditional and fade baselines)?
- ECONOMIC verdict: is the resulting edge large AND frequent enough to
  materially contribute to the ~1%/day portfolio objective? These are
  different questions -- a real, low-correlation edge can be scientifically
  interesting while being economically immaterial to this specific goal.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import ta

from . import db
from . import telegram_bot
from . import analytics
from .historical_backtest import fetch_full_history, fetch_paginated_history, find_outcome_detailed, WARMUP_BARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("btc_displacement")

STRATEGY_TYPE = "btc_displacement"
API_SYMBOL = "BTC/USD"

# Frozen parameters -- do not adjust based on results.
ATR_PERIOD = 14
EVENT_ATR_LOOKBACK = 100
DISPLACEMENT_ATR_MULT = 4.0
STOP_ATR_MULT_4H = 2.0
TP_R_MULT = 3.0  # temporary research control, see module docstring
BASE_SPREAD_PCT = 0.25  # 0.15% spread + 0.10% slippage buffer
COST_SENSITIVITY_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.40]
LOOKAHEAD_HOURS_FOR_OUTCOME = 168  # 7 days

DEV_START, DEV_END = "2023-01-01", "2025-08-31"  # ambitious start date -- the real depth limit is discovered empirically by the fetch itself, same as elsewhere in this project
OOS_START, OOS_END = "2025-09-01", "2025-12-31"

# Falsification thresholds -- frozen, see module docstring for rationale
MIN_EVENTS_NOT_FAILED = 30
MIN_EVENTS_NOT_INCONCLUSIVE = 50
MIN_FREQUENCY_NOT_INCONCLUSIVE = 0.02  # trades/day
CONCENTRATION_FAILURE_THRESHOLD = 0.80  # % of positive R from top 20% of trades
MIN_ECONOMIC_R_PER_DAY = 0.05  # the PROMISING gate -- explicitly NOT "materially useful", just "worth OOS testing"


def find_4h_atr_as_of(df_4h: pd.DataFrame, atr_4h_series: pd.Series, event_time) -> float | None:
    """Finds the most recent 4H candle at or before event_time and
    returns its ATR(14) value."""
    eligible = df_4h[df_4h["datetime"] <= event_time]
    if len(eligible) == 0:
        return None
    idx = eligible.index[-1]
    val = atr_4h_series.iloc[idx]
    return val if pd.notna(val) else None


def detect_displacement_events(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> list[dict]:
    """Scans the 1H series for displacement events per the frozen
    definition, sizing the stop from the 4H ATR as of each event."""
    atr_1h = ta.volatility.AverageTrueRange(df_1h["high"], df_1h["low"], df_1h["close"], window=ATR_PERIOD).average_true_range()
    atr_4h = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"], window=ATR_PERIOD).average_true_range()

    n = len(df_1h)
    events = []
    for i in range(EVENT_ATR_LOOKBACK + 1, n - 1):  # -1 so a next-candle open always exists
        atr_ref = atr_1h.iloc[i - 1]  # ATR as of the candle BEFORE the event, excludes the event itself
        if pd.isna(atr_ref) or atr_ref == 0:
            continue

        candle_return = df_1h["close"].iloc[i] - df_1h["close"].iloc[i - 1]
        if abs(candle_return) <= DISPLACEMENT_ATR_MULT * atr_ref:
            continue

        event_time = df_1h["datetime"].iloc[i]
        direction = "LONG" if df_1h["close"].iloc[i] > df_1h["open"].iloc[i] else "SHORT"
        entry_time = df_1h["datetime"].iloc[i + 1]
        entry_price = df_1h["open"].iloc[i + 1]

        atr_4h_val = find_4h_atr_as_of(df_4h, atr_4h, event_time)
        if atr_4h_val is None or atr_4h_val == 0:
            continue

        sl_distance = STOP_ATR_MULT_4H * atr_4h_val
        if direction == "LONG":
            sl = entry_price - sl_distance
            tp = entry_price + TP_R_MULT * sl_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - TP_R_MULT * sl_distance

        events.append({
            "event_time": event_time, "entry_time": entry_time, "direction": direction,
            "entry": entry_price, "sl": sl, "tp": tp,
            "candle_return": candle_return, "atr_normalized_magnitude": abs(candle_return) / atr_ref,
            "preceding_volatility_atr": atr_ref,
        })
    return events


def compute_baseline_comparisons(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_5m: pd.DataFrame,
                                  events: list[dict], sample_size: int = 300, seed: int = 7) -> dict:
    """Two baselines, per the approved spec:
    (a) UNCONDITIONAL: the same entry/stop/target construction applied to
        a random sample of ALL 1H candles (not just displacement events),
        using each candle's own direction -- isolates whether the SIZE
        of the displacement adds information beyond generic directional
        momentum.
    (b) FADE: the same events, but trading the OPPOSITE direction --
        if fading performs comparably or better, the event does not
        contain genuine continuation information."""
    atr_4h = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"], window=ATR_PERIOD).average_true_range()

    random.seed(seed)
    n = len(df_1h)
    candidate_indices = list(range(EVENT_ATR_LOOKBACK + 1, n - 1))
    sample_indices = random.sample(candidate_indices, min(sample_size, len(candidate_indices)))

    unconditional_r = []
    for i in sample_indices:
        direction = "LONG" if df_1h["close"].iloc[i] > df_1h["open"].iloc[i] else "SHORT"
        entry_time = df_1h["datetime"].iloc[i + 1]
        entry_price = df_1h["open"].iloc[i + 1]
        atr_4h_val = find_4h_atr_as_of(df_4h, atr_4h, df_1h["datetime"].iloc[i])
        if atr_4h_val is None or atr_4h_val == 0:
            continue
        sl_distance = STOP_ATR_MULT_4H * atr_4h_val
        if direction == "LONG":
            sl, tp = entry_price - sl_distance, entry_price + TP_R_MULT * sl_distance
        else:
            sl, tp = entry_price + sl_distance, entry_price - TP_R_MULT * sl_distance
        detail = find_outcome_detailed(df_5m, entry_time, entry_price, sl, tp, None, direction, lookahead_hours=LOOKAHEAD_HOURS_FOR_OUTCOME)
        unconditional_r.append(detail["r_multiple"])

    fade_r = []
    for e in events:
        fade_direction = "SHORT" if e["direction"] == "LONG" else "LONG"
        # mirror the SL/TP around entry for the opposite direction, same distance
        risk = abs(e["entry"] - e["sl"])
        if fade_direction == "LONG":
            sl, tp = e["entry"] - risk, e["entry"] + TP_R_MULT * risk
        else:
            sl, tp = e["entry"] + risk, e["entry"] - TP_R_MULT * risk
        detail = find_outcome_detailed(df_5m, e["entry_time"], e["entry"], sl, tp, None, fade_direction, lookahead_hours=LOOKAHEAD_HOURS_FOR_OUTCOME)
        fade_r.append(detail["r_multiple"])

    return {
        "unconditional_avg_r": sum(unconditional_r) / len(unconditional_r) if unconditional_r else None,
        "unconditional_n": len(unconditional_r),
        "fade_avg_r": sum(fade_r) / len(fade_r) if fade_r else None,
        "fade_n": len(fade_r),
    }


def check_concentration(r_values: list[float]) -> dict:
    """Does a small cluster of trades explain most of the result? Per the
    frozen spec: FAILED if >80% of positive R comes from the top 20% of
    trades (by R value)."""
    n = len(r_values)
    if n == 0:
        return {"concentrated": False, "top_20pct_share": None}
    total_positive = sum(r for r in r_values if r > 0)
    if total_positive <= 0:
        return {"concentrated": False, "top_20pct_share": None}
    top_count = max(1, int(n * 0.2))
    top_sum = sum(sorted(r_values, reverse=True)[:top_count])
    share = top_sum / total_positive
    return {"concentrated": share > CONCENTRATION_FAILURE_THRESHOLD, "top_20pct_share": share}


async def run(period: str = "dev"):
    """period: 'dev' (2025-01-01 to 2025-08-31) or 'oos' (2025-09-01 to
    2025-12-31). OOS should only be run after explicitly reviewing and
    approving the dev-period result -- this function does not gate that
    decision itself, per the two-step process in the approved spec."""
    if period == "dev":
        start_date, end_date = DEV_START, DEV_END
    elif period == "oos":
        start_date, end_date = OOS_START, OOS_END
    else:
        raise ValueError("period must be 'dev' or 'oos'")

    result_symbol = f"BTC/USD [displacement-1h] [{period}] [{start_date} to {end_date}]"
    cleared = db.clear_backtest_data(result_symbol)
    if cleared:
        logger.info("Cleared %d prior evaluation(s) for %s", cleared, result_symbol)

    logger.info("Fetching 1H, 4H, and 5m BTC data for %s period (requested %s to %s)...", period, start_date, end_date)
    target_start = datetime.strptime(start_date, "%Y-%m-%d")
    target_end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    df_1h = fetch_paginated_history(API_SYMBOL, "1h", target_start, target_end)
    df_4h = fetch_paginated_history(API_SYMBOL, "4h", target_start, target_end)
    df_5m = fetch_paginated_history(API_SYMBOL, "5min", target_start, target_end)

    if df_1h is None or len(df_1h) < EVENT_ATR_LOOKBACK + 10:
        logger.error("Not enough 1H data -- aborting")
        return
    if df_4h is None or len(df_4h) < WARMUP_BARS:
        logger.error("Not enough 4H data -- aborting")
        return
    if df_5m is None or len(df_5m) < WARMUP_BARS:
        logger.error("Not enough 5m data -- aborting")
        return

    # The REAL achieved range is whichever timeframe's data is shallowest
    # (5m typically limits this, since the free tier caps candles per
    # request regardless of interval, and 5m needs far more candles to
    # cover the same span). Report and bound everything downstream by
    # this ACTUAL overlap, not the originally requested dates -- a
    # request for 2023 that only achieves 2025 should say so plainly.
    actual_start = max(df_1h["datetime"].min(), df_4h["datetime"].min(), df_5m["datetime"].min())
    actual_end = min(df_1h["datetime"].max(), df_4h["datetime"].max(), df_5m["datetime"].max())
    logger.info("1H: %d candles (%s to %s)", len(df_1h), df_1h["datetime"].min(), df_1h["datetime"].max())
    logger.info("4H: %d candles (%s to %s)", len(df_4h), df_4h["datetime"].min(), df_4h["datetime"].max())
    logger.info("5m: %d candles (%s to %s)", len(df_5m), df_5m["datetime"].min(), df_5m["datetime"].max())
    logger.info("Actual usable overlap across all 3 timeframes: %s to %s", actual_start, actual_end)

    events = detect_displacement_events(df_1h, df_4h)
    logger.info("Detected %d displacement events", len(events))

    n_long = sum(1 for e in events if e["direction"] == "LONG")
    n_short = len(events) - n_long

    trades_evaluated, expired_count = 0, 0
    r_values = []
    trade_records = []
    for event in events:
        if event["entry_time"] + timedelta(hours=LOOKAHEAD_HOURS_FOR_OUTCOME) > df_5m["datetime"].max():
            continue

        eval_id = db.save_evaluation(
            source="backtest", symbol=result_symbol, strategy_type=STRATEGY_TYPE, action=event["direction"],
            confidence=60, entry=event["entry"], sl=event["sl"], tp1=event["tp"], tp2=None,
            details=f"atr_normalized_magnitude={event['atr_normalized_magnitude']:.2f} preceding_vol={event['preceding_volatility_atr']:.2f}",
            evaluated_at=event["entry_time"].isoformat(),
        )
        detail = find_outcome_detailed(
            df_5m, event["entry_time"], event["entry"], event["sl"], event["tp"], None,
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
        r_values.append(detail["r_multiple"])
        trade_records.append({
            "direction": event["direction"], "r": detail["r_multiple"],
            "mae": detail["mae_before_tp1_r"], "mfe": detail["mfe_before_tp1_r"],
            "entry": event["entry"], "sl": event["sl"],
        })
        if detail["outcome"] == "EXPIRED":
            expired_count += 1

    logger.info("Backtest complete: %d events, %d trades evaluated", len(events), trades_evaluated)

    baselines = compute_baseline_comparisons(df_1h, df_4h, df_5m, events)
    concentration = check_concentration(r_values)

    n = len(r_values)
    gross_avg_r = sum(r_values) / n if n else None
    days_span = max((actual_end - actual_start).total_seconds() / 86400, 1e-9)
    trades_per_day = n / days_span if n else 0.0

    # Additional stats requested for the full report: PF, win rate, total
    # net R, MAE/MFE, max drawdown, and long/short breakdown -- all pure
    # reporting additions, none touch the frozen trading rules above.
    wins = sum(1 for r in r_values if r > 0)
    win_rate = wins / n if n else None
    gains = sum(r for r in r_values if r > 0)
    losses_sum = abs(sum(r for r in r_values if r < 0))
    gross_pf = gains / losses_sum if losses_sum > 0 else None

    avg_mae = sum(t["mae"] for t in trade_records if t["mae"] is not None) / n if n else None
    avg_mfe = sum(t["mfe"] for t in trade_records if t["mfe"] is not None) / n if n else None

    cum, peak, max_dd = 0.0, 0.0, 0.0
    for r in r_values:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    long_r = [t["r"] for t in trade_records if t["direction"] == "LONG"]
    short_r = [t["r"] for t in trade_records if t["direction"] == "SHORT"]
    long_avg = sum(long_r) / len(long_r) if long_r else None
    short_avg = sum(short_r) / len(short_r) if short_r else None

    # Cost sensitivity, using the same entry/SL-distance-based model as
    # the rest of this project (reusing the stored entry/SL directly)
    cost_curve = []
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT e.entry, e.sl, o.r_multiple FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.symbol = ? AND e.strategy_type = ? AND o.outcome IN ('WIN','LOSS')""",
            (result_symbol, STRATEGY_TYPE),
        ).fetchall()
    finally:
        conn.close()
    for spread_pct in COST_SENSITIVITY_LEVELS:
        adjusted = []
        for entry, sl, r in rows:
            risk_price = abs(entry - sl)
            if risk_price == 0:
                continue
            cost_r = (entry * spread_pct / 100) / risk_price
            adjusted.append(r - cost_r)
        net_avg = sum(adjusted) / len(adjusted) if adjusted else None
        cost_curve.append({"spread_pct": spread_pct, "net_r": net_avg, "net_r_day": (trades_per_day * net_avg) if net_avg is not None else None})

    base_cost_row = next((c for c in cost_curve if c["spread_pct"] == BASE_SPREAD_PCT), None)
    net_avg_r_base = base_cost_row["net_r"] if base_cost_row else None
    net_r_day_base = base_cost_row["net_r_day"] if base_cost_row else None
    total_net_r_base = net_avg_r_base * n if net_avg_r_base is not None else None
    net_pf_base = None
    net_adjusted_base = []
    for entry, sl, r in rows:
        risk_price = abs(entry - sl)
        if risk_price == 0:
            continue
        cost_r = (entry * BASE_SPREAD_PCT / 100) / risk_price
        net_adjusted_base.append(r - cost_r)
    if net_adjusted_base:
        net_gains = sum(r for r in net_adjusted_base if r > 0)
        net_losses = abs(sum(r for r in net_adjusted_base if r <= 0))
        net_pf_base = net_gains / net_losses if net_losses > 0 else None

    # Monthly breakdown -- reuses the same already-tested analytics
    # function, since results are tagged into the standard schema
    monthly = analytics.get_monthly_breakdown(result_symbol, STRATEGY_TYPE)

    # --- Falsification classification ---
    if gross_avg_r is None or gross_avg_r <= 0:
        classification = "FAILED"
        reason = "Gross expectancy is not positive."
    elif net_avg_r_base is None or net_avg_r_base <= 0:
        classification = "FAILED"
        reason = f"Net expectancy is not positive at the {BASE_SPREAD_PCT}% base cost assumption."
    elif n < MIN_EVENTS_NOT_FAILED:
        classification = "FAILED"
        reason = f"Fewer than {MIN_EVENTS_NOT_FAILED} completed trades ({n})."
    elif concentration["concentrated"]:
        classification = "FAILED"
        reason = f"Result is concentrated: top 20% of trades explain {concentration['top_20pct_share']*100:.0f}% of positive R."
    elif n < MIN_EVENTS_NOT_INCONCLUSIVE or trades_per_day < MIN_FREQUENCY_NOT_INCONCLUSIVE:
        classification = "INCONCLUSIVE"
        reason = f"Positive net expectancy, but sample (n={n}) or frequency ({trades_per_day:.3f}/day) is too small to draw a confident conclusion."
    else:
        classification = "PROMISING" if net_r_day_base and net_r_day_base >= MIN_ECONOMIC_R_PER_DAY else "PROMISING (below the OOS-worthiness gate, but net positive)"
        reason = "Net positive, adequate sample and frequency -- worth reviewing baselines before deciding on OOS testing."

    # --- Scientific verdict (independent of the economic classification) ---
    info_content = None
    if baselines["unconditional_avg_r"] is not None and gross_avg_r is not None:
        edge_over_unconditional = gross_avg_r - baselines["unconditional_avg_r"]
        edge_over_fade = (gross_avg_r - baselines["fade_avg_r"]) if baselines["fade_avg_r"] is not None else None
        info_content = edge_over_unconditional > 0 and (edge_over_fade is None or edge_over_fade > 0)

    lines = [
        f"*⚡ BTC Large-Displacement Momentum ({period.upper()})*",
        f"_Requested {start_date} to {end_date}. Actual usable range achieved: "
        f"{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')} "
        f"({days_span:.0f} days) -- this is the real free-tier depth limit for BTC across all 3 timeframes, "
        f"discovered by the fetch itself, not assumed._",
        "_A large-displacement candle is a PROXY only -- not confirmed liquidation data. All parameters frozen before this test; none adjusted based on this result._\n",
    ]
    lines.append(f"Events detected: {len(events)} ({n_long} LONG, {n_short} SHORT)")
    lines.append(f"Trades evaluated: {trades_evaluated} ({expired_count} expired, {n} resolved WIN/LOSS), {trades_per_day:.3f} trades/day")
    lines.append("")
    if gross_avg_r is not None:
        pf_str = f"{gross_pf:.2f}" if gross_pf is not None else "N/A"
        wr_str = f"{win_rate*100:.0f}%" if win_rate is not None else "N/A"
        net_pf_str = f"{net_pf_base:.2f}" if net_pf_base is not None else "N/A"
        lines.append(f"Gross expectancy: {gross_avg_r:+.3f}R/trade, PF {pf_str}, win rate {wr_str}")
        lines.append(f"Net expectancy (base {BASE_SPREAD_PCT}% cost): {net_avg_r_base:+.3f}R/trade, net PF {net_pf_str}" if net_avg_r_base is not None else "Net expectancy: N/A")
        lines.append(f"Net R/day (base cost): {net_r_day_base:+.3f}" if net_r_day_base is not None else "")
        lines.append(f"Total net R (base cost, all {n} trades): {total_net_r_base:+.2f}R" if total_net_r_base is not None else "")
        if avg_mae is not None:
            lines.append(f"Average MAE: {avg_mae:.2f}R, average MFE: {avg_mfe:.2f}R" if avg_mfe is not None else f"Average MAE: {avg_mae:.2f}R")
        lines.append(f"Max drawdown (gross R): {max_dd:.2f}R")
        if long_avg is not None or short_avg is not None:
            long_str = f"{long_avg:+.3f}R (n={len(long_r)})" if long_avg is not None else "N/A"
            short_str = f"{short_avg:+.3f}R (n={len(short_r)})" if short_avg is not None else "N/A"
            lines.append(f"LONG avg: {long_str} | SHORT avg: {short_str}")
    lines.append("")
    if monthly:
        lines.append("*Monthly results*")
        for m in monthly:
            if m["trades"] == 0:
                continue
            pf_m = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "N/A"
            lines.append(f"  {m['month']}: n={m['trades']}, avg R {m['avg_r']:+.2f}, PF {pf_m}")
        lines.append("")
    lines.append("*Cost sensitivity*")
    for c in cost_curve:
        if c["net_r"] is None:
            lines.append(f"  {c['spread_pct']:.2f}%: no data")
        else:
            lines.append(f"  {c['spread_pct']:.2f}%: net {c['net_r']:+.3f}R/trade, {c['net_r_day']:+.3f} R/day")
    lines.append("")
    lines.append("*Baseline comparison (does the event contain real information?)*")
    if baselines["unconditional_avg_r"] is not None:
        lines.append(f"  Unconditional (any 1H candle, same rules, n={baselines['unconditional_n']}): {baselines['unconditional_avg_r']:+.3f}R avg")
    if baselines["fade_avg_r"] is not None:
        lines.append(f"  Fade (opposite direction on same events, n={baselines['fade_n']}): {baselines['fade_avg_r']:+.3f}R avg")
    lines.append(f"  Event gross: {gross_avg_r:+.3f}R avg" if gross_avg_r is not None else "")
    lines.append("")
    lines.append(f"Concentration check: top 20% of trades = {concentration['top_20pct_share']*100:.0f}% of positive R" if concentration["top_20pct_share"] is not None else "Concentration check: N/A (no positive R)")
    lines.append("")
    lines.append(f"*ECONOMIC classification: {classification}*")
    lines.append(f"_{reason}_")
    lines.append("")
    if info_content is not None:
        sci_verdict = "Event shows genuine information content (beats both unconditional and fade baselines)." if info_content else "Event does NOT clearly beat the baselines -- may just reflect generic BTC momentum/drift, not real event-specific information."
        lines.append(f"*SCIENTIFIC verdict: {sci_verdict}*")
    lines.append(
        "\n_These are two separate questions: the ECONOMIC classification is about whether this specifically helps the "
        "1%/day portfolio objective; the SCIENTIFIC verdict is about whether the mechanism is real at all. A real but "
        "small/rare edge can be scientifically interesting while remaining economically immaterial for this goal._"
    )
    if period == "dev":
        lines.append(
            "\n_Per protocol: do NOT run the out-of-sample period until this dev result has been explicitly reviewed. "
            "No parameters should be changed regardless of this outcome._"
        )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent BTC displacement report (%s)", period)


if __name__ == "__main__":
    asyncio.run(run(period="dev"))
