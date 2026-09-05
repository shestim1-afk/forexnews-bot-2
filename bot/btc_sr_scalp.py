"""BTC Daily/Weekly Support-Resistance Scalp -- frozen specification.

HYPOTHESIS: BTC price bouncing off objectively-defined daily (or weekly)
swing-level support/resistance may produce a tradeable mean-reversion
scalp.

IMPORTANT FRAMING: this shares conceptual DNA with two things already
tested in this project -- the original "range" strategy (support/
resistance bounce with RSI confirmation, tested across XAU/BTC/GBPJPY,
came back weak-to-negative uniformly) and the structural-level-rejection
study on XAU (INCONCLUSIVE, leaning negative). This is being run anyway,
per explicit request, as a genuinely new combination: BTC specifically
(not previously tested with this level methodology), a mean-reversion
bounce entry (not XAU's rejection-continuation entry), and the
already-proven no-lookahead level-construction machinery reused from
xau_level_rejection.py rather than rebuilt from scratch.

FROZEN, PRE-DECLARED TWO-TIER TEST: daily levels run first on 2025 data.
ONLY IF that specific result is net non-positive does the ALSO-frozen
weekly-level variant run next. Both specifications are fully declared
below before either is executed -- this is not parameter search after
seeing a result.

Level construction (identical to the proven xau_level_rejection.py
methodology, just pointed at BTC and, for the weekly variant, weekly
swings instead of daily): 5-bar fractal swings, clustered within 0.5%,
requiring >=2 touches, no-lookahead enforced (only swings confirmed
strictly before the current timestamp are used).

Entry (mean-reversion bounce, NOT rejection-continuation): a 4H candle's
low comes within 0.3x the 4H ATR of an ACTIVE support level AND that same
candle's close is back above the level (a genuine bounce, not just a
touch) -> LONG at close. Mirrored at resistance for SHORT.

Stop: 0.5x the 4H ATR beyond the level (structural).
Exit: FIXED 1.5R target -- explicitly a temporary research control, same
status as every other fixed-target study in this project (compression,
displacement).
Costs: 0.25% round-trip (0.15% spread + 0.10% slippage), the same
realistic BTC exchange assumption established for the displacement study.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import ta

from . import db
from . import telegram_bot
from .historical_backtest import fetch_paginated_history, find_outcome_detailed
from .xau_level_rejection import detect_swing_points, cluster_swings_into_levels, get_active_levels_as_of, MIN_TOUCHES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("btc_sr_scalp")

API_SYMBOL = "BTC/USD"
STRATEGY_TYPE = "sr_scalp"

APPROACH_ATR_MULT_4H = 0.3
STOP_BUFFER_ATR_MULT = 0.5
TP_R_MULT = 1.5  # temporary research control, see module docstring
SPREAD_PCT = 0.25  # 0.15% spread + 0.10% slippage, same as the BTC displacement study

DAILY_LOOKBACK_CANDLES = 252
WEEKLY_LOOKBACK_CANDLES = 104

TEST_START_2025, TEST_END_2025 = "2025-01-01", "2025-12-31"
TEST_START_2024, TEST_END_2024 = "2024-01-01", "2024-12-31"
LEVEL_HISTORY_BUFFER_DAYS = 400  # buffer before the test window for level construction, no lookahead into the test period itself


ADX_REGIME_THRESHOLD = 20  # frozen -- standard "non-trending" threshold, not selected after seeing results
ADX_PERIOD = 14


def find_approach_and_bounce_events(df_4h: pd.DataFrame, df_level_tf: pd.DataFrame,
                                     level_swings: list[dict], lookback_candles: int,
                                     start_date: datetime, end_date: datetime,
                                     cooldown_candles: int = 4) -> list[dict]:
    """Scans 4H candles for a genuine bounce off an active level -- not
    just a touch. Reuses the already-tested no-lookahead level machinery.
    Also computes ADX(14) at each event's own candle (vectorized once,
    then read at each row -- the same no-lookahead-safe pattern already
    proven in ml_features.py and xau_level_rejection.py: a rolling
    indicator's value at row i depends only on rows <= i)."""
    atr_4h = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"], window=14).average_true_range()
    adx_4h = ta.trend.ADXIndicator(df_4h["high"], df_4h["low"], df_4h["close"], window=ADX_PERIOD).adx()
    events = []
    last_event_idx_for_level: dict[float, int] = {}

    mask = (df_4h["datetime"] >= start_date) & (df_4h["datetime"] < end_date)
    for i in df_4h[mask].index:
        atr_val = atr_4h.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue
        adx_val = adx_4h.iloc[i]
        candle_time = df_4h["datetime"].iloc[i]
        candle_low, candle_high, candle_close = df_4h["low"].iloc[i], df_4h["high"].iloc[i], df_4h["close"].iloc[i]
        threshold = APPROACH_ATR_MULT_4H * atr_val

        active_levels = get_active_levels_as_of(level_swings, candle_time, lookback_candles, df_level_tf, min_touches=MIN_TOUCHES)

        for level in active_levels:
            key = round(level["price"], 2)
            if key in last_event_idx_for_level and i - last_event_idx_for_level[key] < cooldown_candles:
                continue

            # Support bounce: low touches within threshold below the level, close back above it
            if level["price"] - threshold <= candle_low <= level["price"] and candle_close > level["price"]:
                sl = level["price"] - threshold - STOP_BUFFER_ATR_MULT * atr_val
                risk = candle_close - sl
                events.append({
                    "event_time": candle_time, "direction": "LONG", "entry": candle_close,
                    "sl": sl, "tp": candle_close + TP_R_MULT * risk, "level_price": level["price"], "level_strength": level["strength"],
                    "adx_at_entry": adx_val,
                })
                last_event_idx_for_level[key] = i
            # Resistance bounce: high touches within threshold above the level, close back below it
            elif level["price"] <= candle_high <= level["price"] + threshold and candle_close < level["price"]:
                sl = level["price"] + threshold + STOP_BUFFER_ATR_MULT * atr_val
                risk = sl - candle_close
                events.append({
                    "event_time": candle_time, "direction": "SHORT", "entry": candle_close,
                    "sl": sl, "tp": candle_close - TP_R_MULT * risk, "level_price": level["price"], "level_strength": level["strength"],
                    "adx_at_entry": adx_val,
                })
                last_event_idx_for_level[key] = i
    return events


async def run(period: str = "2025", level_timeframe: str = "daily"):
    """period: '2025' or '2024'. level_timeframe: 'daily' or 'weekly' --
    weekly should only be run if the daily result on the SAME period is
    net non-positive, per the pre-declared two-tier test."""
    start_date, end_date = (TEST_START_2025, TEST_END_2025) if period == "2025" else (TEST_START_2024, TEST_END_2024)
    result_symbol = f"BTC/USD [sr-scalp-{level_timeframe}] [{period}]"

    cleared = db.clear_backtest_data(result_symbol)
    if cleared:
        logger.info("Cleared %d prior evaluation(s) for %s", cleared, result_symbol)

    level_interval = "1day" if level_timeframe == "daily" else "1week"
    lookback_candles = DAILY_LOOKBACK_CANDLES if level_timeframe == "daily" else WEEKLY_LOOKBACK_CANDLES

    target_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=LEVEL_HISTORY_BUFFER_DAYS)
    target_end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    logger.info("Fetching %s levels and 4H entry data for BTC/USD...", level_timeframe)
    df_level_tf = fetch_paginated_history(API_SYMBOL, level_interval, target_start, target_end)
    df_4h = fetch_paginated_history(API_SYMBOL, "4h", target_start, target_end)

    if df_level_tf is None or len(df_level_tf) == 0 or df_4h is None or len(df_4h) == 0:
        await telegram_bot.send_text(f"*BTC S/R Scalp ({level_timeframe}, {period})*\n\nData fetch failed -- DATA INSUFFICIENT.")
        return

    actual_start = max(df_level_tf["datetime"].min(), df_4h["datetime"].min())
    covers_period = actual_start <= datetime.strptime(start_date, "%Y-%m-%d")
    logger.info("%s: %d candles (%s to %s). 4H: %d candles (%s to %s). Covers requested period: %s",
                level_timeframe, len(df_level_tf), df_level_tf["datetime"].min(), df_level_tf["datetime"].max(),
                len(df_4h), df_4h["datetime"].min(), df_4h["datetime"].max(), covers_period)

    level_swings = detect_swing_points(df_level_tf)
    events = find_approach_and_bounce_events(
        df_4h, df_level_tf, level_swings, lookback_candles,
        datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1),
    )
    logger.info("Detected %d bounce events", len(events))

    trades_evaluated = 0
    resolved_records = []  # one dict per resolved trade: r, net_r, adx_at_entry
    for event in events:
        if event["event_time"] + timedelta(hours=168) > df_4h["datetime"].max():
            continue
        eval_id = db.save_evaluation(
            source="backtest", symbol=result_symbol, strategy_type=STRATEGY_TYPE, action=event["direction"],
            confidence=60, entry=event["entry"], sl=event["sl"], tp1=event["tp"], tp2=None,
            details=f"level_price={event['level_price']:.2f} level_strength={event['level_strength']} adx={event['adx_at_entry']:.2f}",
            evaluated_at=event["event_time"].isoformat(),
        )
        detail = find_outcome_detailed(df_4h, event["event_time"], event["entry"], event["sl"], event["tp"], None,
                                        event["direction"], lookahead_hours=168)
        db.save_backtest_outcome(eval_id, detail["outcome"], detail["exit_price"], detail["r_multiple"],
                                  mae_r=detail["mae_before_tp1_r"], mfe_r=detail["mfe_before_tp1_r"])
        trades_evaluated += 1
        if detail["outcome"] in ("WIN", "LOSS"):
            risk_price = abs(event["entry"] - event["sl"])
            cost_r = (event["entry"] * SPREAD_PCT / 100) / risk_price if risk_price else 0
            resolved_records.append({
                "r": detail["r_multiple"], "net_r": detail["r_multiple"] - cost_r,
                "adx_at_entry": event["adx_at_entry"],
            })

    def _summarize(records: list[dict]) -> dict:
        n = len(records)
        if n == 0:
            return {"n": 0, "gross_avg_r": None, "net_avg_r": None, "net_r_day": None}
        gross_avg_r = sum(r["r"] for r in records) / n
        net_avg_r = sum(r["net_r"] for r in records) / n
        days_span = max((datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days, 1)
        trades_per_day = n / days_span
        return {"n": n, "gross_avg_r": gross_avg_r, "net_avg_r": net_avg_r, "net_r_day": trades_per_day * net_avg_r}

    unfiltered_stats = _summarize(resolved_records)
    regime_filtered_records = [r for r in resolved_records if r["adx_at_entry"] is not None and not pd.isna(r["adx_at_entry"]) and r["adx_at_entry"] < ADX_REGIME_THRESHOLD]
    filtered_stats = _summarize(regime_filtered_records)

    def _fmt_stats(label: str, stats: dict) -> list[str]:
        out = [f"*{label}*"]
        if stats["n"] == 0:
            out.append("  n=0")
            return out
        out.append(f"  n={stats['n']}, gross avg R {stats['gross_avg_r']:+.3f}, net avg R {stats['net_avg_r']:+.3f}, net R/day {stats['net_r_day']:+.3f}")
        return out

    lines = [
        f"*BTC S/R Scalp ({level_timeframe} levels, {period}) -- with ADX regime filter*",
        "Shares conceptual DNA with the already-tested range strategy and XAU structural-level study (both closed "
        "weak/inconclusive) -- run anyway per explicit request. Adds ONE new, frozen filter (ADX(14) < 20 = ranging, "
        "trade allowed; ADX(14) >= 20 = trending/breakout, trade skipped) on top of the already-tested bounce logic.\n",
    ]
    lines.append(f"Data coverage confirmed back to {start_date}: {'YES' if covers_period else 'NO -- results may be incomplete'}")
    lines.append(f"Bounce events detected: {len(events)}, trades evaluated: {trades_evaluated}, resolved: {unfiltered_stats['n']}")
    lines.append("")
    lines += _fmt_stats("Unfiltered (all bounce trades)", unfiltered_stats)
    lines.append("")
    lines += _fmt_stats(f"ADX-filtered (ranging regime only, n retained: {filtered_stats['n']}/{unfiltered_stats['n']})", filtered_stats)
    lines.append("")

    if unfiltered_stats["n"] > 0:
        beats_unfiltered = (filtered_stats["net_r_day"] is not None and unfiltered_stats["net_r_day"] is not None
                             and filtered_stats["net_r_day"] > unfiltered_stats["net_r_day"])
        adequate_sample = filtered_stats["n"] >= 30
        if not adequate_sample:
            lines.append(f"*Filter classification: INCONCLUSIVE -- filtered sample (n={filtered_stats['n']}) below the 30-trade minimum.*")
        elif beats_unfiltered and filtered_stats["net_avg_r"] > 0:
            lines.append("*Filter classification: PROMISING -- ADX regime filter improves on the unfiltered result with an adequate sample.*")
        else:
            lines.append("*Filter classification: FAILED -- ADX regime filter does not improve on the unfiltered result.*")

        base_classification = "NET POSITIVE" if (unfiltered_stats["net_avg_r"] is not None and unfiltered_stats["net_avg_r"] > 0) else "NET NON-POSITIVE"
        lines.append(f"*Unfiltered base classification: {base_classification}*")
        if base_classification == "NET NON-POSITIVE" and level_timeframe == "daily":
            lines.append("Per the pre-declared two-tier test, the weekly-level variant should now be run on the same period.")
    else:
        lines.append("*Classification: INCONCLUSIVE -- no resolved trades.*")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent BTC S/R scalp report (%s, %s)", level_timeframe, period)


if __name__ == "__main__":
    asyncio.run(run())
