"""ML Study 1 -- Feature Engineering for XAU 4H Signal Quality Prediction.

CRITICAL TIMING RULE (deliberately more conservative than the original
historical_backtest.py engine, per explicit agreement): every feature is
computed using ONLY the last 4H candle whose full period had genuinely,
unambiguously closed STRICTLY BEFORE the entry timestamp. The original
backtest engine's slice_up_to() uses an inclusive `datetime <= cutoff_dt`
filter -- since a 4H candle's timestamp marks its START, this can include
a candle whose full period had not actually elapsed yet at the simulated
decision time, because the underlying historical data already contains
that candle's final, complete OHLC values. This does not change the
original strategy's own validated results (untouched here) -- it only
governs how NEW features are computed for this ML study specifically.

Existing stored fields (confidence, entry, sl, tp1, tp2, resolved
outcome/R) are used EXACTLY as originally recorded -- never recomputed.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta

from bot import db
from bot.historical_backtest import fetch_paginated_history
from bot.analytics import get_default_spread_pct, _variant_exclusion_sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_features")

API_SYMBOL = "XAU/USD"
CANDLE_HOURS = 4
ATR_PERCENTILE_LOOKBACK = 100
SWING_LOOKBACK = 50  # candles, for recent swing high/low distance

FEATURE_COLUMNS = [
    "confidence",
    "dist_from_ema20", "dist_from_ema50", "ema20_vs_ema50", "ema20_slope", "ema50_slope",
    "rsi14", "return_1", "return_3", "return_6",
    "atr14", "atr14_over_price", "atr_percentile",
    "prev_body_over_atr", "prev_range_over_atr", "prev_upper_wick_ratio", "prev_lower_wick_ratio", "prev_body_direction",
    "dist_from_swing_high", "dist_from_swing_low",
]


def get_xau_signals(symbol_tag: str, strategy_type: str, start_date: str, end_date: str) -> list[dict]:
    """Pulls existing resolved XAU trend signals EXACTLY as originally
    stored -- entry/sl/action/confidence/resolved outcome/R are never
    recomputed, preserving the real historical signal population
    (including the continuation-blocking already baked into how these
    were generated)."""
    conn = db._connect()
    try:
        rows = conn.execute(
            f"""SELECT e.evaluated_at, e.action, e.entry, e.sl, e.confidence, o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(symbol_tag)}
                 AND o.outcome IN ('WIN', 'LOSS')
                 AND e.evaluated_at >= ? AND e.evaluated_at < ?
               ORDER BY e.evaluated_at ASC""",
            (strategy_type, f"{symbol_tag}%", start_date, end_date),
        ).fetchall()
    finally:
        conn.close()
    cols = ["evaluated_at", "action", "entry", "sl", "confidence", "outcome", "r_multiple"]
    return [dict(zip(cols, row)) for row in rows]


def fetch_xau_4h(start_date: str, end_date: str, warmup_days: int = 60) -> pd.DataFrame | None:
    """Fetches 4H OHLC with a buffer BEFORE start_date so early signals
    still have enough trailing history for ATR-percentile/EMA50/swing
    lookbacks -- the buffer is for feature computation only, never for
    signals themselves."""
    target_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=warmup_days)
    target_end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    return fetch_paginated_history(API_SYMBOL, "4h", target_start, target_end)


def find_last_completed_4h_candle(df_4h: pd.DataFrame, entry_time: datetime) -> int | None:
    """CONSERVATIVE rule: returns the index of the last 4H candle whose
    full period (start + CANDLE_HOURS) closed STRICTLY BEFORE entry_time.
    Never the candle still forming at entry, even if the historical
    dataset contains its final OHLC. Uses strict '<' (not '<=') so this
    stays internally consistent with the downstream timing-audit check,
    which also requires feature_candle_close STRICTLY LESS THAN entry --
    a candle that closes at the exact same instant as entry is treated as
    not yet safely usable, not as a boundary-equal special case."""
    closes_before = df_4h["datetime"] + pd.Timedelta(hours=CANDLE_HOURS) < entry_time
    eligible = df_4h[closes_before]
    if len(eligible) == 0:
        return None
    return eligible.index[-1]


def find_inclusive_4h_candle(df_4h: pd.DataFrame, entry_time: datetime) -> int | None:
    """The ORIGINAL engine's inclusive rule (datetime <= entry_time) --
    used ONLY to quantify how many signals the conservative rule actually
    affects, per the explicit request to measure this rather than assume."""
    eligible = df_4h[df_4h["datetime"] <= entry_time]
    if len(eligible) == 0:
        return None
    return eligible.index[-1]


def compute_features_asof(df_4h: pd.DataFrame, candle_idx: int) -> dict | None:
    """Computes every feature using ONLY df_4h.iloc[:candle_idx+1] -- the
    candle at candle_idx and everything before it. Never touches a later
    index."""
    if candle_idx < max(ATR_PERCENTILE_LOOKBACK, SWING_LOOKBACK, 50) + 14:
        return None  # not enough trailing history for a stable computation

    window = df_4h.iloc[:candle_idx + 1]
    close = window["close"]
    high = window["high"]
    low = window["low"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi()
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    last_close = close.iloc[-1]
    last_ema20, last_ema50 = ema20.iloc[-1], ema50.iloc[-1]
    last_atr = atr14.iloc[-1]
    if pd.isna(last_atr) or last_atr == 0 or pd.isna(last_ema20) or pd.isna(last_ema50) or pd.isna(rsi14.iloc[-1]):
        return None

    atr_window = atr14.iloc[-ATR_PERCENTILE_LOOKBACK:].dropna()
    atr_percentile = 100 * sum(1 for v in atr_window if v <= last_atr) / len(atr_window) if len(atr_window) > 0 else None

    prev = window.iloc[-1]
    prev_range = prev["high"] - prev["low"]
    prev_body = abs(prev["close"] - prev["open"])
    prev_upper_wick = prev["high"] - max(prev["close"], prev["open"])
    prev_lower_wick = min(prev["close"], prev["open"]) - prev["low"]

    swing_window = window.iloc[-SWING_LOOKBACK:]
    swing_high = swing_window["high"].max()
    swing_low = swing_window["low"].min()

    return {
        "dist_from_ema20": (last_close - last_ema20) / last_atr,
        "dist_from_ema50": (last_close - last_ema50) / last_atr,
        "ema20_vs_ema50": (last_ema20 - last_ema50) / last_atr,
        "ema20_slope": (ema20.iloc[-1] - ema20.iloc[-5]) / last_atr if len(ema20) > 5 else None,
        "ema50_slope": (ema50.iloc[-1] - ema50.iloc[-5]) / last_atr if len(ema50) > 5 else None,
        "rsi14": rsi14.iloc[-1],
        "return_1": (close.iloc[-1] - close.iloc[-2]) / last_atr if len(close) > 1 else None,
        "return_3": (close.iloc[-1] - close.iloc[-4]) / last_atr if len(close) > 3 else None,
        "return_6": (close.iloc[-1] - close.iloc[-7]) / last_atr if len(close) > 6 else None,
        "atr14": last_atr,
        "atr14_over_price": last_atr / last_close,
        "atr_percentile": atr_percentile,
        "prev_body_over_atr": prev_body / last_atr,
        "prev_range_over_atr": prev_range / last_atr,
        "prev_upper_wick_ratio": prev_upper_wick / prev_range if prev_range > 0 else 0.0,
        "prev_lower_wick_ratio": prev_lower_wick / prev_range if prev_range > 0 else 0.0,
        "prev_body_direction": 1.0 if prev["close"] > prev["open"] else (-1.0 if prev["close"] < prev["open"] else 0.0),
        "dist_from_swing_high": (swing_high - last_close) / last_atr,
        "dist_from_swing_low": (last_close - swing_low) / last_atr,
        "_feature_candle_close_time": window["datetime"].iloc[-1] + pd.Timedelta(hours=CANDLE_HOURS),
    }


def build_feature_dataset(symbol_tag: str, strategy_type: str, start_date: str, end_date: str,
                           spread_pct: float | None = None) -> dict:
    """Builds the full feature matrix + labels + timing audit + the
    conservative-vs-inclusive discrepancy count, all in one pass."""
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct(API_SYMBOL)
    signals = get_xau_signals(symbol_tag, strategy_type, start_date, end_date)
    if not signals:
        return {"n_signals": 0, "rows": [], "timing_audit_passed": None, "n_affected_by_conservative_rule": None}

    df_4h = fetch_xau_4h(start_date, end_date)
    if df_4h is None or len(df_4h) == 0:
        return {"n_signals": len(signals), "rows": [], "timing_audit_passed": None, "error": "4H fetch failed"}

    rows = []
    n_affected = 0
    n_skipped_insufficient_history = 0
    timing_violations = 0

    for sig in signals:
        entry_time = datetime.fromisoformat(sig["evaluated_at"])
        if entry_time.tzinfo is not None:
            entry_time = entry_time.replace(tzinfo=None)

        conservative_idx = find_last_completed_4h_candle(df_4h, entry_time)
        inclusive_idx = find_inclusive_4h_candle(df_4h, entry_time)
        if conservative_idx is not None and inclusive_idx is not None and conservative_idx != inclusive_idx:
            n_affected += 1

        if conservative_idx is None:
            n_skipped_insufficient_history += 1
            continue

        features = compute_features_asof(df_4h, conservative_idx)
        if features is None:
            n_skipped_insufficient_history += 1
            continue

        feature_close_time = features.pop("_feature_candle_close_time")
        if feature_close_time >= entry_time:
            timing_violations += 1
            continue  # would be a leakage violation -- excluded, never silently included

        risk_price = abs(sig["entry"] - sig["sl"]) if sig["entry"] is not None and sig["sl"] is not None else None
        if not risk_price:
            continue
        cost_r = (sig["entry"] * spread_pct / 100) / risk_price
        net_r = sig["r_multiple"] - cost_r

        row = dict(features)
        row["confidence"] = sig["confidence"]
        row["net_r"] = net_r
        row["gross_r"] = sig["r_multiple"]
        row["label"] = 1 if net_r > 0 else 0
        row["evaluated_at"] = sig["evaluated_at"]
        row["action"] = sig["action"]
        row["entry"] = sig["entry"]
        row["sl"] = sig["sl"]
        row["_feature_candle_close_time"] = feature_close_time.isoformat()
        rows.append(row)

    return {
        "n_signals": len(signals),
        "n_usable_rows": len(rows),
        "n_skipped_insufficient_history": n_skipped_insufficient_history,
        "n_affected_by_conservative_rule": n_affected,
        "timing_violations_excluded": timing_violations,
        "timing_audit_passed": timing_violations == 0,
        "rows": rows,
    }
