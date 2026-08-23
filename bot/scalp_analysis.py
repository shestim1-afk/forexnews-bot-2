"""Multi-symbol scalp signal generator (BTC/USD, Gold, GBP/JPY), using free
Twelve Data market data and free open-source technical indicators via `ta`.

IMPORTANT, read before trusting this: the confidence score is a heuristic
composite of how strongly several independent signals agree RIGHT NOW. It is
NOT a backtested, calibrated probability of a profitable trade. Every
actionable signal generated gets logged to the database, tagged with which
detection engine produced it (trend / range / liquidity_sweep /
breakout_retest), specifically so it can be checked against what price
actually did afterward and broken down by approach. Treat this as
signal-only data collection, especially relevant given leveraged trading
amplifies the cost of an uncalibrated signal.

This script does NOT place any orders. It only analyzes and reports.

The scheduled hourly-ish digest (run()) only messages Telegram when at
least one symbol has an actionable signal -- deliberately quiet otherwise,
rather than a scheduled ping regardless of content. On-demand commands
(/btc, /gold, /gbpjpy) always show the full diagnostic picture since the
person explicitly asked for it.

Data source note: an earlier version used Binance directly and hit HTTP 451
(geo-blocked) from GitHub Actions' shared runners. Twelve Data doesn't have
this issue and is already relied on elsewhere in this project.
"""

import asyncio
import logging
import os
import time

import requests
import pandas as pd
import ta

from . import config
from . import db
from . import risk_controller
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scalp_analysis")

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TIMEFRAMES = {"4h": "4h", "1h": "1h", "15m": "15min", "5m": "5min"}
CANDLE_LIMIT = 250
REQUEST_DELAY_SECONDS = 8  # keeps us under Twelve Data's free-tier 8 req/min limit

SYMBOLS = [
    {"api": "BTC/USD", "display": "BTC/USD", "news_keywords": ["BTC", "Crypto"]},
    {"api": "XAU/USD", "display": "Gold (XAU/USD)", "news_keywords": ["Gold", "XAU"]},
    {"api": "GBP/JPY", "display": "GBP/JPY", "news_keywords": ["GBP/JPY"]},
]
COMMAND_TO_SYMBOL = {"/btc": "BTC/USD", "/gold": "XAU/USD", "/xauusd": "XAU/USD", "/gbpjpy": "GBP/JPY"}


def fetch_klines(api_symbol: str, interval: str) -> pd.DataFrame | None:
    if not TWELVEDATA_API_KEY:
        logger.warning("TWELVEDATA_API_KEY not set -- cannot fetch price data")
        return None
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": api_symbol,
                "interval": interval,
                "outputsize": CANDLE_LIMIT,
                "apikey": TWELVEDATA_API_KEY,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            logger.warning("Twelve Data error for %s %s: %s", api_symbol, interval, data.get("message", data))
            return None
        values = data["values"]
        values.reverse()
        df = pd.DataFrame(values)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
        return df
    except Exception as e:
        logger.warning("Failed to fetch %s %s candles: %s", api_symbol, interval, e)
        return None


def compute_indicators(df: pd.DataFrame) -> dict:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator().iloc[-1]
    adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_obj = ta.trend.MACD(close)
    macd_diff = macd_obj.macd_diff().iloc[-1]
    stochrsi = ta.momentum.StochRSIIndicator(close).stochrsi().iloc[-1]
    roc = ta.momentum.ROCIndicator(close, window=12).roc().iloc[-1]
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    vwap = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume, window=14).volume_weighted_average_price().iloc[-1]
    vol_ma = volume.rolling(20).mean().iloc[-1]

    ich = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    tenkan = ich.ichimoku_conversion_line().iloc[-1]
    kijun = ich.ichimoku_base_line().iloc[-1]
    span_a = ich.ichimoku_a().iloc[-1]
    span_b = ich.ichimoku_b().iloc[-1]
    cloud_top, cloud_bottom = max(span_a, span_b), min(span_a, span_b)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_width = bb.bollinger_wband()
    current_width = bb_width.iloc[-1]
    avg_width = bb_width.iloc[-60:-1].mean() if len(bb_width) > 60 else None
    squeeze = bool(avg_width and pd.notna(current_width) and current_width < 0.5 * avg_width)

    current_close = close.iloc[-1]
    current_volume = volume.iloc[-1]

    if current_close > ema20 > ema50 > ema200 and adx > 20:
        trend = "bullish"
    elif current_close < ema20 < ema50 < ema200 and adx > 20:
        trend = "bearish"
    else:
        trend = "neutral"

    up_votes = sum([rsi > 50, macd_diff > 0, stochrsi > 0.5, roc > 0])
    down_votes = sum([rsi < 50, macd_diff < 0, stochrsi < 0.5, roc < 0])
    if up_votes >= 3:
        momentum = "bullish"
    elif down_votes >= 3:
        momentum = "bearish"
    else:
        momentum = "neutral"

    if current_close > cloud_top:
        cloud_position = "above"
    elif current_close < cloud_bottom:
        cloud_position = "below"
    else:
        cloud_position = "inside"
    tenkan_kijun = "bullish" if tenkan > kijun else "bearish" if tenkan < kijun else "neutral"

    return {
        "close": current_close,
        "trend": trend,
        "momentum": momentum,
        "adx": adx,
        "atr": atr,
        "rsi": rsi,
        "stochrsi": stochrsi,
        "above_vwap": current_close > vwap,
        "volume_confirmed": current_volume > vol_ma * 1.2 if pd.notna(vol_ma) else False,
        "cloud_position": cloud_position,
        "tenkan_kijun": tenkan_kijun,
        "bb_squeeze": squeeze,
    }


def detect_rsi_divergence(df: pd.DataFrame, window: int = 5, lookback: int = 75,
                           min_gap: float = 5, extreme_floor: float = 10, extreme_ceiling: float = 90) -> str | None:
    """Compares the two most recent significant price swing lows/highs
    against RSI at those same points. Known limitation: RSI is bounded at
    0-100, so a comparison starting from an already-extreme reading can look
    like divergence just from having nowhere further to go -- the floor/
    ceiling guards mitigate this, though it's a heuristic, not a perfect fix."""
    close = df["close"]
    rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
    highs = df["high"].tolist()[-lookback:]
    lows = df["low"].tolist()[-lookback:]
    rsi_vals = rsi_series.tolist()[-lookback:]

    swing_low_idxs, swing_high_idxs = [], []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_low_idxs.append(i)
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_high_idxs.append(i)

    divergence = None
    if len(swing_low_idxs) >= 2:
        i1, i2 = swing_low_idxs[-2], swing_low_idxs[-1]
        r1, r2 = rsi_vals[i1], rsi_vals[i2]
        if lows[i2] < lows[i1] and r2 > r1 and r1 >= extreme_floor and (r2 - r1) >= min_gap:
            divergence = "bullish"
    if len(swing_high_idxs) >= 2:
        i1, i2 = swing_high_idxs[-2], swing_high_idxs[-1]
        r1, r2 = rsi_vals[i1], rsi_vals[i2]
        if highs[i2] > highs[i1] and r2 < r1 and r1 <= extreme_ceiling and (r1 - r2) >= min_gap and divergence is None:
            divergence = "bearish"
    return divergence


def find_nearest_levels(df: pd.DataFrame, current_price: float, window: int = 3) -> tuple[float | None, float | None]:
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    swing_highs, swing_lows = [], []

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append(lows[i])

    resistance_candidates = sorted(set(round(h, 5) for h in swing_highs if h > current_price))
    support_candidates = sorted(set(round(l, 5) for l in swing_lows if l < current_price), reverse=True)

    nearest_resistance = resistance_candidates[0] if resistance_candidates else None
    nearest_support = support_candidates[0] if support_candidates else None
    return nearest_support, nearest_resistance


def detect_range_setup(tf_data: dict, dfs: dict, divergence: str | None) -> dict | None:
    h1 = tf_data["1h"]
    if h1["adx"] >= 20:
        return None

    current_price = h1["close"]
    support, resistance = find_nearest_levels(dfs["1h"], current_price)
    if support is None or resistance is None:
        return None

    atr_1h = h1["atr"]
    m15 = tf_data["15m"]

    near_support = (current_price - support) <= 0.5 * atr_1h
    near_resistance = (resistance - current_price) <= 0.5 * atr_1h

    if near_support and m15["rsi"] < 35:
        reason = f"Near range support ({support:.5f}), RSI(15m)={m15['rsi']:.1f} oversold"
        if divergence == "bullish":
            reason += " + RSI bullish divergence confirms"
        return {
            "direction": "LONG", "entry": current_price,
            "sl": support - 0.5 * atr_1h, "tp": resistance,
            "reason": reason,
        }
    if near_resistance and m15["rsi"] > 65:
        reason = f"Near range resistance ({resistance:.5f}), RSI(15m)={m15['rsi']:.1f} overbought"
        if divergence == "bearish":
            reason += " + RSI bearish divergence confirms"
        return {
            "direction": "SHORT", "entry": current_price,
            "sl": resistance + 0.5 * atr_1h, "tp": support,
            "reason": reason,
        }
    return None


def detect_liquidity_sweep(df: pd.DataFrame, atr: float, window: int = 3, recent_candles: int = 3,
                            min_pierce_atr_mult: float = 0.15) -> dict | None:
    """Detects a stop-hunt pattern: price briefly pierces a well-established
    prior swing high/low (where resting stop orders cluster), then quickly
    rejects back inside the range -- a genuine reversal signal, distinct
    from a confirmed breakout. Established levels are computed excluding
    the most recent candles, so the sweep can't be "detecting" its own
    formation. min_pierce_atr_mult requires both the pierce beyond the level
    and the rejection back past it to be a meaningful fraction of ATR, not
    just noise-level wiggle -- without this, the pattern fires constantly
    on pure noise in a flat, quiet market."""
    n = len(df)
    established = df.iloc[:-recent_candles] if n > recent_candles else df
    highs_e = established["high"].tolist()
    lows_e = established["low"].tolist()
    swing_highs, swing_lows = [], []
    for i in range(window, len(established) - window):
        if highs_e[i] == max(highs_e[i - window:i + window + 1]):
            swing_highs.append(highs_e[i])
        if lows_e[i] == min(lows_e[i - window:i + window + 1]):
            swing_lows.append(lows_e[i])

    min_pierce = min_pierce_atr_mult * atr
    recent = df.iloc[-recent_candles:]
    for _, candle in recent.iterrows():
        relevant_lows = [l for l in swing_lows if l < candle["open"]]
        if relevant_lows:
            nearest_low = max(relevant_lows)
            pierced = nearest_low - candle["low"]
            rejected = candle["close"] - nearest_low
            if pierced >= min_pierce and rejected >= min_pierce:
                return {"direction": "bullish", "swept_level": nearest_low}
        relevant_highs = [h for h in swing_highs if h > candle["open"]]
        if relevant_highs:
            nearest_high = min(relevant_highs)
            pierced = candle["high"] - nearest_high
            rejected = nearest_high - candle["close"]
            if pierced >= min_pierce and rejected >= min_pierce:
                return {"direction": "bearish", "swept_level": nearest_high}
    return None


def detect_breakout_retest(df: pd.DataFrame, atr: float, window: int = 3, lookback_recent: int = 20,
                            breakout_atr_mult: float = 0.3, retest_atr_mult: float = 0.3, min_bars_after: int = 2) -> dict | None:
    """Detects: a level broke decisively (close beyond it by a meaningful
    ATR margin), price later pulled back to retest that level without
    closing back through it (confirming it flipped from resistance to
    support, or vice versa), and the current candle is bouncing away from
    the retest -- a continuation entry, not just any pass through a level."""
    established = df.iloc[:-lookback_recent]
    recent = df.iloc[-lookback_recent:]
    highs_e, lows_e = established["high"].tolist(), established["low"].tolist()
    swing_highs, swing_lows = [], []
    for i in range(window, len(established) - window):
        if highs_e[i] == max(highs_e[i - window:i + window + 1]):
            swing_highs.append(highs_e[i])
        if lows_e[i] == min(lows_e[i - window:i + window + 1]):
            swing_lows.append(lows_e[i])

    recent_closes = recent["close"].tolist()
    recent_lows = recent["low"].tolist()
    recent_highs = recent["high"].tolist()
    current_close = recent_closes[-1]

    candidates = [h for h in swing_highs if h < current_close]
    if candidates:
        level = max(candidates)
        breakout_idx = next((i for i, c in enumerate(recent_closes[:-1]) if c > level + breakout_atr_mult * atr), None)
        if breakout_idx is not None:
            for j in range(breakout_idx + min_bars_after, len(recent_closes)):
                ran_higher = max(recent_closes[breakout_idx:j]) > recent_closes[j] + retest_atr_mult * atr * 0.5
                near_level = abs(recent_lows[j] - level) <= retest_atr_mult * atr
                held_above = recent_closes[j] > level
                if ran_higher and near_level and held_above and current_close > recent_closes[j] and current_close > level:
                    return {"direction": "bullish", "level": level}

    candidates2 = [l for l in swing_lows if l > current_close]
    if candidates2:
        level = min(candidates2)
        breakout_idx = next((i for i, c in enumerate(recent_closes[:-1]) if c < level - breakout_atr_mult * atr), None)
        if breakout_idx is not None:
            for j in range(breakout_idx + min_bars_after, len(recent_closes)):
                ran_lower = min(recent_closes[breakout_idx:j]) < recent_closes[j] - retest_atr_mult * atr * 0.5
                near_level = abs(recent_highs[j] - level) <= retest_atr_mult * atr
                held_below = recent_closes[j] < level
                if ran_lower and near_level and held_below and current_close < recent_closes[j] and current_close < level:
                    return {"direction": "bearish", "level": level}
    return None


def get_news_bias(keywords: list[str]) -> tuple[str, str]:
    rows = []
    for kw in keywords:
        rows.extend(db.get_recent_classifications(kw, hours=2))
    if not rows:
        return "neutral", "no recent relevant news"

    bullish_weight = sum(r["confidence"] for r in rows if r["direction"] == "bullish")
    bearish_weight = sum(r["confidence"] for r in rows if r["direction"] == "bearish")

    if bullish_weight > bearish_weight * 1.3:
        return "bullish", f"{len(rows)} recent item(s), net bullish"
    elif bearish_weight > bullish_weight * 1.3:
        return "bearish", f"{len(rows)} recent item(s), net bearish"
    return "neutral", f"{len(rows)} recent item(s), mixed/neutral"


def score_and_decide(tf_data: dict, news_direction: str) -> dict:
    t4h = tf_data["4h"]["trend"]
    t1h = tf_data["1h"]["trend"]

    if t4h == t1h and t4h in ("bullish", "bearish"):
        direction, score = t4h, 3
    elif "bullish" in (t4h, t1h) and "bearish" not in (t4h, t1h):
        direction, score = "bullish", 1
    elif "bearish" in (t4h, t1h) and "bullish" not in (t4h, t1h):
        direction, score = "bearish", 1
    else:
        direction, score = None, 0

    if direction:
        if tf_data["15m"]["momentum"] == direction:
            score += 2
        if tf_data["5m"]["momentum"] == direction:
            score += 2
        if tf_data["15m"]["volume_confirmed"] or tf_data["5m"]["volume_confirmed"]:
            score += 1
        if (direction == "bullish") == tf_data["5m"]["above_vwap"]:
            score += 1
        if news_direction == direction:
            score += 1
        elif news_direction != "neutral" and news_direction != direction:
            score -= 1
        h1 = tf_data["1h"]
        if direction == "bullish" and h1["cloud_position"] == "above" and h1["tenkan_kijun"] == "bullish":
            score += 1
        elif direction == "bearish" and h1["cloud_position"] == "below" and h1["tenkan_kijun"] == "bearish":
            score += 1

    confidence = max(0, min(100, round(score / 11 * 100)))
    if direction is None or confidence < 55:
        action = "NO TRADE"
    else:
        action = "LONG" if direction == "bullish" else "SHORT"

    return {"direction": direction, "action": action, "confidence": confidence}


def compute_trade_levels(action: str, entry: float, atr: float) -> dict:
    if action == "LONG":
        return {"entry": entry, "sl": entry - 1.5 * atr, "tp1": entry + 1.0 * atr, "tp2": entry + 2.0 * atr}
    elif action == "SHORT":
        return {"entry": entry, "sl": entry + 1.5 * atr, "tp1": entry - 1.0 * atr, "tp2": entry - 2.0 * atr}
    return {"entry": None, "sl": None, "tp1": None, "tp2": None}


def compute_sweep_breakout_levels(direction: str, level: float, entry: float, atr: float,
                                   sl_buffer_atr_mult: float = 0.3, min_stop_atr_mult: float = 0.5,
                                   tp1_atr_mult: float = 1.5) -> dict:
    """Computes SL/TP1 for liquidity-sweep and breakout-retest signals,
    anchored to the swept/broken level but with a MINIMUM stop distance
    from entry enforced. Without this floor: right after a sweep, entry
    (the current price) can already be very close to the level, so a
    purely level-anchored stop can end up tighter than realistic spread/
    slippage -- making backtested wins look far more achievable than
    they'd actually be in live trading. This affects both live signals and
    the historical backtest identically, since both call this function."""
    if direction == "bullish":
        level_anchored_sl = level - sl_buffer_atr_mult * atr
        min_sl = entry - min_stop_atr_mult * atr
        sl = min(level_anchored_sl, min_sl)  # whichever is further from entry
        tp1 = entry + tp1_atr_mult * atr
    else:
        level_anchored_sl = level + sl_buffer_atr_mult * atr
        min_sl = entry + min_stop_atr_mult * atr
        sl = max(level_anchored_sl, min_sl)
        tp1 = entry - tp1_atr_mult * atr
    return {"sl": sl, "tp1": tp1}


def calculate_position_size(display_symbol: str, entry: float | None, sl: float | None) -> str | None:
    """Given the account size and risk % configured in config.py (or via
    ACCOUNT_SIZE_USD / RISK_PCT_PER_TRADE env vars), computes the position
    size that risks exactly that amount if SL is hit. Forex sizing is
    approximate -- verify against your broker's own calculator before using
    real numbers, since precise pip-value conversion depends on your
    account's currency."""
    if entry is None or sl is None or entry == sl:
        return None
    risk_amount = config.ACCOUNT_SIZE_USD * (config.RISK_PCT_PER_TRADE / 100)
    price_risk_per_unit = abs(entry - sl)
    units = risk_amount / price_risk_per_unit

    if "BTC" in display_symbol:
        return f"{units:.4f} BTC (risking ${risk_amount:.2f})"
    elif "Gold" in display_symbol or "XAU" in display_symbol:
        return f"{units:.2f} oz (risking ${risk_amount:.2f})"
    else:
        lots = units / 100000
        return f"{units:.0f} units (~{lots:.4f} std lots) -- approx, verify with your broker (risking ${risk_amount:.2f})"


LABEL_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def format_symbol_block(display_symbol: str, tf_data: dict, decision: dict, levels: dict,
                         news_direction: str, news_summary: str, range_setup: dict | None,
                         divergence: str | None, sweep: dict | None, breakout_retest: dict | None) -> str:
    """Always builds the FULL diagnostic picture (used as-is for on-demand
    /commands). The scheduled digest decides separately whether to include
    this block based on is_actionable()."""
    lines = [f"*{display_symbol}*"]
    for tf_label in ["4h", "1h", "15m", "5m"]:
        d = tf_data[tf_label]
        emoji = LABEL_EMOJI.get(d["trend"], "⚪")
        lines.append(f"{tf_label.upper()}: {emoji} {d['trend'].capitalize()}")

    h1 = tf_data["1h"]
    cloud_emoji = {"above": "🟢", "below": "🔴", "inside": "⚪"}[h1["cloud_position"]]
    lines.append(f"Ichimoku(1H): {cloud_emoji} price {h1['cloud_position']} cloud, Tenkan/Kijun {h1['tenkan_kijun']}")

    if h1["bb_squeeze"]:
        lines.append("⚡ BB squeeze on 1H -- volatility compressed, breakout may be near")

    if divergence:
        div_emoji = "🟢" if divergence == "bullish" else "🔴"
        lines.append(f"{div_emoji} RSI(15m) {divergence} divergence -- possible exhaustion/reversal warning")

    news_emoji = LABEL_EMOJI.get(news_direction, "⚪")
    lines.append(f"News: {news_emoji} {news_summary}")

    action_emoji = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}[decision["action"]]
    lines.append(f"{action_emoji} Trend: *{decision['action']}*" + (f" ({decision['confidence']}%)" if decision["action"] != "NO TRADE" else ""))
    if decision["action"] != "NO TRADE":
        lines.append(f"  Entry {levels['entry']:.5f} | SL {levels['sl']:.5f} | TP1 {levels['tp1']:.5f} | TP2 {levels['tp2']:.5f}")
        size = calculate_position_size(display_symbol, levels["entry"], levels["sl"])
        if size:
            lines.append(f"  Size: {size}")

    if range_setup:
        rs_emoji = "🟢" if range_setup["direction"] == "LONG" else "🔴"
        lines.append(f"{rs_emoji} Range: *{range_setup['direction']}* -- {range_setup['reason']}")
        lines.append(f"  Entry {range_setup['entry']:.5f} | SL {range_setup['sl']:.5f} | TP {range_setup['tp']:.5f}")
        size = calculate_position_size(display_symbol, range_setup["entry"], range_setup["sl"])
        if size:
            lines.append(f"  Size: {size}")
    elif tf_data["1h"]["adx"] < 20:
        lines.append("⚪ Range: 1H ranging, no trigger nearby yet")

    if sweep:
        sw_emoji = "🟢" if sweep["direction"] == "bullish" else "🔴"
        action_word = "LONG" if sweep["direction"] == "bullish" else "SHORT"
        lines.append(f"{sw_emoji} Liquidity sweep: *{action_word}* -- swept {sweep['swept_level']:.5f} and rejected")
        entry = tf_data["15m"]["close"]
        atr15 = tf_data["15m"]["atr"]
        sw_levels = compute_sweep_breakout_levels(sweep["direction"], sweep["swept_level"], entry, atr15)
        sl, tp1 = sw_levels["sl"], sw_levels["tp1"]
        lines.append(f"  Entry {entry:.5f} | SL {sl:.5f} | TP1 {tp1:.5f}")
        size = calculate_position_size(display_symbol, entry, sl)
        if size:
            lines.append(f"  Size: {size}")

    if breakout_retest:
        br_emoji = "🟢" if breakout_retest["direction"] == "bullish" else "🔴"
        action_word = "LONG" if breakout_retest["direction"] == "bullish" else "SHORT"
        lines.append(f"{br_emoji} Breakout+retest: *{action_word}* -- level {breakout_retest['level']:.5f} held on retest")
        entry = tf_data["15m"]["close"]
        atr15 = tf_data["15m"]["atr"]
        br_levels = compute_sweep_breakout_levels(breakout_retest["direction"], breakout_retest["level"], entry, atr15)
        sl, tp1 = br_levels["sl"], br_levels["tp1"]
        lines.append(f"  Entry {entry:.5f} | SL {sl:.5f} | TP1 {tp1:.5f}")
        size = calculate_position_size(display_symbol, entry, sl)
        if size:
            lines.append(f"  Size: {size}")

    return "\n".join(lines)


def is_actionable(decision: dict, range_setup: dict | None, sweep: dict | None, breakout_retest: dict | None) -> bool:
    return decision["action"] != "NO TRADE" or range_setup is not None or sweep is not None or breakout_retest is not None


def analyze_symbol(symbol_cfg: dict) -> tuple[str, list[dict], bool]:
    """Runs the full pipeline for one symbol. Returns (message_block,
    actionable_signals, is_actionable) where actionable_signals is a list of
    dicts ready for db.save_scalp_signal(), one per fired signal type (a
    symbol can fire more than one simultaneously, e.g. a trend signal AND a
    liquidity sweep). Used by both the scheduled digest and on-demand
    Telegram commands."""
    api_symbol = symbol_cfg["api"]
    display = symbol_cfg["display"]

    tf_data, dfs = {}, {}
    for i, (label, interval) in enumerate(TIMEFRAMES.items()):
        df = fetch_klines(api_symbol, interval)
        if df is None or len(df) < 205:
            logger.warning("Insufficient data for %s %s", display, label)
            return f"*{display}*\n⚠️ Could not fetch enough price data this cycle.", [], False
        tf_data[label] = compute_indicators(df)
        dfs[label] = df
        if i < len(TIMEFRAMES) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    divergence = detect_rsi_divergence(dfs["15m"])
    news_direction, news_summary = get_news_bias(symbol_cfg["news_keywords"])
    decision = score_and_decide(tf_data, news_direction)
    range_setup = detect_range_setup(tf_data, dfs, divergence)
    sweep = detect_liquidity_sweep(dfs["15m"], tf_data["15m"]["atr"])
    breakout_retest = detect_breakout_retest(dfs["15m"], tf_data["15m"]["atr"])

    entry_price = tf_data["5m"]["close"]
    atr_5m = tf_data["5m"]["atr"]
    levels = compute_trade_levels(decision["action"], entry_price, atr_5m)

    block = format_symbol_block(display, tf_data, decision, levels, news_direction, news_summary,
                                 range_setup, divergence, sweep, breakout_retest)

    actionable_signals = []
    if decision["action"] != "NO TRADE":
        actionable_signals.append({
            "symbol": display, "action": decision["action"], "entry": levels["entry"], "sl": levels["sl"],
            "tp1": levels["tp1"], "tp2": levels["tp2"], "confidence": decision["confidence"],
            "details": f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction} divergence={divergence}",
            "strategy_type": "trend",
        })
    if range_setup:
        actionable_signals.append({
            "symbol": display, "action": range_setup["direction"],
            "entry": range_setup["entry"], "sl": range_setup["sl"], "tp1": range_setup["tp"], "tp2": None,
            "confidence": 60, "details": range_setup["reason"], "strategy_type": "range",
        })
    if sweep:
        atr15 = tf_data["15m"]["atr"]
        entry = tf_data["15m"]["close"]
        sw_levels = compute_sweep_breakout_levels(sweep["direction"], sweep["swept_level"], entry, atr15)
        actionable_signals.append({
            "symbol": display, "action": "LONG" if sweep["direction"] == "bullish" else "SHORT",
            "entry": entry, "sl": sw_levels["sl"], "tp1": sw_levels["tp1"], "tp2": None, "confidence": 60,
            "details": f"swept {sweep['swept_level']:.5f}", "strategy_type": "liquidity_sweep",
        })
    if breakout_retest:
        atr15 = tf_data["15m"]["atr"]
        entry = tf_data["15m"]["close"]
        br_levels = compute_sweep_breakout_levels(breakout_retest["direction"], breakout_retest["level"], entry, atr15)
        actionable_signals.append({
            "symbol": display, "action": "LONG" if breakout_retest["direction"] == "bullish" else "SHORT",
            "entry": entry, "sl": br_levels["sl"], "tp1": br_levels["tp1"], "tp2": None, "confidence": 60,
            "details": f"retested {breakout_retest['level']:.5f}", "strategy_type": "breakout_retest",
        })

    actionable = is_actionable(decision, range_setup, sweep, breakout_retest)

    # Log the primary trend decision EVERY cycle, including NO TRADE ones --
    # not just the actionable signals above. This is what lets us later ask
    # whether our filtering is actually improving quality, or just throwing
    # away good setups along with bad ones.
    db.save_evaluation(
        source="live", symbol=display, strategy_type="trend", action=decision["action"],
        confidence=decision["confidence"], entry=levels["entry"], sl=levels["sl"],
        tp1=levels["tp1"], tp2=levels["tp2"],
        details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction}",
    )

    return block, actionable_signals, actionable


async def run():
    """Scheduled scan: analyzes all tracked symbols, but only messages
    Telegram if at least one symbol has an actionable signal -- silent
    otherwise, rather than a scheduled digest regardless of content."""
    blocks = []
    for i, symbol_cfg in enumerate(SYMBOLS):
        block, actionable_signals, actionable = analyze_symbol(symbol_cfg)
        risk_lines = []
        for sig in actionable_signals:
            open_signal = db.get_open_signal(sig["symbol"], sig["strategy_type"])
            is_continuation = open_signal is not None and open_signal["action"] == sig["action"]

            if is_continuation:
                risk_lines.append(f"↻ {sig['strategy_type']}: still the same open trade from before, not a new entry")
                continue

            risk_usd = config.ACCOUNT_SIZE_USD * (config.RISK_PCT_PER_TRADE / 100)
            allowed, reason = risk_controller.check_and_reserve(risk_usd)
            db.save_scalp_signal(
                symbol=sig["symbol"], action=sig["action"], entry=sig["entry"], sl=sig["sl"],
                tp1=sig["tp1"], tp2=sig["tp2"], confidence=sig["confidence"], details=sig["details"],
                strategy_type=sig["strategy_type"], risk_allowed=allowed,
            )
            status_emoji = "✅" if allowed else "⛔"
            status_text = "would be taken" if allowed else f"SKIPPED -- {reason}"
            evidence = db.get_strategy_evidence_label(symbol_cfg["api"], sig["strategy_type"])
            risk_lines.append(f"{status_emoji} {sig['strategy_type']}: {status_text}\n   {evidence}")

        if actionable:
            if risk_lines:
                block = block + "\n\n_Risk check (paper-trading simulation):_\n" + "\n".join(risk_lines)
            blocks.append(block)
        else:
            logger.info("%s: nothing actionable this cycle", symbol_cfg["display"])
        if i < len(SYMBOLS) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not blocks:
        logger.info("No actionable signals across any symbol -- staying quiet this cycle")
        return

    message = "*📡 Scalp Alert*\n\n" + "\n\n".join(blocks) + "\n\n_Heuristic score, not backtested -- signal-only, not financial advice._"
    await telegram_bot.send_text(message)
    logger.info("Sent alert covering %d actionable symbol(s)", len(blocks))


if __name__ == "__main__":
    asyncio.run(run())
