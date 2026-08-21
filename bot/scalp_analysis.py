"""Multi-symbol scalp signal generator (BTC/USD, Gold, GBP/JPY), using free
Twelve Data market data and free open-source technical indicators via `ta`.

IMPORTANT, read before trusting this: the confidence score is a heuristic
composite of how strongly several independent signals agree RIGHT NOW. It is
NOT a backtested, calibrated probability of a profitable trade. Every signal
this script generates -- including "NO TRADE" calls -- gets logged to the
database via db.save_scalp_signal(), specifically so it can be checked
against what price actually did afterward. Treat this as signal-only data
collection, especially relevant given leveraged trading amplifies the cost
of an uncalibrated signal.

This script does NOT place any orders. It only analyzes and reports.

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

from . import db
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


def compute_trade_levels(action: str, entry: float, atr_5m: float) -> dict:
    if action == "LONG":
        return {"entry": entry, "sl": entry - 1.5 * atr_5m, "tp1": entry + 1.0 * atr_5m, "tp2": entry + 2.0 * atr_5m}
    elif action == "SHORT":
        return {"entry": entry, "sl": entry + 1.5 * atr_5m, "tp1": entry - 1.0 * atr_5m, "tp2": entry - 2.0 * atr_5m}
    return {"entry": None, "sl": None, "tp1": None, "tp2": None}


LABEL_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def format_symbol_block(display_symbol: str, tf_data: dict, decision: dict, levels: dict,
                         news_direction: str, news_summary: str, range_setup: dict | None,
                         divergence: str | None) -> str:
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

    if range_setup:
        rs_emoji = "🟢" if range_setup["direction"] == "LONG" else "🔴"
        lines.append(f"{rs_emoji} Range: *{range_setup['direction']}* -- {range_setup['reason']}")
        lines.append(f"  Entry {range_setup['entry']:.5f} | SL {range_setup['sl']:.5f} | TP {range_setup['tp']:.5f}")
    elif tf_data["1h"]["adx"] < 20:
        lines.append("⚪ Range: 1H ranging, no trigger nearby yet")

    return "\n".join(lines)


def analyze_symbol(symbol_cfg: dict) -> tuple[str, dict, dict | None]:
    api_symbol = symbol_cfg["api"]
    display = symbol_cfg["display"]

    tf_data, dfs = {}, {}
    for i, (label, interval) in enumerate(TIMEFRAMES.items()):
        df = fetch_klines(api_symbol, interval)
        if df is None or len(df) < 205:
            logger.warning("Insufficient data for %s %s", display, label)
            return f"*{display}*\n⚠️ Could not fetch enough price data this cycle.", None, None
        tf_data[label] = compute_indicators(df)
        dfs[label] = df
        if i < len(TIMEFRAMES) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    divergence = detect_rsi_divergence(dfs["15m"])

    news_direction, news_summary = get_news_bias(symbol_cfg["news_keywords"])
    decision = score_and_decide(tf_data, news_direction)
    range_setup = detect_range_setup(tf_data, dfs, divergence)

    entry_price = tf_data["5m"]["close"]
    atr_5m = tf_data["5m"]["atr"]
    levels = compute_trade_levels(decision["action"], entry_price, atr_5m)

    block = format_symbol_block(display, tf_data, decision, levels, news_direction, news_summary, range_setup, divergence)

    signal_data = {
        "symbol": display, "action": decision["action"], "entry": levels["entry"],
        "sl": levels["sl"], "tp1": levels["tp1"], "tp2": levels["tp2"], "confidence": decision["confidence"],
        "details": f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction} divergence={divergence}"
                   + (f" | range={range_setup['direction']}@{range_setup['entry']:.5f}" if range_setup else ""),
    }
    return block, signal_data, range_setup


async def run():
    """Hourly digest: analyzes all tracked symbols and sends one consolidated message."""
    blocks = []
    for i, symbol_cfg in enumerate(SYMBOLS):
        block, signal_data, _ = analyze_symbol(symbol_cfg)
        blocks.append(block)
        if signal_data:
            db.save_scalp_signal(
                symbol=signal_data["symbol"], action=signal_data["action"],
                entry=signal_data["entry"], sl=signal_data["sl"], tp1=signal_data["tp1"], tp2=signal_data["tp2"],
                confidence=signal_data["confidence"], details=signal_data["details"],
            )
        if i < len(SYMBOLS) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    message = "*📡 Hourly Scalp Scan*\n\n" + "\n\n".join(blocks) + "\n\n_Heuristic score, not backtested -- signal-only, not financial advice._"
    await telegram_bot.send_text(message)
    logger.info("Sent hourly digest covering %d symbols", len(SYMBOLS))


if __name__ == "__main__":
    asyncio.run(run())
