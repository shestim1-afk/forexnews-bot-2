"""Hourly BTC scalp signal generator, using free Twelve Data market data
(same free provider already used for the daily forex/gold analysis -- proven
working, no geo-blocking issues) and free open-source technical indicators
via the `ta` library.

IMPORTANT, read before trusting this: the confidence score is a heuristic
composite of how strongly several independent signals agree RIGHT NOW. It is
NOT a backtested, calibrated probability of a profitable trade. Every signal
this script generates -- including "NO TRADE" calls -- gets logged to the
database via db.save_scalp_signal(), specifically so that after enough
signals accumulate, we can go back and check what price actually did
afterward. That is the only way to find out whether an "80% confidence"
signal has actually meant anything. Treat this as signal-only / paper-trading
data collection until that validation has been done -- especially relevant
given leveraged trading amplifies the cost of an uncalibrated signal.

This script does NOT place any orders. It only analyzes and reports.

Note: an earlier version used Binance's public API directly. That was
switched out after Binance returned HTTP 451 (geo-blocked) when called from
GitHub Actions' shared runners -- a real, documented restriction, not a bug.
Twelve Data doesn't have this issue and is already relied on elsewhere in
this project.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

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

SYMBOL = "BTC/USD"  # Twelve Data crypto symbol format
DISPLAY_SYMBOL = "BTC/USD"
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TIMEFRAMES = {"4h": "4h", "1h": "1h", "15m": "15min", "5m": "5min"}
CANDLE_LIMIT = 250


def fetch_klines(interval: str) -> pd.DataFrame | None:
    if not TWELVEDATA_API_KEY:
        logger.warning("TWELVEDATA_API_KEY not set -- cannot fetch price data")
        return None
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": SYMBOL,
                "interval": interval,
                "outputsize": CANDLE_LIMIT,
                "apikey": TWELVEDATA_API_KEY,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            logger.warning("Twelve Data error for %s: %s", interval, data.get("message", data))
            return None
        values = data["values"]
        values.reverse()
        df = pd.DataFrame(values)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
        return df
    except Exception as e:
        logger.warning("Failed to fetch %s candles: %s", interval, e)
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

    return {
        "close": current_close,
        "trend": trend,
        "momentum": momentum,
        "adx": adx,
        "atr": atr,
        "above_vwap": current_close > vwap,
        "volume_confirmed": current_volume > vol_ma * 1.2 if pd.notna(vol_ma) else False,
    }


def get_news_bias() -> tuple[str, str]:
    rows = db.get_recent_classifications("BTC", hours=2) + db.get_recent_classifications("Crypto", hours=2)
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
    higher_tf_trend_4h = tf_data["4h"]["trend"]
    higher_tf_trend_1h = tf_data["1h"]["trend"]

    if higher_tf_trend_4h == higher_tf_trend_1h and higher_tf_trend_4h in ("bullish", "bearish"):
        direction = higher_tf_trend_4h
        score = 3
    elif "bullish" in (higher_tf_trend_4h, higher_tf_trend_1h) and "bearish" not in (higher_tf_trend_4h, higher_tf_trend_1h):
        direction = "bullish"
        score = 1
    elif "bearish" in (higher_tf_trend_4h, higher_tf_trend_1h) and "bullish" not in (higher_tf_trend_4h, higher_tf_trend_1h):
        direction = "bearish"
        score = 1
    else:
        direction = None
        score = 0

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

    max_score = 10
    confidence = max(0, min(100, round(score / max_score * 100)))

    if direction is None or confidence < 55:
        action = "NO TRADE"
    else:
        action = "LONG" if direction == "bullish" else "SHORT"

    return {"direction": direction, "action": action, "confidence": confidence}


def compute_trade_levels(action: str, entry: float, atr_5m: float) -> dict:
    if action == "LONG":
        return {
            "entry": entry,
            "sl": entry - 1.5 * atr_5m,
            "tp1": entry + 1.0 * atr_5m,
            "tp2": entry + 2.0 * atr_5m,
        }
    elif action == "SHORT":
        return {
            "entry": entry,
            "sl": entry + 1.5 * atr_5m,
            "tp1": entry - 1.0 * atr_5m,
            "tp2": entry - 2.0 * atr_5m,
        }
    return {"entry": None, "sl": None, "tp1": None, "tp2": None}


LABEL_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def format_message(tf_data: dict, decision: dict, levels: dict, news_direction: str, news_summary: str) -> str:
    lines = [f"*{DISPLAY_SYMBOL}* — hourly scan\n"]
    for tf_label in ["4h", "1h", "15m", "5m"]:
        d = tf_data[tf_label]
        emoji = LABEL_EMOJI.get(d["trend"], "⚪")
        lines.append(f"{tf_label.upper()}: {emoji} {d['trend'].capitalize()}")

    news_emoji = LABEL_EMOJI.get(news_direction, "⚪")
    lines.append(f"News: {news_emoji} {news_summary}")
    lines.append("")

    action_emoji = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}[decision["action"]]
    lines.append(f"{action_emoji} *{decision['action']}*")

    if decision["action"] != "NO TRADE":
        lines.append(f"Entry: {levels['entry']:.2f}")
        lines.append(f"SL: {levels['sl']:.2f}")
        lines.append(f"TP1: {levels['tp1']:.2f}")
        lines.append(f"TP2: {levels['tp2']:.2f}")
    else:
        agree = tf_data["4h"]["trend"] == tf_data["1h"]["trend"]
        lines.append(f"4H {tf_data['4h']['trend']} / 1H {tf_data['1h']['trend']}" + (" -- higher timeframes disagree" if not agree else " -- confluence too weak"))

    lines.append(f"Confidence: {decision['confidence']}%")
    lines.append("")
    lines.append("_Heuristic score, not a backtested probability -- signal-only, not financial advice._")

    return "\n".join(lines)


async def run():
    tf_data = {}
    for label, interval in TIMEFRAMES.items():
        df = fetch_klines(interval)
        if df is None or len(df) < 205:
            logger.warning("Insufficient data for %s, aborting this cycle", label)
            return
        tf_data[label] = compute_indicators(df)

    news_direction, news_summary = get_news_bias()
    decision = score_and_decide(tf_data, news_direction)

    entry_price = tf_data["5m"]["close"]
    atr_5m = tf_data["5m"]["atr"]
    levels = compute_trade_levels(decision["action"], entry_price, atr_5m)

    message = format_message(tf_data, decision, levels, news_direction, news_summary)

    db.save_scalp_signal(
        symbol=DISPLAY_SYMBOL,
        action=decision["action"],
        entry=levels["entry"],
        sl=levels["sl"],
        tp1=levels["tp1"],
        tp2=levels["tp2"],
        confidence=decision["confidence"],
        details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction}",
    )

    await telegram_bot.send_text(message)
    logger.info("Sent scalp signal: %s (%d%% confidence)", decision["action"], decision["confidence"])


if __name__ == "__main__":
    asyncio.run(run())
