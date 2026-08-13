"""Daily technical analysis report: for each tracked symbol, fetches real
45-minute candle data from Twelve Data's free API, computes trend direction
and support/resistance levels deterministically from actual price data (not
guessed by the AI), then has Gemini write a concise analyst-style summary
grounded in those real numbers. Sends one consolidated digest to Telegram.

This is intentionally a separate, once-daily script from the news bot's
5-minute loop -- run via its own GitHub Actions schedule.

Free tier notes: Twelve Data's free plan allows 800 requests/day and 8/min,
comfortably covering 8 symbols once a day. Sign up free at
https://twelvedata.com/apikey and put the key in TWELVEDATA_API_KEY.
"""

import asyncio
import logging
import os
import requests

from . import config
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_analysis")

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

SYMBOLS = [
    ("EUR/USD", "EUR/USD"),
    ("GBP/USD", "GBP/USD"),
    ("USD/JPY", "USD/JPY"),
    ("USD/CHF", "USD/CHF"),
    ("AUD/USD", "AUD/USD"),
    ("USD/CAD", "USD/CAD"),
    ("XAU/USD", "Gold (XAU/USD)"),
    ("BTC/USD", "BTC"),
]

INTERVAL = "45min"
CANDLE_COUNT = 100


def fetch_candles(symbol: str) -> list[dict]:
    if not TWELVEDATA_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "outputsize": CANDLE_COUNT,
                "apikey": TWELVEDATA_API_KEY,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            logger.warning("Twelve Data error for %s: %s", symbol, data.get("message", data))
            return []
        candles = data["values"]
        candles.reverse()
        return [
            {
                "close": float(c["close"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "datetime": c["datetime"],
            }
            for c in candles
        ]
    except Exception as e:
        logger.warning("Failed to fetch candles for %s: %s", symbol, e)
        return []


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def find_swing_levels(candles: list[dict], current_price: float, window: int = 3, max_levels: int = 3) -> tuple[list[float], list[float]]:
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    swing_highs, swing_lows = [], []

    for i in range(window, len(candles) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append(lows[i])

    resistance = sorted(set(round(h, 5) for h in swing_highs if h > current_price))[:max_levels]
    support = sorted(set(round(l, 5) for l in swing_lows if l < current_price), reverse=True)[:max_levels]

    return sorted(support), sorted(resistance)


def analyze_symbol(display_name: str, candles: list[dict]) -> dict | None:
    if len(candles) < 55:
        logger.warning("Not enough candle data for %s (%d candles)", display_name, len(candles))
        return None

    closes = [c["close"] for c in candles]
    ema20 = ema(closes, 20)[-1]
    ema50 = ema(closes, 50)[-1]
    current_price = closes[-1]

    if ema20 > ema50 and current_price > ema20:
        trend = "bullish"
    elif ema20 < ema50 and current_price < ema20:
        trend = "bearish"
    else:
        trend = "mixed/ranging"

    support, resistance = find_swing_levels(candles, current_price)

    return {
        "name": display_name,
        "current_price": current_price,
        "trend": trend,
        "ema20": ema20,
        "ema50": ema50,
        "support": support,
        "resistance": resistance,
    }


def build_narrative_prompt(analysis: dict) -> str:
    support_text = ", ".join(f"{s:.5f}" for s in analysis["support"]) or "none nearby -- price is below all recent swing lows"
    resistance_text = ", ".join(f"{r:.5f}" for r in analysis["resistance"]) or "none nearby -- price is making new highs versus recent swings"
    return f"""You are a professional technical analyst writing a concise daily brief.
Asset: {analysis['name']}
Timeframe: 45-minute chart
Current price: {analysis['current_price']:.5f}
20-period EMA: {analysis['ema20']:.5f}
50-period EMA: {analysis['ema50']:.5f}
Computed trend (from EMA relationship): {analysis['trend']}
Recent support levels: {support_text}
Recent resistance levels: {resistance_text}

Write a 2-3 sentence analyst note using ONLY the data above. State the trend,
the nearest support and resistance levels (or note their absence if none
nearby), and a near-term bullish/bearish/neutral bias with brief reasoning.
Do not invent any numbers not given above. No preamble, just the analysis."""


def get_narrative(analysis: dict) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY if hasattr(config, "GEMINI_API_KEY") else os.getenv("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel(model_name=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
        resp = model.generate_content(build_narrative_prompt(analysis))
        return resp.text.strip()
    except Exception as e:
        logger.warning("Narrative generation failed for %s: %s", analysis["name"], e)
        return (
            f"Trend: {analysis['trend']}. Support near {analysis['support']}, "
            f"resistance near {analysis['resistance']}."
        )


async def run():
    sections = []
    for symbol, display_name in SYMBOLS:
        candles = fetch_candles(symbol)
        analysis = analyze_symbol(display_name, candles)
        if not analysis:
            continue
        narrative = get_narrative(analysis)
        emoji = {"bullish": "🟢", "bearish": "🔴", "mixed/ranging": "⚪"}[analysis["trend"]]
        sections.append(f"{emoji} *{display_name}* ({analysis['trend'].upper()})\n{narrative}")

    if not sections:
        logger.warning("No analysis produced for any symbol -- check TWELVEDATA_API_KEY")
        return

    digest = "*📊 Daily Technical Analysis (45min charts)*\n\n" + "\n\n".join(sections)
    await telegram_bot.send_text(digest)
    logger.info("Sent daily analysis digest covering %d symbols", len(sections))


if __name__ == "__main__":
    asyncio.run(run())
