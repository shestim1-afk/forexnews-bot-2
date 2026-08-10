"""Classifies a NewsItem into asset impact (bullish/bearish/neutral) using an
LLM. Three backends are supported:

  - "gemini" (RECOMMENDED, default): Google Gemini API via AI Studio. This
    is the only major LLM provider with a genuinely permanent free tier --
    no credit card, no expiration, ~1,500 requests/day, which comfortably
    covers a bot polling every few minutes. Get a free key at
    https://aistudio.google.com/apikey and put it in GEMINI_API_KEY in .env.
  - "anthropic": uses the Claude API. New accounts get a one-time ~$5 free
    trial credit that does NOT renew -- expect to need real payment within
    days of continuous polling. Prefer "gemini" unless you specifically
    want Claude's classification quality and are OK paying afterward.
  - "ollama": uses a locally-run, fully free open-source model via Ollama
    (https://ollama.com). No API cost ever, no daily request cap, but needs
    a machine with enough RAM/CPU to run the model, and quality is
    generally lower. More setup work than Gemini, so only worth it if you
    want zero reliance on any external API.

Set CLASSIFIER_BACKEND in .env to choose ("gemini" | "anthropic" | "ollama").
"""

import json
import logging
import os
import requests

from . import config
from .db import NewsItem

logger = logging.getLogger(__name__)

BACKEND = os.getenv("CLASSIFIER_BACKEND", "gemini")  # "gemini" | "anthropic" | "ollama"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# gemini-2.5-flash-lite was deprecated for new users -- gemini-3.1-flash-lite
# is the current recommended free-tier model (as of Aug 2026) and also has a
# slightly higher free rate limit (15 requests/min vs 10 for gemini-3-flash).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
# Free tier is rate-limited per minute, not just per day -- this delay keeps
# us comfortably under that limit. 4.5s -> ~13 requests/min, safely under the
# 15 RPM cap for flash-lite. Raise this if you still see 429 errors.
CLASSIFY_DELAY_SECONDS = float(os.getenv("CLASSIFY_DELAY_SECONDS", "4.5"))

SYSTEM_PROMPT = """You are a macro/forex market analyst. Given a single news \
item, decide which tradeable assets it plausibly affects and whether the \
likely near-term market reaction is bullish, bearish, or neutral for each.

Only tag assets from this universe:
- Forex pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/JPY, GBP/JPY, USD/CNH
- Metals: Gold (XAU/USD), Silver (XAG/USD)
- Crypto: BTC, ETH, Crypto (broad market)
- Stocks/sectors: Energy sector, Defense sector, Tech sector (broad), US indices (S&P500/Nasdaq/Dow), or a specific named stock if the news is about that company directly

Respond with ONLY valid JSON (no markdown fences, no preamble), matching \
this schema exactly:

{
  "relevant": true/false,
  "impacts": [
    {"asset": "<asset name from the universe above>", "direction": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, "reason": "<one short sentence>"}
  ]
}

If the news item has no plausible market relevance, return {"relevant": false, "impacts": []}.
Be conservative with confidence -- only use confidence above 0.7 for major, unambiguous events (e.g. a central bank rate decision, a war escalation, a surprise policy announcement from a head of state)."""


def _build_user_prompt(item: NewsItem) -> str:
    return f"""Source: {item.source}
Category: {item.category}
Headline: {item.title}
Body: {item.body[:1500]}"""


def _classify_anthropic(item: NewsItem) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(item)}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return _safe_parse(text)


def _classify_gemini(item: NewsItem) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content(_build_user_prompt(item))
    return _safe_parse(resp.text)


def _classify_ollama(item: NewsItem) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(item)}",
        "stream": False,
        "format": "json",
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
    r.raise_for_status()
    text = r.json().get("response", "")
    return _safe_parse(text)


def _safe_parse(text: str) -> dict:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse classifier output as JSON: %s", cleaned[:200])
        return {"relevant": False, "impacts": []}


def classify(item: NewsItem) -> dict:
    """Returns {"relevant": bool, "impacts": [{"asset","direction","confidence","reason"}]}"""
    try:
        if BACKEND == "gemini":
            return _classify_gemini(item)
        if BACKEND == "ollama":
            return _classify_ollama(item)
        return _classify_anthropic(item)
    except Exception as e:
        logger.error("Classification failed for '%s': %s", item.title[:80], e)
        return {"relevant": False, "impacts": []}
