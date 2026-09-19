"""2-of-3 Confluence Strategy Backtest -- frozen specification.

Combines three well-documented, independently-real trading systems:
- Donchian Channel Breakout (20-day entry / 10-day exit) -- the original
  "Turtle Trading" system (Dennis & Eckhardt, 1980s), real documented
  track record.
- 50/200-day Moving Average Crossover ("Golden Cross" / "Death Cross") --
  one of the most widely used trend-following signals by CTAs.
- MACD (12/26/9) crossover -- another standard, widely used momentum
  signal, mechanically distinct from the other two.

ENTRY RULE (frozen BEFORE any results were seen): a trade triggers when
at least 2 of these 3 signals agree on direction (2-of-3 majority, not
unanimous -- chosen for sample-size reasons: requiring all 3 given their
shared trend-following nature would likely cut frequency too far for a
trustworthy sample in a 2-year window).

EXIT: 2x ATR stop-loss, or exit when the majority signal flips --
whichever comes first.

SPEED DESIGN CHOICE: uses DAILY candles only (not the multi-timeframe
intraday data that caused repeated rate-limit/pagination problems
elsewhere in this project today) -- 2 years of daily data is ~730
candles per instrument, fitting in a single API call with no pagination
needed at all.

Instruments: XAU/USD and BTC/USD, BOTH reported regardless of result.
Costs: 0.10% round-trip, consistent with other studies in this project.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from . import config
from .historical_backtest import fetch_full_history

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("confluence_backtest")

DONCHIAN_ENTRY_DAYS = 20
DONCHIAN_EXIT_DAYS = 10
MA_FAST = 50
MA_SLOW = 200
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0
SPREAD_PCT = 0.10

INSTRUMENTS = [("XAU/USD", "Gold"), ("BTC/USD", "Bitcoin")]


def _send_telegram_direct(text: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Cannot send: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=20,
        )
        if r.status_code == 200 and r.json().get("ok") is True:
            return True
        logger.error("Telegram send failed: HTTP %d, response: %s", r.status_code, r.text[:300])
        return False
    except Exception as e:
        logger.error("Telegram send raised an exception: %s", e)
        return False


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    # Donchian channels
    df["donchian_high_entry"] = df["high"].rolling(DONCHIAN_ENTRY_DAYS).max().shift(1)
    df["donchian_low_entry"] = df["low"].rolling(DONCHIAN_ENTRY_DAYS).min().shift(1)
    df["donchian_high_exit"] = df["high"].rolling(DONCHIAN_EXIT_DAYS).max().shift(1)
    df["donchian_low_exit"] = df["low"].rolling(DONCHIAN_EXIT_DAYS).min().shift(1)

    # Moving averages
    df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()

    # MACD
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    # ATR
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    # Individual signal directions: +1 bullish, -1 bearish, 0 neutral
    df["sig_donchian"] = 0
    df.loc[df["close"] > df["donchian_high_entry"], "sig_donchian"] = 1
    df.loc[df["close"] < df["donchian_low_entry"], "sig_donchian"] = -1

    df["sig_ma"] = 0
    df.loc[df["ma_fast"] > df["ma_slow"], "sig_ma"] = 1
    df.loc[df["ma_fast"] < df["ma_slow"], "sig_ma"] = -1

    df["sig_macd"] = 0
    df.loc[df["macd_line"] > df["macd_signal"], "sig_macd"] = 1
    df.loc[df["macd_line"] < df["macd_signal"], "sig_macd"] = -1

    bull_votes = (df[["sig_donchian", "sig_ma", "sig_macd"]] == 1).sum(axis=1)
    bear_votes = (df[["sig_donchian", "sig_ma", "sig_macd"]] == -1).sum(axis=1)
    df["majority"] = 0
    df.loc[bull_votes >= 2, "majority"] = 1
    df.loc[bear_votes >= 2, "majority"] = -1

    return df


def run_backtest_on_df(df: pd.DataFrame) -> dict:
    df = compute_signals(df)
    trades = []
    position = None  # {"direction", "entry_idx", "entry_price", "stop_price"}

    for i in range(max(MA_SLOW, DONCHIAN_ENTRY_DAYS) + 1, len(df)):
        row = df.iloc[i]
        if pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        if position is not None:
            direction = position["direction"]
            exit_now, exit_price, exit_reason = False, None, None

            if direction == 1:
                if row["low"] <= position["stop_price"]:
                    exit_now, exit_price, exit_reason = True, position["stop_price"], "stop"
                elif row["majority"] != 1:
                    exit_now, exit_price, exit_reason = True, row["close"], "signal_flip"
            else:
                if row["high"] >= position["stop_price"]:
                    exit_now, exit_price, exit_reason = True, position["stop_price"], "stop"
                elif row["majority"] != -1:
                    exit_now, exit_price, exit_reason = True, row["close"], "signal_flip"

            if exit_now:
                gross_pct = 100 * (exit_price - position["entry_price"]) / position["entry_price"]
                if direction == -1:
                    gross_pct = -gross_pct
                net_pct = gross_pct - SPREAD_PCT
                risk_pct = 100 * abs(position["entry_price"] - position["stop_price"]) / position["entry_price"]
                r_multiple = net_pct / risk_pct if risk_pct else 0.0
                trades.append({"net_pct": net_pct, "r_multiple": r_multiple, "exit_reason": exit_reason})
                position = None

        if position is None and row["majority"] != 0:
            direction = row["majority"]
            entry_price = row["close"]
            stop_price = entry_price - direction * ATR_STOP_MULT * row["atr"]
            position = {"direction": direction, "entry_price": entry_price, "stop_price": stop_price}

    n = len(trades)
    if n == 0:
        return {"n": 0}
    wins = [t for t in trades if t["r_multiple"] > 0]
    avg_r = sum(t["r_multiple"] for t in trades) / n
    win_rate = len(wins) / n
    gross_win = sum(t["r_multiple"] for t in wins)
    gross_loss = abs(sum(t["r_multiple"] for t in trades if t["r_multiple"] <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else None

    cum_r, peak, max_dd = 0.0, 0.0, 0.0
    for t in trades:
        cum_r += t["r_multiple"]
        peak = max(peak, cum_r)
        max_dd = max(max_dd, peak - cum_r)

    return {"n": n, "win_rate": win_rate, "avg_r": avg_r, "profit_factor": pf, "max_dd_r": max_dd}


async def run():
    lines = [
        "*2-of-3 Confluence Strategy Backtest (Donchian + MA Crossover + MACD)*",
        f"Entry: 2-of-3 majority agreement. Exit: {ATR_STOP_MULT}x ATR stop or signal flip. "
        f"Daily candles, 2 years, {SPREAD_PCT}% round-trip cost.\n",
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2 + 250)  # extra buffer for 200-day MA warmup

    for api_symbol, name in INSTRUMENTS:
        logger.info("Fetching daily data for %s...", api_symbol)
        df = fetch_full_history(api_symbol, "1day", outputsize=1000)
        if df is None or len(df) < MA_SLOW + 50:
            lines.append(f"*{name} ({api_symbol})*: insufficient data, skipped.")
            continue

        df = df[df["datetime"] >= start_date].reset_index(drop=True)
        result = run_backtest_on_df(df)

        lines.append(f"*{name} ({api_symbol})*")
        if result["n"] == 0:
            lines.append("  No trades triggered in this period.")
        else:
            pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] is not None else "N/A"
            lines.append(
                f"  n={result['n']}, win rate={result['win_rate']*100:.0f}%, avg R={result['avg_r']:+.3f}, "
                f"PF={pf_str}, max DD={result['max_dd_r']:.2f}R"
            )
        lines.append("")

    lines.append(
        "Reminder: this is a single, frozen specification tested once -- no parameter tuning after seeing "
        "results. Both instruments reported regardless of outcome."
    )

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent confluence backtest report")


if __name__ == "__main__":
    asyncio.run(run())
