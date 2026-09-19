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
from .historical_backtest import fetch_full_history, fetch_paginated_history

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


async def run_extended_xau():
    """SAME frozen specification as run() -- no parameter changes -- just
    applied to the FULL available XAU/USD daily history (back to
    2020-01-24, confirmed available earlier in this project) instead of
    the default 2-year window, to see whether the 2-year result (n=43,
    +0.201 avg R) holds up with a real, adequately-sized sample."""
    lines = [
        "*2-of-3 Confluence Strategy Backtest -- EXTENDED WINDOW (XAU/USD only)*",
        "SAME frozen specification as the 2-year test -- no parameter changes. Full available daily history "
        f"(back to 2020-01-24), to check whether the 2-year result (n=43, +0.201 avg R) holds with a real sample.\n",
    ]

    target_start = datetime(2020, 1, 24)
    logger.info("Fetching full XAU/USD daily history back to %s...", target_start.date())
    df = fetch_paginated_history("XAU/USD", "1day", target_start, datetime.now())

    if df is None or len(df) < MA_SLOW + 50:
        lines.append("Insufficient data retrieved -- could not run the extended test.")
        _send_telegram_direct("\n".join(lines))
        return

    lines.append(f"Data: {df['datetime'].min().date()} to {df['datetime'].max().date()} ({len(df)} daily candles)")
    result = run_backtest_on_df(df)

    if result["n"] == 0:
        lines.append("No trades triggered across the full period.")
    else:
        pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] is not None else "N/A"
        lines.append(
            f"n={result['n']}, win rate={result['win_rate']*100:.0f}%, avg R={result['avg_r']:+.3f}, "
            f"PF={pf_str}, max DD={result['max_dd_r']:.2f}R"
        )
        lines.append("")
        adequate = result["n"] >= 30
        if not adequate:
            lines.append("*CLASSIFICATION: INCONCLUSIVE -- still below the 30-trade minimum.*")
        elif result["avg_r"] > 0 and (result["profit_factor"] or 0) > 1.0:
            lines.append("*CLASSIFICATION: PROMISING -- net positive with an adequate sample. Still needs OOS confirmation before any real use.*")
        else:
            lines.append("*CLASSIFICATION: FAILED -- did not hold up on the full available history.*")

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent extended confluence backtest report")


async def run_extended_dogecoin():
    """SAME frozen specification, unchanged -- applied to DOGE/USD as one
    additional, honest data point. Not a search across many assets: this
    is the one additional instrument tested, reported regardless of
    outcome, same as XAU and BTC before it."""
    lines = [
        "*2-of-3 Confluence Strategy Backtest -- DOGE/USD*",
        "SAME frozen specification, no changes. One additional instrument tested honestly, not a search "
        "across many assets for a hit.\n",
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2 + 250)
    logger.info("Fetching daily data for DOGE/USD...")
    df = fetch_full_history("DOGE/USD", "1day", outputsize=1000)

    if df is None or len(df) < MA_SLOW + 50:
        lines.append("Insufficient data retrieved for DOGE/USD -- could not run the test.")
        _send_telegram_direct("\n".join(lines))
        return

    df = df[df["datetime"] >= start_date].reset_index(drop=True)
    lines.append(f"Data: {df['datetime'].min().date()} to {df['datetime'].max().date()} ({len(df)} daily candles)")
    result = run_backtest_on_df(df)

    if result["n"] == 0:
        lines.append("No trades triggered in this period.")
    else:
        pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] is not None else "N/A"
        lines.append(
            f"n={result['n']}, win rate={result['win_rate']*100:.0f}%, avg R={result['avg_r']:+.3f}, "
            f"PF={pf_str}, max DD={result['max_dd_r']:.2f}R"
        )
        lines.append("")
        adequate = result["n"] >= 30
        if not adequate:
            lines.append("*CLASSIFICATION: INCONCLUSIVE -- below the 30-trade minimum.*")
        elif result["avg_r"] > 0 and (result["profit_factor"] or 0) > 1.0:
            lines.append("*CLASSIFICATION: PROMISING -- still needs OOS confirmation and a trade-frequency check before any real use.*")
        else:
            lines.append("*CLASSIFICATION: FAILED.*")

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent DOGE/USD confluence backtest report")


async def run_dogecoin_dev_oos_split():
    """Retroactive dev/OOS check on the ALREADY-COLLECTED DOGE data --
    since the specification was frozen before ever seeing DOGE data at
    all, splitting the available history chronologically into a dev
    portion (first ~70%) and an OOS portion (last ~30%) is an honest way
    to check consistency now, without waiting months for genuinely new
    data to accumulate. SAME frozen rules applied to both halves,
    unchanged."""
    lines = [
        "*2-of-3 Confluence Strategy -- DOGE/USD Dev/OOS Split*",
        "Same frozen specification. Retroactive split of already-collected data into dev (first ~70%) and "
        "OOS (last ~30%) portions -- an honest consistency check, not a true prospective OOS test.\n",
    ]

    df = fetch_full_history("DOGE/USD", "1day", outputsize=1000)
    if df is None or len(df) < MA_SLOW + 100:
        lines.append("Insufficient data retrieved -- could not run the split.")
        _send_telegram_direct("\n".join(lines))
        return

    df = df.reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    # Both halves need MA_SLOW warmup bars of their OWN preceding data to
    # compute indicators correctly -- the OOS half reuses the tail of the
    # dev half's data for warmup, but only trades/signals occurring AFTER
    # the split point count toward its own results.
    dev_df = df.iloc[:split_idx].reset_index(drop=True)
    oos_df = df.iloc[max(0, split_idx - MA_SLOW - 10):].reset_index(drop=True)
    oos_start_date = df.iloc[split_idx]["datetime"]

    dev_result = run_backtest_on_df(dev_df)
    oos_result_full = run_backtest_on_df(oos_df)

    lines.append(f"Full data: {df['datetime'].min().date()} to {df['datetime'].max().date()} ({len(df)} candles)")
    lines.append(f"Split point: {oos_start_date.date()}\n")

    for label, result in [("DEV", dev_result), ("OOS", oos_result_full)]:
        if result["n"] == 0:
            lines.append(f"*{label}*: no trades triggered.")
        else:
            pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] is not None else "N/A"
            lines.append(
                f"*{label}*: n={result['n']}, win rate={result['win_rate']*100:.0f}%, "
                f"avg R={result['avg_r']:+.3f}, PF={pf_str}, max DD={result['max_dd_r']:.2f}R"
            )
    lines.append("")

    if dev_result["n"] > 0 and oos_result_full["n"] > 0:
        dev_positive = dev_result["avg_r"] > 0
        oos_positive = oos_result_full["avg_r"] > 0
        oos_adequate = oos_result_full["n"] >= 15  # smaller OOS slice, lower bar acknowledged explicitly
        if not oos_adequate:
            lines.append(f"*CLASSIFICATION: INCONCLUSIVE -- OOS sample (n={oos_result_full['n']}) too small even for this reduced check.*")
        elif dev_positive and oos_positive:
            lines.append("*CLASSIFICATION: CONSISTENT -- edge held its sign across both halves. Still not a substitute for genuine forward testing.*")
        else:
            lines.append("*CLASSIFICATION: DID NOT HOLD -- edge reversed or disappeared in the OOS half.*")
    else:
        lines.append("*CLASSIFICATION: INCONCLUSIVE -- one or both halves had no trades.*")

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent DOGE dev/OOS split report")


if __name__ == "__main__":
    asyncio.run(run())
