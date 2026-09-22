"""2-of-3 Confluence Strategy Backtest -- frozen specification.

Combines three well-documented, independently-real trading systems:
Donchian Channel Breakout, 50/200-day MA Crossover, MACD crossover.
Entry: 2-of-3 majority. Exit: 2x ATR stop or signal flip.
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
SMALLER_CAP_CANDIDATES = ["AVAX/USD", "LTC/USD", "JTO/USD", "ENA/USD"]


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

    df["donchian_high_entry"] = df["high"].rolling(DONCHIAN_ENTRY_DAYS).max().shift(1)
    df["donchian_low_entry"] = df["low"].rolling(DONCHIAN_ENTRY_DAYS).min().shift(1)

    df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()

    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

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
    position = None

    for i in range(max(MA_SLOW, DONCHIAN_ENTRY_DAYS) + 1, len(df)):
        row = df.iloc[i]
        if pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        if position is not None:
            direction = position["direction"]
            exit_now, exit_price = False, None

            if direction == 1:
                if row["low"] <= position["stop_price"]:
                    exit_now, exit_price = True, position["stop_price"]
                elif row["majority"] != 1:
                    exit_now, exit_price = True, row["close"]
            else:
                if row["high"] >= position["stop_price"]:
                    exit_now, exit_price = True, position["stop_price"]
                elif row["majority"] != -1:
                    exit_now, exit_price = True, row["close"]

            if exit_now:
                gross_pct = 100 * (exit_price - position["entry_price"]) / position["entry_price"]
                if direction == -1:
                    gross_pct = -gross_pct
                net_pct = gross_pct - SPREAD_PCT
                risk_pct = 100 * abs(position["entry_price"] - position["stop_price"]) / position["entry_price"]
                r_multiple = net_pct / risk_pct if risk_pct else 0.0
                trades.append({"net_pct": net_pct, "r_multiple": r_multiple})
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
    start_date = end_date - timedelta(days=365 * 2 + 250)

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
    lines = [
        "*2-of-3 Confluence Strategy Backtest -- EXTENDED WINDOW (XAU/USD only)*",
        "SAME frozen specification as the 2-year test -- no parameter changes. Full available daily history "
        f"(back to 2020-01-24).\n",
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
        if result["n"] < 30:
            lines.append("*CLASSIFICATION: INCONCLUSIVE -- still below the 30-trade minimum.*")
        elif result["avg_r"] > 0 and (result["profit_factor"] or 0) > 1.0:
            lines.append("*CLASSIFICATION: PROMISING -- net positive with an adequate sample. Still needs OOS confirmation before any real use.*")
        else:
            lines.append("*CLASSIFICATION: FAILED -- did not hold up on the full available history.*")
    _send_telegram_direct("\n".join(lines))
    logger.info("Sent extended confluence backtest report")


async def run_extended_dogecoin():
    lines = [
        "*2-of-3 Confluence Strategy Backtest -- DOGE/USD*",
        "SAME frozen specification, no changes. One additional instrument tested honestly.\n",
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
        if result["n"] < 30:
            lines.append("*CLASSIFICATION: INCONCLUSIVE -- below the 30-trade minimum.*")
        elif result["avg_r"] > 0 and (result["profit_factor"] or 0) > 1.0:
            lines.append("*CLASSIFICATION: PROMISING -- still needs OOS confirmation and a trade-frequency check before any real use.*")
        else:
            lines.append("*CLASSIFICATION: FAILED.*")
    _send_telegram_direct("\n".join(lines))
    logger.info("Sent DOGE/USD confluence backtest report")


async def run_dogecoin_dev_oos_split():
    lines = [
        "*2-of-3 Confluence Strategy -- DOGE/USD Dev/OOS Split*",
        "Same frozen specification. Retroactive split of already-collected data into dev (first ~70%) and "
        "OOS (last ~30%) portions.\n",
    ]
    df = fetch_full_history("DOGE/USD", "1day", outputsize=1000)
    if df is None or len(df) < MA_SLOW + 100:
        lines.append("Insufficient data retrieved -- could not run the split.")
        _send_telegram_direct("\n".join(lines))
        return
    df = df.reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
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
        oos_adequate = oos_result_full["n"] >= 15
        if not oos_adequate:
            lines.append(f"*CLASSIFICATION: INCONCLUSIVE -- OOS sample (n={oos_result_full['n']}) too small.*")
        elif dev_positive and oos_positive:
            lines.append("*CLASSIFICATION: CONSISTENT -- edge held its sign across both halves.*")
        else:
            lines.append("*CLASSIFICATION: DID NOT HOLD -- edge reversed or disappeared in the OOS half.*")
    else:
        lines.append("*CLASSIFICATION: INCONCLUSIVE -- one or both halves had no trades.*")
    _send_telegram_direct("\n".join(lines))
    logger.info("Sent DOGE dev/OOS split report")


async def run_smaller_cap_check():
    """Applies the SAME frozen confluence specification, unchanged, to
    smaller, more volatile, less liquid instruments -- the category the
    user's screenshot was trading (JTO, AVAX, ENA, LTC). Checks real data
    availability first (smaller-cap tokens like JTO and ENA are recent,
    2023-2024 launches, and may have limited or no coverage on Twelve
    Data) rather than assuming it."""
    lines = [
        "*2-of-3 Confluence Strategy -- Smaller-Cap/Less-Liquid Instruments*",
        "SAME frozen specification, no changes. Data availability checked first, not assumed -- smaller-cap "
        "tokens are recent and may not have reliable historical coverage.\n",
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2 + 250)

    for api_symbol in SMALLER_CAP_CANDIDATES:
        logger.info("Checking data availability for %s...", api_symbol)
        df = fetch_full_history(api_symbol, "1day", outputsize=1000)

        if df is None or len(df) < MA_SLOW + 50:
            n_available = len(df) if df is not None else 0
            lines.append(f"*{api_symbol}*: DATA INSUFFICIENT ({n_available} candles available, need {MA_SLOW + 50}+). Skipped.")
            lines.append("")
            continue

        df = df[df["datetime"] >= start_date].reset_index(drop=True)
        lines.append(f"*{api_symbol}*")
        lines.append(f"  Data: {df['datetime'].min().date()} to {df['datetime'].max().date()} ({len(df)} candles)")

        result = run_backtest_on_df(df)
        if result["n"] == 0:
            lines.append("  No trades triggered in this period.")
        else:
            pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] is not None else "N/A"
            lines.append(
                f"  n={result['n']}, win rate={result['win_rate']*100:.0f}%, avg R={result['avg_r']:+.3f}, "
                f"PF={pf_str}, max DD={result['max_dd_r']:.2f}R"
            )
            if result["n"] >= 30 and result["avg_r"] > 0 and (result["profit_factor"] or 0) > 1.0:
                lines.append("  CLASSIFICATION: PROMISING (adequate sample) -- still needs OOS confirmation.")
            elif result["n"] < 30:
                lines.append(f"  CLASSIFICATION: INCONCLUSIVE (n={result['n']}, below the 30-trade minimum).")
            else:
                lines.append("  CLASSIFICATION: FAILED.")
        lines.append("")

    lines.append(
        "Reminder: all instruments checked and reported above, including any DATA INSUFFICIENT or FAILED "
        "results -- not just whichever looked best."
    )

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent smaller-cap confluence check report")


if __name__ == "__main__":
    asyncio.run(run())
