"""Two independent filter tests on the REAL, already-validated trend
strategy -- using the actual scalp_analysis.py functions directly, not
an approximation.

TEST 1 -- Volume confirmation, isolated: 'volume_confirmed' is already
one of several factors inside score_and_decide's composite score. This
tests it ALONE: does it separate winning trend trades from losing ones
when checked independently, not just as +1 of 11 possible points?

TEST 2 -- Support/resistance proximity as a TREND filter (not a
standalone strategy -- that was already tested as detect_range_setup
and FAILED, n=920, avg R -0.12, over the same 2.8-year period). This
asks a different, narrower question: does avoiding a trend-LONG signal
that's priced right under known resistance (or a trend-SHORT right
above known support) improve the EXISTING trend edge specifically?

Both tests run on the SAME real signal stream, split after the fact --
not two separate strategies, two separate honest cuts of one dataset.

Scope: ~4 months of XAU/USD, 4h/1h/15m/5m data -- a faster first look
given today's full-period multi-timeframe backtests have taken 45-60+
minutes. If either filter looks genuinely interesting, the same
2year-then-extended pattern used for the confluence tests applies here.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from . import config, scalp_analysis
from .historical_backtest import fetch_paginated_history, find_outcome_detailed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("filter_tests")

WARMUP_BARS = 250
REPLAY_STEP_MINUTES = 30
SPREAD_PCT_AS_ATR_FRACTION = None  # costs already baked into compute_trade_levels' SL/TP distances


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


def slice_up_to(df: pd.DataFrame, cutoff_dt, window: int = WARMUP_BARS):
    sliced = df[df["datetime"] <= cutoff_dt]
    if len(sliced) < window:
        return None
    return sliced.iloc[-window:].reset_index(drop=True)


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    wins = [t for t in trades if t["r_multiple"] > 0]
    avg_r = sum(t["r_multiple"] for t in trades) / n
    win_rate = len(wins) / n
    gross_win = sum(t["r_multiple"] for t in wins)
    gross_loss = abs(sum(t["r_multiple"] for t in trades if t["r_multiple"] <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else None
    return {"n": n, "win_rate": win_rate, "avg_r": avg_r, "profit_factor": pf}


def fmt(label: str, stats: dict) -> str:
    if stats["n"] == 0:
        return f"*{label}*: no trades"
    pf_str = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "N/A"
    return f"*{label}*: n={stats['n']}, win rate={stats['win_rate']*100:.0f}%, avg R={stats['avg_r']:+.3f}, PF={pf_str}"


async def run():
    lines = [
        "*Two Independent Filter Tests on the Real Trend Strategy*",
        "Using the actual live strategy functions directly. ~4-month window, XAU/USD, real multi-timeframe "
        "data. Volume confirmation and S/R proximity tested separately -- not combined.\n",
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=130)
    fetch_buffer_start = start_date - timedelta(days=30)  # extra bars for warmup before replay begins

    logger.info("Fetching multi-timeframe XAU/USD data...")
    dfs_full = {}
    for label, interval in scalp_analysis.TIMEFRAMES.items():
        df = fetch_paginated_history("XAU/USD", interval, fetch_buffer_start, end_date)
        if df is None or len(df) < WARMUP_BARS:
            lines.append(f"Insufficient {label} data -- aborting.")
            _send_telegram_direct("\n".join(lines))
            return
        dfs_full[label] = df
        logger.info("%s: %d candles, %s to %s", label, len(df), df["datetime"].min(), df["datetime"].max())

    earliest_start = max(df["datetime"].min() for df in dfs_full.values())
    latest_end = min(df["datetime"].max() for df in dfs_full.values())
    replay_start = max(earliest_start + timedelta(minutes=WARMUP_BARS * 5), start_date)
    replay_end = latest_end - timedelta(hours=24)

    if replay_start >= replay_end:
        lines.append("Not enough range to replay after warmup/lookahead.")
        _send_telegram_direct("\n".join(lines))
        return

    logger.info("Replaying %s to %s...", replay_start, replay_end)

    volume_confirmed_trades, volume_unconfirmed_trades = [], []
    sr_clear_trades, sr_crowded_trades = [], []
    tracker = {"direction": None, "exit_time": None}

    step = timedelta(minutes=REPLAY_STEP_MINUTES)
    t = replay_start
    n_signals = 0

    while t <= replay_end:
        tf_data = {}
        dfs = {}
        ok = True
        for label in ["4h", "1h", "15m", "5m"]:
            sliced = slice_up_to(dfs_full[label], t)
            if sliced is None:
                ok = False
                break
            tf_data[label] = scalp_analysis.compute_indicators(sliced)
            dfs[label] = sliced
        if not ok:
            t += step
            continue

        decision = scalp_analysis.score_and_decide(tf_data, "neutral")
        action = decision["action"]

        if action == "NO TRADE":
            tracker["direction"], tracker["exit_time"] = None, None
            t += step
            continue

        is_continuation = (
            tracker["direction"] == action
            and tracker["exit_time"] is not None
            and t <= tracker["exit_time"]
        )
        if is_continuation:
            t += step
            continue

        entry_price = tf_data["5m"]["close"]
        levels = scalp_analysis.compute_trade_levels(action, entry_price, tf_data["5m"]["atr"])
        detail = find_outcome_detailed(
            dfs_full["5m"], t, levels["entry"], levels["sl"], levels["tp1"], levels["tp2"], action, lookahead_hours=24
        )
        tracker["direction"] = action
        tracker["exit_time"] = t + timedelta(hours=24)
        n_signals += 1

        trade_record = {"r_multiple": detail["r_multiple"]}

        # TEST 1: volume confirmation, checked in isolation -- same
        # condition score_and_decide itself checks, but reported on its
        # own rather than bundled into the composite score.
        volume_confirmed = tf_data["15m"]["volume_confirmed"] or tf_data["5m"]["volume_confirmed"]
        if volume_confirmed:
            volume_confirmed_trades.append(trade_record)
        else:
            volume_unconfirmed_trades.append(trade_record)

        # TEST 2: S/R proximity as a trend filter -- "crowded" means a
        # LONG signal priced within 0.5x ATR of known resistance, or a
        # SHORT within 0.5x ATR of known support (using the real
        # find_nearest_levels function, on the 1h timeframe).
        support, resistance = scalp_analysis.find_nearest_levels(dfs["1h"], tf_data["1h"]["close"])
        atr_1h = tf_data["1h"]["atr"]
        crowded = False
        if action == "LONG" and resistance is not None:
            crowded = (resistance - tf_data["1h"]["close"]) <= 0.5 * atr_1h
        elif action == "SHORT" and support is not None:
            crowded = (tf_data["1h"]["close"] - support) <= 0.5 * atr_1h

        if crowded:
            sr_crowded_trades.append(trade_record)
        else:
            sr_clear_trades.append(trade_record)

        t += step

    lines.append(f"Signals evaluated: {n_signals}\n")

    lines.append("*TEST 1 -- Volume Confirmation (isolated)*")
    lines.append(fmt("Volume confirmed", summarize(volume_confirmed_trades)))
    lines.append(fmt("Volume NOT confirmed", summarize(volume_unconfirmed_trades)))
    lines.append("")

    lines.append("*TEST 2 -- S/R Proximity Filter (on trend signals)*")
    lines.append(fmt("Clear of S/R", summarize(sr_clear_trades)))
    lines.append(fmt("Crowded (near opposing S/R)", summarize(sr_crowded_trades)))
    lines.append("")

    lines.append(
        "Reminder: this is a ~4-month first look, not full validation -- same discipline as the confluence "
        "tests. Every filter tried so far today (ADX, confidence, session, structural levels) failed to "
        "improve the base edge; both results above are reported regardless of outcome."
    )

    _send_telegram_direct("\n".join(lines))
    logger.info("Sent filter test report")


if __name__ == "__main__":
    asyncio.run(run())
