"""Higher-Timeframe Structural Level Rejection -- Phases 1-3.

HYPOTHESIS: XAU/USD price approaching significant daily/weekly structural
levels may produce a measurable reaction that can improve trade
expectancy. This is a genuinely different mechanism from every other
strategy tested in this project -- it uses higher-timeframe (daily/
weekly) historical structure as the signal, not 4H price action patterns.

PHASE GATING, exactly as specified: Phase 1 (data audit) must pass before
anything else runs. Phase 2 (level definition) is FROZEN below, before
any performance data was examined. Phase 3 (information content) must
show genuine predictive value BEFORE any trading rule is defined -- this
module does NOT build a trading rule; that is an explicitly separate,
later step gated on this phase's result.

FROZEN LEVEL DEFINITION (Phase 2, documented before any results seen):
- Swing point: a 5-bar fractal (candle's high/low is the extreme among
  the 5 candles before AND after it) on daily and weekly XAU/USD.
- Level construction lookback: trailing 252 daily candles (~1yr) for
  daily levels, trailing 104 weekly candles (~2yr) for weekly levels.
- Clustering: swing points within 0.5% of each other merge into one level.
- Strength: a level qualifies as "structural" only with >=2 distinct
  swing points clustered into it.
- Expiration: a level is no longer active once price has closed beyond
  it by more than 2x the daily ATR (measured at its most recent touch).
- Approach event: a 4H candle whose high/low comes within 0.5x the 4H
  ATR of an ACTIVE level, without closing beyond it.
- NO-LOOKAHEAD: at any timestamp, only swing points strictly BEFORE that
  timestamp are used to build the active level set for that timestamp.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import ta

from . import db
from . import telegram_bot
from .historical_backtest import fetch_paginated_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xau_level_rejection")

API_SYMBOL = "XAU/USD"
FRACTAL_WINDOW = 5
DAILY_LOOKBACK_CANDLES = 252
WEEKLY_LOOKBACK_CANDLES = 104
CLUSTER_PCT = 0.5
MIN_TOUCHES = 2
EXPIRATION_ATR_MULT = 2.0
APPROACH_ATR_MULT_4H = 0.5

DEV_START, DEV_END = "2025-01-01", "2025-09-01"
OOS_START, OOS_END = "2025-09-01", "2026-01-01"
# Level construction needs history BEFORE the dev period starts, per the
# no-lookahead rule -- request well before 2025-01-01 so daily/weekly
# levels active AT the start of dev are built from real prior data, not
# an empty/warm-up-starved set.
LEVEL_HISTORY_START = "2022-01-01"


async def run_data_audit():
    """Phase 1 -- STOP here if this fails, per protocol. Checks daily and
    weekly XAU/USD availability and depth via the same feed the rest of
    this project already uses."""
    lines = [
        "*📐 Structural Level Rejection -- Phase 1: Data Audit*",
        "_Checking daily/weekly XAU/USD availability. No strategy or backtest yet._\n",
    ]

    target_start = datetime.strptime(LEVEL_HISTORY_START, "%Y-%m-%d")
    target_end = datetime.strptime(OOS_END, "%Y-%m-%d") + timedelta(days=1)

    df_daily = fetch_paginated_history(API_SYMBOL, "1day", target_start, target_end)
    df_weekly = fetch_paginated_history(API_SYMBOL, "1week", target_start, target_end)

    daily_ok = df_daily is not None and len(df_daily) > DAILY_LOOKBACK_CANDLES + 200
    weekly_ok = df_weekly is not None and len(df_weekly) > WEEKLY_LOOKBACK_CANDLES + 50

    if df_daily is not None:
        lines.append(f"Daily: {len(df_daily)} candles, {df_daily['datetime'].min()} to {df_daily['datetime'].max()}")
    else:
        lines.append("Daily: FAILED to fetch")
    if df_weekly is not None:
        lines.append(f"Weekly: {len(df_weekly)} candles, {df_weekly['datetime'].min()} to {df_weekly['datetime'].max()}")
    else:
        lines.append("Weekly: FAILED to fetch")

    dev_start_dt = datetime.strptime(DEV_START, "%Y-%m-%d")
    oos_end_dt = datetime.strptime(OOS_END, "%Y-%m-%d")
    covers_dev_oos = daily_ok and df_daily["datetime"].min() <= dev_start_dt and df_daily["datetime"].max() >= oos_end_dt - timedelta(days=2)

    lines.append("")
    lines.append(f"Sufficient daily history (pre-dev lookback + dev + OOS): {'✅ yes' if daily_ok else '❌ no'}")
    lines.append(f"Sufficient weekly history: {'✅ yes' if weekly_ok else '❌ no'}")
    lines.append(f"Covers full dev+OOS window: {'✅ yes' if covers_dev_oos else '❌ no'}")

    passed = daily_ok and weekly_ok and covers_dev_oos
    lines.append("")
    if passed:
        lines.append("*RESULT: DATA SUFFICIENT -- proceeding to Phase 2/3 is justified.*")
    else:
        lines.append("*RESULT: DATA SOURCE INSUFFICIENT -- per protocol, STOP here.*")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent data audit report, passed=%s", passed)
    return passed


def detect_swing_points(df: pd.DataFrame, window: int = FRACTAL_WINDOW) -> list[dict]:
    """5-bar fractal swing highs/lows. Each swing point's index in df is
    recorded so no-lookahead filtering can later restrict to swings
    whose CONFIRMING candles (both sides of the fractal) occurred before
    a given timestamp."""
    swings = []
    n = len(df)
    for i in range(window, n - window):
        window_slice_high = df["high"].iloc[i - window:i + window + 1]
        window_slice_low = df["low"].iloc[i - window:i + window + 1]
        if df["high"].iloc[i] == window_slice_high.max():
            swings.append({"index": i, "datetime": df["datetime"].iloc[i], "confirmed_at": df["datetime"].iloc[i + window], "price": df["high"].iloc[i], "type": "high"})
        if df["low"].iloc[i] == window_slice_low.min():
            swings.append({"index": i, "datetime": df["datetime"].iloc[i], "confirmed_at": df["datetime"].iloc[i + window], "price": df["low"].iloc[i], "type": "low"})
    return swings


def cluster_swings_into_levels(swings: list[dict], cluster_pct: float = CLUSTER_PCT) -> list[dict]:
    """Merges swing points within cluster_pct of each other into a single
    level, tracking how many distinct swings support it (its 'strength')
    and the most recent swing's confirmation time."""
    if not swings:
        return []
    sorted_swings = sorted(swings, key=lambda s: s["price"])
    levels = []
    current_cluster = [sorted_swings[0]]

    for s in sorted_swings[1:]:
        cluster_center = sum(c["price"] for c in current_cluster) / len(current_cluster)
        if abs(s["price"] - cluster_center) / cluster_center * 100 <= cluster_pct:
            current_cluster.append(s)
        else:
            levels.append(_finalize_cluster(current_cluster))
            current_cluster = [s]
    levels.append(_finalize_cluster(current_cluster))
    return levels


def _finalize_cluster(cluster: list[dict]) -> dict:
    return {
        "price": sum(c["price"] for c in cluster) / len(cluster),
        "strength": len(cluster),
        "last_confirmed_at": max(c["confirmed_at"] for c in cluster),
        "swing_datetimes": [c["datetime"] for c in cluster],
    }


def get_active_levels_as_of(all_swings: list[dict], as_of_time, lookback_candles: int,
                             df_for_lookback: pd.DataFrame, min_touches: int = MIN_TOUCHES) -> list[dict]:
    """NO-LOOKAHEAD: restricts to swing points whose confirmation time is
    STRICTLY BEFORE as_of_time, and whose origin candle falls within the
    trailing lookback_candles window as measured from as_of_time. This is
    the single most important correctness property in this module."""
    eligible_swings = [s for s in all_swings if s["confirmed_at"] < as_of_time]
    if not eligible_swings:
        return []

    # Restrict further to the trailing lookback window (in candles, using
    # df_for_lookback's own index positions as of as_of_time)
    candles_before = df_for_lookback[df_for_lookback["datetime"] < as_of_time]
    if len(candles_before) == 0:
        return []
    lookback_start_idx = max(0, len(candles_before) - lookback_candles)
    lookback_start_time = df_for_lookback["datetime"].iloc[lookback_start_idx]
    eligible_swings = [s for s in eligible_swings if s["datetime"] >= lookback_start_time]

    levels = cluster_swings_into_levels(eligible_swings)
    return [lvl for lvl in levels if lvl["strength"] >= min_touches]


def is_level_expired(level: dict, current_price: float, daily_atr_at_touch: float) -> bool:
    """A level expires once price has closed beyond it by more than
    EXPIRATION_ATR_MULT x the daily ATR (measured at its last touch)."""
    return abs(current_price - level["price"]) > EXPIRATION_ATR_MULT * daily_atr_at_touch


if __name__ == "__main__":
    asyncio.run(run_data_audit())
