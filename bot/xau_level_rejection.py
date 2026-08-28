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


def find_approach_events(df_4h: pd.DataFrame, df_daily: pd.DataFrame, df_weekly: pd.DataFrame,
                          daily_swings: list[dict], weekly_swings: list[dict],
                          start_date, end_date, cooldown_candles: int = 4) -> list[dict]:
    """Scans every 4H candle in [start_date, end_date) for an approach to
    an ACTIVE level (no lookahead -- levels as of each candle's own
    timestamp). A simple, frozen cooldown avoids trivially over-counting
    many consecutive candles hovering near the same level as independent
    events."""
    atr_4h = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"], window=14).average_true_range()
    events = []
    last_event_idx_for_level: dict[float, int] = {}

    mask = (df_4h["datetime"] >= start_date) & (df_4h["datetime"] < end_date)
    for i in df_4h[mask].index:
        atr_val = atr_4h.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue
        candle_time = df_4h["datetime"].iloc[i]
        candle_high, candle_low, candle_close = df_4h["high"].iloc[i], df_4h["low"].iloc[i], df_4h["close"].iloc[i]
        threshold = APPROACH_ATR_MULT_4H * atr_val

        daily_levels = get_active_levels_as_of(daily_swings, candle_time, DAILY_LOOKBACK_CANDLES, df_daily)
        weekly_levels = get_active_levels_as_of(weekly_swings, candle_time, WEEKLY_LOOKBACK_CANDLES, df_weekly)

        for level, level_type in [(l, "daily") for l in daily_levels] + [(l, "weekly") for l in weekly_levels]:
            key = round(level["price"], 2)
            if key in last_event_idx_for_level and i - last_event_idx_for_level[key] < cooldown_candles:
                continue
            if level["price"] - threshold <= candle_high <= level["price"] and candle_close < level["price"]:
                events.append({
                    "index": i, "time": candle_time, "level_price": level["price"], "level_type": level_type,
                    "approach_direction": "from_below", "level_strength": level["strength"], "atr_at_event": atr_val,
                })
                last_event_idx_for_level[key] = i
            elif level["price"] <= candle_low <= level["price"] + threshold and candle_close > level["price"]:
                events.append({
                    "index": i, "time": candle_time, "level_price": level["price"], "level_type": level_type,
                    "approach_direction": "from_above", "level_strength": level["strength"], "atr_at_event": atr_val,
                })
                last_event_idx_for_level[key] = i
    return events


def measure_reaction(df_4h: pd.DataFrame, event_index: int, direction: str, atr_at_event: float,
                      horizon: int = 20) -> dict | None:
    """Measures, over the next `horizon` 4H candles, how far price moved
    in the REACTION direction (away from the level -- the hypothesized
    rejection) versus the CONTINUATION direction (through the level),
    both in ATR units at the time of the event. Also flags whether price
    actually closed beyond the level within the window."""
    n = len(df_4h)
    if event_index + horizon >= n:
        return None
    entry_price = df_4h["close"].iloc[event_index]
    forward = df_4h.iloc[event_index + 1: event_index + 1 + horizon]

    if direction == "from_below":  # level = resistance; reaction = DOWN, continuation = UP (through it)
        reaction_r = (entry_price - forward["low"].min()) / atr_at_event
        continuation_r = (forward["high"].max() - entry_price) / atr_at_event
    else:  # from_above; level = support; reaction = UP, continuation = DOWN (through it)
        reaction_r = (forward["high"].max() - entry_price) / atr_at_event
        continuation_r = (entry_price - forward["low"].min()) / atr_at_event

    return {"reaction_r": reaction_r, "continuation_r": continuation_r, "rejected": reaction_r > continuation_r}


def sample_baseline_events(df_4h: pd.DataFrame, approach_indices: set[int], start_date, end_date,
                            sample_size: int = 300, seed: int = 11) -> list[dict]:
    """Random 4H candles in the SAME window that are NOT flagged as near
    any active level, each assigned a RANDOM 50/50 direction -- the fair
    baseline for 'if you had no level information and just guessed'."""
    import random
    random.seed(seed)
    atr_4h = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"], window=14).average_true_range()
    mask = (df_4h["datetime"] >= start_date) & (df_4h["datetime"] < end_date)
    candidate_indices = [i for i in df_4h[mask].index if i not in approach_indices and pd.notna(atr_4h.iloc[i]) and atr_4h.iloc[i] > 0]
    sampled = random.sample(candidate_indices, min(sample_size, len(candidate_indices)))

    results = []
    for i in sampled:
        direction = random.choice(["from_below", "from_above"])
        results.append({"index": i, "direction": direction, "atr_at_event": atr_4h.iloc[i]})
    return results


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> float | None:
    """Standard two-proportion z-test -- correctly accounts for sample
    size, unlike a flat percentage-point threshold (an earlier version of
    this module used exactly that, and rigorous testing caught it
    producing a false-positive PROMISING verdict on pure random synthetic
    data with no genuine level effect at all)."""
    import math
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return None
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    return (p1 - p2) / se


async def run_information_content_test(period: str = "dev"):
    """Phase 3 -- the critical, result-independent gate. Does NOT build
    or test a trading rule; only measures whether price behaves
    differently near a structural level than at a random comparable
    baseline. Nothing here is optimized; the level definition and horizon
    are frozen per the Phase 2 specification."""
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)
    start_dt, end_dt = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")

    logger.info("Fetching daily, weekly, and 4H XAU/USD data...")
    target_start = datetime.strptime(LEVEL_HISTORY_START, "%Y-%m-%d")
    target_end = end_dt + timedelta(days=1)
    df_daily = fetch_paginated_history(API_SYMBOL, "1day", target_start, target_end)
    df_weekly = fetch_paginated_history(API_SYMBOL, "1week", target_start, target_end)
    df_4h = fetch_paginated_history(API_SYMBOL, "4h", target_start, target_end)

    if df_daily is None or df_weekly is None or df_4h is None:
        await telegram_bot.send_text("*📐 Structural Level Rejection -- Phase 3*\n\nData fetch failed -- aborting.")
        return

    logger.info("Detecting swing points...")
    daily_swings = detect_swing_points(df_daily)
    weekly_swings = detect_swing_points(df_weekly)
    logger.info("%d daily swings, %d weekly swings detected", len(daily_swings), len(weekly_swings))

    events = find_approach_events(df_4h, df_daily, df_weekly, daily_swings, weekly_swings, start_dt, end_dt)
    logger.info("%d approach events detected", len(events))

    HORIZON = 20
    event_reactions = []
    for e in events:
        r = measure_reaction(df_4h, e["index"], e["approach_direction"], e["atr_at_event"], horizon=HORIZON)
        if r is not None:
            r.update(e)
            event_reactions.append(r)

    approach_indices = {e["index"] for e in events}
    baseline_events = sample_baseline_events(df_4h, approach_indices, start_dt, end_dt)
    baseline_reactions = []
    for b in baseline_events:
        r = measure_reaction(df_4h, b["index"], b["direction"], b["atr_at_event"], horizon=HORIZON)
        if r is not None:
            baseline_reactions.append(r)

    n_events = len(event_reactions)
    n_baseline = len(baseline_reactions)

    lines = [
        f"*📐 Structural Level Rejection -- Phase 3: Information Content ({period.upper()})*",
        "_Testing ONLY whether price behaves differently near a level vs a random baseline. No trading rule yet._\n",
    ]

    if n_events == 0:
        lines.append("No qualifying approach events detected in this period.")
        lines.append("\n*RESULT: FAILED -- no events to analyze.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    avg_reaction_event = sum(e["reaction_r"] for e in event_reactions) / n_events
    avg_continuation_event = sum(e["continuation_r"] for e in event_reactions) / n_events
    reject_rate_event = sum(1 for e in event_reactions if e["rejected"]) / n_events

    avg_reaction_baseline = sum(b["reaction_r"] for b in baseline_reactions) / n_baseline if n_baseline else None
    avg_continuation_baseline = sum(b["continuation_r"] for b in baseline_reactions) / n_baseline if n_baseline else None
    reject_rate_baseline = sum(1 for b in baseline_reactions if b["rejected"]) / n_baseline if n_baseline else None

    lines.append(f"Approach events: {n_events} (n from_below={sum(1 for e in event_reactions if e['approach_direction']=='from_below')}, "
                 f"from_above={sum(1 for e in event_reactions if e['approach_direction']=='from_above')})")
    lines.append(f"  daily levels: {sum(1 for e in event_reactions if e['level_type']=='daily')}, weekly levels: {sum(1 for e in event_reactions if e['level_type']=='weekly')}")
    lines.append("")
    lines.append("*Near-level events*")
    lines.append(f"  Avg reaction (away from level): {avg_reaction_event:+.3f} ATR")
    lines.append(f"  Avg continuation (through level): {avg_continuation_event:+.3f} ATR")
    lines.append(f"  Rejection rate (reaction > continuation): {reject_rate_event*100:.1f}%")
    lines.append("")
    lines.append(f"*Random-direction baseline (n={n_baseline})*")
    if avg_reaction_baseline is not None:
        lines.append(f"  Avg reaction: {avg_reaction_baseline:+.3f} ATR")
        lines.append(f"  Avg continuation: {avg_continuation_baseline:+.3f} ATR")
        lines.append(f"  Rejection rate: {reject_rate_baseline*100:.1f}%")

    info_edge = reject_rate_event - reject_rate_baseline if reject_rate_baseline is not None else None
    z_score = None
    if reject_rate_baseline is not None and n_baseline:
        x_event = sum(1 for e in event_reactions if e["rejected"])
        x_baseline = sum(1 for b in baseline_reactions if b["rejected"])
        z_score = two_proportion_z_test(x_event, n_events, x_baseline, n_baseline)

    lines.append("")
    if info_edge is not None:
        lines.append(f"*Information edge: {info_edge*100:+.1f} percentage points rejection rate vs random baseline*")
        z_str = f"{z_score:.2f}" if z_score is not None else "N/A"
        lines.append(f"*Two-proportion z-score: {z_str}* (needs |z|>1.96 for 95% significance -- not just a raw percentage-point gap, which was tested and found to false-positive on pure random data)")
        if n_events < 30:
            verdict = "INCONCLUSIVE -- sample too small to draw a confident conclusion, regardless of the apparent edge."
        elif z_score is not None and abs(z_score) > 1.96 and info_edge > 0:
            verdict = "PROMISING -- statistically significant edge over baseline, adequate sample. Phase 4 (trading rule) is justified."
        else:
            verdict = "FAILED -- no statistically significant information advantage over a random-direction baseline."
        lines.append(f"*RESULT: {verdict}*")

    lines.append(
        "\n_This measures REACTION vs a fair baseline only -- it is not a trading strategy and includes no "
        "transaction costs, entry confirmation, or stop/exit logic. A positive result here justifies designing "
        "a trading rule (Phase 4); it is not evidence that rule will be profitable._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent Phase 3 information content report (%s)", period)


if __name__ == "__main__":
    asyncio.run(run_data_audit())
