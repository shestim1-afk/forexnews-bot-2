"""EUR/USD as a USD-Strength Proxy: does it contain incremental
predictive information for the ALREADY-VALIDATED XAU/USD 4H trend signal?

This is a REFINEMENT test, not a second independent strategy. The XAU
entry/stop/exit/position-sizing are completely unchanged -- only whether
we ACT on a given signal is conditioned on EUR/USD's state. Per the
frozen specification: a positive result is NOT additive R/day to the
portfolio; it can only potentially improve the existing 0.403 R/day
XAU figure.

FROZEN definitions (before any results were examined):
- Candidate variable: the EUR/USD 4H candle that CLOSED immediately
  before a given XAU entry candle opened (strictly no lookahead).
  USD-weakness state: that candle's close > open (EUR/USD rose).
  USD-strength state: that candle's close < open (EUR/USD fell).
  Neutral: exact tie (close == open), reported separately.
- Alignment: XAU LONG + USD-weakness = aligned; XAU LONG + USD-strength
  = opposed. XAU SHORT + USD-strength = aligned; XAU SHORT + USD-weakness
  = opposed.
- Cost model: XAU-only (0.03% default XAU spread) -- EUR/USD is an
  information input, not a second traded leg, so no extra cost applied.
- Success requires ALL of: aligned net R/day > baseline net R/day,
  Welch's t-test p<0.05 (aligned mean R vs baseline mean R), n>=30 in
  the aligned bucket, concentration check passes (top 20% of trades
  <=80% of positive R), AND the result survives OOS unchanged.
- Interpretation rule: a positive result means EUR/USD state has
  predictive information CONDITIONAL ON the existing XAU signal -- not
  that EUR/USD "causes" better XAU trades. That is the strongest claim
  this experiment can legitimately make.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from scipy import stats

from . import telegram_bot
from .historical_backtest import fetch_paginated_history
from .xau_diagnostics import get_xau_trend_trades, compute_bucket_stats, MIN_SAMPLE_SIZE
from .analytics import get_default_spread_pct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eurusd_xau_proxy")

DEV_START, DEV_END = "2025-01-01", "2025-09-01"
OOS_START, OOS_END = "2025-09-01", "2026-01-01"
CONCENTRATION_FAILURE_THRESHOLD = 0.80
SIGNIFICANCE_LEVEL = 0.05


def classify_eurusd_state(eurusd_candle: dict) -> str:
    if eurusd_candle["close"] > eurusd_candle["open"]:
        return "weakness"  # EUR/USD rose -> USD weakness
    elif eurusd_candle["close"] < eurusd_candle["open"]:
        return "strength"  # EUR/USD fell -> USD strength
    return "neutral"


def find_prior_eurusd_candle(eurusd_candles: list[dict], xau_entry_time: str) -> dict | None:
    """Finds the EUR/USD 4H candle that closed strictly BEFORE the XAU
    entry time -- the most recent one whose own timestamp is earlier.
    eurusd_candles must be pre-sorted ascending by datetime. Normalizes
    to timezone-naive (stripping any tzinfo), matching the naive-datetime
    convention Twelve Data timestamps use everywhere else in this
    project -- otherwise a stored timezone-aware evaluated_at string
    fails to compare against the naive candle timestamps at all."""
    entry_dt = datetime.fromisoformat(xau_entry_time)
    if entry_dt.tzinfo is not None:
        entry_dt = entry_dt.replace(tzinfo=None)
    prior = None
    for c in eurusd_candles:
        candle_dt = c["datetime"]
        if hasattr(candle_dt, "tzinfo") and candle_dt.tzinfo is not None:
            candle_dt = candle_dt.replace(tzinfo=None)
        if candle_dt < entry_dt:
            prior = c
        else:
            break
    return prior


def classify_alignment(xau_action: str, eurusd_state: str) -> str:
    if eurusd_state == "neutral":
        return "neutral"
    if xau_action == "LONG":
        return "aligned" if eurusd_state == "weakness" else "opposed"
    else:  # SHORT
        return "aligned" if eurusd_state == "strength" else "opposed"


def check_concentration(r_values: list[float]) -> dict:
    n = len(r_values)
    if n == 0:
        return {"concentrated": False, "top_20pct_share": None}
    total_positive = sum(r for r in r_values if r > 0)
    if total_positive <= 0:
        return {"concentrated": False, "top_20pct_share": None}
    top_count = max(1, int(n * 0.2))
    top_sum = sum(sorted(r_values, reverse=True)[:top_count])
    share = top_sum / total_positive
    return {"concentrated": share > CONCENTRATION_FAILURE_THRESHOLD, "top_20pct_share": share}


async def run_test(period: str = "dev", spread_pct: float | None = None):
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct("XAU/USD")

    xau_trades = get_xau_trend_trades(start_date, end_date)
    if not xau_trades:
        await telegram_bot.send_text(f"*💱 EUR/USD USD-Proxy Test ({period.upper()})*\n\nNo XAU trades found for {start_date} to {end_date}.")
        return

    logger.info("Fetching EUR/USD 4H data for %s to %s...", start_date, end_date)
    target_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5)  # small buffer so the FIRST XAU trade has a prior EUR/USD candle
    target_end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    df_eurusd = fetch_paginated_history("EUR/USD", "4h", target_start, target_end)

    if df_eurusd is None or len(df_eurusd) == 0:
        await telegram_bot.send_text(f"*💱 EUR/USD USD-Proxy Test ({period.upper()})*\n\nEUR/USD data fetch failed -- DATA INSUFFICIENT.")
        return

    eurusd_candles = df_eurusd.sort_values("datetime").to_dict("records")

    # get_xau_trend_trades doesn't include the trade's action (LONG/SHORT)
    # -- alignment depends on direction per the frozen spec, so pull the
    # SAME data directly with action included.
    from . import db
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT e.evaluated_at, e.action, e.entry, e.sl, o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type='trend'
                 AND e.symbol LIKE 'XAU/USD [4h-ATR]%' AND e.symbol NOT LIKE '%(R:R%'
                 AND o.outcome IN ('WIN', 'LOSS')
                 AND e.evaluated_at >= ? AND e.evaluated_at < ?
               ORDER BY e.evaluated_at ASC""",
            (start_date, end_date),
        ).fetchall()
    finally:
        conn.close()
    cols = ["evaluated_at", "action", "entry", "sl", "outcome", "r_multiple"]
    xau_trades_with_action = [dict(zip(cols, row)) for row in rows]

    aligned, opposed, neutral = [], [], []
    for t in xau_trades_with_action:
        prior = find_prior_eurusd_candle(eurusd_candles, t["evaluated_at"])
        if prior is None:
            continue
        state = classify_eurusd_state(prior)
        bucket = classify_alignment(t["action"], state)
        if bucket == "aligned":
            aligned.append(t)
        elif bucket == "opposed":
            opposed.append(t)
        else:
            neutral.append(t)

    days_span = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    baseline_stats = compute_bucket_stats(xau_trades_with_action, days_span, spread_pct)
    aligned_stats = compute_bucket_stats(aligned, days_span, spread_pct)
    opposed_stats = compute_bucket_stats(opposed, days_span, spread_pct)

    lines = [
        f"*💱 EUR/USD USD-Proxy Refinement Test ({period.upper()}): {start_date} to {end_date}*",
        "_Testing whether EUR/USD state adds incremental information to the EXISTING XAU signal -- not a new strategy. "
        "A positive result is NOT additive R/day; it can only refine the existing 0.403 R/day figure._\n",
    ]

    lines.append("*Baseline (unfiltered XAU)*")
    lines.append(f"  n={baseline_stats['n']}, net R/day {baseline_stats['net_r_day']:+.3f}, avg R/trade {baseline_stats['net_avg_r']:+.3f}")
    lines.append("")
    lines.append(f"*Aligned (n={aligned_stats.get('n',0)})*")
    if aligned_stats.get("n", 0) > 0:
        lines.append(f"  WR {aligned_stats['win_rate']*100:.0f}%, avg R {aligned_stats['net_avg_r']:+.3f}, net R/day {aligned_stats['net_r_day']:+.3f}, PF {aligned_stats['profit_factor']:.2f}" if aligned_stats["profit_factor"] else f"  WR {aligned_stats['win_rate']*100:.0f}%, avg R {aligned_stats['net_avg_r']:+.3f}")
    lines.append(f"*Opposed (n={opposed_stats.get('n',0)})*")
    if opposed_stats.get("n", 0) > 0:
        lines.append(f"  WR {opposed_stats['win_rate']*100:.0f}%, avg R {opposed_stats['net_avg_r']:+.3f}, net R/day {opposed_stats['net_r_day']:+.3f}")
    lines.append(f"Neutral: n={len(neutral)}")

    # Welch's t-test: aligned mean R vs baseline mean R
    aligned_r = [t["r_multiple"] for t in aligned]
    baseline_r = [t["r_multiple"] for t in xau_trades_with_action]
    t_stat, p_value = (None, None)
    if len(aligned_r) >= 2 and len(baseline_r) >= 2:
        t_stat, p_value = stats.ttest_ind(aligned_r, baseline_r, equal_var=False)

    concentration = check_concentration(aligned_r) if aligned_r else {"concentrated": False, "top_20pct_share": None}

    lines.append("")
    lines.append("*Statistical test (Welch's t-test, aligned vs baseline mean R)*")
    if p_value is not None:
        lines.append(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f} (need <{SIGNIFICANCE_LEVEL} for significance)")
    else:
        lines.append("  Insufficient data for t-test.")
    if concentration["top_20pct_share"] is not None:
        lines.append(f"  Concentration: top 20% of aligned trades = {concentration['top_20pct_share']*100:.0f}% of positive R")

    n_aligned = aligned_stats.get("n", 0)
    beats_baseline = aligned_stats.get("net_r_day", -999) > baseline_stats["net_r_day"] if n_aligned > 0 else False
    significant = p_value is not None and p_value < SIGNIFICANCE_LEVEL
    adequate_sample = n_aligned >= MIN_SAMPLE_SIZE
    not_concentrated = not concentration["concentrated"]

    lines.append("")
    lines.append("*Success criteria check*")
    lines.append(f"  Beats baseline net R/day: {'✅' if beats_baseline else '❌'}")
    lines.append(f"  Statistically significant (p<0.05): {'✅' if significant else '❌'}")
    lines.append(f"  Adequate sample (n>={MIN_SAMPLE_SIZE}): {'✅' if adequate_sample else '❌'}")
    lines.append(f"  Not concentration-dominated: {'✅' if not_concentrated else '❌'}")

    if not adequate_sample:
        classification = "INCONCLUSIVE"
        reason = f"Aligned sample too small (n={n_aligned})."
    elif beats_baseline and significant and not_concentrated:
        classification = "DEVELOPMENT PROMISING" if period == "dev" else "IMPROVEMENT VALIDATED"
        reason = "All success criteria met." + (" OOS confirmed the development result." if period == "oos" else " OOS test required before this can be trusted.")
    else:
        classification = "FAILED"
        reason = "One or more required success criteria were not met."

    lines.append("")
    lines.append(f"*CLASSIFICATION: {classification}*")
    lines.append(f"_{reason}_")
    lines.append(
        "\n_Interpretation rule: a positive result means EUR/USD state has predictive information CONDITIONAL ON the "
        "existing XAU signal -- not that EUR/USD causes better XAU trades. This is a refinement test; even a validated "
        "improvement leaves a substantial gap to the 0.5-1%/day objective, since it modifies 0.403 R/day, not adds to it._"
    )
    if period == "dev":
        lines.append("\n_Per protocol: OOS is only run if this development result is DEVELOPMENT PROMISING. No parameter changes regardless of outcome._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent EUR/USD proxy test report (%s), classification=%s", period, classification)


if __name__ == "__main__":
    asyncio.run(run_test(period="dev"))
