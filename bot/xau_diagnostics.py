"""Diagnostic studies on the validated XAU/USD 4H-ATR trend strategy:
does existing information (session timing, confidence score) already
contain predictive value we're not using?

These are DIAGNOSTIC studies, not parameter optimization. The live bot is
not touched by anything here. Session boundaries and confidence bands are
FROZEN below, defined before any results were examined -- neither is
adjusted based on what the data shows.

Development period: 2025-01-01 to 2025-08-31 (matches the split already
established for the BTC displacement study). Out-of-sample: 2025-09-01 to
2025-12-31 -- NOT analyzed in this module's default run; only pulled in
as an explicit follow-up if a development finding is classified
DEVELOPMENT PROMISING.

Session boundaries (UTC, non-overlapping, standard market-session
convention -- documented before any results were seen):
  Asian:              00:00-08:00
  London:             08:00-13:00
  London/NY Overlap:  13:00-17:00
  New York:           17:00-22:00
  Late/Sydney:        22:00-24:00

Confidence bands (frozen, matching the pre-specified request):
  55-59, 60-64, 65-69, 70-74, 75+

Decision taxonomy for every finding:
  NO INFORMATION / INTERESTING BUT INSUFFICIENT / DEVELOPMENT PROMISING /
  OOS CONFIRMED / FORWARD VALIDATION REQUIRED
Nothing is called "validated" from historical data alone.
"""

import asyncio
import logging
from datetime import datetime

from . import db
from . import telegram_bot
from .analytics import get_default_spread_pct, percentile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xau_diagnostics")

XAU_TAG_PREFIX = "XAU/USD [4h-ATR]"
STRATEGY_TYPE = "trend"

DEV_START, DEV_END = "2025-01-01", "2025-09-01"  # end is exclusive
OOS_START, OOS_END = "2025-09-01", "2026-01-01"

MIN_SAMPLE_SIZE = 30  # consistent with the floor used throughout this project

# Frozen, non-overlapping, UTC session boundaries (hour, inclusive-start/exclusive-end)
SESSION_BOUNDARIES = [
    ("Asian", 0, 8),
    ("London", 8, 13),
    ("London/NY Overlap", 13, 17),
    ("New York", 17, 22),
    ("Late/Sydney", 22, 24),
]

# Frozen confidence bands
CONFIDENCE_BANDS = [(55, 59), (60, 64), (65, 69), (70, 74), (75, 1000)]


def classify_session(evaluated_at: str) -> str:
    hour = datetime.fromisoformat(evaluated_at).hour
    for name, start_h, end_h in SESSION_BOUNDARIES:
        if start_h <= hour < end_h:
            return name
    return "Unknown"


def classify_confidence_band(confidence: float) -> str:
    for lo, hi in CONFIDENCE_BANDS:
        if lo <= confidence <= hi:
            return f"{lo}-{hi}" if hi < 1000 else f"{lo}+"
    return "below 55"


def get_xau_trend_trades(start_date: str, end_date: str) -> list[dict]:
    """Pulls XAU/USD 4H-ATR trend trades from already-collected backtest
    data, restricted to [start_date, end_date) -- filters the existing
    full-year dataset down to the dev or OOS sub-period."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT e.evaluated_at, e.confidence, e.entry, e.sl, o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ? AND e.symbol NOT LIKE '%(R:R%'
                 AND o.outcome IN ('WIN', 'LOSS')
                 AND e.evaluated_at >= ? AND e.evaluated_at < ?
               ORDER BY e.evaluated_at ASC""",
            (STRATEGY_TYPE, f"{XAU_TAG_PREFIX}%", start_date, end_date),
        ).fetchall()
    finally:
        conn.close()
    cols = ["evaluated_at", "confidence", "entry", "sl", "outcome", "r_multiple"]
    return [dict(zip(cols, row)) for row in rows]


def compute_bucket_stats(trades: list[dict], days_span: float, spread_pct: float) -> dict:
    """Full stats for one bucket (or the unfiltered whole set): n, win
    rate, gross/net R per trade, net R/day, PF, max drawdown."""
    n = len(trades)
    if n == 0:
        return {"n": 0}

    gross_r = [t["r_multiple"] for t in trades]
    net_r = []
    for t in trades:
        risk_price = abs(t["entry"] - t["sl"]) if t["entry"] is not None and t["sl"] is not None else None
        if risk_price:
            cost_r = (t["entry"] * spread_pct / 100) / risk_price
            net_r.append(t["r_multiple"] - cost_r)
        else:
            net_r.append(t["r_multiple"])

    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    win_rate = wins / n
    gross_avg = sum(gross_r) / n
    net_avg = sum(net_r) / n
    gains = sum(r for r in net_r if r > 0)
    losses = abs(sum(r for r in net_r if r < 0))
    pf = gains / losses if losses > 0 else None

    cum, peak, max_dd = 0.0, 0.0, 0.0
    for r in net_r:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    trades_per_day = n / days_span if days_span > 0 else 0.0
    net_r_day = trades_per_day * net_avg

    return {
        "n": n, "win_rate": win_rate, "gross_avg_r": gross_avg, "net_avg_r": net_avg,
        "profit_factor": pf, "max_drawdown_r": max_dd, "trades_per_day": trades_per_day,
        "net_r_day": net_r_day, "insufficient_sample": n < MIN_SAMPLE_SIZE,
    }


def compare_to_baseline(candidate: dict, baseline: dict) -> str:
    """Classifies a candidate bucket/filter against the unfiltered
    baseline on the 6 required dimensions. A filter only earns
    DEVELOPMENT PROMISING if it doesn't just have a higher win rate on a
    small remaining sample -- it must genuinely beat the baseline's net
    R/day without materially worsening PF or drawdown."""
    if candidate.get("n", 0) == 0:
        return "NO INFORMATION"
    if candidate["insufficient_sample"]:
        return "INTERESTING BUT INSUFFICIENT" if candidate["net_r_day"] > baseline["net_r_day"] else "NO INFORMATION"

    beats_r_day = candidate["net_r_day"] > baseline["net_r_day"]
    pf_ok = candidate["profit_factor"] is not None and baseline["profit_factor"] is not None and candidate["profit_factor"] >= baseline["profit_factor"] * 0.9
    dd_ok = candidate["max_drawdown_r"] <= baseline["max_drawdown_r"] * 1.25

    if beats_r_day and pf_ok and dd_ok:
        return "DEVELOPMENT PROMISING"
    elif beats_r_day:
        return "INTERESTING BUT INSUFFICIENT"
    else:
        return "NO INFORMATION"


async def run_session_study(period: str = "dev", spread_pct: float | None = None):
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct("XAU/USD")
    trades = get_xau_trend_trades(start_date, end_date)

    if not trades:
        await telegram_bot.send_text(f"*🕐 XAU Session Study ({period.upper()})*\n\nNo trades found for {start_date} to {end_date}.")
        return

    days_span = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).total_seconds() / 86400
    baseline = compute_bucket_stats(trades, days_span, spread_pct)

    buckets: dict[str, list[dict]] = {name: [] for name, _, _ in SESSION_BOUNDARIES}
    for t in trades:
        buckets[classify_session(t["evaluated_at"])].append(t)

    lines = [
        f"*🕐 XAU Session Study ({period.upper()}): {start_date} to {end_date}*",
        "_Diagnostic only -- live bot unchanged. Session boundaries frozen before any results were examined._\n",
    ]
    lines.append("*Unfiltered baseline*")
    lines.append(f"  n={baseline['n']}, {baseline['trades_per_day']:.2f} trades/day, WR {baseline['win_rate']*100:.0f}%")
    lines.append(f"  Gross R/trade {baseline['gross_avg_r']:+.3f}, Net R/trade {baseline['net_avg_r']:+.3f}, Net R/day {baseline['net_r_day']:+.3f}")
    pf_str = f"{baseline['profit_factor']:.2f}" if baseline["profit_factor"] is not None else "N/A"
    lines.append(f"  PF {pf_str}, max DD {baseline['max_drawdown_r']:.2f}R")
    lines.append("")

    lines.append("*By session*")
    for name, _, _ in SESSION_BOUNDARIES:
        bucket_trades = buckets[name]
        stats = compute_bucket_stats(bucket_trades, days_span, spread_pct)
        pct_retained = 100 * stats.get("n", 0) / baseline["n"]
        if stats.get("n", 0) == 0:
            lines.append(f"  {name}: no trades")
            continue
        classification = compare_to_baseline(stats, baseline)
        insufficient_tag = " [INSUFFICIENT SAMPLE]" if stats["insufficient_sample"] else ""
        pf_s = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "N/A"
        lines.append(
            f"  *{name}* (n={stats['n']}, {pct_retained:.0f}% of trades){insufficient_tag}: "
            f"WR {stats['win_rate']*100:.0f}%, net R/trade {stats['net_avg_r']:+.3f}, "
            f"net R/day {stats['net_r_day']:+.3f}, PF {pf_s}, max DD {stats['max_drawdown_r']:.2f}R"
        )
        lines.append(f"    -> {classification}")

    lines.append("")
    lines.append(
        "_A session is only genuinely interesting if it beats the unfiltered baseline's net R/day without "
        "materially worse PF or drawdown -- a high win rate on a shrunken sample alone is not sufficient._"
    )
    if period == "dev":
        lines.append("\n_Per protocol: OOS is only checked if a finding here is classified DEVELOPMENT PROMISING._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent XAU session study (%s)", period)


async def run_confidence_study(period: str = "dev", spread_pct: float | None = None):
    start_date, end_date = (DEV_START, DEV_END) if period == "dev" else (OOS_START, OOS_END)
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct("XAU/USD")
    trades = get_xau_trend_trades(start_date, end_date)

    if not trades:
        await telegram_bot.send_text(f"*🎯 XAU Confidence Study ({period.upper()})*\n\nNo trades found for {start_date} to {end_date}.")
        return

    days_span = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).total_seconds() / 86400
    baseline = compute_bucket_stats(trades, days_span, spread_pct)

    buckets: dict[str, list[dict]] = {}
    band_order = []
    for lo, hi in CONFIDENCE_BANDS:
        label = f"{lo}-{hi}" if hi < 1000 else f"{lo}+"
        buckets[label] = []
        band_order.append(label)
    for t in trades:
        buckets[classify_confidence_band(t["confidence"])].append(t)

    lines = [
        f"*🎯 XAU Confidence Study ({period.upper()}): {start_date} to {end_date}*",
        "_Diagnostic only -- live bot unchanged. Confidence bands frozen before any results were examined._\n",
    ]
    lines.append("*Unfiltered baseline*")
    lines.append(f"  n={baseline['n']}, Net R/trade {baseline['net_avg_r']:+.3f}, Net R/day {baseline['net_r_day']:+.3f}")
    lines.append("")

    lines.append("*By confidence band*")
    band_avgs = []
    for label in band_order:
        bucket_trades = buckets[label]
        stats = compute_bucket_stats(bucket_trades, days_span, spread_pct)
        if stats.get("n", 0) == 0:
            lines.append(f"  {label}: no trades")
            band_avgs.append(None)
            continue
        classification = compare_to_baseline(stats, baseline)
        insufficient_tag = " [INSUFFICIENT SAMPLE]" if stats["insufficient_sample"] else ""
        pf_s = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "N/A"
        total_net_r = stats["net_avg_r"] * stats["n"]
        lines.append(
            f"  *{label}* (n={stats['n']}){insufficient_tag}: WR {stats['win_rate']*100:.0f}%, "
            f"gross R {stats['gross_avg_r']:+.3f}, net R {stats['net_avg_r']:+.3f}, PF {pf_s}, "
            f"total net R {total_net_r:+.2f}, net R/day contribution {stats['net_r_day']:+.3f}"
        )
        lines.append(f"    -> {classification}")
        band_avgs.append(stats["net_avg_r"])

    lines.append("")
    valid_avgs = [(label, avg) for label, avg in zip(band_order, band_avgs) if avg is not None]
    is_monotonic = all(valid_avgs[i][1] <= valid_avgs[i + 1][1] for i in range(len(valid_avgs) - 1)) if len(valid_avgs) >= 2 else None
    if is_monotonic is not None:
        lines.append(f"*Monotonicity check: higher confidence -> better outcome?* {'YES, monotonic' if is_monotonic else 'NO -- not monotonic, see bands above'}")
        if not is_monotonic:
            lines.append("_Do not assume higher confidence is automatically better -- check which specific bands break the pattern above._")

    lines.append("")
    lines.append("_A band is only genuinely interesting if it beats the unfiltered baseline's net R/day without materially worse PF or drawdown._")
    if period == "dev":
        lines.append("\n_Per protocol: OOS is only checked if a finding here is classified DEVELOPMENT PROMISING._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent XAU confidence study (%s)", period)


async def run_portfolio_projection(candidate_net_r_day: float | None = None, candidate_label: str = "Filtered"):
    """Converts baseline vs a candidate filter's net R/day into expected
    account return at the 4 standard risk levels, plus the % improvement
    -- purely a factual projection, not a recommendation to increase
    risk. If candidate_net_r_day is None, only the baseline is shown."""
    trades = get_xau_trend_trades(DEV_START, DEV_END)
    days_span = (datetime.fromisoformat(DEV_END) - datetime.fromisoformat(DEV_START)).total_seconds() / 86400
    baseline = compute_bucket_stats(trades, days_span, get_default_spread_pct("XAU/USD"))

    lines = ["*📊 Portfolio Projection: Baseline vs Candidate Filter*\n"]
    lines.append(f"Baseline net R/day: {baseline['net_r_day']:+.3f}")
    if candidate_net_r_day is not None:
        improvement_pct = 100 * (candidate_net_r_day - baseline["net_r_day"]) / abs(baseline["net_r_day"]) if baseline["net_r_day"] != 0 else None
        lines.append(f"{candidate_label} net R/day: {candidate_net_r_day:+.3f}")
        if improvement_pct is not None:
            lines.append(f"Improvement: {improvement_pct:+.1f}%")
    lines.append("")
    lines.append("Risk/trade | Baseline %/day | Candidate %/day")
    for risk_pct in [0.25, 0.50, 0.75, 1.00]:
        baseline_pct = baseline["net_r_day"] * risk_pct
        if candidate_net_r_day is not None:
            candidate_pct = candidate_net_r_day * risk_pct
            lines.append(f"  {risk_pct:.2f}%: {baseline_pct:.3f}% | {candidate_pct:.3f}%")
        else:
            lines.append(f"  {risk_pct:.2f}%: {baseline_pct:.3f}%")
    lines.append(
        "\n_This is a factual projection, not a recommendation. Risk sizing should never be increased "
        "merely to make a historical result approach the 1%/day objective._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent portfolio projection")


if __name__ == "__main__":
    asyncio.run(run_session_study(period="dev"))
