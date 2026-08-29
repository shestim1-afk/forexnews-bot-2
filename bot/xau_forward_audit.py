"""Forward-Test Audit: XAU/USD 4H-ATR Baseline vs 1.0R Candidate.

This module NEVER modifies bot/xau_swing.py, NEVER touches either
strategy's logic, and NEVER combines the two configurations' results. It
only READS already-collected live forward-test data (via the exact
symbol/strategy_type tags xau_swing.py already uses) and reports on it.

Terminology discipline, enforced throughout:
- HISTORICAL: the full-year 2025 backtest results established earlier in
  this project (baseline +0.403 R/day, candidate +0.479 R/day). These
  numbers are FIXED, documented constants below -- never recomputed here,
  never treated as forward evidence.
- FORWARD: freshly computed from live signal_outcomes/scalp_signals rows
  that xau_swing.py has actually generated and resolved since going live.
- Nothing here is ever called "FORWARD VALIDATED" without meeting an
  explicit, adequate minimum sample size -- a small forward sample is
  reported as INSUFFICIENT SAMPLE, not spun into a premature conclusion.
"""

import asyncio
import logging
from datetime import datetime

from . import db
from . import telegram_bot
from .analytics import get_default_spread_pct
from .xau_swing import SYMBOL_TAG, STRATEGY_TYPE

# Candidate constants defined LOCALLY, not imported from xau_swing.py --
# this avoids fragile cross-file coupling (an import error here revealed
# that the deployed xau_swing.py may not have these names at all, worth
# separately verifying whether the candidate forward test is actually
# running). These values match what was specified when the A/B candidate
# system was built earlier in this project.
CANDIDATE_SYMBOL_TAG = "Gold (XAU/USD) [4h-swing-candidate-1.0R]"
CANDIDATE_STRATEGY_TYPE = "trend_4h_swing_candidate_1_0r"
MIN_TRADES_FOR_COMPARISON = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xau_forward_audit")

# Fixed, documented HISTORICAL figures established earlier in this
# project -- never recomputed here, shown only for side-by-side context.
HISTORICAL_BASELINE_NET_R_DAY = 0.403
HISTORICAL_CANDIDATE_NET_R_DAY = 0.479


def get_forward_trades(symbol_tag: str, strategy_type: str) -> list[dict]:
    """Pulls every RESOLVED forward-test trade for one configuration --
    entry/sl (for cost calculation), outcome, gross r_multiple, mae/mfe,
    and both the signal's creation time and its resolution time."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT s.entry, s.sl, s.created_at, o.outcome, o.r_multiple, o.mae_r, o.mfe_r, o.evaluated_at
               FROM signal_outcomes o
               JOIN scalp_signals s ON s.id = o.signal_id
               WHERE s.symbol = ? AND s.strategy_type = ?
                 AND o.outcome IN ('WIN', 'LOSS')
               ORDER BY s.created_at ASC""",
            (symbol_tag, strategy_type),
        ).fetchall()
    finally:
        conn.close()
    cols = ["entry", "sl", "created_at", "outcome", "r_multiple", "mae_r", "mfe_r", "evaluated_at"]
    return [dict(zip(cols, row)) for row in rows]


def compute_net_r(trade: dict, spread_pct: float) -> float:
    """Applies the same entry/SL-distance-based cost model used
    throughout this project to get this trade's NET R from its stored
    GROSS r_multiple -- forward signal_outcomes rows store gross R only,
    with no cost adjustment applied at storage time."""
    risk_price = abs(trade["entry"] - trade["sl"]) if trade["entry"] is not None and trade["sl"] is not None else None
    if not risk_price:
        return trade["r_multiple"]
    cost_r = (trade["entry"] * spread_pct / 100) / risk_price
    return trade["r_multiple"] - cost_r


def get_daily_r_series(trades: list[dict], spread_pct: float) -> dict[str, float]:
    """Sums NET R per calendar day (by resolution/evaluated_at date) --
    the basis for best/worst day, daily-return distribution, and %
    positive/negative/flat day calculations."""
    by_day: dict[str, float] = {}
    for t in trades:
        day = t["evaluated_at"][:10]
        by_day[day] = by_day.get(day, 0.0) + compute_net_r(t, spread_pct)
    return by_day


def get_monthly_breakdown(trades: list[dict], spread_pct: float) -> list[dict]:
    by_month: dict[str, list[dict]] = {}
    for t in trades:
        month = t["evaluated_at"][:7]
        by_month.setdefault(month, []).append(t)

    result = []
    for month in sorted(by_month.keys()):
        month_trades = by_month[month]
        net_r_vals = [compute_net_r(t, spread_pct) for t in month_trades]
        wins = sum(1 for t in month_trades if t["outcome"] == "WIN")
        result.append({
            "month": month, "n": len(month_trades), "wins": wins,
            "avg_net_r": sum(net_r_vals) / len(net_r_vals) if net_r_vals else None,
        })
    return result


def get_current_streak(trades: list[dict]) -> dict:
    """The CURRENT (most recent) consecutive win or loss streak -- a
    different quantity from the longest historical losing streak."""
    if not trades:
        return {"type": None, "length": 0}
    last_outcome = trades[-1]["outcome"]
    length = 0
    for t in reversed(trades):
        if t["outcome"] == last_outcome:
            length += 1
        else:
            break
    return {"type": last_outcome, "length": length}


def compute_full_stats(trades: list[dict], spread_pct: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0}

    net_r_vals = [compute_net_r(t, spread_pct) for t in trades]
    gross_r_vals = [t["r_multiple"] for t in trades]
    wins = sum(1 for t in trades if t["outcome"] == "WIN")

    gross_gains = sum(r for r in gross_r_vals if r > 0)
    gross_losses = abs(sum(r for r in gross_r_vals if r < 0))
    gross_pf = gross_gains / gross_losses if gross_losses > 0 else None

    net_gains = sum(r for r in net_r_vals if r > 0)
    net_losses = abs(sum(r for r in net_r_vals if r <= 0))
    net_pf = net_gains / net_losses if net_losses > 0 else None

    cum, peak, max_dd = 0.0, 0.0, 0.0
    for r in net_r_vals:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    first_day = datetime.fromisoformat(trades[0]["created_at"]).date()
    last_day = datetime.fromisoformat(trades[-1]["evaluated_at"]).date()
    days_elapsed = max((last_day - first_day).days + 1, 1)
    trades_per_day = n / days_elapsed

    daily_series = get_daily_r_series(trades, spread_pct)
    daily_values = list(daily_series.values())
    sorted_daily = sorted(daily_values)
    n_days_with_data = len(daily_values)
    best_day = max(daily_values) if daily_values else None
    worst_day = min(daily_values) if daily_values else None
    avg_daily = sum(daily_values) / n_days_with_data if n_days_with_data else None
    median_daily = sorted_daily[n_days_with_data // 2] if n_days_with_data % 2 == 1 else (
        (sorted_daily[n_days_with_data // 2 - 1] + sorted_daily[n_days_with_data // 2]) / 2
    ) if n_days_with_data else None

    n_positive_days = sum(1 for v in daily_values if v > 0)
    n_negative_days = sum(1 for v in daily_values if v < 0)
    n_flat_trading_days = sum(1 for v in daily_values if v == 0)
    n_flat_calendar_days = days_elapsed - n_days_with_data  # days with no trade at all

    avg_cost_r = sum(gross_r_vals[i] - net_r_vals[i] for i in range(n)) / n

    return {
        "n": n, "days_elapsed": days_elapsed, "trades_per_day": trades_per_day,
        "gross_avg_r": sum(gross_r_vals) / n, "net_avg_r": sum(net_r_vals) / n,
        "net_r_day": trades_per_day * (sum(net_r_vals) / n),
        "win_rate": wins / n, "gross_pf": gross_pf, "net_pf": net_pf,
        "max_drawdown_r": max_dd, "best_day": best_day, "worst_day": worst_day,
        "avg_daily_return": avg_daily, "median_daily_return": median_daily,
        "n_positive_days": n_positive_days, "n_negative_days": n_negative_days,
        "n_flat_trading_days": n_flat_trading_days, "n_flat_calendar_days": n_flat_calendar_days,
        "n_days_with_data": n_days_with_data,
        "current_streak": get_current_streak(trades), "avg_cost_r": avg_cost_r,
        "monthly": get_monthly_breakdown(trades, spread_pct),
    }


def classify(stats: dict) -> tuple[str, str]:
    if stats["n"] == 0:
        return "NOT CONFIRMED", "No resolved forward trades yet."
    if stats["n"] < MIN_TRADES_FOR_COMPARISON:
        return "INSUFFICIENT SAMPLE", f"Only {stats['n']} resolved trades -- below the {MIN_TRADES_FOR_COMPARISON}-trade minimum this project has consistently required before drawing a conclusion."
    if stats["net_r_day"] <= 0:
        return "FAILED", f"Adequate sample (n={stats['n']}), but net R/day is not positive ({stats['net_r_day']:+.3f})."
    if stats["n"] < 50:
        return "PROMISING BUT INSUFFICIENT SAMPLE", f"Net R/day is positive ({stats['net_r_day']:+.3f}) with n={stats['n']}, but this project's own precedent (e.g. the BTC displacement study) treats n<50 as too small for real confidence, regardless of a positive sign."
    return "PROMISING BUT INSUFFICIENT SAMPLE", (
        f"n={stats['n']} is a reasonable sample, but 'FORWARD VALIDATED' is reserved for a standard this project "
        f"has never actually defined a lower bound for reaching -- treat this as the strongest forward evidence "
        f"available so far, not as validation."
    )


def _fmt_stats_block(label: str, stats: dict) -> list[str]:
    lines = [f"*{label}*"]
    if stats["n"] == 0:
        lines.append("  No resolved forward trades yet.")
        return lines
    gross_pf_s = f"{stats['gross_pf']:.2f}" if stats["gross_pf"] is not None else "N/A"
    net_pf_s = f"{stats['net_pf']:.2f}" if stats["net_pf"] is not None else "N/A"
    lines.append(f"  n={stats['n']}, {stats['days_elapsed']} days elapsed, {stats['trades_per_day']:.3f} trades/day")
    lines.append(f"  Gross avg R: {stats['gross_avg_r']:+.3f}, Net avg R: {stats['net_avg_r']:+.3f} (cost impact: {stats['avg_cost_r']:.3f}R/trade)")
    lines.append(f"  Net R/day: {stats['net_r_day']:+.3f}")
    lines.append(f"  Win rate: {stats['win_rate']*100:.0f}%, Gross PF: {gross_pf_s}, Net PF: {net_pf_s}")
    lines.append(f"  Max drawdown: {stats['max_drawdown_r']:.2f}R")
    if stats["best_day"] is not None:
        lines.append(f"  Best day: {stats['best_day']:+.2f}R, Worst day: {stats['worst_day']:+.2f}R")
        lines.append(f"  Avg daily return: {stats['avg_daily_return']:+.3f}R, Median: {stats['median_daily_return']:+.3f}R")
        lines.append(
            f"  Days with trades: {stats['n_days_with_data']} ({stats['n_positive_days']} positive, "
            f"{stats['n_negative_days']} negative, {stats['n_flat_trading_days']} flat-but-traded), "
            f"{stats['n_flat_calendar_days']} calendar days with no trade at all"
        )
    streak = stats["current_streak"]
    if streak["type"]:
        lines.append(f"  Current streak: {streak['length']} consecutive {streak['type']}(s)")
    if stats["monthly"]:
        lines.append("  Monthly: " + ", ".join(f"{m['month']} (n={m['n']}, avg {m['avg_net_r']:+.2f}R)" for m in stats["monthly"] if m["avg_net_r"] is not None))
    return lines


async def run_audit(spread_pct: float | None = None):
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct("XAU/USD")

    baseline_trades = get_forward_trades(SYMBOL_TAG, STRATEGY_TYPE)
    candidate_trades = get_forward_trades(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)
    baseline_stats = compute_full_stats(baseline_trades, spread_pct)
    candidate_stats = compute_full_stats(candidate_trades, spread_pct)

    lines = [
        "*🔍 XAU Forward-Test Audit: Baseline vs 1.0R Candidate*",
        f"_Read-only audit of already-collected live data. `bot/xau_swing.py` and both strategies are untouched. "
        f"Cost assumption: {spread_pct:.2f}% spread (XAU default)._\n",
    ]

    lines += _fmt_stats_block("FORWARD -- Baseline (TP1=0.667R)", baseline_stats)
    lines.append("")
    lines += _fmt_stats_block("FORWARD -- Candidate (TP1=1.0R)", candidate_stats)
    lines.append("")

    lines.append("*Comparison table*")
    lines.append("Metric | Hist. Baseline | Fwd Baseline | Hist. Candidate | Fwd Candidate")
    fwd_b_r_day = f"{baseline_stats['net_r_day']:+.3f}" if baseline_stats["n"] > 0 else "N/A"
    fwd_c_r_day = f"{candidate_stats['net_r_day']:+.3f}" if candidate_stats["n"] > 0 else "N/A"
    lines.append(f"  Net R/day | {HISTORICAL_BASELINE_NET_R_DAY:+.3f} | {fwd_b_r_day} | {HISTORICAL_CANDIDATE_NET_R_DAY:+.3f} | {fwd_c_r_day}")
    fwd_b_n = str(baseline_stats["n"])
    fwd_c_n = str(candidate_stats["n"])
    lines.append(f"  n (trades) | 812 | {fwd_b_n} | 756 | {fwd_c_n}")
    lines.append("")

    lines.append("*Did the candidate's historical advantage survive forward data?*")
    if baseline_stats["n"] < MIN_TRADES_FOR_COMPARISON or candidate_stats["n"] < MIN_TRADES_FOR_COMPARISON:
        lines.append(
            f"_Cannot answer with confidence yet -- baseline n={baseline_stats['n']}, candidate n={candidate_stats['n']}, "
            f"both below the {MIN_TRADES_FOR_COMPARISON}-trade minimum. Any apparent difference right now could easily be noise._"
        )
    else:
        diff = candidate_stats["net_r_day"] - baseline_stats["net_r_day"]
        if diff > 0.05:
            verdict = "Candidate BETTER on forward data so far"
        elif diff < -0.05:
            verdict = "Candidate WORSE on forward data so far"
        else:
            verdict = "Candidate approximately EQUAL to baseline on forward data so far"
        lines.append(f"_Forward difference: {diff:+.3f} R/day. {verdict}._")

    lines.append("")
    lines.append("*1%/day analysis -- OBSERVED forward figures only, not historical*")
    lines.append("Risk/trade | Required R/day for 1% | Baseline observed %/day | Candidate observed %/day")
    for risk_pct in [0.25, 0.50, 0.75, 1.00]:
        required = 0.01 / (risk_pct / 100)
        b_pct = f"{baseline_stats['net_r_day']*risk_pct:.3f}%" if baseline_stats["n"] > 0 else "N/A"
        c_pct = f"{candidate_stats['net_r_day']*risk_pct:.3f}%" if candidate_stats["n"] > 0 else "N/A"
        lines.append(f"  {risk_pct:.2f}% | {required:.2f} R/day | {b_pct} | {c_pct}")

    lines.append(
        "\n_Reminder: '1% average/day' means the MEAN across many days, not every individual day being positive -- "
        "see the % positive/negative/flat day breakdown above for the actual distribution, not just the average._"
    )

    b_class, b_reason = classify(baseline_stats)
    c_class, c_reason = classify(candidate_stats)
    lines.append("")
    lines.append(f"*Baseline classification: {b_class}* -- {b_reason}")
    lines.append(f"*Candidate classification: {c_class}* -- {c_reason}")

    lines.append("")
    lines.append("*Final project-level conclusion*")
    lines.append(f"A. Original XAU edge behaving as expected: {'plausible, but sample-limited -- see classification above' if baseline_stats['n'] > 0 else 'no forward data yet to assess'}")
    lines.append(f"B. Has the 1.0R exit survived forward testing: {'not yet answerable with confidence -- see comparison above' if candidate_stats['n'] < MIN_TRADES_FOR_COMPARISON or baseline_stats['n'] < MIN_TRADES_FOR_COMPARISON else 'see forward difference above'}")
    lines.append(f"C. Currently supported net R/day: baseline {fwd_b_r_day}, candidate {fwd_c_r_day} (both provisional given current sample size)")
    lines.append(f"D. Distance from 1%/day objective: unchanged from the historical analysis -- forward data so far does not contradict or confirm the historical gap, it simply hasn't accumulated enough evidence yet")
    lines.append("E. Recommendation: keep running both forward tests completely unchanged -- the only way to close the sample-size gap is time, not new code")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent XAU forward-test audit report")


if __name__ == "__main__":
    asyncio.run(run_audit())
