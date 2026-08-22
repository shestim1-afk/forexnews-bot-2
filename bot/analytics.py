"""Exploratory analytics on already-collected backtest data. Every result
this module produces is explicitly labeled EXPLORATORY -- hypothesis
material for future validation on a larger sample, NOT evidence to change
the live bot's actual behavior. The live bot's 55% confidence threshold
stays fixed regardless of what these reports show.

Statistical guardrail, stated plainly: with roughly 170-270 actionable
trades per symbol from a single 17-day window, splitting further by
threshold shrinks each bucket further. A threshold that looks better here
could easily just be a smaller, noisier sample looking better by chance --
that's exactly the trap this analysis exists to avoid falling into, not
evidence to act on.
"""

import asyncio
import logging

from . import db
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analytics")

CONFIDENCE_THRESHOLDS = [55, 60, 65, 70]
EXCURSION_THRESHOLDS_R = [1.0, 1.5, 2.0]


def compute_metrics(rows: list[tuple]) -> dict:
    """rows: (outcome, r_multiple) tuples, in chronological order.
    'avg_r' here IS the per-trade expectancy in R-multiples -- the standard
    win_rate*avg_win - loss_rate*avg_loss formula collapses to exactly this
    when computed directly from the realized R sequence."""
    wins = sum(1 for o, r in rows if o == "WIN")
    losses = sum(1 for o, r in rows if o == "LOSS")
    resolved = wins + losses
    r_seq = [r for o, r in rows if o in ("WIN", "LOSS")]

    win_rate = wins / resolved if resolved else None
    avg_r = sum(r_seq) / resolved if resolved else None
    gains = sum(r for r in r_seq if r > 0)
    loss_sum = abs(sum(r for r in r_seq if r < 0))
    profit_factor = gains / loss_sum if loss_sum > 0 else None

    cum, peak, max_dd = 0.0, 0.0, 0.0
    for r in r_seq:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    max_streak, streak = 0, 0
    for o, r in rows:
        if o == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        elif o == "WIN":
            streak = 0

    return {
        "trades": resolved, "wins": wins, "losses": losses,
        "win_rate": win_rate, "avg_r": avg_r, "profit_factor": profit_factor,
        "max_drawdown_r": max_dd, "max_consecutive_losses": max_streak,
    }


def get_backtest_rows(symbol: str, min_confidence: float = 0) -> list[tuple]:
    conn = db._connect()
    try:
        return conn.execute(
            """SELECT o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.symbol = ? AND e.confidence >= ?
               ORDER BY o.evaluated_at ASC""",
            (symbol, min_confidence),
        ).fetchall()
    finally:
        conn.close()


def get_actionable_count(symbol: str) -> int:
    conn = db._connect()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM all_evaluations WHERE source='backtest' AND symbol=? AND action != 'NO TRADE'",
            (symbol,),
        ).fetchone()[0]
    finally:
        conn.close()


async def run_confidence_threshold_analysis(symbols: list[str] | None = None):
    symbols = symbols or ["BTC/USD", "XAU/USD"]
    lines = [
        "*🔬 Confidence Threshold Analysis*",
        "_EXPLORATORY -- insufficient sample for production optimization. "
        "A hypothesis for future validation, not a strategy change. "
        "The live bot's 55% threshold is unchanged._\n",
    ]

    for symbol in symbols:
        total_actionable = get_actionable_count(symbol)
        if total_actionable == 0:
            lines.append(f"*{symbol}*: no backtest data found -- run the Historical Backtest workflow for this symbol first.\n")
            continue

        lines.append(f"*{symbol}*")
        for thresh in CONFIDENCE_THRESHOLDS:
            rows = get_backtest_rows(symbol, thresh)
            m = compute_metrics(rows)
            if m["trades"] == 0:
                lines.append(f"  ≥{thresh}%: 0 trades")
                continue
            filtered_pct = 100 * (1 - m["trades"] / total_actionable)
            pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "N/A"
            wr = f"{m['win_rate']*100:.0f}%" if m["win_rate"] is not None else "N/A"
            lines.append(
                f"  ≥{thresh}%: {m['trades']} trades ({filtered_pct:.0f}% filtered out), "
                f"WR {wr}, expectancy {m['avg_r']:+.2f}R, PF {pf}, "
                f"max DD {m['max_drawdown_r']:.2f}R, longest losing streak {m['max_consecutive_losses']}"
            )
        lines.append("")

    lines.append(
        "_Note: range/liquidity-sweep/breakout-retest signals all carry a fixed 60% confidence tag "
        "(not a computed score like trend signals), so thresholds above 60% filter them out as a block, "
        "not gradually -- this test mainly varies the 'trend' signal count._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent confidence threshold analysis")


def percentile(sorted_values: list[float], pct: float) -> float | None:
    """Simple linear-interpolation percentile, no numpy dependency needed.
    pct in [0, 100]. Returns None for an empty list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def get_excursion_rows(symbol: str | None = None, strategy_type: str | None = None) -> list[dict]:
    """Returns per-trade excursion data for the requested slice (by symbol
    and/or strategy_type -- pass None to not filter on that dimension)."""
    conditions, params = ["e.source = 'backtest'"], []
    if symbol:
        conditions.append("e.symbol = ?")
        params.append(symbol)
    if strategy_type:
        conditions.append("e.strategy_type = ?")
        params.append(strategy_type)

    conn = db._connect()
    try:
        rows = conn.execute(
            f"""SELECT o.mae_before_tp1_r, o.mfe_before_tp1_r, o.tp1_hit, o.tp2_hit, o.mfe_after_tp1_r
                FROM backtest_outcomes o
                JOIN all_evaluations e ON e.id = o.evaluation_id
                WHERE {' AND '.join(conditions)}""",
            params,
        ).fetchall()
    finally:
        conn.close()

    cols = ["mae_before_tp1_r", "mfe_before_tp1_r", "tp1_hit", "tp2_hit", "mfe_after_tp1_r"]
    return [dict(zip(cols, row)) for row in rows]


def compute_excursion_stats(rows: list[dict]) -> dict | None:
    """Median/25th/75th percentile MAE & MFE, plus % of trades reaching
    TP1, TP2, and 1R/1.5R/2R (using the trade's best overall favorable
    excursion -- mfe_after_tp1_r when available and larger, since that
    reflects the true peak measured from entry, otherwise mfe_before_tp1_r)."""
    if not rows:
        return None

    n = len(rows)
    mae_vals = sorted(r["mae_before_tp1_r"] for r in rows if r["mae_before_tp1_r"] is not None)
    mfe_vals = sorted(r["mfe_before_tp1_r"] for r in rows if r["mfe_before_tp1_r"] is not None)

    overall_mfe = []
    for r in rows:
        if r["tp1_hit"] and r["mfe_after_tp1_r"] is not None:
            overall_mfe.append(max(r["mfe_before_tp1_r"] or 0.0, r["mfe_after_tp1_r"]))
        else:
            overall_mfe.append(r["mfe_before_tp1_r"] or 0.0)

    tp1_hit_count = sum(1 for r in rows if r["tp1_hit"])
    tp2_hit_count = sum(1 for r in rows if r["tp2_hit"])

    pct_reaching = {}
    for thresh in EXCURSION_THRESHOLDS_R:
        pct_reaching[thresh] = 100 * sum(1 for m in overall_mfe if m >= thresh) / n

    return {
        "n": n,
        "mae_median": percentile(mae_vals, 50), "mae_p25": percentile(mae_vals, 25), "mae_p75": percentile(mae_vals, 75),
        "mfe_median": percentile(mfe_vals, 50), "mfe_p25": percentile(mfe_vals, 25), "mfe_p75": percentile(mfe_vals, 75),
        "pct_reaching_tp1": 100 * tp1_hit_count / n,
        "pct_reaching_tp2": 100 * tp2_hit_count / n,
        "pct_reaching_r": pct_reaching,
    }


async def run_excursion_report(symbols: list[str] | None = None, strategy_types: list[str] | None = None):
    """Aggregate MAE/MFE report -- median/percentiles and % reaching various
    milestones, broken down by symbol and strategy type. EXPLORATORY, same
    caveats as the confidence threshold analysis: don't draw conclusions
    from small slices, this is data collection, not optimization."""
    symbols = symbols or ["BTC/USD", "XAU/USD"]
    strategy_types = strategy_types or ["trend"]  # currently the only type logged by the historical backtester

    lines = [
        "*📐 MAE/MFE Excursion Report*",
        "_EXPLORATORY -- data collection, not optimization. Do not conclude a TP change is warranted from this alone._\n",
    ]

    for symbol in symbols:
        lines.append(f"*{symbol}*")
        for strategy_type in strategy_types:
            rows = get_excursion_rows(symbol, strategy_type)
            stats = compute_excursion_stats(rows)
            if stats is None:
                lines.append(f"  {strategy_type}: no data")
                continue
            r_pct_str = ", ".join(f"{t}R: {stats['pct_reaching_r'][t]:.0f}%" for t in EXCURSION_THRESHOLDS_R)
            lines.append(
                f"  {strategy_type} (n={stats['n']}): "
                f"MAE median {stats['mae_median']:.2f}R (p25 {stats['mae_p25']:.2f}, p75 {stats['mae_p75']:.2f}), "
                f"MFE median {stats['mfe_median']:.2f}R (p25 {stats['mfe_p25']:.2f}, p75 {stats['mfe_p75']:.2f})"
            )
            lines.append(
                f"    Reached TP1: {stats['pct_reaching_tp1']:.0f}% | TP2: {stats['pct_reaching_tp2']:.0f}% | {r_pct_str}"
            )
        lines.append("")

    lines.append("_MAE/MFE measured only from actual post-entry price action, before the trade's real exit. Overall MFE combines before/after-TP1 peaks where applicable._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent excursion report")


if __name__ == "__main__":
    asyncio.run(run_confidence_threshold_analysis())
