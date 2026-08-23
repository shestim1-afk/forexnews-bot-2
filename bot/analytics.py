"""Exploratory analytics on already-collected backtest data. Every result
this module produces is explicitly labeled EXPLORATORY -- hypothesis
material for future validation on a larger sample, NOT evidence to change
the live bot's actual behavior. The live bot's 55% confidence threshold
stays fixed regardless of what these reports show.
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
        "so thresholds above 60% filter them out as a block, not gradually._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent confidence threshold analysis")


def percentile(sorted_values: list[float], pct: float) -> float | None:
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
    symbols = symbols or ["BTC/USD", "XAU/USD"]
    strategy_types = strategy_types or ["trend"]

    lines = [
        "*📐 MAE/MFE Excursion Report*",
        "_EXPLORATORY -- data collection, not optimization._\n",
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
            lines.append(f"    Reached TP1: {stats['pct_reaching_tp1']:.0f}% | TP2: {stats['pct_reaching_tp2']:.0f}% | {r_pct_str}")
        lines.append("")

    lines.append("_MAE/MFE measured only from actual post-entry price action._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent excursion report")


# Round-trip spread cost estimates, as a % of price -- ROUGH STARTING POINTS,
# not guaranteed figures. Real costs vary significantly by broker/exchange;
# override with your own actual spread if you know it, for a meaningful answer.
DEFAULT_SPREAD_PCT = {
    "BTC/USD": 0.05,
    "XAU/USD": 0.03,
    "GBP/JPY": 0.02,
}


def get_raw_stats_for_symbol_prefix(api_symbol: str, strategy_type: str) -> dict | None:
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ? AND e.symbol NOT LIKE '%(R:R%'
                 AND o.outcome IN ('WIN', 'LOSS')""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchall()
    finally:
        conn.close()

    n = len(rows)
    if n == 0:
        return None
    r_seq = [r for _, r in rows]
    avg_r = sum(r_seq) / n
    gains = sum(r for r in r_seq if r > 0)
    loss_sum = abs(sum(r for r in r_seq if r < 0))
    pf = gains / loss_sum if loss_sum > 0 else None
    return {"n": n, "avg_r": avg_r, "profit_factor": pf}


def get_cost_adjusted_stats(api_symbol: str, strategy_type: str, spread_pct: float | None = None) -> dict | None:
    """Recomputes win/loss/PF using each trade's OWN real stored entry/SL
    distance (not an estimate) to convert an assumed spread cost into an
    R-multiple, then subtracts that from every trade's realized R."""
    spread_pct = spread_pct if spread_pct is not None else DEFAULT_SPREAD_PCT.get(api_symbol, 0.05)
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT e.entry, e.sl, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ? AND e.symbol NOT LIKE '%(R:R%'
                 AND o.outcome IN ('WIN', 'LOSS')""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchall()
    finally:
        conn.close()

    adjusted_r_seq, cost_r_seq = [], []
    for entry, sl, r in rows:
        if entry is None or sl is None:
            continue
        risk_price = abs(entry - sl)
        if risk_price == 0:
            continue
        spread_cost_price = entry * (spread_pct / 100)
        cost_r = spread_cost_price / risk_price
        adjusted_r_seq.append(r - cost_r)
        cost_r_seq.append(cost_r)

    n = len(adjusted_r_seq)
    if n == 0:
        return None

    wins = sum(1 for r in adjusted_r_seq if r > 0)
    losses = n - wins
    avg_r = sum(adjusted_r_seq) / n
    gains = sum(r for r in adjusted_r_seq if r > 0)
    loss_sum = abs(sum(r for r in adjusted_r_seq if r <= 0))
    pf = gains / loss_sum if loss_sum > 0 else None
    avg_cost_r = sum(cost_r_seq) / len(cost_r_seq)

    return {
        "n": n, "wins": wins, "losses": losses, "win_rate": wins / n,
        "avg_r": avg_r, "profit_factor": pf, "spread_pct_used": spread_pct, "avg_cost_r": avg_cost_r,
    }


async def run_cost_adjusted_report(api_symbol: str, spread_pct: float | None = None):
    strategy_types = ["trend", "range", "liquidity_sweep", "breakout_retest"]
    used_pct = spread_pct if spread_pct is not None else DEFAULT_SPREAD_PCT.get(api_symbol, 0.05)

    lines = [
        f"*💸 Transaction-Cost-Adjusted Backtest: {api_symbol}*",
        f"_Assuming {used_pct:.2f}% round-trip spread cost (estimate). "
        f"Each trade's own actual entry/SL distance is used to convert this into an R-cost._\n",
    ]

    for strategy_type in strategy_types:
        raw = get_raw_stats_for_symbol_prefix(api_symbol, strategy_type)
        adj = get_cost_adjusted_stats(api_symbol, strategy_type, spread_pct)

        if adj is None or raw is None or raw["n"] == 0:
            lines.append(f"*{strategy_type}*: no data")
            continue

        raw_pf = f"{raw['profit_factor']:.2f}" if raw["profit_factor"] is not None else "N/A"
        adj_pf = f"{adj['profit_factor']:.2f}" if adj["profit_factor"] is not None else "N/A"
        survived = adj["avg_r"] > 0
        verdict = "✅ edge survives" if survived else "❌ edge erased by costs"

        lines.append(f"*{strategy_type}* (n={raw['n']}, avg cost {adj['avg_cost_r']:.3f}R/trade):")
        lines.append(f"  Before costs: avg R {raw['avg_r']:+.3f}, PF {raw_pf}")
        lines.append(f"  After costs:  avg R {adj['avg_r']:+.3f}, PF {adj_pf}  -- {verdict}")
        lines.append("")

    lines.append("_Simplified model (flat cost subtracted from realized R) -- treat as directional, not exact._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent cost-adjusted report for %s", api_symbol)


def calculate_required_risk_for_target(trades_per_day: float, avg_r_after_cost: float, target_daily_profit: float) -> dict:
    """Given a real (cost-adjusted) edge and trade frequency, computes the
    risk-per-trade needed to reach a target daily profit -- and the account
    size that risk implies at 1% risk/trade. This is the honest way to
    answer "what does it take to hit my target", using a MEASURED edge,
    not a hoped-for one. Flags infeasibility if the edge is zero or
    negative, since no amount of position sizing fixes a losing edge --
    only a real, positive edge scales with risk."""
    if avg_r_after_cost <= 0 or trades_per_day <= 0:
        return {
            "feasible": False,
            "reason": "Edge is zero or negative after costs -- no position sizing fixes a losing edge. "
                      "A larger risk per trade would only lose money faster, not slower.",
        }

    expected_daily_r = trades_per_day * avg_r_after_cost
    required_risk_per_trade = target_daily_profit / expected_daily_r
    implied_account_size_at_1pct = required_risk_per_trade / 0.01

    return {
        "feasible": True,
        "trades_per_day": trades_per_day,
        "avg_r_after_cost": avg_r_after_cost,
        "expected_daily_r": expected_daily_r,
        "required_risk_per_trade": required_risk_per_trade,
        "implied_account_size_at_1pct_risk": implied_account_size_at_1pct,
    }


async def run_target_calculator(api_symbol: str, target_daily_profit: float = 30.0,
                                 trades_per_day: float | None = None, spread_pct: float | None = None):
    """Pulls trend's REAL cost-adjusted edge and trade frequency from
    already-collected backtest data and reports what risk-per-trade and
    account size would be needed to reach the target, using measured
    numbers rather than assumptions."""
    if trades_per_day is None:
        conn = db._connect()
        try:
            row = conn.execute(
                """SELECT MIN(evaluated_at), MAX(evaluated_at), COUNT(*)
                   FROM all_evaluations
                   WHERE source='backtest' AND strategy_type='trend' AND action != 'NO TRADE'
                     AND symbol LIKE ? AND symbol NOT LIKE '%(R:R%'""",
                (f"{api_symbol}%",),
            ).fetchone()
        finally:
            conn.close()
        if row is None or row[2] == 0:
            await telegram_bot.send_text(f"*🎯 Target Calculator: {api_symbol}*\n\nNo trend backtest data found for this symbol -- run a Historical Backtest first.")
            return
        import datetime as dt
        start = dt.datetime.fromisoformat(row[0])
        end = dt.datetime.fromisoformat(row[1])
        days = max((end - start).total_seconds() / 86400, 1e-9)
        trades_per_day = row[2] / days

    adj = get_cost_adjusted_stats(api_symbol, "trend", spread_pct)
    if adj is None:
        await telegram_bot.send_text(f"*🎯 Target Calculator: {api_symbol}*\n\nNo resolved trend trades found to compute a cost-adjusted edge.")
        return

    calc = calculate_required_risk_for_target(trades_per_day, adj["avg_r"], target_daily_profit)

    lines = [f"*🎯 Target Calculator: {api_symbol}*", f"_Target: €{target_daily_profit:.0f}/day net, using trend's REAL measured cost-adjusted edge._\n"]
    lines.append(f"Measured: {trades_per_day:.1f} trades/day, avg R after costs {adj['avg_r']:+.3f} (spread assumption {adj['spread_pct_used']:.2f}%)")

    if not calc["feasible"]:
        lines.append("")
        lines.append(f"❌ *Not feasible at any risk size.* {calc['reason']}")
    else:
        lines.append("")
        lines.append(f"Expected daily R: {calc['expected_daily_r']:+.3f}")
        lines.append(f"Required risk per trade: €{calc['required_risk_per_trade']:.2f}")
        lines.append(f"Implied account size (at 1% risk/trade): €{calc['implied_account_size_at_1pct_risk']:.0f}")
        lines.append("")
        lines.append(
            "_This assumes the measured edge holds going forward, which is never guaranteed -- "
            "treat this as 'what it would take IF the edge is real', not a promise it will happen. "
            "Taxes are not modeled here; the target should be set net of taxes if that matters to you._"
        )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent target calculator for %s", api_symbol)


if __name__ == "__main__":
    asyncio.run(run_confidence_threshold_analysis())
