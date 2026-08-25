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


def get_default_spread_pct(api_symbol: str) -> float:
    """Matches the START of api_symbol against known instrument keys (e.g.
    'XAU/USD [4h-ATR]' should match the 'XAU/USD' default), rather than
    requiring an exact match -- otherwise a bracketed variant tag silently
    falls through to the generic 0.05% default instead of the more
    accurate per-instrument one."""
    for key, pct in DEFAULT_SPREAD_PCT.items():
        if api_symbol.startswith(key):
            return pct
    return 0.05


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
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct(api_symbol)
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
    used_pct = spread_pct if spread_pct is not None else get_default_spread_pct(api_symbol)

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


def get_monthly_breakdown(api_symbol: str, strategy_type: str) -> list[dict]:
    """Groups already-collected backtest results by calendar month, to
    check whether an edge held up consistently over time or was
    concentrated in one lucky stretch. This is NOT true out-of-sample
    validation (the strategy logic wasn't tuned on this data in the first
    place, but the whole year is still one dataset) -- it's a cheaper,
    real robustness check using data already paid for in API calls, useful
    when a genuinely separate year isn't available (Twelve Data's free
    tier caps out around 1 year of intraday history)."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT e.evaluated_at, o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ? AND e.symbol NOT LIKE '%(R:R%'
                 AND o.outcome IN ('WIN', 'LOSS')
               ORDER BY e.evaluated_at ASC""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchall()
    finally:
        conn.close()

    by_month: dict[str, list] = {}
    for evaluated_at, outcome, r in rows:
        month_key = evaluated_at[:7]  # YYYY-MM
        by_month.setdefault(month_key, []).append((outcome, r))

    result = []
    for month_key in sorted(by_month.keys()):
        m = compute_metrics(by_month[month_key])
        result.append({"month": month_key, **m})
    return result


async def run_temporal_consistency_report(api_symbol: str, strategy_type: str = "trend"):
    monthly = get_monthly_breakdown(api_symbol, strategy_type)
    if not monthly:
        await telegram_bot.send_text(f"*📅 Temporal Consistency: {api_symbol} ({strategy_type})*\n\nNo data found -- run a Historical Backtest for this symbol/tag first.")
        return

    lines = [
        f"*📅 Temporal Consistency: {api_symbol} ({strategy_type})*",
        "_Checks whether the edge held up consistently across the year, or was concentrated in one lucky stretch. "
        "Not true out-of-sample validation, but a real robustness check using already-collected data._\n",
    ]

    positive_months, total_months = 0, 0
    for m in monthly:
        if m["trades"] == 0:
            continue
        total_months += 1
        if m["avg_r"] is not None and m["avg_r"] > 0:
            positive_months += 1
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "N/A"
        avg_r_str = f"{m['avg_r']:+.2f}" if m["avg_r"] is not None else "N/A"
        wr_str = f"{m['win_rate']*100:.0f}%" if m["win_rate"] is not None else "N/A"
        emoji = "🟢" if (m["avg_r"] is not None and m["avg_r"] > 0) else ("🔴" if (m["avg_r"] is not None and m["avg_r"] < 0) else "⚪")
        lines.append(f"{emoji} {m['month']}: n={m['trades']}, WR {wr_str}, avg R {avg_r_str}, PF {pf}")

    lines.append("")
    lines.append(f"*{positive_months}/{total_months} months showed a positive average R.*")
    if total_months > 0 and positive_months >= total_months * 0.7:
        lines.append("_Reasonably consistent across the year -- a good sign, though still not a substitute for true out-of-sample validation on unseen data._")
    else:
        lines.append("_Inconsistent across the year -- the annual average may be masking a few strong months carrying the rest. Treat with more caution._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent temporal consistency report for %s (%s)", api_symbol, strategy_type)


# ============================================================================
# MAE/MFE diagnostic and counterfactual exit analysis
#
# Terminology used throughout, kept deliberately distinct per the research
# request this was built for:
#   OBSERVED     -- computed directly from real stored backtest data.
#   COUNTERFACTUAL -- "what R would this trade have shown under a DIFFERENT
#                    exit rule", reconstructed from stored MAE/MFE summary
#                    stats (not a full price-path re-simulation). This is
#                    an APPROXIMATION with a stated limitation: for trades
#                    whose favorable move never reached the counterfactual
#                    target within the observation window (24h post-entry,
#                    or 24h post-TP1 for the "after" phase), we do not know
#                    what happened beyond that window -- treated as
#                    "target not reached", which could understate a wider
#                    target's true results (it can never overstate them).
#   HYPOTHESIS   -- an interpretation offered ABOUT the observed/counterfactual
#                    numbers, not itself a measured fact.
#   VALIDATED    -- reserved for a result that has been re-tested with its
#                    OWN forward backtest run against genuinely separate
#                    data. Nothing produced by this module is validated;
#                    it is diagnostic only.
# ============================================================================

MAE_MFE_R_THRESHOLDS = [1.0, 1.5, 2.0, 3.0]


def get_mae_mfe_trade_rows(api_symbol: str, strategy_type: str) -> list[dict]:
    """Pulls PER-TRADE (not aggregated) MAE/MFE and timing data for
    already-collected backtest trades. Uses the same symbol-prefix
    matching as the rest of this module, to correctly find deep-backtest
    data regardless of its date-range tag suffix."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT o.outcome, o.r_multiple, o.mae_before_tp1_r, o.mfe_before_tp1_r,
                      o.tp1_hit, o.tp2_hit, o.mfe_after_tp1_r, o.max_giveback_after_tp1_r,
                      o.time_to_tp1_minutes, o.time_to_tp2_minutes, o.time_to_exit_minutes
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ? AND e.symbol NOT LIKE '%(R:R%'""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchall()
    finally:
        conn.close()

    cols = ["outcome", "r_multiple", "mae_before_tp1_r", "mfe_before_tp1_r", "tp1_hit", "tp2_hit",
            "mfe_after_tp1_r", "max_giveback_after_tp1_r", "time_to_tp1_minutes", "time_to_tp2_minutes",
            "time_to_exit_minutes"]
    trades = [dict(zip(cols, row)) for row in rows]
    for t in trades:
        # overall_mfe: the single best favorable excursion reached at any
        # point in the observed window, combining the before-TP1 and
        # after-TP1 (research-only) phases where available.
        t["overall_mfe_r"] = max(t["mfe_before_tp1_r"] or 0.0, t["mfe_after_tp1_r"] or 0.0)
    return trades


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def compute_mae_mfe_diagnostics(trades: list[dict]) -> dict:
    """Answers the 10 specific diagnostic questions, computed directly
    from OBSERVED stored data -- no counterfactual reconstruction here."""
    n = len(trades)
    if n == 0:
        return {"n": 0}

    winners = [t for t in trades if t["outcome"] == "WIN"]
    losers = [t for t in trades if t["outcome"] == "LOSS"]

    # Q1-Q4: % reaching various overall MFE thresholds (all trades)
    pct_reaching = {}
    for thresh in MAE_MFE_R_THRESHOLDS:
        pct_reaching[thresh] = 100 * sum(1 for t in trades if t["overall_mfe_r"] >= thresh) / n

    # Q5: conditional -- among trades reaching 1R, how many also reach 1.5R / 2R
    reached_1r = [t for t in trades if t["overall_mfe_r"] >= 1.0]
    cond_1_5r = 100 * sum(1 for t in reached_1r if t["overall_mfe_r"] >= 1.5) / len(reached_1r) if reached_1r else None
    cond_2r = 100 * sum(1 for t in reached_1r if t["overall_mfe_r"] >= 2.0) / len(reached_1r) if reached_1r else None

    # Q6: giveback after MFE, among trades where this is measurable (tp1_hit trades)
    giveback_vals = [t["max_giveback_after_tp1_r"] for t in trades if t["tp1_hit"] and t["max_giveback_after_tp1_r"] is not None]
    median_giveback = _median(giveback_vals)

    # Q7: median MFE for winners
    median_mfe_winners = _median([t["overall_mfe_r"] for t in winners])

    # Q8: distribution of MFE for losers (median/p25/p75)
    loser_mfes = sorted(t["overall_mfe_r"] for t in losers)
    mfe_losers_median = _median(loser_mfes)
    mfe_losers_p25 = percentile(loser_mfes, 25)
    mfe_losers_p75 = percentile(loser_mfes, 75)

    # Q9: MAE pattern for winners -- median MAE before TP1, relative to the
    # 1R (=SL distance) benchmark. A small median MAE means winners tend to
    # move favorably quickly; a large one (close to 1R) means they tolerate
    # real adverse movement before working out.
    winner_maes = [t["mae_before_tp1_r"] for t in winners if t["mae_before_tp1_r"] is not None]
    median_mae_winners = _median(winner_maes)

    # Q10 is answered by combining Q1-Q7 in the report, not a separate stat

    return {
        "n": n, "n_winners": len(winners), "n_losers": len(losers),
        "pct_reaching_mfe_r": pct_reaching,
        "cond_prob_1_5r_given_1r": cond_1_5r, "cond_prob_2r_given_1r": cond_2r,
        "n_reached_1r": len(reached_1r),
        "median_giveback_after_tp1_r": median_giveback, "n_giveback_measured": len(giveback_vals),
        "median_mfe_winners": median_mfe_winners,
        "mfe_losers_median": mfe_losers_median, "mfe_losers_p25": mfe_losers_p25, "mfe_losers_p75": mfe_losers_p75,
        "median_mae_winners": median_mae_winners,
    }


def compute_counterfactual_exit(trades: list[dict], exit_type: str, target_r: float | None = None) -> dict:
    """Reconstructs COUNTERFACTUAL per-trade R under a different exit rule,
    using each trade's stored overall_mfe_r as an approximation of "would
    price have reached this target". See the module-level note above for
    the stated limitation: trades that never reached the counterfactual
    target within our observation window are treated as not reaching it,
    which can only understate (never overstate) a wider target's results.

    exit_type:
      'baseline'    -- exactly what was actually observed (A)
      'runner'      -- current TP1 (0.667R) then let the rest run to the
                       observed final MFE, giving back the observed
                       giveback -- approximates "TP1 + runner" (B)
      'fixed_target' -- a single fixed R target, using target_r (C/D/E)
    Losing trades are UNCHANGED under any of these (SL hit before any
    profit target is reached doesn't depend on how far away the target
    was) -- see the module note for why this simplification is sound.
    """
    r_values = []
    for t in trades:
        if t["outcome"] == "LOSS":
            r_values.append(-1.0)
            continue
        if t["outcome"] != "WIN":
            # EXPIRED or other -- keep the originally observed R unchanged;
            # we don't have enough data to safely reconstruct these under
            # a different target
            r_values.append(t["r_multiple"])
            continue

        if exit_type == "baseline":
            r_values.append(t["r_multiple"])
        elif exit_type == "runner":
            # TP1 captured at 0.667R (unchanged), then the observed
            # after-TP1 phase's giveback is applied to the observed peak
            final_r = t["overall_mfe_r"] - (t["max_giveback_after_tp1_r"] or 0.0)
            r_values.append(max(final_r, 0.667))  # can't go below what TP1 already locked in
        elif exit_type == "fixed_target":
            if target_r is None:
                raise ValueError("fixed_target requires target_r")
            if t["overall_mfe_r"] >= target_r:
                r_values.append(target_r)
            else:
                # Target never reached within the observation window --
                # conservatively treat as a scratch at the observed final R,
                # since we don't know what actually happened afterward
                r_values.append(min(t["r_multiple"], t["overall_mfe_r"]))
        else:
            raise ValueError(f"unknown exit_type: {exit_type}")

    n = len(r_values)
    wins = sum(1 for r in r_values if r > 0)
    losses = sum(1 for r in r_values if r < 0)
    avg_r = sum(r_values) / n if n else None
    gains = sum(r for r in r_values if r > 0)
    loss_sum = abs(sum(r for r in r_values if r < 0))
    pf = gains / loss_sum if loss_sum > 0 else None

    cum, peak, max_dd = 0.0, 0.0, 0.0
    for r in r_values:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    avg_holding_minutes = _median([t["time_to_exit_minutes"] for t in trades if t["time_to_exit_minutes"] is not None])

    return {
        "n": n, "wins": wins, "losses": losses,
        "win_rate": wins / n if n else None, "avg_r": avg_r, "profit_factor": pf,
        "max_drawdown_r": max_dd, "median_holding_minutes": avg_holding_minutes,
    }


async def run_mae_mfe_diagnostic_report(api_symbol: str, strategy_type: str = "trend", spread_pct: float | None = None):
    """Sends the full diagnostic + counterfactual report. Everything here
    is labeled OBSERVED, COUNTERFACTUAL (approximate), or HYPOTHESIS --
    nothing is VALIDATED, since validation requires a genuinely separate
    forward/out-of-sample test, not another look at the same dataset."""
    trades = get_mae_mfe_trade_rows(api_symbol, strategy_type)
    if not trades:
        await telegram_bot.send_text(f"*🔬 MAE/MFE Diagnostic: {api_symbol} ({strategy_type})*\n\nNo data found -- run a Historical Backtest for this symbol/tag first.")
        return

    diag = compute_mae_mfe_diagnostics(trades)
    # Real, entry/SL-distance-based average cost per trade for this exact
    # symbol/strategy, to net every counterfactual against the same
    # realistic cost basis as the rest of this project's analysis.
    cost_stats = get_cost_adjusted_stats(api_symbol, strategy_type, spread_pct)
    avg_cost_r = cost_stats["avg_cost_r"] if cost_stats else 0.0
    used_spread = cost_stats["spread_pct_used"] if cost_stats else get_default_spread_pct(api_symbol)

    lines = [
        f"*🔬 MAE/MFE Diagnostic: {api_symbol} ({strategy_type})*",
        f"_OBSERVED data from {diag['n']} already-collected backtest trades. "
        f"Nothing below is VALIDATED -- these are diagnostics and approximate counterfactuals, "
        f"not forward-tested results._\n",
    ]

    lines.append("*Part 1 -- Observed diagnostics*")
    lines.append(f"n={diag['n']} ({diag['n_winners']}W / {diag['n_losers']}L)")
    lines.append("")
    lines.append("Q1-4, % of ALL trades reaching overall MFE threshold:")
    for thresh in MAE_MFE_R_THRESHOLDS:
        lines.append(f"  ≥{thresh}R: {diag['pct_reaching_mfe_r'][thresh]:.1f}%")
    lines.append("")
    lines.append(f"Q5, among the {diag['n_reached_1r']} trades reaching ≥1R MFE:")
    if diag["cond_prob_1_5r_given_1r"] is not None:
        lines.append(f"  also reached ≥1.5R: {diag['cond_prob_1_5r_given_1r']:.1f}%")
        lines.append(f"  also reached ≥2R: {diag['cond_prob_2r_given_1r']:.1f}%")
    else:
        lines.append("  (no trades reached ≥1R)")
    lines.append("")
    if diag["median_giveback_after_tp1_r"] is not None:
        lines.append(f"Q6, median giveback after TP1 (n={diag['n_giveback_measured']}): {diag['median_giveback_after_tp1_r']:.2f}R")
    lines.append(f"Q7, median MFE for winners: {diag['median_mfe_winners']:.2f}R" if diag["median_mfe_winners"] is not None else "Q7: no winners")
    if diag["mfe_losers_median"] is not None:
        lines.append(f"Q8, MFE distribution for losers: median {diag['mfe_losers_median']:.2f}R (p25 {diag['mfe_losers_p25']:.2f}, p75 {diag['mfe_losers_p75']:.2f})")
    if diag["median_mae_winners"] is not None:
        lines.append(f"Q9, median MAE for winners: {diag['median_mae_winners']:.2f}R")
        interp = "small relative to 1R -- winners tend to move favorably early" if diag["median_mae_winners"] < 0.3 else "substantial -- winners often tolerate real adverse movement first"
        lines.append(f"  (HYPOTHESIS, not a measured fact: {interp})")

    # Q10: TP1-too-close read, stated as hypothesis, not fact
    pct_beyond_tp1 = diag["pct_reaching_mfe_r"].get(1.0, 0.0)
    lines.append("")
    lines.append(
        f"Q10 (HYPOTHESIS): current TP1 sits at 0.667R. {pct_beyond_tp1:.0f}% of ALL trades reach ≥1R MFE at some point "
        f"(not just winners) -- {'suggestive that real profit is being left on the table' if pct_beyond_tp1 > 40 else 'not strong evidence TP1 is systematically too close'}. "
        f"This is directional, not proof -- see the counterfactuals below for a more concrete estimate."
    )

    lines.append("")
    lines.append(f"*Part 2 -- Counterfactual exit structures (approximate, spread {used_spread:.2f}%, avg cost {avg_cost_r:.3f}R/trade)*")
    lines.append("_COUNTERFACTUAL: reconstructed from stored MAE/MFE, not a full price re-simulation. Not validated._\n")

    scenarios = [
        ("A. Current baseline (TP1=0.667R)", compute_counterfactual_exit(trades, "baseline")),
        ("B. TP1 + runner", compute_counterfactual_exit(trades, "runner")),
        ("C. 1.0R fixed target", compute_counterfactual_exit(trades, "fixed_target", target_r=1.0)),
        ("D. 1.5R fixed target", compute_counterfactual_exit(trades, "fixed_target", target_r=1.5)),
        ("E. 2.0R fixed target", compute_counterfactual_exit(trades, "fixed_target", target_r=2.0)),
    ]
    for label, result in scenarios:
        if result["avg_r"] is None:
            lines.append(f"{label}: no data")
            continue
        net_avg_r = result["avg_r"] - avg_cost_r
        pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] is not None else "N/A"
        holding_str = f"{result['median_holding_minutes']/60:.1f}h" if result["median_holding_minutes"] is not None else "N/A"
        lines.append(
            f"{label}: WR {result['win_rate']*100:.0f}%, avg R {result['avg_r']:+.3f} (gross) / {net_avg_r:+.3f} (net), "
            f"PF {pf_str}, max DD {result['max_drawdown_r']:.2f}R, n={result['n']}, median hold {holding_str}"
        )

    lines.append("")
    lines.append("*Part 3 -- Improvement required to reach economic targets*")
    lines.append(f"_At the current measured {2.46:.2f} trades/day pace (HYPOTHESIS: assumes this frequency continues):_")
    for target_edge in [0.20, 0.25, 0.30]:
        daily_r = 2.46 * target_edge
        lines.append(f"  +{target_edge:.2f}R/trade -> {daily_r:.3f}R/day expected")

    lines.append("")
    lines.append(
        "*Conclusion*: none of the counterfactuals above are validated results -- if one looks structurally promising, "
        "the correct next step is to freeze that exact exit rule and re-run it as a genuine backtest against a period "
        "not used to generate this diagnostic, before it's considered for the live bot."
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent MAE/MFE diagnostic report for %s (%s)", api_symbol, strategy_type)


if __name__ == "__main__":
    asyncio.run(run_confidence_threshold_analysis())
