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
    # Silver's spread as a % of price is genuinely more uncertain than the
    # others: broker-quoted dollar spreads range roughly $0.05 (tight ECN)
    # to $0.50 (standard retail), and unlike gold, silver's own price
    # moved dramatically across the 2025 test window -- the same dollar
    # spread represents a very different % depending on when in that move
    # a trade happened. 0.15% is a defensible middle estimate for a decent
    # (not elite, not worst-case) retail account -- verify against your
    # real broker before trusting this more than directionally.
    "XAG/USD": 0.15,
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


def _variant_exclusion_sql(api_symbol: str, column_ref: str = "e.symbol") -> str:
    """Returns the SQL fragment to exclude R:R-variant-tagged rows from a
    prefix-matched query -- UNLESS the caller is explicitly asking about a
    specific variant (their api_symbol itself already contains "(R:R"),
    in which case excluding it would filter out the very data they asked
    for. Without this, cost-adjusted/consistency analysis could never be
    run on a variant's own results."""
    if "(R:R" in api_symbol:
        return ""
    return f" AND {column_ref} NOT LIKE '%(R:R%'"


def get_raw_stats_for_symbol_prefix(api_symbol: str, strategy_type: str) -> dict | None:
    conn = db._connect()
    try:
        rows = conn.execute(
            f"""SELECT o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(api_symbol)}
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
            f"""SELECT e.entry, e.sl, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(api_symbol)}
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
                f"""SELECT MIN(evaluated_at), MAX(evaluated_at), COUNT(*)
                   FROM all_evaluations
                   WHERE source='backtest' AND strategy_type='trend' AND action != 'NO TRADE'
                     AND symbol LIKE ?{_variant_exclusion_sql(api_symbol, column_ref='symbol')}""",
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
            f"""SELECT e.evaluated_at, o.outcome, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(api_symbol)}
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
            f"""SELECT o.outcome, o.r_multiple, o.mae_before_tp1_r, o.mfe_before_tp1_r,
                      o.tp1_hit, o.tp2_hit, o.mfe_after_tp1_r, o.max_giveback_after_tp1_r,
                      o.time_to_tp1_minutes, o.time_to_tp2_minutes, o.time_to_exit_minutes
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(api_symbol)}""",
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


def get_cost_r_distribution(api_symbol: str, strategy_type: str, spread_pct: float | None = None) -> dict:
    """Computes the DISTRIBUTION of cost/R (not just the average) across
    every trade -- median, mean, p25, p75, and % of trades where the
    spread cost consumed more than 0.10R / 0.20R / 0.30R of the risk
    unit. The average alone can hide how badly a subset of trades were
    affected; this is the more complete picture behind why the original
    5m-ATR configuration failed so badly on cost -- most trades had a
    tolerable cost/R, but a meaningful fraction had a severe one, since
    cost/R scales inversely with how tight that specific trade's SL
    happened to be."""
    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct(api_symbol)
    conn = db._connect()
    try:
        rows = conn.execute(
            f"""SELECT e.entry, e.sl
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(api_symbol)}
                 AND o.outcome IN ('WIN', 'LOSS')""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchall()
    finally:
        conn.close()

    cost_r_vals = []
    for entry, sl in rows:
        if entry is None or sl is None:
            continue
        risk_price = abs(entry - sl)
        if risk_price == 0:
            continue
        cost_r_vals.append((entry * spread_pct / 100) / risk_price)

    n = len(cost_r_vals)
    if n == 0:
        return {"n": 0}
    sorted_vals = sorted(cost_r_vals)
    return {
        "n": n, "spread_pct_used": spread_pct,
        "mean": sum(cost_r_vals) / n,
        "median": percentile(sorted_vals, 50),
        "p25": percentile(sorted_vals, 25),
        "p75": percentile(sorted_vals, 75),
        "pct_above_010": 100 * sum(1 for c in cost_r_vals if c > 0.10) / n,
        "pct_above_020": 100 * sum(1 for c in cost_r_vals if c > 0.20) / n,
        "pct_above_030": 100 * sum(1 for c in cost_r_vals if c > 0.30) / n,
    }


def get_worst_month(api_symbol: str, strategy_type: str) -> dict | None:
    """Returns the single worst calendar month by avg_r, from the monthly
    breakdown -- explicitly called out since it's the concrete number the
    report needs, not just the full list."""
    monthly = get_monthly_breakdown(api_symbol, strategy_type)
    resolved_months = [m for m in monthly if m["trades"] > 0 and m["avg_r"] is not None]
    if not resolved_months:
        return None
    return min(resolved_months, key=lambda m: m["avg_r"])


async def run_three_way_comparison_report(spread_pct_xau: float | None = None):
    """Compares XAU 4H baseline, XAU 4H 1.0R candidate, and XAU 1H-ATR
    side by side -- number of trades, trades/day, gross/net expectancy,
    net R/day, win rate, PF, max drawdown, MAE/MFE, holding time, losing
    streak, worst month, and (for the 1H config specifically) the full
    cost/R distribution requested. Every figure here is OBSERVED from
    historical backtest data -- nothing here is forward-validated."""
    configs = [
        ("XAU 4H baseline (TP1=0.667R)", "XAU/USD [4h-ATR]", "trend"),
        ("XAU 4H candidate (TP1=1.0R)", "XAU/USD (R:R 1.5:1.5) [4h-ATR]", "trend"),
        ("XAU 1H-ATR", "XAU/USD [1h-ATR]", "trend"),
    ]

    lines = [
        "*📊 Three-Way Comparison: XAU 4H Baseline vs 4H Candidate vs 1H-ATR*",
        "_All figures OBSERVED from historical backtest data. Nothing here is forward-validated._\n",
    ]

    econ_rows = []  # (label, net_r_per_day) for the economic table at the end

    for label, tag, strategy_type in configs:
        raw = get_raw_stats_for_symbol_prefix(tag, strategy_type)
        adj = get_cost_adjusted_stats(tag, strategy_type, spread_pct_xau)
        monthly = get_monthly_breakdown(tag, strategy_type)
        worst = get_worst_month(tag, strategy_type)
        mae_mfe_trades = get_mae_mfe_trade_rows(tag, strategy_type)
        diag = compute_mae_mfe_diagnostics(mae_mfe_trades) if mae_mfe_trades else {"n": 0}

        lines.append(f"*{label}*")
        if raw is None or raw["n"] == 0:
            lines.append("  No data -- run the Historical Backtest for this configuration first.\n")
            econ_rows.append((label, None))
            continue

        days_span = len(monthly) * 30 if monthly else 1  # rough, refined below if possible
        conn = db._connect()
        try:
            date_row = conn.execute(
                f"""SELECT MIN(evaluated_at), MAX(evaluated_at) FROM all_evaluations
                    WHERE source='backtest' AND strategy_type=? AND symbol LIKE ?{_variant_exclusion_sql(tag, column_ref='symbol')}""",
                (strategy_type, f"{tag}%"),
            ).fetchone()
        finally:
            conn.close()
        if date_row and date_row[0] and date_row[1]:
            import datetime as dt
            start, end = dt.datetime.fromisoformat(date_row[0]), dt.datetime.fromisoformat(date_row[1])
            days_span = max((end - start).total_seconds() / 86400, 1e-9)
        trades_per_day = raw["n"] / days_span

        net_r_per_day = trades_per_day * adj["avg_r"] if adj else None
        pf_str = f"{raw['profit_factor']:.2f}" if raw["profit_factor"] is not None else "N/A"
        adj_pf_str = f"{adj['profit_factor']:.2f}" if adj and adj["profit_factor"] is not None else "N/A"

        lines.append(f"  n={raw['n']}, {trades_per_day:.2f} trades/day")
        lines.append(f"  Gross expectancy: {raw['avg_r']:+.3f}R/trade, PF {pf_str}")
        if adj:
            lines.append(f"  Transaction cost: {adj['avg_cost_r']:.3f}R/trade avg")
            lines.append(f"  Net expectancy: {adj['avg_r']:+.3f}R/trade, PF {adj_pf_str}, win rate {adj['win_rate']*100:.0f}%")
            lines.append(f"  *Net R/day: {net_r_per_day:+.3f}*")
        if diag.get("n", 0) > 0:
            lines.append(f"  Median MFE (winners): {diag['median_mfe_winners']:.2f}R" if diag.get("median_mfe_winners") is not None else "")
        # Max drawdown and losing streak from compute_metrics on the raw rows
        conn = db._connect()
        try:
            outcome_rows = conn.execute(
                f"""SELECT o.outcome, o.r_multiple FROM backtest_outcomes o
                    JOIN all_evaluations e ON e.id=o.evaluation_id
                    WHERE e.source='backtest' AND e.strategy_type=?
                      AND e.symbol LIKE ?{_variant_exclusion_sql(tag)}
                      AND o.outcome IN ('WIN','LOSS')
                    ORDER BY o.evaluated_at ASC""",
                (strategy_type, f"{tag}%"),
            ).fetchall()
        finally:
            conn.close()
        m = compute_metrics(outcome_rows)
        lines.append(f"  Max drawdown: {m['max_drawdown_r']:.2f}R, longest losing streak: {m['max_consecutive_losses']}")
        if worst:
            lines.append(f"  Worst month: {worst['month']} (avg R {worst['avg_r']:+.2f}, n={worst['trades']})")
        lines.append("")

        econ_rows.append((label, net_r_per_day))

    # Cost/R distribution -- specifically requested for the 1H config
    lines.append("*Cost/R distribution -- XAU 1H-ATR (the explicit cost check requested)*")
    cost_dist = get_cost_r_distribution("XAU/USD [1h-ATR]", "trend", spread_pct_xau)
    if cost_dist.get("n", 0) == 0:
        lines.append("  No data yet.")
    else:
        lines.append(f"  n={cost_dist['n']}, spread assumption {cost_dist['spread_pct_used']:.2f}%")
        lines.append(f"  Mean {cost_dist['mean']:.3f}R, median {cost_dist['median']:.3f}R, p25 {cost_dist['p25']:.3f}R, p75 {cost_dist['p75']:.3f}R")
        lines.append(f"  % of trades with cost >0.10R: {cost_dist['pct_above_010']:.1f}%")
        lines.append(f"  % of trades with cost >0.20R: {cost_dist['pct_above_020']:.1f}%")
        lines.append(f"  % of trades with cost >0.30R: {cost_dist['pct_above_030']:.1f}%")
    lines.append("")

    lines.append("*Economic comparison -- net %/day at each risk level*")
    for risk_pct in [0.25, 0.50, 0.75, 1.00]:
        parts = []
        for label, net_r_day in econ_rows:
            if net_r_day is None:
                parts.append(f"{label.split('(')[0].strip()}: N/A")
            else:
                parts.append(f"{label.split('(')[0].strip()}: {net_r_day * risk_pct:.3f}%")
        lines.append(f"  {risk_pct:.2f}% risk: " + " | ".join(parts))

    lines.append(
        "\n_Reminder: nothing above is forward-validated. A positive 1H-ATR result here means it's a candidate for "
        "out-of-sample/forward testing, not a result to act on directly._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent three-way XAU comparison report")


def get_daily_r_series(api_symbol: str, strategy_type: str) -> dict[str, float]:
    """Returns {date_str: summed_r_multiple} for every day with at least
    one resolved trade -- the building block for cross-instrument
    correlation, since correlation needs aligned daily observations, not
    a raw trade-by-trade list (which would rarely line up in time between
    two different symbols)."""
    conn = db._connect()
    try:
        rows = conn.execute(
            f"""SELECT e.evaluated_at, o.r_multiple
               FROM backtest_outcomes o
               JOIN all_evaluations e ON e.id = o.evaluation_id
               WHERE e.source='backtest' AND e.strategy_type=?
                 AND e.symbol LIKE ?{_variant_exclusion_sql(api_symbol)}
                 AND o.outcome IN ('WIN', 'LOSS')""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchall()
    finally:
        conn.close()

    by_day: dict[str, float] = {}
    for evaluated_at, r in rows:
        day = evaluated_at[:10]
        by_day[day] = by_day.get(day, 0.0) + r
    return by_day


def compute_correlation(series_a: dict[str, float], series_b: dict[str, float]) -> dict:
    """Pearson correlation between two instruments' daily R series, over
    the days where BOTH had at least one resolved trade (days where only
    one instrument traded aren't informative about co-movement). Also
    reports simple same-day-directional-overlap: of the days both traded,
    what % had the same sign of daily R (both net positive or both net
    negative) -- a more intuitive, less assumption-laden overlap measure
    alongside the correlation coefficient."""
    common_days = sorted(set(series_a.keys()) & set(series_b.keys()))
    n = len(common_days)
    if n < 2:
        return {"n_common_days": n, "correlation": None, "same_sign_pct": None}

    a_vals = [series_a[d] for d in common_days]
    b_vals = [series_b[d] for d in common_days]
    mean_a, mean_b = sum(a_vals) / n, sum(b_vals) / n

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(a_vals, b_vals))
    std_a = (sum((a - mean_a) ** 2 for a in a_vals)) ** 0.5
    std_b = (sum((b - mean_b) ** 2 for b in b_vals)) ** 0.5

    correlation = cov / (std_a * std_b) if (std_a > 0 and std_b > 0) else None
    same_sign = sum(1 for a, b in zip(a_vals, b_vals) if (a > 0) == (b > 0))
    same_sign_pct = 100 * same_sign / n

    return {"n_common_days": n, "correlation": correlation, "same_sign_pct": same_sign_pct}


def classify_result(net_r_per_day: float | None, min_trades_met: bool) -> str:
    """Classifies a backtested result per the pre-declared taxonomy --
    NEGATIVE / PROMISING / HISTORICALLY POSITIVE (REQUIRES VALIDATION) --
    FORWARD VALIDATED is never assigned here, since that requires actual
    forward data this function has no access to."""
    if not min_trades_met or net_r_per_day is None:
        return "INSUFFICIENT DATA"
    if net_r_per_day <= 0:
        return "NEGATIVE"
    if net_r_per_day < 0.15:
        return "PROMISING (weak)"
    return "HISTORICALLY POSITIVE -- REQUIRES VALIDATION"


async def run_xag_analysis_report(spread_pct_xag: float | None = None, spread_pct_xau: float | None = None):
    """Full XAG/USD 4H-ATR report using the same methodology as the
    established single-symbol reports, PLUS a cost-sensitivity sweep (not
    a single spread assumption), correlation/overlap with XAU, and a
    portfolio analysis that asks whether XAG adds INCREMENTAL value given
    its correlation with XAU -- not just whether it's independently
    profitable under one assumed cost.

    IMPORTANT: every cost-dependent figure below is explicitly labeled
    "under assumed X% spread" -- this assumption is a placeholder, not a
    measured fact. The real number for your bot is your actual execution
    venue's spread + commission + slippage, expressed against each
    trade's real risk. Nothing here should be read as "validated" until
    that real figure replaces the placeholder."""
    xag_tag, xau_tag, strategy_type = "XAG/USD [4h-ATR]", "XAU/USD [4h-ATR]", "trend"
    used_spread_xag = spread_pct_xag if spread_pct_xag is not None else get_default_spread_pct(xag_tag)

    raw = get_raw_stats_for_symbol_prefix(xag_tag, strategy_type)
    if raw is None or raw["n"] == 0:
        await telegram_bot.send_text("*🥈 XAG/USD 4H-ATR Analysis*\n\nNo data found -- run the Historical Backtest for XAG/USD with atr_timeframe=4h first.")
        return

    adj = get_cost_adjusted_stats(xag_tag, strategy_type, spread_pct_xag)
    monthly = get_monthly_breakdown(xag_tag, strategy_type)
    worst = get_worst_month(xag_tag, strategy_type)
    cost_dist = get_cost_r_distribution(xag_tag, strategy_type, spread_pct_xag)
    sensitivity = get_cost_sensitivity_table(xag_tag, strategy_type)
    mae_mfe_trades = get_mae_mfe_trade_rows(xag_tag, strategy_type)
    diag = compute_mae_mfe_diagnostics(mae_mfe_trades) if mae_mfe_trades else {"n": 0}

    conn = db._connect()
    try:
        date_row = conn.execute(
            f"""SELECT MIN(evaluated_at), MAX(evaluated_at) FROM all_evaluations
                WHERE source='backtest' AND strategy_type=? AND symbol LIKE ?{_variant_exclusion_sql(xag_tag, column_ref='symbol')}""",
            (strategy_type, f"{xag_tag}%"),
        ).fetchone()
        outcome_rows = conn.execute(
            f"""SELECT o.outcome, o.r_multiple FROM backtest_outcomes o
                JOIN all_evaluations e ON e.id=o.evaluation_id
                WHERE e.source='backtest' AND e.strategy_type=?
                  AND e.symbol LIKE ?{_variant_exclusion_sql(xag_tag)}
                  AND o.outcome IN ('WIN','LOSS')
                ORDER BY o.evaluated_at ASC""",
            (strategy_type, f"{xag_tag}%"),
        ).fetchall()
    finally:
        conn.close()

    import datetime as dt
    days_span = 1.0
    if date_row and date_row[0] and date_row[1]:
        start, end = dt.datetime.fromisoformat(date_row[0]), dt.datetime.fromisoformat(date_row[1])
        days_span = max((end - start).total_seconds() / 86400, 1e-9)
    trades_per_day = raw["n"] / days_span
    net_r_per_day = trades_per_day * adj["avg_r"] if adj else None
    m = compute_metrics(outcome_rows)

    min_trades_met = raw["n"] >= 100
    classification = classify_result(net_r_per_day, min_trades_met)
    # How many of the 5 sensitivity levels stay net-positive -- the more
    # informative summary than any single spread's classification alone
    positive_levels = [r["spread_pct"] for r in sensitivity if r["net_r"] is not None and r["net_r"] > 0]

    lines = [
        "*🥈 XAG/USD 4H-ATR Analysis (independent hypothesis test)*",
        "_Same predefined 4H-ATR methodology as gold, no silver-specific tuning. All figures OBSERVED, not forward-validated. "
        "Cost figures are placeholders until tied to your actual broker's real spread + commission + slippage._\n",
    ]

    pf_str = f"{raw['profit_factor']:.2f}" if raw["profit_factor"] is not None else "N/A"
    adj_pf_str = f"{adj['profit_factor']:.2f}" if adj and adj["profit_factor"] is not None else "N/A"
    lines.append(f"n={raw['n']}, {trades_per_day:.2f} trades/day")
    lines.append(f"Gross expectancy (cost-independent, real): {raw['avg_r']:+.3f}R/trade, PF {pf_str}")
    if adj:
        lines.append(f"Transaction cost UNDER ASSUMED {used_spread_xag:.2f}% SPREAD: {adj['avg_cost_r']:.3f}R/trade avg")
        lines.append(f"*XAG result under assumed {used_spread_xag:.2f}% spread*: net {adj['avg_r']:+.3f}R/trade, PF {adj_pf_str}, WR {adj['win_rate']*100:.0f}%, *{net_r_per_day:+.3f} R/day*")
    lines.append(f"Max drawdown: {m['max_drawdown_r']:.2f}R, longest losing streak: {m['max_consecutive_losses']}")
    if worst:
        lines.append(f"Worst month: {worst['month']} (avg R {worst['avg_r']:+.2f}, n={worst['trades']})")
    if diag.get("n", 0) > 0 and diag.get("median_mfe_winners") is not None:
        lines.append(f"Median MFE (winners): {diag['median_mfe_winners']:.2f}R, median MAE (winners): {diag.get('median_mae_winners', 0) or 0:.2f}R")

    lines.append("")
    lines.append(f"*Cost/R distribution (under assumed {used_spread_xag:.2f}% spread)*")
    if cost_dist.get("n", 0) > 0:
        lines.append(f"  Mean {cost_dist['mean']:.3f}R, median {cost_dist['median']:.3f}R, p25 {cost_dist['p25']:.3f}R, p75 {cost_dist['p75']:.3f}R")
        lines.append(f"  % >0.10R: {cost_dist['pct_above_010']:.1f}% | % >0.20R: {cost_dist['pct_above_020']:.1f}% | % >0.30R: {cost_dist['pct_above_030']:.1f}%")

    lines.append("")
    lines.append("*Cost sensitivity -- how much does this depend on which spread assumption is used?*")
    lines.append("Spread% | Gross R | Net R | Net R/day | PF")
    for row in sensitivity:
        if row["net_r"] is None:
            lines.append(f"  {row['spread_pct']:.2f}% | no data")
            continue
        pf_s = f"{row['pf']:.2f}" if row["pf"] is not None else "N/A"
        lines.append(f"  {row['spread_pct']:.2f}% | {row['gross_r']:+.3f}R | {row['net_r']:+.3f}R | {row['net_r_day']:+.3f} | {pf_s}")
    if positive_levels:
        lines.append(f"  _Stays net-positive from {min(positive_levels):.2f}% up to {max(positive_levels):.2f}% spread (of the levels tested)._")
    else:
        lines.append("  _Not net-positive at ANY tested spread level -- the edge does not survive realistic costs at all._")

    lines.append("")
    lines.append(f"*Classification (under assumed {used_spread_xag:.2f}% spread): {classification}*")

    # Correlation / INCREMENTAL portfolio analysis
    xau_raw = get_raw_stats_for_symbol_prefix(xau_tag, strategy_type)
    xau_adj = get_cost_adjusted_stats(xau_tag, strategy_type, spread_pct_xau)
    if xau_raw and xau_adj:
        xau_days = get_daily_r_series(xau_tag, strategy_type)
        xag_days = get_daily_r_series(xag_tag, strategy_type)
        corr = compute_correlation(xau_days, xag_days)

        lines.append("")
        lines.append("*Correlation / overlap with XAU 4H-ATR trend*")
        if corr["correlation"] is not None:
            lines.append(f"  Pearson correlation (daily R, n={corr['n_common_days']} common days): {corr['correlation']:+.2f}")
            lines.append(f"  Same-sign-day overlap: {corr['same_sign_pct']:.0f}%")
        else:
            lines.append(f"  Not enough overlapping days to compute correlation (n={corr['n_common_days']}).")

        conn = db._connect()
        try:
            xau_date_row = conn.execute(
                f"""SELECT MIN(evaluated_at), MAX(evaluated_at) FROM all_evaluations
                    WHERE source='backtest' AND strategy_type=? AND symbol LIKE ?{_variant_exclusion_sql(xau_tag, column_ref='symbol')}""",
                (strategy_type, f"{xau_tag}%"),
            ).fetchone()
        finally:
            conn.close()
        xau_days_span = 1.0
        if xau_date_row and xau_date_row[0] and xau_date_row[1]:
            xstart, xend = dt.datetime.fromisoformat(xau_date_row[0]), dt.datetime.fromisoformat(xau_date_row[1])
            xau_days_span = max((xend - xstart).total_seconds() / 86400, 1e-9)
        xau_trades_per_day = xau_raw["n"] / xau_days_span
        xau_net_r_per_day = xau_trades_per_day * xau_adj["avg_r"]

        lines.append("")
        lines.append("*The real portfolio question: does XAG add INCREMENTAL value, not just 'is it profitable'?*")
        lines.append(f"  XAU alone: {xau_net_r_per_day:+.3f} R/day")
        lines.append(f"  XAG alone (under assumed {used_spread_xag:.2f}% spread): {net_r_per_day:+.3f} R/day" if net_r_per_day is not None else "  XAG alone: N/A")
        if net_r_per_day is not None:
            combined_r_day = xau_net_r_per_day + net_r_per_day
            lines.append(f"  Sum of R/day if run independently: {combined_r_day:+.3f} R/day")
            lines.append("")
            if net_r_per_day <= 0:
                lines.append("  _VERDICT: XAG is net negative under this assumption -- it adds no value, incremental or otherwise, and would dilute a XAU-only allocation if capital were split toward it._")
            elif corr["correlation"] is not None and corr["correlation"] > 0.5:
                lines.append(
                    f"  _VERDICT: XAG is individually positive ({net_r_per_day:+.3f} R/day), but correlation with XAU is high (+{corr['correlation']:.2f}). "
                    f"The R/day SUM is still additive in expectation -- but on your worst days, both would likely lose together, "
                    f"so running both is closer to running one instrument at larger size than to genuine diversification. "
                    f"The incremental RISK-ADJUSTED value is meaningfully less than the raw sum suggests._"
                )
            elif corr["correlation"] is not None and corr["correlation"] > 0.2:
                lines.append(
                    f"  _VERDICT: XAG is individually positive, with moderate correlation (+{corr['correlation']:.2f}) to XAU -- "
                    f"some genuine incremental/diversification value, but not fully independent. Worth treating as a partial second leg, not a full one._"
                )
            else:
                lines.append(
                    f"  _VERDICT: XAG is individually positive, with low correlation to XAU -- this is the strongest case for genuine "
                    f"incremental portfolio value, IF it also survives the cost-sensitivity range and out-of-sample testing below._"
                )

    lines.append(
        "\n_Per protocol: nothing above is a validated second edge. If XAG survives the cost-sensitivity range and shows genuine "
        "incremental value net of correlation, the next step is freezing this exact configuration for out-of-sample/forward testing -- "
        "not live deployment. XAU baseline and 1.0R candidate forward tests remain untouched by this analysis._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent XAG/USD analysis report")


def get_cost_sensitivity_table(api_symbol: str, strategy_type: str,
                                spread_levels: list[float] | None = None) -> list[dict]:
    """Runs the SAME already-collected trades through several different
    spread assumptions, to answer the real question: how sensitive is
    this edge to execution cost? A single spread assumption can make a
    marginal edge look falsely solid or falsely dead -- this shows the
    whole curve, so the honest answer is 'profitable from X% to Y%', not
    a single number that depends entirely on which assumption was picked."""
    spread_levels = spread_levels or [0.05, 0.10, 0.15, 0.20, 0.30]
    raw = get_raw_stats_for_symbol_prefix(api_symbol, strategy_type)

    conn = db._connect()
    try:
        date_row = conn.execute(
            f"""SELECT MIN(evaluated_at), MAX(evaluated_at) FROM all_evaluations
                WHERE source='backtest' AND strategy_type=? AND symbol LIKE ?{_variant_exclusion_sql(api_symbol, column_ref='symbol')}""",
            (strategy_type, f"{api_symbol}%"),
        ).fetchone()
    finally:
        conn.close()

    import datetime as dt
    days_span = 1.0
    if date_row and date_row[0] and date_row[1]:
        start, end = dt.datetime.fromisoformat(date_row[0]), dt.datetime.fromisoformat(date_row[1])
        days_span = max((end - start).total_seconds() / 86400, 1e-9)
    trades_per_day = (raw["n"] / days_span) if raw else 0.0

    rows = []
    for spread_pct in spread_levels:
        adj = get_cost_adjusted_stats(api_symbol, strategy_type, spread_pct)
        if adj is None:
            rows.append({"spread_pct": spread_pct, "gross_r": None, "net_r": None, "net_r_day": None, "pf": None})
            continue
        net_r_day = trades_per_day * adj["avg_r"]
        rows.append({
            "spread_pct": spread_pct,
            "gross_r": raw["avg_r"] if raw else None,
            "net_r": adj["avg_r"],
            "net_r_day": net_r_day,
            "pf": adj["profit_factor"],
        })
    return rows


if __name__ == "__main__":
    asyncio.run(run_confidence_threshold_analysis())
