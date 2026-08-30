"""XAU Forward-Test Checkpoint -- comprehensive evidence review.

Read-only. Does NOT modify bot/xau_swing.py or either live configuration.
Reuses already-tested helpers from xau_forward_audit.py where possible;
adds the additional analyses this specific checkpoint requires (open
trades, LONG/SHORT split, holding-time distribution, longest streaks,
concentration, candidate-vs-baseline pairing/lag mechanics, extended
risk-level economics, and the decision tree).
"""

import asyncio
import logging
from datetime import datetime

from . import db
from . import telegram_bot
from .analytics import get_default_spread_pct
from .xau_forward_audit import get_forward_trades, compute_net_r, get_monthly_breakdown
from .xau_swing import SYMBOL_TAG, STRATEGY_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("xau_checkpoint")

# Candidate constants defined LOCALLY (not imported from xau_swing.py) --
# same defensive pattern established in xau_forward_audit.py, avoiding
# fragile cross-file coupling regardless of what the deployed file
# currently contains.
CANDIDATE_SYMBOL_TAG = "Gold (XAU/USD) [4h-swing-candidate-1.0R]"
CANDIDATE_STRATEGY_TYPE = "trend_4h_swing_candidate_1_0r"

RISK_LEVELS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
HISTORICAL_MAX_DD_R = 25.67  # established full-year backtest figure, at 1.0R-equivalent baseline scaling


def get_trades_with_action(symbol_tag: str, strategy_type: str) -> list[dict]:
    """Same as get_forward_trades but also includes action (LONG/SHORT)
    -- a separate query rather than modifying the already-tested
    xau_forward_audit.get_forward_trades."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT s.entry, s.sl, s.created_at, s.action, o.outcome, o.r_multiple, o.mae_r, o.mfe_r, o.evaluated_at
               FROM signal_outcomes o
               JOIN scalp_signals s ON s.id = o.signal_id
               WHERE s.symbol = ? AND s.strategy_type = ?
                 AND o.outcome IN ('WIN', 'LOSS')
               ORDER BY s.created_at ASC""",
            (symbol_tag, strategy_type),
        ).fetchall()
    finally:
        conn.close()
    cols = ["entry", "sl", "created_at", "action", "outcome", "r_multiple", "mae_r", "mfe_r", "evaluated_at"]
    return [dict(zip(cols, row)) for row in rows]


def get_open_trade_count(symbol_tag: str, strategy_type: str) -> int:
    conn = db._connect()
    try:
        row = conn.execute(
            """SELECT COUNT(*) FROM scalp_signals s
               WHERE s.symbol = ? AND s.strategy_type = ? AND s.action != 'NO TRADE'
                 AND s.id NOT IN (SELECT signal_id FROM signal_outcomes)""",
            (symbol_tag, strategy_type),
        ).fetchone()
        return row[0]
    finally:
        conn.close()


def compute_holding_time_stats(trades: list[dict]) -> dict:
    hours = []
    for t in trades:
        try:
            created = datetime.fromisoformat(t["created_at"])
            evaluated = datetime.fromisoformat(t["evaluated_at"])
            hours.append((evaluated - created).total_seconds() / 3600)
        except Exception:
            continue
    if not hours:
        return {"n": 0, "median_hours": None, "mean_hours": None, "max_hours": None, "min_hours": None}
    sorted_hours = sorted(hours)
    n = len(sorted_hours)
    median = sorted_hours[n // 2] if n % 2 == 1 else (sorted_hours[n // 2 - 1] + sorted_hours[n // 2]) / 2
    return {"n": n, "median_hours": median, "mean_hours": sum(hours) / n, "max_hours": max(hours), "min_hours": min(hours)}


def compute_longest_streaks(trades: list[dict]) -> dict:
    """Trades must be pre-sorted ascending by created_at (get_forward_trades/
    get_trades_with_action already do this). Longest streaks, not just current."""
    longest_win, longest_loss = 0, 0
    cur_win, cur_loss = 0, 0
    for t in trades:
        if t["outcome"] == "WIN":
            cur_win += 1
            cur_loss = 0
            longest_win = max(longest_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            longest_loss = max(longest_loss, cur_loss)
    return {"longest_win_streak": longest_win, "longest_loss_streak": longest_loss}


def compute_concentration(net_r_values: list[float]) -> dict:
    n = len(net_r_values)
    if n == 0:
        return {"top10": None, "top20": None, "top30": None}
    total_positive = sum(r for r in net_r_values if r > 0)
    if total_positive <= 0:
        return {"top10": None, "top20": None, "top30": None}
    sorted_desc = sorted(net_r_values, reverse=True)
    def top_pct_share(pct):
        count = max(1, int(n * pct))
        return sum(sorted_desc[:count]) / total_positive
    return {"top10": top_pct_share(0.10), "top20": top_pct_share(0.20), "top30": top_pct_share(0.30)}


def compute_direction_breakdown(trades: list[dict], spread_pct: float) -> dict:
    result = {}
    for direction in ["LONG", "SHORT"]:
        sub = [t for t in trades if t["action"] == direction]
        n = len(sub)
        if n == 0:
            result[direction] = {"n": 0}
            continue
        net_r_vals = [compute_net_r(t, spread_pct) for t in sub]
        wins = sum(1 for t in sub if t["outcome"] == "WIN")
        result[direction] = {"n": n, "win_rate": wins / n, "avg_net_r": sum(net_r_vals) / n}
    return result


def match_candidate_to_baseline(baseline_trades: list[dict], candidate_trades: list[dict],
                                 all_baseline_incl_open: list[dict], all_candidate_incl_open: list[dict]) -> dict:
    """Mechanical lag check: baseline and candidate are generated from the
    IDENTICAL entry/SL each cycle (per xau_swing.py's shared-fetch A/B
    design) -- so a baseline signal and a candidate signal sharing the
    same entry price within a tight time window represent the SAME
    underlying market decision, differing only in which resolved first.
    This directly tests whether the candidate's lower resolved count is
    mechanically explained by its wider TP taking longer, not by fewer
    underlying signals."""
    def round_entry(e):
        return round(e, 2) if e is not None else None

    baseline_by_key = {}
    for t in all_baseline_incl_open:
        key = (t["created_at"][:16], round_entry(t["entry"]))  # match to the minute
        baseline_by_key[key] = t
    candidate_by_key = {}
    for t in all_candidate_incl_open:
        key = (t["created_at"][:16], round_entry(t["entry"]))
        candidate_by_key[key] = t

    shared_keys = set(baseline_by_key.keys()) & set(candidate_by_key.keys())
    resolved_both = sum(1 for k in shared_keys if baseline_by_key[k].get("outcome") and candidate_by_key[k].get("outcome"))
    resolved_baseline_only = sum(1 for k in shared_keys if baseline_by_key[k].get("outcome") and not candidate_by_key[k].get("outcome"))
    resolved_candidate_only = sum(1 for k in shared_keys if candidate_by_key[k].get("outcome") and not baseline_by_key[k].get("outcome"))
    unresolved_both = sum(1 for k in shared_keys if not baseline_by_key[k].get("outcome") and not candidate_by_key[k].get("outcome"))

    return {
        "n_shared_signals": len(shared_keys),
        "resolved_both": resolved_both,
        "resolved_baseline_only": resolved_baseline_only,
        "resolved_candidate_only": resolved_candidate_only,
        "unresolved_both": unresolved_both,
    }


def get_all_trades_incl_open(symbol_tag: str, strategy_type: str) -> list[dict]:
    """Every non-NO-TRADE signal, resolved or not, with 'outcome' present
    only if resolved -- used for the candidate/baseline pairing check."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT s.entry, s.sl, s.created_at, s.action, o.outcome, o.r_multiple, o.evaluated_at
               FROM scalp_signals s
               LEFT JOIN signal_outcomes o ON o.signal_id = s.id
               WHERE s.symbol = ? AND s.strategy_type = ? AND s.action != 'NO TRADE'
               ORDER BY s.created_at ASC""",
            (symbol_tag, strategy_type),
        ).fetchall()
    finally:
        conn.close()
    cols = ["entry", "sl", "created_at", "action", "outcome", "r_multiple", "evaluated_at"]
    return [dict(zip(cols, row)) for row in rows]


def compute_extended_risk_table(net_r_day: float | None, max_dd_r: float | None) -> list[dict]:
    rows = []
    for risk_pct in RISK_LEVELS:
        required_r_day = 0.01 / (risk_pct / 100)
        observed_pct_day = (net_r_day * risk_pct) if net_r_day is not None else None
        dd_pct = (max_dd_r * risk_pct) if max_dd_r is not None else None
        rows.append({
            "risk_pct": risk_pct, "required_r_day_for_1pct": required_r_day,
            "observed_pct_day": observed_pct_day, "estimated_dd_pct": dd_pct,
        })
    return rows


def classify_reliability(n: int) -> str:
    if n < 10:
        return "still too small to draw any conclusion"
    elif n < 20:
        return "informative but preliminary"
    elif n < 50:
        return "reasonably persuasive, but below this project's own 50-trade comfort threshold"
    else:
        return "strong enough to meaningfully inform a risk decision"


def decide(baseline_stats: dict, candidate_stats: dict, baseline_reliability: str, candidate_reliability: str) -> tuple[str, str]:
    """Implements the decision tree, Cases A-E, mechanically."""
    b_n = baseline_stats.get("n", 0)
    c_n = candidate_stats.get("n", 0)

    if b_n < 20:
        return "INSUFFICIENT FOR DECISION -- KEEP FORWARD TESTING", "Baseline sample still below the 20-trade minimum this project has consistently required."

    b_net_r_day = baseline_stats.get("net_r_day")
    if b_net_r_day is None or b_net_r_day <= 0:
        return "BASELINE EDGE NOT YET CONFIRMED", "Forward net R/day is not positive with an adequate sample -- treat as a possible degradation signal, not evidence to increase risk."

    # Required R/day for 1% at the most conservative acceptable risk (1.00%) vs observed
    required_at_1pct = 1.00
    if b_net_r_day >= required_at_1pct:
        return "CONTROLLED RISK ESCALATION MAY BE CONSIDERED", "Observed forward net R/day already meets or exceeds the 1.00 R/day needed for 1%/day at 1% risk."
    elif b_net_r_day >= required_at_1pct * 0.5:
        return "1% OBJECTIVE NOT REALISTIC WITH CURRENT EDGE", "Edge is real and positive, but closing the gap to 1%/day would require risk levels beyond what this project treats as reasonable."
    else:
        return "1% OBJECTIVE NOT REALISTIC WITH CURRENT EDGE", "Observed forward edge is well short of what any reasonable risk level could close."


def _fmt_full_report_block(label: str, trades: list[dict], stats: dict, spread_pct: float,
                            open_count: int, holding: dict, streaks: dict, concentration: dict,
                            direction: dict) -> list[str]:
    lines = [f"*{label}*"]
    if stats.get("n", 0) == 0:
        lines.append("  No resolved trades yet.")
        lines.append(f"  Open/unresolved trades: {open_count}")
        return lines

    lines.append(f"  Resolved: {stats['n']} | Open/unresolved: {open_count}")
    pf_str = f"{stats['profit_factor']:.2f}" if stats.get("profit_factor") is not None else "N/A"
    lines.append(f"  Win rate: {stats['win_rate']*100:.0f}%, PF: {pf_str}")
    lines.append(f"  Net R/day: {stats['net_r_day']:+.3f}, Net avg R/trade: {stats['net_avg_r']:+.3f}")
    lines.append(f"  Max drawdown: {stats['max_drawdown_r']:.2f}R")
    if holding.get("n", 0) > 0:
        lines.append(f"  Holding time -- median: {holding['median_hours']:.1f}h, mean: {holding['mean_hours']:.1f}h, longest: {holding['max_hours']:.1f}h, shortest: {holding['min_hours']:.1f}h")
    lines.append(f"  Longest winning streak: {streaks['longest_win_streak']}, longest losing streak: {streaks['longest_loss_streak']}")
    if concentration.get("top20") is not None:
        lines.append(f"  Concentration -- top 10%: {concentration['top10']*100:.0f}%, top 20%: {concentration['top20']*100:.0f}%, top 30%: {concentration['top30']*100:.0f}% of positive R")
    long_d, short_d = direction.get("LONG", {}), direction.get("SHORT", {})
    if long_d.get("n", 0) > 0 or short_d.get("n", 0) > 0:
        long_str = f"n={long_d['n']}, WR {long_d['win_rate']*100:.0f}%, avg R {long_d['avg_net_r']:+.3f}" if long_d.get("n", 0) > 0 else "n=0"
        short_str = f"n={short_d['n']}, WR {short_d['win_rate']*100:.0f}%, avg R {short_d['avg_net_r']:+.3f}" if short_d.get("n", 0) > 0 else "n=0"
        lines.append(f"  LONG: {long_str} | SHORT: {short_str}")
    return lines


async def run_checkpoint(spread_pct: float | None = None):
    from .xau_forward_audit import compute_full_stats

    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct("XAU/USD")

    baseline_trades = get_trades_with_action(SYMBOL_TAG, STRATEGY_TYPE)
    candidate_trades = get_trades_with_action(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)
    baseline_stats = compute_full_stats(baseline_trades, spread_pct)
    candidate_stats = compute_full_stats(candidate_trades, spread_pct)

    baseline_open = get_open_trade_count(SYMBOL_TAG, STRATEGY_TYPE)
    candidate_open = get_open_trade_count(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)
    baseline_holding = compute_holding_time_stats(baseline_trades)
    candidate_holding = compute_holding_time_stats(candidate_trades)
    baseline_streaks = compute_longest_streaks(baseline_trades)
    candidate_streaks = compute_longest_streaks(candidate_trades)
    baseline_net_r_vals = [compute_net_r(t, spread_pct) for t in baseline_trades]
    candidate_net_r_vals = [compute_net_r(t, spread_pct) for t in candidate_trades]
    baseline_conc = compute_concentration(baseline_net_r_vals)
    candidate_conc = compute_concentration(candidate_net_r_vals)
    baseline_dir = compute_direction_breakdown(baseline_trades, spread_pct)
    candidate_dir = compute_direction_breakdown(candidate_trades, spread_pct)

    # Message 1: detailed stats for both configurations
    lines1 = [
        "*XAU Forward-Test Checkpoint -- Evidence Review (1 of 3)*",
        "Read-only. `bot/xau_swing.py` and both live configurations are untouched.\n",
    ]
    lines1 += _fmt_full_report_block("Baseline (TP1=0.667R)", baseline_trades, baseline_stats, spread_pct,
                                      baseline_open, baseline_holding, baseline_streaks, baseline_conc, baseline_dir)
    lines1.append("")
    lines1 += _fmt_full_report_block("Candidate (TP1=1.0R)", candidate_trades, candidate_stats, spread_pct,
                                      candidate_open, candidate_holding, candidate_streaks, candidate_conc, candidate_dir)
    await telegram_bot.send_text("\n".join(lines1))

    # Message 2: candidate-vs-baseline mechanical pairing + extended risk tables
    baseline_incl_open = get_all_trades_incl_open(SYMBOL_TAG, STRATEGY_TYPE)
    candidate_incl_open = get_all_trades_incl_open(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)
    pairing = match_candidate_to_baseline(baseline_trades, candidate_trades, baseline_incl_open, candidate_incl_open)

    lines2 = ["*XAU Forward-Test Checkpoint (2 of 3): Candidate-lag mechanics + risk economics*\n"]
    lines2.append("*Candidate/baseline pairing (same underlying entry/SL, matched by timestamp)*")
    lines2.append(f"  Shared signals identified: {pairing['n_shared_signals']}")
    lines2.append(f"  Both resolved: {pairing['resolved_both']}")
    lines2.append(f"  Baseline resolved, candidate still open (mechanical lag): {pairing['resolved_baseline_only']}")
    lines2.append(f"  Candidate resolved, baseline still open: {pairing['resolved_candidate_only']}")
    lines2.append(f"  Both still open: {pairing['unresolved_both']}")
    if pairing["resolved_baseline_only"] > 0:
        lines2.append("  -> Candidate's lower resolved count is at least partly mechanically explained by its wider TP taking longer to resolve, not necessarily fewer real signals.")
    lines2.append("")

    lines2.append("*Extended risk-level economics -- Baseline*")
    lines2.append("Risk% | Required R/day for 1% | Observed %/day | Est. drawdown %")
    baseline_table = compute_extended_risk_table(baseline_stats.get("net_r_day"), baseline_stats.get("max_drawdown_r"))
    for row in baseline_table:
        obs = f"{row['observed_pct_day']:+.3f}%" if row["observed_pct_day"] is not None else "N/A"
        dd = f"{row['estimated_dd_pct']:.1f}%" if row["estimated_dd_pct"] is not None else "N/A"
        lines2.append(f"  {row['risk_pct']:.2f}% | {row['required_r_day_for_1pct']:.2f} R/day | {obs} | {dd}")

    lines2.append("")
    lines2.append("*Extended risk-level economics -- Candidate*")
    lines2.append("Risk% | Required R/day for 1% | Observed %/day | Est. drawdown %")
    candidate_table = compute_extended_risk_table(candidate_stats.get("net_r_day"), candidate_stats.get("max_drawdown_r"))
    for row in candidate_table:
        obs = f"{row['observed_pct_day']:+.3f}%" if row["observed_pct_day"] is not None else "N/A"
        dd = f"{row['estimated_dd_pct']:.1f}%" if row["estimated_dd_pct"] is not None else "N/A"
        lines2.append(f"  {row['risk_pct']:.2f}% | {row['required_r_day_for_1pct']:.2f} R/day | {obs} | {dd}")

    lines2.append("\nNote: increasing risk per trade magnifies both returns and losses equally -- it does not create edge. These figures show what risk WOULD be required, not a recommendation to use it.")
    await telegram_bot.send_text("\n".join(lines2))

    # Message 3: final structured decision report
    b_reliability = classify_reliability(baseline_stats.get("n", 0))
    c_reliability = classify_reliability(candidate_stats.get("n", 0))
    decision, decision_reason = decide(baseline_stats, candidate_stats, b_reliability, c_reliability)

    if baseline_stats.get("n", 0) < 20 or candidate_stats.get("n", 0) < 20:
        candidate_verdict = "INCONCLUSIVE -- insufficient sample on at least one side"
    else:
        diff = candidate_stats["net_r_day"] - baseline_stats["net_r_day"]
        if abs(diff) < 0.05:
            candidate_verdict = "statistically indistinguishable from baseline"
        elif diff > 0:
            candidate_verdict = "superior on current evidence (still requires more data to confirm)"
        else:
            candidate_verdict = "inferior on current evidence"

    lines3 = ["*XAU Forward-Test Checkpoint (3 of 3): Decision*\n"]
    lines3.append("*CURRENT EVIDENCE*")
    lines3.append(f"Baseline: n={baseline_stats.get('n',0)} resolved ({b_reliability}), Candidate: n={candidate_stats.get('n',0)} resolved ({c_reliability})")
    lines3.append("")
    lines3.append("*ACCOUNT ECONOMICS*")
    if baseline_stats.get("net_r_day") is not None:
        lines3.append(f"Baseline observed net R/day: {baseline_stats['net_r_day']:+.3f} -- see message 2 for the full risk-level table")
    if candidate_stats.get("net_r_day") is not None:
        lines3.append(f"Candidate observed net R/day: {candidate_stats['net_r_day']:+.3f}")
    lines3.append("")
    lines3.append("*1% OBJECTIVE*")
    lines3.append(decision_reason)
    lines3.append("")
    lines3.append("*DRAWDOWN / RISK*")
    lines3.append(f"Historical full-year max drawdown: {HISTORICAL_MAX_DD_R:.2f}R. Forward-observed max drawdown: baseline {baseline_stats.get('max_drawdown_r', 0):.2f}R, candidate {candidate_stats.get('max_drawdown_r', 0):.2f}R (small-sample, not yet comparable to the historical figure).")
    lines3.append("")
    lines3.append("*BASELINE vs CANDIDATE*")
    lines3.append(f"Candidate classified as: {candidate_verdict}")
    lines3.append("")
    lines3.append(f"*DECISION: {decision}*")
    lines3.append("")
    lines3.append("*NEXT CHECKPOINT*")
    needed_b = max(0, 30 - baseline_stats.get("n", 0))
    needed_c = max(0, 30 - candidate_stats.get("n", 0))
    lines3.append(f"At least 30-50 resolved trades per variant before the next review (currently need {needed_b} more for baseline, {needed_c} more for candidate). No changes to either configuration until then.")

    await telegram_bot.send_text("\n".join(lines3))
    logger.info("Sent full checkpoint report (3 messages), decision=%s", decision)


if __name__ == "__main__":
    asyncio.run(run_checkpoint())
