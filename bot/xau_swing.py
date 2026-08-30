"""Live forward test of the validated gold trend configuration, PLUS an
isolated A/B exit-structure experiment.

BASELINE (unchanged from before): TP1 = 0.667R (entry +/- 1.0xATR, SL at
1.5xATR). This is the original validated configuration. Its logic, tags,
risk tracking, and behavior are NOT modified by anything below.

CANDIDATE (new): TP1 = 1.0R (entry +/- 1.0x the SL distance itself, i.e.
+/- 1.5xATR). This tests whether the MAE/MFE-diagnostic-suggested
improvement (backtested at +0.190R net vs baseline's +0.149R net, on the
same 2025 data) survives on genuinely unseen forward data.

CRITICAL DESIGN CONSTRAINT: both configurations must receive EXACTLY the
same underlying signal -- same entry price, same SL, same timestamp, same
market read. This is enforced by computing the market state (indicators,
decision, entry, SL) exactly ONCE per cycle in _fetch_and_decide(), then
branching into baseline and candidate logging from that single shared
result. Two independently-timed API fetches could return slightly
different prices and would violate the "same signal, different exit only"
requirement -- this is an A/B exit test, not two different strategies.

Each configuration has its OWN isolated risk tracking (separately
namespaced) and its OWN independent continuation-tracking (since the
candidate's wider TP1 takes longer to resolve, blocking new candidate
entries for longer than baseline entries are blocked) -- but both share
the identical entry/SL/risk-per-trade/spread-cost model, so the only
variable being tested is genuinely just the exit.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from . import db
from . import risk_controller
from . import scalp_analysis
from . import telegram_bot
from .historical_backtest import find_outcome_detailed, fetch_full_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xau_swing")

API_SYMBOL = "XAU/USD"

# Baseline -- UNCHANGED tags/config from the original forward test
SYMBOL_TAG = "Gold (XAU/USD) [4h-swing]"
STRATEGY_TYPE = "trend_4h_swing"
RISK_NAMESPACE_PREFIX = "xau_swing"

# Candidate -- new, fully isolated from baseline
CANDIDATE_SYMBOL_TAG = "Gold (XAU/USD) [4h-swing-candidate-1.0R]"
CANDIDATE_STRATEGY_TYPE = "trend_4h_swing_candidate_1_0r"
CANDIDATE_RISK_NAMESPACE_PREFIX = "xau_swing_candidate10r"

MIN_TRADES_FOR_COMPARISON = 20  # per your own instruction: don't declare a winner on a small sample


def _today_namespaced_key(prefix: str = RISK_NAMESPACE_PREFIX) -> str:
    return f"{prefix}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


async def _fetch_and_decide():
    """The SINGLE source of truth for 'what did the market look like this
    cycle' -- fetches once, decides once. Both the baseline signal and the
    candidate signal are derived from this same call's result, so they can
    never drift apart due to timing. Returns None if data was insufficient."""
    tf_data, dfs = {}, {}
    for label, interval in scalp_analysis.TIMEFRAMES.items():
        df = scalp_analysis.fetch_klines(API_SYMBOL, interval)
        if df is None or len(df) < 205:
            logger.warning("Insufficient data for XAU/USD 4h-swing check (%s)", label)
            return None
        tf_data[label] = scalp_analysis.compute_indicators(df)
        dfs[label] = df
        time.sleep(scalp_analysis.REQUEST_DELAY_SECONDS)

    divergence = scalp_analysis.detect_rsi_divergence(dfs["15m"])
    news_direction, news_summary = scalp_analysis.get_news_bias(["Gold", "XAU"])
    decision = scalp_analysis.score_and_decide(tf_data, news_direction)

    return {
        "tf_data": tf_data, "decision": decision, "divergence": divergence,
        "news_direction": news_direction, "news_summary": news_summary,
    }


def _log_no_trade_evaluation(symbol_tag: str, strategy_type: str, state: dict) -> None:
    tf_data, decision, news_direction = state["tf_data"], state["decision"], state["news_direction"]
    db.save_evaluation(
        source="live", symbol=symbol_tag, strategy_type=strategy_type, action=decision["action"],
        confidence=decision["confidence"], entry=None, sl=None, tp1=None, tp2=None,
        details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction}",
    )


def _try_open_signal(symbol_tag: str, strategy_type: str, risk_namespace_prefix: str,
                      state: dict, entry_price: float, sl: float, tp1: float, tp2: float,
                      risk_pct: float, label: str) -> str:
    """Shared logic for checking continuation, checking risk, and logging
    a signal -- used identically by both baseline and candidate, so
    neither gets special-cased handling. Returns a short status string for
    the combined Telegram message."""
    tf_data, decision, divergence, news_direction = state["tf_data"], state["decision"], state["divergence"], state["news_direction"]

    open_signal = db.get_open_signal(symbol_tag, strategy_type)
    if open_signal is not None and open_signal["action"] == decision["action"]:
        logger.info("%s: still the same open trade from before, not a new entry", label)
        return f"↻ {label}: still the same open trade, not a new entry"

    risk_usd = config.ACCOUNT_SIZE_USD * (risk_pct / 100)
    allowed, reason = risk_controller.check_and_reserve(risk_usd, date_key=_today_namespaced_key(risk_namespace_prefix))

    db.save_scalp_signal(
        symbol=symbol_tag, action=decision["action"], entry=entry_price, sl=sl, tp1=tp1, tp2=tp2,
        confidence=decision["confidence"],
        details=f"4h={tf_data['4h']['trend']} 1h={tf_data['1h']['trend']} 15m={tf_data['15m']['trend']} 5m={tf_data['5m']['trend']} news={news_direction} divergence={divergence}",
        strategy_type=strategy_type, risk_allowed=allowed,
    )

    status_emoji = "✅" if allowed else "⛔"
    status_text = "taken" if allowed else f"SKIPPED -- {reason}"
    price_risk_per_unit = abs(entry_price - sl)
    units = risk_usd / price_risk_per_unit if price_risk_per_unit else 0
    size_str = f"{units:.2f} oz (risking ${risk_usd:.2f})" if price_risk_per_unit else "N/A"
    return f"{status_emoji} {label}: {status_text}, entry {entry_price:.2f} SL {sl:.2f} TP1 {tp1:.2f} | {size_str}"


async def analyze_and_signal():
    """BASELINE ONLY -- unchanged behavior from before this A/B test
    existed. Kept for anyone still calling it directly; the scheduled
    workflow now calls analyze_and_signal_ab() instead, which reproduces
    this exact same baseline logic plus the candidate, from one shared
    fetch."""
    state = await _fetch_and_decide()
    if state is None:
        return

    _log_no_trade_evaluation(SYMBOL_TAG, STRATEGY_TYPE, state)

    decision = state["decision"]
    if decision["action"] == "NO TRADE":
        logger.info("XAU 4h-swing: no signal this cycle")
        return

    tf_data = state["tf_data"]
    entry_price = tf_data["5m"]["close"]
    atr_4h = tf_data[config.XAU_SWING_ATR_TIMEFRAME]["atr"]
    sl = entry_price - 1.5 * atr_4h if decision["action"] == "LONG" else entry_price + 1.5 * atr_4h
    tp1 = entry_price + 1.0 * atr_4h if decision["action"] == "LONG" else entry_price - 1.0 * atr_4h
    tp2 = entry_price + 2.0 * atr_4h if decision["action"] == "LONG" else entry_price - 2.0 * atr_4h

    status_line = _try_open_signal(
        SYMBOL_TAG, STRATEGY_TYPE, RISK_NAMESPACE_PREFIX, state,
        entry_price, sl, tp1, tp2, config.XAU_SWING_RISK_PCT, "Baseline (TP1=0.667R)",
    )
    if status_line.startswith("↻"):
        return

    message = (
        f"*🥇 XAU/USD 4H-Swing Forward Test*\n\n{status_line}\n\n"
        f"4H={tf_data['4h']['trend']} 1H={tf_data['1h']['trend']} 15M={tf_data['15m']['trend']} 5M={tf_data['5m']['trend']}\n"
        f"News: {state['news_summary']}\n\n"
        f"_This is the validated 4H-ATR configuration, at {config.XAU_SWING_RISK_PCT:.2f}% risk. "
        f"Forward test only -- backtest validation, not a guarantee._"
    )
    await telegram_bot.send_text(message)
    logger.info("Sent XAU 4h-swing baseline signal: %s", decision["action"])


async def analyze_and_signal_ab():
    """Fetches and decides ONCE, then opens (or continues) BOTH the
    baseline and candidate positions from that single shared result --
    this is what the scheduled workflow actually calls. Reproduces
    baseline's exact original logic exactly, plus the isolated candidate."""
    state = await _fetch_and_decide()
    if state is None:
        return

    _log_no_trade_evaluation(SYMBOL_TAG, STRATEGY_TYPE, state)
    _log_no_trade_evaluation(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE, state)

    decision = state["decision"]
    if decision["action"] == "NO TRADE":
        logger.info("XAU 4h-swing A/B: no signal this cycle")
        return

    tf_data = state["tf_data"]
    entry_price = tf_data["5m"]["close"]
    atr_4h = tf_data[config.XAU_SWING_ATR_TIMEFRAME]["atr"]
    sl_distance = 1.5 * atr_4h
    sl = entry_price - sl_distance if decision["action"] == "LONG" else entry_price + sl_distance

    # Baseline: TP1 = 0.667R (entry +/- 1.0xATR)
    baseline_tp1 = entry_price + 1.0 * atr_4h if decision["action"] == "LONG" else entry_price - 1.0 * atr_4h
    baseline_tp2 = entry_price + 2.0 * atr_4h if decision["action"] == "LONG" else entry_price - 2.0 * atr_4h

    # Candidate: TP1 = 1.0R (entry +/- the SAME distance as SL) -- the ONLY
    # difference from baseline. Same entry, same SL, same everything else.
    candidate_tp1 = entry_price + sl_distance if decision["action"] == "LONG" else entry_price - sl_distance
    candidate_tp2 = entry_price + 2 * sl_distance if decision["action"] == "LONG" else entry_price - 2 * sl_distance

    baseline_status = _try_open_signal(
        SYMBOL_TAG, STRATEGY_TYPE, RISK_NAMESPACE_PREFIX, state,
        entry_price, sl, baseline_tp1, baseline_tp2, config.XAU_SWING_RISK_PCT, "Baseline (TP1=0.667R)",
    )
    candidate_status = _try_open_signal(
        CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE, CANDIDATE_RISK_NAMESPACE_PREFIX, state,
        entry_price, sl, candidate_tp1, candidate_tp2, config.XAU_SWING_RISK_PCT, "Candidate (TP1=1.0R)",
    )

    if baseline_status.startswith("↻") and candidate_status.startswith("↻"):
        return  # both just continuing -- nothing new to report

    message = (
        f"*🥇 XAU/USD 4H-Swing A/B Forward Test*\n\n"
        f"{baseline_status}\n{candidate_status}\n\n"
        f"4H={tf_data['4h']['trend']} 1H={tf_data['1h']['trend']} 15M={tf_data['15m']['trend']} 5M={tf_data['5m']['trend']}\n"
        f"News: {state['news_summary']}\n\n"
        f"_Same entry/SL for both -- only TP1 differs. Neither result should be trusted until "
        f"at least {MIN_TRADES_FOR_COMPARISON} resolved trades exist per side._"
    )
    await telegram_bot.send_text(message)
    logger.info("Sent XAU 4h-swing A/B signal: %s", decision["action"])


async def evaluate_outcomes(symbol_tag: str = SYMBOL_TAG, strategy_type: str = STRATEGY_TYPE,
                             risk_namespace_prefix: str = RISK_NAMESPACE_PREFIX):
    """Evaluates unresolved signals for ONE configuration (baseline by
    default; pass the candidate's tags to evaluate it instead). Reuses
    find_outcome_detailed directly -- same evaluation logic as the
    validated backtest, for both configurations equally."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT id, action, entry, sl, tp1, tp2, created_at
               FROM scalp_signals
               WHERE symbol = ? AND strategy_type = ? AND action != 'NO TRADE'
                 AND id NOT IN (SELECT signal_id FROM signal_outcomes)
               ORDER BY created_at ASC""",
            (symbol_tag, strategy_type),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("No unresolved signals to evaluate for %s / %s", symbol_tag, strategy_type)
        return

    for signal_id, action, entry, sl, tp1, tp2, created_at in rows:
        created_dt = datetime.fromisoformat(created_at)
        if created_dt.tzinfo is not None:
            # Normalize to naive UTC -- matches the naive-datetime convention
            # Twelve Data's own fetched dataframes use everywhere else in
            # this project. Comparing a tz-aware created_dt directly against
            # find_outcome_detailed's naive df_5m datetime column crashes
            # unconditionally with "Cannot compare tz-naive and tz-aware
            # datetime-like objects" -- confirmed happening on every call.
            created_dt = created_dt.replace(tzinfo=None)
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        age_hours = (now_naive - created_dt).total_seconds() / 3600
        window_fully_elapsed = age_hours >= config.XAU_SWING_LOOKAHEAD_HOURS

        df_5m = fetch_full_history(API_SYMBOL, "5min")
        if df_5m is None:
            continue

        detail = find_outcome_detailed(df_5m, created_dt, entry, sl, tp1, tp2, action, lookahead_hours=config.XAU_SWING_LOOKAHEAD_HOURS)

        if detail["outcome"] == "EXPIRED" and not window_fully_elapsed:
            continue

        db.save_signal_outcome(
            signal_id, detail["outcome"], detail["exit_price"], detail["r_multiple"],
            mae_r=detail["mae_before_tp1_r"], mfe_r=detail["mfe_before_tp1_r"],
        )
        logger.info("%s signal #%d: %s, R=%.2f", symbol_tag, signal_id, detail["outcome"], detail["r_multiple"])

        risk_usd = config.ACCOUNT_SIZE_USD * (config.XAU_SWING_RISK_PCT / 100)
        signal_date_key = f"{risk_namespace_prefix}:{created_dt.strftime('%Y-%m-%d')}"
        risk_controller.record_outcome(detail["r_multiple"], risk_usd, date_key=signal_date_key)


async def evaluate_all_outcomes():
    """Evaluates BOTH configurations -- what the scheduled evaluation
    workflow calls."""
    await evaluate_outcomes(SYMBOL_TAG, STRATEGY_TYPE, RISK_NAMESPACE_PREFIX)
    await evaluate_outcomes(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE, CANDIDATE_RISK_NAMESPACE_PREFIX)


def get_forward_test_summary(risk_namespace_prefix: str = RISK_NAMESPACE_PREFIX) -> dict:
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT date, cumulative_usd, trades_taken FROM daily_risk_state
               WHERE date LIKE ? ORDER BY date ASC""",
            (f"{risk_namespace_prefix}:%",),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"days": 0, "total_pnl_usd": 0.0, "max_drawdown_usd": 0.0, "trades_taken": 0}

    cum, peak, max_dd = 0.0, 0.0, 0.0
    total_trades = 0
    for date_key, daily_pnl, trades_taken in rows:
        cum += daily_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        total_trades += trades_taken

    return {
        "days": len(rows), "total_pnl_usd": cum, "max_drawdown_usd": max_dd,
        "trades_taken": total_trades, "first_day": rows[0][0].split(":")[1], "last_day": rows[-1][0].split(":")[1],
    }


def get_resolved_trade_stats(symbol_tag: str, strategy_type: str) -> dict:
    """Pulls resolved-trade-level stats (win rate, avg R, PF, avg MAE/MFE,
    avg holding time, max consecutive losses) for ONE configuration,
    directly from signal_outcomes -- used for the A/B comparison report."""
    conn = db._connect()
    try:
        rows = conn.execute(
            """SELECT o.outcome, o.r_multiple, o.mae_r, o.mfe_r, s.created_at, o.evaluated_at
               FROM signal_outcomes o
               JOIN scalp_signals s ON s.id = o.signal_id
               WHERE s.symbol = ? AND s.strategy_type = ?
               ORDER BY s.created_at ASC""",
            (symbol_tag, strategy_type),
        ).fetchall()
    finally:
        conn.close()

    resolved = [r for r in rows if r[0] in ("WIN", "LOSS")]
    n = len(resolved)
    if n == 0:
        return {"n": 0}

    wins = sum(1 for r in resolved if r[0] == "WIN")
    losses = n - wins
    r_seq = [r[1] for r in resolved]
    avg_r = sum(r_seq) / n
    gains = sum(r for r in r_seq if r > 0)
    loss_sum = abs(sum(r for r in r_seq if r < 0))
    pf = gains / loss_sum if loss_sum > 0 else None

    mae_vals = [r[2] for r in resolved if r[2] is not None]
    mfe_vals = [r[3] for r in resolved if r[3] is not None]
    avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else None
    avg_mfe = sum(mfe_vals) / len(mfe_vals) if mfe_vals else None

    holding_hours = []
    for r in resolved:
        try:
            created = datetime.fromisoformat(r[4])
            evaluated = datetime.fromisoformat(r[5])
            holding_hours.append((evaluated - created).total_seconds() / 3600)
        except Exception:
            continue
    avg_holding_hours = sum(holding_hours) / len(holding_hours) if holding_hours else None

    cum, peak, max_dd = 0.0, 0.0, 0.0
    max_streak, streak = 0, 0
    for outcome, r in [(r[0], r[1]) for r in resolved]:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        if outcome == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "n": n, "wins": wins, "losses": losses, "win_rate": wins / n,
        "avg_r": avg_r, "profit_factor": pf, "cumulative_r": cum, "max_drawdown_r": max_dd,
        "max_consecutive_losses": max_streak, "avg_mae_r": avg_mae, "avg_mfe_r": avg_mfe,
        "avg_holding_hours": avg_holding_hours,
    }


async def send_ab_comparison_status():
    """The main comparison report: baseline vs candidate, side by side,
    plus the delta -- gated by MIN_TRADES_FOR_COMPARISON before offering
    any comparative read, per the explicit decision-rule requirement."""
    baseline = get_resolved_trade_stats(SYMBOL_TAG, STRATEGY_TYPE)
    candidate = get_resolved_trade_stats(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)
    baseline_pnl = get_forward_test_summary(RISK_NAMESPACE_PREFIX)
    candidate_pnl = get_forward_test_summary(CANDIDATE_RISK_NAMESPACE_PREFIX)

    lines = ["*🥇 XAU 4H-Swing A/B Forward Test: Baseline vs Candidate*\n"]

    def _fmt(stats: dict, pnl: dict, label: str) -> list[str]:
        out = [f"*{label}*"]
        if stats.get("n", 0) == 0:
            out.append("  No resolved trades yet")
            return out
        pf_str = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "N/A"
        out.append(f"  n={stats['n']} ({stats['wins']}W/{stats['losses']}L, {stats['win_rate']*100:.0f}% WR)")
        out.append(f"  avg R {stats['avg_r']:+.3f}, PF {pf_str}, cumulative {stats['cumulative_r']:+.2f}R")
        out.append(f"  max DD {stats['max_drawdown_r']:.2f}R, longest losing streak {stats['max_consecutive_losses']}")
        if stats["avg_mae_r"] is not None:
            mfe_part = f", avg MFE {stats['avg_mfe_r']:.2f}R" if stats["avg_mfe_r"] is not None else ""
            out.append(f"  avg MAE {stats['avg_mae_r']:.2f}R{mfe_part}")
        if stats["avg_holding_hours"] is not None:
            out.append(f"  avg holding time {stats['avg_holding_hours']:.1f}h")
        out.append(f"  simulated P&L: ${pnl['total_pnl_usd']:+.2f} (${pnl['max_drawdown_usd']:.2f} max DD)")
        return out

    lines += _fmt(baseline, baseline_pnl, "Baseline (TP1=0.667R)")
    lines.append("")
    lines += _fmt(candidate, candidate_pnl, "Candidate (TP1=1.0R)")
    lines.append("")

    min_n = min(baseline.get("n", 0), candidate.get("n", 0))
    if min_n < MIN_TRADES_FOR_COMPARISON:
        lines.append(
            f"*⚠️ Not enough data for a comparison yet* ({min_n}/{MIN_TRADES_FOR_COMPARISON} minimum resolved trades on the smaller side). "
            f"No verdict should be drawn from either side's numbers until this threshold is reached."
        )
    else:
        delta_r = candidate["avg_r"] - baseline["avg_r"]
        delta_dd = candidate["max_drawdown_r"] - baseline["max_drawdown_r"]
        lines.append(f"*Delta (candidate - baseline): {delta_r:+.3f}R avg, drawdown {delta_dd:+.2f}R*")
        if delta_r > 0 and delta_dd <= baseline["max_drawdown_r"] * 0.5:
            lines.append("_Candidate is outperforming without materially worse drawdown -- a legitimate candidate to keep monitoring, not yet a decision to switch._")
        elif delta_r <= 0:
            lines.append("_The backtested advantage has NOT persisted on forward data -- per the pre-declared decision rule, keep the baseline._")
        else:
            lines.append("_Candidate shows a higher return but with meaningfully worse drawdown -- risk-adjusted, this is not a clear win. Keep monitoring._")

    lines.append("\n_Per the pre-declared protocol: neither configuration is being modified during this test. No 0.8R/0.9R/1.1R variants are being explored mid-experiment._")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent A/B comparison status")


async def send_economic_analysis(risk_levels: list[float] | None = None):
    """Once enough forward data exists for the candidate, shows expected
    daily/weekly/monthly return at several risk levels, using the REAL
    measured forward avg_r -- factually, without recommending any
    particular level, per the explicit instruction not to select a risk
    size merely to reach a target."""
    risk_levels = risk_levels or [0.25, 0.50, 0.75, 1.00]
    candidate = get_resolved_trade_stats(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)

    if candidate.get("n", 0) < MIN_TRADES_FOR_COMPARISON:
        await telegram_bot.send_text(
            f"*📊 Economic Analysis: XAU 4H-Swing Candidate*\n\n"
            f"Only {candidate.get('n', 0)} resolved trades so far -- fewer than the "
            f"{MIN_TRADES_FOR_COMPARISON}-trade minimum before this analysis is meaningful."
        )
        return

    forward_test_summary = get_forward_test_summary(CANDIDATE_RISK_NAMESPACE_PREFIX)
    days_elapsed = max(forward_test_summary["days"], 1)
    trades_per_day = candidate["n"] / days_elapsed

    lines = [
        "*📊 Economic Analysis: XAU 4H-Swing Candidate (forward data)*",
        f"_Measured: {candidate['n']} resolved trades over {days_elapsed} day(s) ({trades_per_day:.2f}/day), "
        f"avg R {candidate['avg_r']:+.3f} (gross, not yet cost-adjusted against real forward spread)._\n",
    ]
    for risk_pct in risk_levels:
        expected_daily_r = trades_per_day * candidate["avg_r"]
        expected_daily_pct = expected_daily_r * risk_pct
        expected_weekly_pct = expected_daily_pct * 5
        expected_monthly_pct = ((1 + expected_daily_pct / 100) ** 21 - 1) * 100
        lines.append(
            f"{risk_pct:.2f}% risk/trade: ~{expected_daily_pct:.3f}%/day, "
            f"~{expected_weekly_pct:.2f}%/week, ~{expected_monthly_pct:.2f}%/month (compounded)"
        )
    lines.append(
        "\n_This is a factual projection of the measured edge at each risk size, not a recommendation. "
        "Risk should be increased only after sustained forward evidence, never to reach a target._"
    )
    await telegram_bot.send_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(analyze_and_signal_ab())
