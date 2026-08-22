"""Daily risk controller: simulates what WOULD happen if live signals were
actually traded with a real risk framework -- fixed risk per trade, a hard
daily loss ceiling, a break after too many losses in a row, and a cap on
trades per day. This does NOT execute real trades. It's paper-trading
bookkeeping, built specifically so the numbers (would this have breached a
daily limit? what's the simulated running P&L today?) are trustworthy
before ever connecting this system to a real account.

Philosophy, stated plainly: don't design around a daily profit target --
markets don't pay a fixed salary, and forcing trades to hit one is a fast
way to destroy a working edge. A day with zero valid setups should show
$0, not a forced trade. This module's job is purely to enforce the ceiling,
never to manufacture activity to reach a target.
"""

import logging
from datetime import datetime, timezone

from . import config
from . import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("risk_controller")


def today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_daily_state(date_key: str | None = None) -> dict:
    date_key = date_key or today_key()
    conn = db._connect()
    try:
        conn.execute("INSERT OR IGNORE INTO daily_risk_state (date) VALUES (?)", (date_key,))
        conn.commit()
        row = conn.execute(
            """SELECT date, trades_taken, consecutive_losses, cumulative_r, cumulative_usd,
                      reserved_risk_usd, kill_switch_triggered, kill_switch_reason
               FROM daily_risk_state WHERE date = ?""",
            (date_key,),
        ).fetchone()
    finally:
        conn.close()
    cols = ["date", "trades_taken", "consecutive_losses", "cumulative_r", "cumulative_usd",
            "reserved_risk_usd", "kill_switch_triggered", "kill_switch_reason"]
    state = dict(zip(cols, row))
    state["kill_switch_triggered"] = bool(state["kill_switch_triggered"])
    return state


def _save_state(state: dict) -> None:
    conn = db._connect()
    try:
        conn.execute(
            """UPDATE daily_risk_state SET trades_taken=?, consecutive_losses=?, cumulative_r=?,
               cumulative_usd=?, reserved_risk_usd=?, kill_switch_triggered=?, kill_switch_reason=?
               WHERE date=?""",
            (state["trades_taken"], state["consecutive_losses"], state["cumulative_r"],
             state["cumulative_usd"], state["reserved_risk_usd"], int(state["kill_switch_triggered"]),
             state["kill_switch_reason"], state["date"]),
        )
        conn.commit()
    finally:
        conn.close()


def check_and_reserve(risk_usd: float, date_key: str | None = None) -> tuple[bool, str]:
    """Call this when a new actionable signal is generated, BEFORE treating
    it as "would be taken". Returns (allowed, reason). If allowed, this
    trade's risk is reserved against the daily budget so subsequent signals
    the same day see an accurately reduced remaining budget -- release it
    later via record_outcome() once the trade resolves."""
    date_key = date_key or today_key()
    state = get_daily_state(date_key)

    if state["kill_switch_triggered"]:
        return False, f"kill switch active today: {state['kill_switch_reason']}"
    if state["trades_taken"] >= config.MAX_TRADES_PER_DAY:
        return False, f"max trades/day reached ({config.MAX_TRADES_PER_DAY})"
    if state["consecutive_losses"] >= config.MAX_CONSECUTIVE_LOSSES:
        return False, f"max consecutive losses reached ({config.MAX_CONSECUTIVE_LOSSES})"

    max_daily_risk_usd = config.ACCOUNT_SIZE_USD * (config.MAX_DAILY_RISK_PCT / 100)
    if state["reserved_risk_usd"] + risk_usd > max_daily_risk_usd:
        return False, f"daily risk budget would be exceeded (limit ${max_daily_risk_usd:.2f})"

    state["trades_taken"] += 1
    state["reserved_risk_usd"] += risk_usd
    _save_state(state)
    return True, "ok"


def record_outcome(r_multiple: float, risk_usd: float, date_key: str | None = None) -> dict:
    """Call once a previously-reserved trade's real outcome (WIN/LOSS/EXPIRED)
    is known. Releases its reserved risk, updates cumulative P&L and the
    consecutive-loss streak, and trips the kill switch if either the
    consecutive-loss limit or the daily loss ceiling is breached. Returns
    the updated state."""
    date_key = date_key or today_key()
    state = get_daily_state(date_key)

    realized_usd = r_multiple * risk_usd
    state["cumulative_r"] += r_multiple
    state["cumulative_usd"] += realized_usd
    state["reserved_risk_usd"] = max(0.0, state["reserved_risk_usd"] - risk_usd)

    if r_multiple < 0:
        state["consecutive_losses"] += 1
    else:
        state["consecutive_losses"] = 0

    max_daily_risk_usd = config.ACCOUNT_SIZE_USD * (config.MAX_DAILY_RISK_PCT / 100)
    if not state["kill_switch_triggered"]:
        if state["consecutive_losses"] >= config.MAX_CONSECUTIVE_LOSSES:
            state["kill_switch_triggered"] = True
            state["kill_switch_reason"] = f"{state['consecutive_losses']} consecutive losses"
            logger.warning("Kill switch triggered for %s: %s consecutive losses", date_key, state["consecutive_losses"])
        elif state["cumulative_usd"] <= -max_daily_risk_usd:
            state["kill_switch_triggered"] = True
            state["kill_switch_reason"] = f"daily loss limit reached (${state['cumulative_usd']:.2f})"
            logger.warning("Kill switch triggered for %s: daily loss limit reached", date_key)

    _save_state(state)
    return state


def format_status(date_key: str | None = None) -> str:
    date_key = date_key or today_key()
    state = get_daily_state(date_key)
    max_daily_risk_usd = config.ACCOUNT_SIZE_USD * (config.MAX_DAILY_RISK_PCT / 100)

    lines = [f"*🛡️ Daily Risk Status -- {date_key}*\n"]
    pnl_emoji = "🟢" if state["cumulative_usd"] > 0 else ("🔴" if state["cumulative_usd"] < 0 else "⚪")
    lines.append(f"{pnl_emoji} Simulated P&L: ${state['cumulative_usd']:+.2f} ({state['cumulative_r']:+.2f}R)")
    lines.append(f"Trades taken: {state['trades_taken']}/{config.MAX_TRADES_PER_DAY}")
    lines.append(f"Consecutive losses: {state['consecutive_losses']}/{config.MAX_CONSECUTIVE_LOSSES}")
    lines.append(f"Risk used: ${state['reserved_risk_usd'] + max(0, -state['cumulative_usd']):.2f} / ${max_daily_risk_usd:.2f} daily limit")
    if state["kill_switch_triggered"]:
        lines.append(f"\n⛔ *Kill switch ACTIVE*: {state['kill_switch_reason']}")
        lines.append("No further trades would be taken today under this framework.")
    else:
        lines.append("\n✅ Within all limits -- new signals would still be taken.")
    lines.append("\n_Paper-trading simulation only -- no real trades are placed. This tracks what WOULD happen under these risk limits, not a guarantee._")
    return "\n".join(lines)
