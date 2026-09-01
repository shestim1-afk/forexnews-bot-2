"""XAU Forward-Test Checkpoint Readiness Monitor.

MONITORING/NOTIFICATION ONLY. This module does not touch bot/xau_swing.py,
signal generation, entries, stops, targets, position sizing, or risk. It
does not run the checkpoint analysis itself and does not declare either
strategy validated. It only counts genuinely RESOLVED trades for each
configuration and sends exactly one Telegram notification the first time
both cross the 30-trade threshold simultaneously.

Persistence: a small, self-contained table in the SAME shared SQLite
database everything else already uses (created here, not in bot/db.py,
to keep this feature fully isolated). Survives restarts because the
database itself is already persisted via the existing GitHub Actions
cache mechanism every other workflow relies on.
"""

import asyncio
import logging
from datetime import datetime, timezone

import requests

from . import db
from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("xau_checkpoint_monitor")

# Local constants -- not imported from xau_swing.py, same defensive
# pattern already established elsewhere in this project.
SYMBOL_TAG = "Gold (XAU/USD) [4h-swing]"
STRATEGY_TYPE = "trend_4h_swing"
CANDIDATE_SYMBOL_TAG = "Gold (XAU/USD) [4h-swing-candidate-1.0R]"
CANDIDATE_STRATEGY_TYPE = "trend_4h_swing_candidate_1_0r"

CHECKPOINT_NAME = "xau_30_trade_checkpoint"
THRESHOLD = 30


def _ensure_table():
    conn = db._connect()
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS checkpoint_notifications (
                checkpoint_name TEXT PRIMARY KEY,
                notified INTEGER NOT NULL DEFAULT 0,
                notified_at TEXT
            )"""
        )
        conn.commit()
    finally:
        conn.close()


def get_resolved_count(symbol_tag: str, strategy_type: str) -> int:
    conn = db._connect()
    try:
        row = conn.execute(
            """SELECT COUNT(*) FROM signal_outcomes o
               JOIN scalp_signals s ON s.id = o.signal_id
               WHERE s.symbol = ? AND s.strategy_type = ? AND o.outcome IN ('WIN', 'LOSS')""",
            (symbol_tag, strategy_type),
        ).fetchone()
        return row[0]
    finally:
        conn.close()


def get_open_count(symbol_tag: str, strategy_type: str) -> int:
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


def is_notified(checkpoint_name: str = CHECKPOINT_NAME) -> bool:
    _ensure_table()
    conn = db._connect()
    try:
        row = conn.execute(
            "SELECT notified FROM checkpoint_notifications WHERE checkpoint_name = ?",
            (checkpoint_name,),
        ).fetchone()
        return bool(row[0]) if row else False
    finally:
        conn.close()


def mark_notified(checkpoint_name: str = CHECKPOINT_NAME):
    """Only ever called AFTER a Telegram send has succeeded -- a failed
    send must never reach this, so a retry on the next scheduled run
    remains possible."""
    _ensure_table()
    conn = db._connect()
    try:
        conn.execute(
            """INSERT INTO checkpoint_notifications (checkpoint_name, notified, notified_at)
               VALUES (?, 1, ?)
               ON CONFLICT(checkpoint_name) DO UPDATE SET notified=1, notified_at=excluded.notified_at""",
            (checkpoint_name, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _send_telegram_direct(text: str) -> bool:
    """Self-contained Telegram send with GENUINELY verifiable success/
    failure -- does NOT depend on bot/telegram_bot.py's send_text, whose
    exact error-handling behavior could not be confirmed (production
    logs show it catches and logs Telegram API errors internally, which
    would make its return value indistinguishable between success and
    failure if reused here). Checking the raw HTTP response directly
    is the only way to make the safety-critical notification-state
    transition (mark_notified) trustworthy."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Cannot send: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("ok") is True:
                return True
        logger.error("Telegram send failed: HTTP %d, response: %s", r.status_code, r.text[:300])
        return False
    except Exception as e:
        logger.error("Telegram send raised an exception: %s", e)
        return False


async def check_and_notify():
    """The only function this feature's scheduled workflow calls. Reads
    resolved counts, compares against the frozen 30-trade threshold,
    and sends exactly one notification on the NOT READY -> READY
    transition. Never opens/closes trades, never touches xau_swing.py,
    never runs the full checkpoint analysis itself."""
    baseline_resolved = get_resolved_count(SYMBOL_TAG, STRATEGY_TYPE)
    candidate_resolved = get_resolved_count(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)
    baseline_open = get_open_count(SYMBOL_TAG, STRATEGY_TYPE)
    candidate_open = get_open_count(CANDIDATE_SYMBOL_TAG, CANDIDATE_STRATEGY_TYPE)

    ready = baseline_resolved >= THRESHOLD and candidate_resolved >= THRESHOLD
    already_notified = is_notified()

    if not ready:
        baseline_needed = max(0, THRESHOLD - baseline_resolved)
        candidate_needed = max(0, THRESHOLD - candidate_resolved)
        logger.info(
            "Checkpoint NOT READY -- baseline %d/%d (%d more needed), candidate %d/%d (%d more needed)",
            baseline_resolved, THRESHOLD, baseline_needed, candidate_resolved, THRESHOLD, candidate_needed,
        )
        return

    if already_notified:
        logger.info(
            "Checkpoint READY (baseline %d, candidate %d) but already notified previously -- no duplicate message sent.",
            baseline_resolved, candidate_resolved,
        )
        return

    # Transition NOT READY -> READY, and not yet notified: send exactly once.
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = (
        "*XAU FORWARD CHECKPOINT READY*\n\n"
        f"Baseline: {baseline_resolved} resolved trades ({baseline_open} still open)\n"
        f"1.0R candidate: {candidate_resolved} resolved trades ({candidate_open} still open)\n\n"
        f"Both variants have crossed the {THRESHOLD}-trade minimum, as of {now_str}.\n\n"
        "This means READY TO ANALYZE -- it does NOT mean either strategy is validated, profitable, "
        "or that risk should change. No strategy or risk changes should be made before running the "
        "full checkpoint analysis.\n\n"
        "Run the XAU checkpoint audit now."
    )

    try:
        send_succeeded = _send_telegram_direct(message)
    except Exception as e:
        logger.error("Unexpected error while attempting to send checkpoint-ready notification: %s", e)
        send_succeeded = False

    if send_succeeded:
        mark_notified()
        logger.info("Checkpoint-ready notification sent successfully and marked notified.")
    else:
        logger.warning("Checkpoint-ready notification send FAILED -- state NOT marked notified, will retry next run.")


BASELINE_PULSE_CHECKPOINT_NAME = "xau_baseline_pulse_15"
BASELINE_PULSE_THRESHOLD = 15


async def check_and_notify_baseline_pulse(spread_pct: float | None = None):
    """A DELIBERATELY narrower, faster, lower-bar check than the real
    30-trade checkpoint: does the baseline's forward performance look
    roughly consistent with its historical +0.403 R/day, or does it look
    like something has changed? This does NOT compare against the
    candidate, does NOT constitute a checkpoint, and is explicitly
    labeled PRELIMINARY every time it fires -- it answers a narrower
    question (has the baseline degraded) sooner, precisely because that
    question doesn't require candidate parity to be meaningful. The real
    30-trade checkpoint (check_and_notify) is unaffected and still the
    only thing that triggers a full baseline-vs-candidate decision."""
    from .analytics import get_default_spread_pct
    from .xau_forward_audit import get_forward_trades, compute_full_stats, HISTORICAL_BASELINE_NET_R_DAY

    spread_pct = spread_pct if spread_pct is not None else get_default_spread_pct("XAU/USD")
    baseline_resolved = get_resolved_count(SYMBOL_TAG, STRATEGY_TYPE)

    if baseline_resolved < BASELINE_PULSE_THRESHOLD:
        logger.info("Baseline pulse NOT READY -- %d/%d (%d more needed)",
                    baseline_resolved, BASELINE_PULSE_THRESHOLD, BASELINE_PULSE_THRESHOLD - baseline_resolved)
        return

    if is_notified(BASELINE_PULSE_CHECKPOINT_NAME):
        logger.info("Baseline pulse threshold reached (%d) but already notified previously -- no duplicate.", baseline_resolved)
        return

    trades = get_forward_trades(SYMBOL_TAG, STRATEGY_TYPE)
    stats = compute_full_stats(trades, spread_pct)
    net_r_day = stats.get("net_r_day")

    if net_r_day is None:
        direction_note = "Could not compute a net R/day figure -- insufficient underlying data despite the trade count."
    elif net_r_day >= HISTORICAL_BASELINE_NET_R_DAY * 0.5:
        direction_note = "Roughly consistent with the historical figure so far -- not evidence of degradation, but still far too small a sample to confirm the edge is intact."
    elif net_r_day > 0:
        direction_note = "Positive, but meaningfully below the historical figure -- worth watching, not yet evidence of a real problem at this sample size."
    else:
        direction_note = "Currently negative -- worth watching closely, but n is still small enough that this could easily reverse with a few more trades."

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    net_r_day_str = f"{net_r_day:+.3f}" if net_r_day is not None else "N/A"
    pf_str = f"{stats['profit_factor']:.2f}" if stats.get("profit_factor") is not None else "N/A"
    wr_str = f"{stats['win_rate']*100:.0f}%" if stats.get("win_rate") is not None else "N/A"

    message = (
        "*XAU BASELINE PRELIMINARY PULSE CHECK*\n\n"
        f"PRELIMINARY -- baseline has reached {baseline_resolved} resolved trades ({BASELINE_PULSE_THRESHOLD}-trade "
        f"early-look threshold), as of {now_str}.\n\n"
        "This is NOT the real 30-trade checkpoint and does NOT include the candidate. It exists only to give an "
        "early, low-confidence read on whether the baseline still looks broadly consistent with its historical "
        "performance -- nothing here should inform any strategy or risk decision.\n\n"
        f"Forward net R/day: {net_r_day_str} (historical: {HISTORICAL_BASELINE_NET_R_DAY:+.3f})\n"
        f"Win rate: {wr_str}, Profit factor: {pf_str}\n\n"
        f"{direction_note}\n\n"
        "This message will not repeat. The real checkpoint notification arrives separately once both baseline "
        "and candidate reach 30 resolved trades."
    )

    try:
        send_succeeded = _send_telegram_direct(message)
    except Exception as e:
        logger.error("Unexpected error while attempting to send baseline pulse notification: %s", e)
        send_succeeded = False

    if send_succeeded:
        mark_notified(BASELINE_PULSE_CHECKPOINT_NAME)
        logger.info("Baseline pulse notification sent successfully and marked notified.")
    else:
        logger.warning("Baseline pulse notification send FAILED -- state NOT marked notified, will retry next run.")


async def run_all_checks():
    """Runs both the real 30-trade checkpoint check and the narrower,
    faster baseline-only pulse check -- called by the scheduled
    workflow's entry point."""
    await check_and_notify()
    await check_and_notify_baseline_pulse()


if __name__ == "__main__":
    asyncio.run(run_all_checks())
