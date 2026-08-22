"""On-demand Telegram command handler. Checks for new messages containing
commands like /btc, /gold, /gbpjpy, /stats, or /help, and replies with a
fresh analysis using the same engine as the scheduled scalp scan.

Supports two modes:
- Short-poll (timeout=0, the default): used by the old GitHub Actions
  periodic-check workflow. Replies arrive within a few minutes.
- Long-poll (timeout=25): used by bot/railway_main.py's persistent process.
  Replies arrive within seconds, since the request stays open waiting for
  a new message rather than checking briefly on a schedule.
"""

import asyncio
import logging
import os
import requests

from . import config
from . import db
from . import scalp_analysis
from . import backtest_signals
from . import risk_controller
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telegram_commands")

HELP_TEXT = (
    "*Available commands:*\n"
    "/btc -- fresh BTC/USD scan\n"
    "/gold or /xauusd -- fresh Gold scan\n"
    "/gbpjpy -- fresh GBP/JPY scan\n"
    "/stats -- current win/loss record so far\n"
    "/riskstatus -- today's simulated risk/P&L status\n"
    "/help -- this message\n\n"
    "_Replies arrive within a few minutes, not instantly -- this checks for new messages periodically rather than listening live._"
)


def format_stats_message() -> str:
    return backtest_signals.format_full_stats_message()


def get_updates(offset: int | None, timeout: int = 0) -> list[dict]:
    """timeout=0 (default) is a quick, non-blocking check -- used by the
    old GitHub Actions periodic-check workflow. timeout=25 makes this a
    genuine Telegram long-poll call (the request blocks server-side until
    a new message arrives or 25s elapses) -- only viable from a persistent
    process like Railway, since GitHub Actions jobs can't stay open that
    way. The HTTP client timeout must exceed Telegram's long-poll timeout,
    or our own request would time out first."""
    if not config.TELEGRAM_BOT_TOKEN:
        return []
    try:
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        r = requests.get(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates",
            params=params, timeout=timeout + 10,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            logger.warning("getUpdates failed: %s", data)
            return []
        return data.get("result", [])
    except Exception as e:
        logger.warning("Failed to poll Telegram for updates: %s", e)
        return []


def get_last_update_id() -> int | None:
    conn = db._connect()
    try:
        cur = conn.execute("SELECT value FROM telegram_state WHERE key = 'last_update_id'")
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def save_last_update_id(update_id: int) -> None:
    conn = db._connect()
    try:
        conn.execute(
            "INSERT INTO telegram_state (key, value) VALUES ('last_update_id', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(update_id),),
        )
        conn.commit()
    finally:
        conn.close()


async def reply_to(chat_id: str, text: str) -> None:
    try:
        bot = telegram_bot._get_bot()
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception as e:
        logger.error("Failed to reply to chat %s: %s", chat_id, e)


async def handle_command(chat_id: str, command: str) -> None:
    command = command.lower().split("@")[0]  # strip @botname if present

    if command == "/help" or command == "/start":
        await reply_to(chat_id, HELP_TEXT)
        return

    if command == "/stats":
        await reply_to(chat_id, format_stats_message())
        return

    if command == "/riskstatus":
        await reply_to(chat_id, risk_controller.format_status())
        return

    api_symbol = scalp_analysis.COMMAND_TO_SYMBOL.get(command)
    if not api_symbol:
        await reply_to(chat_id, f"Unknown command `{command}`. Send /help for the list.")
        return

    symbol_cfg = next(s for s in scalp_analysis.SYMBOLS if s["api"] == api_symbol)
    await reply_to(chat_id, f"Running fresh analysis for {symbol_cfg['display']}, one moment...")

    block, actionable_signals, _ = scalp_analysis.analyze_symbol(symbol_cfg)
    for sig in actionable_signals:
        db.save_scalp_signal(
            symbol=sig["symbol"], action=sig["action"], entry=sig["entry"], sl=sig["sl"],
            tp1=sig["tp1"], tp2=sig["tp2"], confidence=sig["confidence"],
            details=sig["details"] + " (on-demand)", strategy_type=sig["strategy_type"],
        )
    await reply_to(chat_id, block)


async def run():
    last_id = get_last_update_id()
    offset = last_id + 1 if last_id is not None else None
    updates = get_updates(offset)

    if not updates:
        logger.info("No new Telegram messages")
        return

    for update in updates:
        update_id = update["update_id"]
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = str(message.get("chat", {}).get("id", ""))

        if text.startswith("/") and chat_id:
            logger.info("Handling command '%s' from chat %s", text, chat_id)
            await handle_command(chat_id, text)

        save_last_update_id(update_id)


if __name__ == "__main__":
    asyncio.run(run())
