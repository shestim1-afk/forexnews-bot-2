"""On-demand Telegram command handler. Checks for new messages containing
commands like /btc, /gold, /gbpjpy, or /help, and replies with a fresh
analysis for that specific symbol using the same engine as the hourly digest.

This is NOT truly instant -- it works by periodically checking Telegram for
new messages (polling), since a true instant-reply webhook setup needs a
persistent server or more fragile free-tier workarounds. Run this on a
frequent schedule (e.g. every 3-5 minutes) via its own GitHub Actions
workflow, and you'll get a reply within roughly that window of sending a
command.
"""

import asyncio
import logging
import os
import requests

from . import config
from . import db
from . import scalp_analysis
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
    "/help -- this message\n\n"
    "_Replies arrive within a few minutes, not instantly -- this checks for new messages periodically rather than listening live._"
)


def format_stats_message() -> str:
    stats = db.get_outcome_stats()
    if not stats:
        return "No signals have been evaluated yet. Signals need to be at least 4 hours old before they're checked -- check back once some have had time to play out."

    lines = ["*📊 Win/Loss Record So Far*\n"]
    for s in stats:
        if s["win_rate"] is not None:
            lines.append(
                f"*{s['symbol']}*: {s['wins']}W / {s['losses']}L "
                f"({s['win_rate']*100:.0f}% win rate), avg R: {s['avg_r']:+.2f}"
                + (f", {s['expired']} expired" if s["expired"] else "")
            )
        else:
            lines.append(f"*{s['symbol']}*: no resolved signals yet" + (f" ({s['expired']} expired)" if s["expired"] else ""))

    lines.append("")
    lines.append("_TP1 vs SL, whichever hit first in 15m candles. Treat with appropriate skepticism until many signals have accumulated._")
    return "\n".join(lines)


def get_updates(offset: int | None) -> list[dict]:
    if not config.TELEGRAM_BOT_TOKEN:
        return []
    try:
        params = {"timeout": 0}
        if offset is not None:
            params["offset"] = offset
        r = requests.get(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates",
            params=params, timeout=15,
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

    api_symbol = scalp_analysis.COMMAND_TO_SYMBOL.get(command)
    if not api_symbol:
        await reply_to(chat_id, f"Unknown command `{command}`. Send /help for the list.")
        return

    symbol_cfg = next(s for s in scalp_analysis.SYMBOLS if s["api"] == api_symbol)
    await reply_to(chat_id, f"Running fresh analysis for {symbol_cfg['display']}, one moment...")

    block, signal_data, _ = scalp_analysis.analyze_symbol(symbol_cfg)
    if signal_data:
        db.save_scalp_signal(
            symbol=signal_data["symbol"], action=signal_data["action"],
            entry=signal_data["entry"], sl=signal_data["sl"], tp1=signal_data["tp1"], tp2=signal_data["tp2"],
            confidence=signal_data["confidence"], details=signal_data["details"] + " (on-demand)",
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
