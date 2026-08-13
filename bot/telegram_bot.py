import logging
from telegram import Bot
from telegram.constants import ParseMode

from . import config
from .db import NewsItem

logger = logging.getLogger(__name__)

_bot = None


def _get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    return _bot


DIRECTION_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def _format_message(item: NewsItem, classification: dict) -> str:
    lines = [f"*{_escape(item.title)}*", f"_{item.source} · {item.category}_", ""]

    for impact in classification.get("impacts", []):
        emoji = DIRECTION_EMOJI.get(impact.get("direction", "neutral"), "⚪")
        conf_pct = round(float(impact.get("confidence", 0)) * 100)
        lines.append(
            f"{emoji} *{_escape(impact.get('asset',''))}* — "
            f"{impact.get('direction','?').upper()} ({conf_pct}%)"
        )
        lines.append(f"    {_escape(impact.get('reason',''))}")

    if item.url:
        lines.append("")
        lines.append(f"[source]({item.url})")

    return "\n".join(lines)


def _escape(text: str) -> str:
    # Minimal escaping for Telegram MarkdownV1-safe output
    for ch in ["_", "*", "`", "["]:
        text = text.replace(ch, f"\\{ch}") if ch != "_" else text
    return text


async def send_alert(item: NewsItem, classification: dict) -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured -- printing alert to console instead:")
        print(_format_message(item, classification))
        return

    try:
        bot = _get_bot()
        await bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=_format_message(item, classification),
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=False,
        )
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)
async def send_text(text: str) -> None:
    """Sends a pre-formatted plain message (Markdown-enabled), for reports
    like the daily technical analysis digest that aren't tied to a single
    NewsItem/classification pair."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured -- printing message to console instead:")
        print(text)
        return

    try:
        bot = _get_bot()
        await bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)
async def send_text(text: str) -> None:
    """Sends a pre-formatted plain message (Markdown-enabled), for reports
    like the daily technical analysis digest that aren't tied to a single
    NewsItem/classification pair."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured -- printing message to console instead:")
        print(text)
        return

    try:
        bot = _get_bot()
        await bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)
