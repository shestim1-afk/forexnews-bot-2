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
