"""Entry point for running this bot as ONE continuously-running process on
Railway, instead of the separate GitHub Actions scheduled workflows.

What this changes vs. the GitHub Actions setup:
- All jobs (news, scalp scan, daily analysis, backtest) run on their own
  schedule inside one persistent process, using APScheduler.
- Telegram commands (/btc, /gold, etc.) now use real long-polling: the
  request sits open waiting for a new message for up to 25 seconds, so
  replies arrive within seconds of you sending a command, not "within a
  few minutes" like the old periodic-check version. This genuinely
  requires a persistent process -- it's not achievable from GitHub Actions'
  short-lived scheduled runs.

Required Railway setup (see README for full steps):
- All the same environment variables as the GitHub Secrets (TELEGRAM_BOT_TOKEN,
  TELEGRAM_CHAT_ID, GEMINI_API_KEY, TWELVEDATA_API_KEY, FRED_API_KEY optional)
  set as Railway "Variables" instead.
- A Railway Volume mounted at the `data/` folder path, so the SQLite
  database (dedup history, signal log, backtest outcomes) survives
  redeploys instead of resetting every time.
- IMPORTANT: once this is confirmed working, disable or delete the old
  GitHub Actions workflow files (.github/workflows/*.yml) -- otherwise
  both Railway and GitHub Actions will run the same jobs independently,
  doubling your API usage and sending every alert twice.
"""

import asyncio
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from . import config
from . import main as news_main
from . import scalp_analysis
from . import daily_analysis
from . import backtest_signals
from . import telegram_commands
from .db import count_seen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("railway_main")


class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP endpoint so Railway (and you, in a browser) can confirm
    the process is alive. Not part of the bot's actual functionality."""
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK - forexnews-bot is running")

    def log_message(self, format, *args):
        pass  # silence default per-request HTTP logging noise


def start_health_server():
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info("Health check server listening on port %d", port)
    server.serve_forever()


async def news_job():
    try:
        first_run = count_seen() == 0
        await news_main.run_cycle(skip_classification=first_run)
    except Exception as e:
        logger.error("News job crashed (will retry next interval): %s", e)


async def scalp_job():
    try:
        await scalp_analysis.run()
    except Exception as e:
        logger.error("Scalp job crashed (will retry next interval): %s", e)


async def daily_job():
    try:
        await daily_analysis.run()
    except Exception as e:
        logger.error("Daily analysis job crashed: %s", e)


async def backtest_job():
    try:
        await backtest_signals.run()
    except Exception as e:
        logger.error("Backtest job crashed: %s", e)


async def telegram_command_loop():
    """Runs forever, using Telegram long-polling for near-instant replies --
    the main practical benefit of moving to a persistent process."""
    logger.info("Starting Telegram command long-poll loop (near-instant replies)")
    while True:
        try:
            last_id = telegram_commands.get_last_update_id()
            offset = last_id + 1 if last_id is not None else None
            updates = telegram_commands.get_updates(offset, timeout=25)
            for update in updates:
                update_id = update["update_id"]
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = str(message.get("chat", {}).get("id", ""))
                if text.startswith("/") and chat_id:
                    logger.info("Handling command '%s' from chat %s", text, chat_id)
                    await telegram_commands.handle_command(chat_id, text)
                telegram_commands.save_last_update_id(update_id)
        except Exception as e:
            logger.error("Telegram command loop error (retrying in 5s): %s", e)
            await asyncio.sleep(5)


async def main():
    threading.Thread(target=start_health_server, daemon=True).start()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(news_job, "interval", seconds=config.POLL_INTERVAL_SECONDS, id="news")
    scheduler.add_job(scalp_job, "interval", minutes=30, id="scalp")
    scheduler.add_job(daily_job, "cron", hour=7, minute=0, id="daily_analysis")
    scheduler.add_job(backtest_job, "cron", hour=3, minute=17, id="backtest")
    scheduler.start()
    logger.info(
        "Scheduler started: news every %ds, scalp every 30min, daily analysis at 07:00 UTC, backtest at 03:17 UTC",
        config.POLL_INTERVAL_SECONDS,
    )

    # Kick off an immediate news check on startup rather than waiting a full
    # interval for the first one.
    asyncio.create_task(news_job())

    await telegram_command_loop()  # runs forever -- this is what keeps the process alive


if __name__ == "__main__":
    asyncio.run(main())
