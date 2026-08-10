"""Entry point for running the bot as a single check (one cycle, then exit),
instead of an infinite loop -- built for GitHub Actions, which handles the
'run every 5 minutes' scheduling itself rather than the bot looping forever.

Usage: python -m bot.run_once
"""

import asyncio
import logging

from .db import count_seen
from .main import run_cycle  # reuses the exact same cycle logic as main.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_once")


async def main():
    first_run = count_seen() == 0
    if first_run:
        logger.info("No history found (first run, or cache was empty) -- this check will file away the current backlog without classifying it, to avoid wasting API quota on old news.")
    await run_cycle(skip_classification=first_run)


if __name__ == "__main__":
    asyncio.run(main())
