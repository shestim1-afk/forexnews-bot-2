import asyncio
import logging
import time

from . import config
from .db import is_new, mark_seen, count_seen
from .collectors import rss_collector, twitter_collector, fred_collector, truthsocial_collector, forexfactory_collector
from . import classifier
from . import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def collect_all():
    """Runs every collector, logging and continuing past individual failures
    so one dead source never takes down the whole cycle."""
    items = []
    for collector, name in [
        (rss_collector, "rss"),
        (twitter_collector, "twitter"),
        (forexfactory_collector, "forexfactory_calendar"),
        (truthsocial_collector, "truthsocial"),
        (fred_collector, "fred"),
    ]:
        try:
            fetched = collector.fetch_all()
            logger.info("Collector '%s' returned %d items", name, len(fetched))
            items.extend(fetched)
        except Exception as e:
            logger.error("Collector '%s' crashed entirely: %s", name, e)
    return items


async def run_cycle(skip_classification: bool = False):
    all_items = collect_all()
    new_items = [i for i in all_items if is_new(i)]
    logger.info("%d new items out of %d fetched this cycle", len(new_items), len(all_items))

    if skip_classification:
        # First-ever run: don't burn API quota classifying a huge backlog of
        # old news that was already published before the bot existed. Mark
        # it all as seen so future cycles only see genuinely new items.
        for item in new_items:
            mark_seen(item)
        logger.info(
            "First run: marked %d backlog items as seen without classifying them. "
            "From the next cycle onward, only genuinely new items will be classified.",
            len(new_items),
        )
        return

    # Cap how many we classify per cycle to control cost/rate limits
    new_items = new_items[: config.MAX_ITEMS_PER_CYCLE]

    for item in new_items:
        mark_seen(item)  # mark seen immediately so a crash mid-cycle doesn't reprocess forever

        result = classifier.classify(item)

        # Pace requests to stay under the AI provider's per-minute rate limit
        # (free tiers, e.g. Gemini, allow only ~10-15 requests/minute).
        await asyncio.sleep(config.CLASSIFY_DELAY_SECONDS)

        if not result.get("relevant"):
            continue

        max_conf = max((i.get("confidence", 0) for i in result.get("impacts", [])), default=0)
        if max_conf < config.MIN_CONFIDENCE_TO_ALERT:
            logger.info("Skipping low-confidence item: %s (%.2f)", item.title[:60], max_conf)
            continue

        await telegram_bot.send_alert(item, result)
        logger.info("Sent alert: %s", item.title[:80])


async def main_loop():
    logger.info("Starting macro news bot. Poll interval: %ds", config.POLL_INTERVAL_SECONDS)
    first_run = count_seen() == 0
    while True:
        start = time.time()
        try:
            await run_cycle(skip_classification=first_run)
            first_run = False  # only skip classification on the very first cycle ever
        except Exception as e:
            logger.error("Cycle crashed (bot will retry next interval): %s", e)

        elapsed = time.time() - start
        sleep_for = max(0, config.POLL_INTERVAL_SECONDS - elapsed)
        logger.info("Cycle done in %.1fs, sleeping %.1fs", elapsed, sleep_for)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    asyncio.run(main_loop())
