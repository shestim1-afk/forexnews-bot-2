"""SEC Schedule 13D/13G Individual Stake Disclosure DATA-AVAILABILITY PROBE.

Explicitly NOT a strategy, NOT a backtest. Tests whether SEC EDGAR's
real-time filing feed can reliably surface 13D/13G beneficial-ownership
disclosures (the mechanism that caught the Markiplier/GoPro case),
filtered to individual filers (not institutions) taking large stakes
(>=5%) in small/micro-cap companies.

IMPORTANT, stated honestly: this hypothesis has WEAKER evidentiary
grounding than the insider-cluster or Congress-trading studies. Those
were backed by real academic literature before any code was written.
This one is closer to an educated guess based on a single dramatic
anecdote (GoPro). Expect a noisier, less certain result even if the
data checks out.

SEC EDGAR full-text search / daily filing index is used here -- the
same official, unauthenticated public API family already confirmed
working in bot/insider_cluster.py and bot/congress_insider_probe.py.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta

import requests

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stake_disclosure_probe")

SEC_USER_AGENT = "Research Probe research-probe@example.com"
EDGAR_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"


def sanitize_for_telegram(text: str) -> str:
    """Strips every Telegram Markdown special character from arbitrary,
    unpredictable external text (e.g. a raw API response) before it is
    ever embedded in a message. FIX, applied after a real send failure:
    the raw SEC API response was embedded directly in a message and
    happened to contain characters (likely underscores/asterisks/
    brackets in JSON field names or values) that broke Telegram's
    parser -- confirmed by a genuine HTTP 400 in a live run, which the
    code then incorrectly logged as a success (see the second fix in
    run() below). Arbitrary external content should never be trusted to
    be Markdown-safe; strip the special characters entirely rather than
    trying to escape or wrap them, since nesting/escaping rules have
    already proven unreliable elsewhere in this project."""
    return re.sub(r"[*_`\[\]]", "", text)


def _send_telegram_direct(text: str) -> bool:
    """Self-contained send with genuinely verifiable success/failure --
    the same fix already applied in xau_checkpoint_monitor.py, needed
    here for the same reason: bot/telegram_bot.py's send_text catches
    and logs Telegram API errors internally, so its return value cannot
    reliably distinguish success from failure. A prior run here logged
    "Sent" immediately after the actual send had already failed with a
    real HTTP 400 -- this fixes that."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Cannot send: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=20,
        )
        if r.status_code == 200 and r.json().get("ok") is True:
            return True
        logger.error("Telegram send failed: HTTP %d, response: %s", r.status_code, r.text[:300])
        return False
    except Exception as e:
        logger.error("Telegram send raised an exception: %s", e)
        return False


def check_edgar_fulltext_search() -> dict:
    """Checks whether SEC's full-text search API can return recent
    13D/13G filings with real filer names and dates. FIX (applied after
    the first real run returned a suspicious 0 hits in 14 days -- 13D/13G
    filings happen constantly across the market, so a genuine two-week
    gap would be extraordinary): the original query required the literal
    phrase "beneficial ownership" to appear in the filing text, which was
    almost certainly too restrictive. This version filters by form type
    only, with no text requirement, and captures the raw response body
    when the hit count looks suspiciously low so a future run can
    actually diagnose the real cause instead of silently reporting a
    false DATA SUFFICIENT."""
    try:
        r = requests.get(
            EDGAR_FULLTEXT_SEARCH_URL,
            params={"forms": "SC 13D,SC 13G",
                    "startdt": (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
                    "enddt": datetime.now().strftime("%Y-%m-%d")},
            headers={"User-Agent": SEC_USER_AGENT}, timeout=20,
        )
        if r.status_code != 200:
            return {"available": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        data = r.json()
    except Exception as e:
        return {"available": False, "error": str(e)}

    hits = data.get("hits", {}).get("hits", [])
    total = data.get("hits", {}).get("total", {})
    n_hits = total.get("value", len(hits)) if isinstance(total, dict) else len(hits)

    result = {"available": True, "n_recent_filings": n_hits, "sample": hits[:3]}
    if n_hits < 10:
        # Suspiciously low for a real two-week market-wide window --
        # capture the raw response so the actual cause is visible,
        # rather than accepting this at face value.
        result["suspicious"] = True
        result["raw_response_sample"] = str(data)[:500]
    return result


async def run():
    lines = [
        "*SEC 13D/13G Individual Stake Disclosure Probe (feasibility check ONLY)*",
        "No strategy built, no backtest run, production bot untouched. WEAKER evidentiary grounding than the "
        "insider-cluster or Congress studies -- based on one anecdote (GoPro), not established literature. "
        "Expect a noisier result even if data is sufficient.\n",
    ]

    logger.info("Checking SEC EDGAR full-text search for 13D/13G filings...")
    result = check_edgar_fulltext_search()

    if not result["available"]:
        lines.append(f"FAILED: {result.get('error')}")
        lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT*")
    elif result.get("suspicious"):
        lines.append(f"Connected successfully, but only {result['n_recent_filings']} filings found in 14 days -- SUSPICIOUS.")
        lines.append(
            "13D/13G filings happen constantly across the market -- a genuine two-week gap this low would be "
            "extraordinary. This is very likely a query construction problem, not a real absence of filings."
        )
        lines.append(f"Raw response sample: {sanitize_for_telegram(result.get('raw_response_sample', 'N/A'))}")
        lines.append("\n*CLASSIFICATION: NEEDS INVESTIGATION -- do not treat as DATA SUFFICIENT until the query is confirmed correct.*")
    else:
        lines.append(f"Connected successfully. Recent 13D/13G filings found (last 14 days): {result['n_recent_filings']}")
        lines.append(
            "\nNOTE: this confirms the filing feed itself is reachable, but does NOT yet confirm we can "
            "reliably distinguish individual filers from institutions, or small/micro-caps from large caps, "
            "at scale -- that requires parsing each filing's actual content, a meaningfully bigger next step "
            "than this existence check."
        )
        lines.append("\n*CLASSIFICATION: DATA SUFFICIENT (existence only) -- further filtering-feasibility work required before Phase 2.*")

    send_succeeded = _send_telegram_direct("\n".join(lines))
    if send_succeeded:
        logger.info("Sent stake disclosure probe report")
    else:
        logger.warning("Stake disclosure probe report FAILED to send -- see the error above.")


if __name__ == "__main__":
    asyncio.run(run())
