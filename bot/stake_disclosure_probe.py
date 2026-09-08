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
from datetime import datetime, timedelta

import requests

from . import telegram_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stake_disclosure_probe")

SEC_USER_AGENT = "Research Probe research-probe@example.com"
EDGAR_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"


def check_edgar_fulltext_search() -> dict:
    """Checks whether SEC's full-text search API can return recent
    13D/13G filings with real filer names and dates -- the piece needed
    to eventually filter for individual (not institutional) filers."""
    try:
        r = requests.get(
            EDGAR_FULLTEXT_SEARCH_URL,
            params={"q": "beneficial ownership", "forms": "SC 13D,SC 13G", "dateRange": "custom",
                    "startdt": (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
                    "enddt": datetime.now().strftime("%Y-%m-%d")},
            headers={"User-Agent": SEC_USER_AGENT}, timeout=20,
        )
        if r.status_code != 200:
            return {"available": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        data = r.json()
    except Exception as e:
        return {"available": False, "error": str(e)}

    hits = data.get("hits", {}).get("hits", [])
    return {"available": True, "n_recent_filings": len(hits), "sample": hits[:3]}


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
    else:
        lines.append(f"Connected successfully. Recent 13D/13G filings found (last 14 days): {result['n_recent_filings']}")
        lines.append(
            "\nNOTE: this confirms the filing feed itself is reachable, but does NOT yet confirm we can "
            "reliably distinguish individual filers from institutions, or small/micro-caps from large caps, "
            "at scale -- that requires parsing each filing's actual content, a meaningfully bigger next step "
            "than this existence check."
        )
        lines.append("\n*CLASSIFICATION: DATA SUFFICIENT (existence only) -- further filtering-feasibility work required before Phase 2.*")

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent stake disclosure probe report")


if __name__ == "__main__":
    asyncio.run(run())
