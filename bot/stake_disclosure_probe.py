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

SEC EDGAR daily index files are used here -- a simpler, decades-stable
plain-text format, switched to after two failed attempts against SEC's
full-text search API (see check_daily_index's docstring for details).
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


def sanitize_for_telegram(text: str) -> str:
    """Strips every Telegram Markdown special character from arbitrary,
    unpredictable external text (e.g. a raw API response) before it is
    ever embedded in a message."""
    return re.sub(r"[*_`\[\]]", "", text)


def _send_telegram_direct(text: str) -> bool:
    """Self-contained send with genuinely verifiable success/failure."""
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


def check_daily_index(check_date: datetime) -> dict:
    """METHOD CHANGE, applied after two failed attempts against SEC's
    full-text search API (efts.sec.gov) -- that endpoint returned
    well-formed, error-free responses with genuinely zero hits both
    times, most likely because it is a text-search engine that requires
    specific query syntax that could not be verified without live
    testing access (blocked from the development sandbox). Rather than
    keep guessing at that API's exact requirements, this switches to
    SEC's daily index files -- a much simpler, decades-stable plain-text
    format listing every filing submitted on a given day, with no query-
    syntax ambiguity at all."""
    quarter = (check_date.month - 1) // 3 + 1
    date_str = check_date.strftime("%Y%m%d")
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{check_date.year}/QTR{quarter}/form.{date_str}.idx"

    try:
        r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
        if r.status_code != 200:
            return {"available": False, "error": f"HTTP {r.status_code}", "date": date_str}
        content = r.text
    except Exception as e:
        return {"available": False, "error": str(e), "date": date_str}

    lines = content.splitlines()
    form4_count = sum(1 for l in lines if l.startswith("4 "))
    sc13d_count = sum(1 for l in lines if l.startswith("SC 13D"))
    sc13g_count = sum(1 for l in lines if l.startswith("SC 13G"))

    # DIAGNOSTIC, added after a second real run showed working Form 4
    # counts but genuinely zero SC 13D/13G -- rather than guess a third
    # parsing fix blind, capture a few RAW lines (a data-row sample, not
    # the header) so the actual live format can be inspected directly
    # instead of assumed.
    data_rows = [l for l in lines if l and not l.startswith("-") and not l.startswith("Form Type") and not l.startswith("Description") and not l.startswith("Last Data") and not l.startswith("Comments")]
    raw_sample_lines = data_rows[5:10] if len(data_rows) > 10 else data_rows[:5]

    return {
        "available": True, "date": date_str, "total_lines": len(lines),
        "form4_count": form4_count, "sc13d_count": sc13d_count, "sc13g_count": sc13g_count,
        "raw_sample_lines": raw_sample_lines,
    }


async def run():
    lines = [
        "*SEC 13D/13G Individual Stake Disclosure Probe (feasibility check ONLY) -- daily index method*",
        "No strategy built, no backtest run, production bot untouched. WEAKER evidentiary grounding than the "
        "insider-cluster or Congress studies -- based on one anecdote (GoPro), not established literature.\n",
    ]

    # Extended from 5 to 20 filing days, and up to 35 calendar-day
    # attempts to find them -- added after a 5-day window showed zero
    # 13D/13G, to distinguish "genuinely rare, needs a longer window"
    # from "still zero even with a much bigger sample."
    results = []
    check_date = datetime.now()
    checked = 0
    attempts = 0
    while checked < 20 and attempts < 35:
        attempts += 1
        result = check_daily_index(check_date - timedelta(days=attempts))
        if result["available"]:
            results.append(result)
            checked += 1

    if not results:
        lines.append("FAILED -- could not retrieve any daily index files.")
        lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT*")
    else:
        total_13d = sum(r["sc13d_count"] for r in results)
        total_13g = sum(r["sc13g_count"] for r in results)
        total_form4 = sum(r["form4_count"] for r in results)
        lines.append(f"Checked {len(results)} recent filing days (showing 5 most recent):")
        for r in results[:5]:
            lines.append(f"  {r['date']}: {r['total_lines']} total filings, {r['sc13d_count']} SC 13D, {r['sc13g_count']} SC 13G, {r['form4_count']} Form 4 (sanity reference)")
        lines.append(f"\nTotals across all {len(results)} days -- SC 13D: {total_13d}, SC 13G: {total_13g}, Form 4: {total_form4}")

        if total_13d + total_13g == 0:
            lines.append(
                "\nStill zero 13D/13G filings found via the daily-index method. Form 4 counts above are clearly "
                "working (thousands/day, as expected), which rules out a broken connection or URL. This could "
                "mean either (a) a subtle text-matching issue specific to the SC 13D/SC 13G prefix, or (b) these "
                "filing types are genuinely much rarer day-to-day than assumed (they require crossing a 5% "
                "ownership threshold -- a much rarer event than routine Form 4 activity). Raw sample lines from "
                "the most recent day, to inspect the real format directly rather than guess again:"
            )
            for sample_line in results[0].get("raw_sample_lines", []):
                lines.append(f"  {sanitize_for_telegram(sample_line)}")
            lines.append("\n*CLASSIFICATION: NEEDS INVESTIGATION -- inspect the raw lines above before concluding DATA INSUFFICIENT.*")
        else:
            lines.append(
                "\nThis confirms the filing feed and form-type identification work correctly. Distinguishing "
                "individual filers from institutions, and small/micro-caps from large caps, still requires "
                "parsing each filing's actual content -- a meaningfully bigger next step than this check."
            )
            lines.append("\n*CLASSIFICATION: DATA SUFFICIENT (existence only) -- further filtering-feasibility work required before Phase 2.*")

    send_succeeded = _send_telegram_direct("\n".join(lines))
    if send_succeeded:
        logger.info("Sent stake disclosure probe report")
    else:
        logger.warning("Stake disclosure probe report FAILED to send -- see the error above.")


if __name__ == "__main__":
    asyncio.run(run())
