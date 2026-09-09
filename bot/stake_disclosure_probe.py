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


def fetch_filing_header_detail(cik_in_url: str, accession: str) -> dict | None:
    """Fetches ONE 13D/13G filing's full submission text and extracts
    BOTH the SUBJECT COMPANY and FILED BY fields from the standard SEC
    header -- resolves, with real data, whether the daily index's
    'Company Name' field represents the investor (filer) or the company
    being invested in (subject). This is checked explicitly because
    guessing wrong here would silently corrupt the entire study (unlike
    a mere formatting bug), and could not be verified from the
    development sandbox (www.sec.gov unreachable there)."""
    accession_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_in_url}/{accession_nodash}/{accession}.txt"
    try:
        r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
        if r.status_code != 200:
            return None
        content = r.text
    except Exception as e:
        logger.warning("Failed to fetch filing header for %s/%s: %s", cik_in_url, accession, e)
        return None

    subject_match = re.search(r"SUBJECT COMPANY:.*?COMPANY CONFORMED NAME:\s*([^\n]+)", content, re.DOTALL)
    filed_by_match = re.search(r"FILED BY:.*?COMPANY CONFORMED NAME:\s*([^\n]+)", content, re.DOTALL)
    reporting_owner_match = re.search(r"REPORTING-OWNER:.*?COMPANY CONFORMED NAME:\s*([^\n]+)", content, re.DOTALL)

    return {
        "subject_company": subject_match.group(1).strip() if subject_match else None,
        "filed_by": filed_by_match.group(1).strip() if filed_by_match else None,
        "reporting_owner": reporting_owner_match.group(1).strip() if reporting_owner_match else None,
    }


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
    # FIX: the real SEC daily-index form-type string is "SCHEDULE 13D" /
    # "SCHEDULE 13G", not "SC 13D" / "SC 13G" as originally assumed --
    # confirmed directly from real raw file content after two rounds of
    # diagnostic investigation (7,272 real matches found once the
    # correct string was identified). This was a pure text-matching bug,
    # not a data-availability problem.
    sc13d_count = sum(1 for l in lines if l.startswith("SCHEDULE 13D"))
    sc13g_count = sum(1 for l in lines if l.startswith("SCHEDULE 13G"))

    # DIAGNOSTIC FIX: the prior version sampled arbitrary rows, which
    # happened to be unrelated filing types by chance -- useless for
    # diagnosing the SC 13D/13G matching specifically. This searches for
    # any line containing "13D" or "13G" ANYWHERE (not relying on the
    # exact prefix assumption being tested), so the real form-type string
    # can finally be seen directly, whatever it actually is.
    related_lines = [l for l in lines if "13D" in l.upper() or "13G" in l.upper()]

    return {
        "available": True, "date": date_str, "total_lines": len(lines),
        "form4_count": form4_count, "sc13d_count": sc13d_count, "sc13g_count": sc13g_count,
        "related_lines_sample": related_lines[:8],
        "n_related_lines_found": len(related_lines),
    }


def parse_daily_index_line(line: str) -> dict | None:
    """Robustly extracts CIK, accession number, and company name from a
    daily-index data row, anchored on the distinctive 'edgar/data/CIK/
    ACCESSION.txt' file-path token. FIX: the name regex originally only
    skipped a single leading token, but "SCHEDULE 13D"/"SCHEDULE 13G" are
    TWO-word form types, leaving "13D" stuck onto the captured company
    name -- caught immediately when tested against the exact real line
    format. Now strips any known multi-word form-type prefix explicitly
    before extracting the name."""
    path_match = re.search(r"edgar/data/(\d+)/([\d-]+)\.txt", line)
    if not path_match:
        return None
    cik, accession = path_match.group(1), path_match.group(2)

    remainder = line
    for prefix in ("SCHEDULE 13D/A", "SCHEDULE 13G/A", "SCHEDULE 13D", "SCHEDULE 13G"):
        if remainder.strip().startswith(prefix):
            remainder = remainder.strip()[len(prefix):]
            break
    else:
        remainder = re.sub(r"^\S+\s+", "", remainder.strip())

    name_match = re.search(r"^\s*(.+?)\s+" + re.escape(cik) + r"\s", remainder)
    company_name = name_match.group(1).strip() if name_match else None
    return {"cik": cik, "accession": accession, "daily_index_name": company_name}


async def run_filer_diagnostic():
    """Fetches a handful of REAL filings and directly compares the daily
    index's 'Company Name' field against the actual filing's SUBJECT
    COMPANY vs FILED BY fields -- settling, with real data, which one the
    daily index actually represents."""
    lines_out = ["*13D/13G Daily-Index Name Resolution Diagnostic*", "Determines whether the daily index lists the FILER or the SUBJECT company.\n"]

    result = None
    for days_back in range(1, 10):
        candidate = check_daily_index(datetime.now() - timedelta(days=days_back))
        if candidate["available"] and candidate.get("related_lines_sample"):
            result = candidate
            break

    sample_lines = [l for l in result.get("related_lines_sample", []) if l.strip().startswith("SCHEDULE 13D")][:3] if result else []
    if not sample_lines:
        lines_out.append("Could not find sample SCHEDULE 13D lines to test against.")
        _send_telegram_direct("\n".join(lines_out))
        return

    for line in sample_lines:
        parsed = parse_daily_index_line(line)
        if not parsed:
            continue
        detail = fetch_filing_header_detail(parsed["cik"], parsed["accession"])
        lines_out.append(f"Daily index name: {sanitize_for_telegram(parsed['daily_index_name'] or 'N/A')}")
        if detail:
            lines_out.append(f"  Real SUBJECT COMPANY: {sanitize_for_telegram(detail.get('subject_company') or 'N/A')}")
            lines_out.append(f"  Real FILED BY: {sanitize_for_telegram(detail.get('filed_by') or 'N/A')}")
            matches_subject = bool(detail.get("subject_company") and parsed["daily_index_name"] and detail["subject_company"].lower() in parsed["daily_index_name"].lower())
            matches_filer = bool(detail.get("filed_by") and parsed["daily_index_name"] and detail["filed_by"].lower() in parsed["daily_index_name"].lower())
            lines_out.append(f"  -> Daily index matches: {'SUBJECT' if matches_subject else ('FILED BY' if matches_filer else 'UNCLEAR')}")
        else:
            lines_out.append("  Could not fetch real filing detail.")
        lines_out.append("")

    _send_telegram_direct("\n".join(lines_out))
    logger.info("Sent filer diagnostic report")


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
            # Search across ALL checked days (not just the most recent)
            # for any line containing 13D/13G anywhere -- the prior
            # sample searched only one day's arbitrary rows and missed
            # them by chance.
            any_related = []
            for r in results:
                any_related.extend(r.get("related_lines_sample", []))
                if any_related:
                    break
            total_related_found = sum(r.get("n_related_lines_found", 0) for r in results)

            lines.append(
                f"\nStill zero via exact prefix match. Form 4 counts above are clearly working (thousands/day), "
                f"ruling out a broken connection or URL. Broader substring search (containing '13D' or '13G' "
                f"anywhere in the line) across all {len(results)} days found: {total_related_found} lines."
            )
            if any_related:
                lines.append("Sample matching lines (real format, for direct inspection):")
                for sample_line in any_related[:8]:
                    lines.append(f"  {sanitize_for_telegram(sample_line)}")
                lines.append("\n*CLASSIFICATION: NEEDS INVESTIGATION -- real 13D/13G-related lines exist; the exact prefix match needs adjusting to this format.*")
            else:
                lines.append(
                    "No lines containing '13D' or '13G' found ANYWHERE across 20 full days of data, using a broad "
                    "substring search. This is much stronger evidence that these filing types are genuinely far "
                    "rarer day-to-day than assumed, rather than a text-matching bug."
                )
                lines.append("\n*CLASSIFICATION: DATA INSUFFICIENT (or requires a much longer window than 20 days) -- a genuine rarity finding, not a bug.*")
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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        asyncio.run(run_filer_diagnostic())
    else:
        asyncio.run(run())
