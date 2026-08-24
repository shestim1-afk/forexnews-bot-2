import os
from dotenv import load_dotenv

load_dotenv()

def _safe_float_env(key: str, default: str) -> float:
    """Like float(os.getenv(key, default)), but also falls back to the
    default when the env var is SET but empty -- GitHub Actions sets a
    referenced secret to an empty string (not "unset") if that secret
    doesn't exist in the repo yet, which would otherwise crash here."""
    val = os.getenv(key, "").strip()
    return float(val) if val else float(default)

# --- Secrets / required ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Optional ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")

# --- Tuning ---
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))
MIN_CONFIDENCE_TO_ALERT = float(os.getenv("MIN_CONFIDENCE_TO_ALERT", "0.7"))
MAX_ITEMS_PER_CYCLE = int(os.getenv("MAX_ITEMS_PER_CYCLE", "25"))
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
# Delay between AI classification calls, to stay under free-tier rate limits
# (e.g. Gemini free tier allows only ~10-15 requests/minute).
CLASSIFY_DELAY_SECONDS = float(os.getenv("CLASSIFY_DELAY_SECONDS", "4.5"))

# --- Position sizing (used by the scalp bot's risk engine) ---
# These are placeholders -- personalize to your actual account before
# trusting the position sizes shown in any signal.
ACCOUNT_SIZE_USD = float(os.getenv("ACCOUNT_SIZE_USD", "1000"))
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "1.0"))

# --- Daily risk controller (paper-trading simulation) ---
# These mirror the standard risk-management guardrails: a hard ceiling on
# how much of the account can be at risk in a single day, a break after too
# many losses in a row, and a cap on how many trades get taken per day --
# regardless of how many signals the bot generates. None of this executes
# real trades; it simulates what WOULD happen if these signals were traded
# with this risk framework, so the numbers are trustworthy before ever
# connecting this to a real account.
MAX_DAILY_RISK_PCT = float(os.getenv("MAX_DAILY_RISK_PCT", str(RISK_PCT_PER_TRADE * 3)))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))

# --- XAU/USD 4H-ATR swing forward test ---
# This is the ONE configuration in this project with a real, validated,
# cost-adjusted positive edge (gold trend-following, sized with 4H ATR
# instead of the tight 5m default). Deliberately isolated with its OWN
# risk setting, separate from RISK_PCT_PER_TRADE above -- starting
# cautiously at 0.25%, not the general default, per the forward-test plan:
# confirm live behavior matches the backtest before scaling risk up.
XAU_SWING_RISK_PCT = float(os.getenv("XAU_SWING_RISK_PCT", "0.25"))
XAU_SWING_ATR_TIMEFRAME = "4h"
XAU_SWING_LOOKAHEAD_HOURS = 168  # 7 days -- matches the validated backtest configuration

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seen.sqlite3")

# --- RSS feeds, grouped by category. Add/remove freely. ---
# URLs below were verified live via web search/fetch as of Aug 2026 where
# noted; sites do restructure their feeds occasionally, so if a collector
# logs "Feed unreachable or malformed" for one of these, check the site's
# own /rss or /feeds page for the current path.
RSS_FEEDS = {
    "forex_macro": [
        # ForexLive rebranded to "InvestingLive" in 2026 -- these are their
        # current official feeds (verified live).
        "https://investinglive.com/feed/news/",
        "https://investinglive.com/feed/centralbank/",  # dedicated central-bank feed, very useful
        "https://investinglive.com/feed/forexorders/",
        "https://www.fxstreet.com/rss/news",
    ],
    "geopolitics_war": [
        # Reuters discontinued its free public RSS feeds; BBC World News is
        # the official, reliable free replacement (verified live).
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",  # verified live
    ],
    "energy": [
        "https://oilprice.com/rss/main",
        "https://www.eia.gov/rss/todayinenergy.xml",
    ],
    "central_banks": [
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.ecb.europa.eu/rss/press.html",
        "https://www.bankofengland.co.uk/rss/news",
        "https://www.boj.or.jp/en/rss/whatsnew.xml",
        # InvestingLive's dedicated central-bank feed above is a good
        # cross-check/backup if any official bank feed goes stale.
    ],
}

# --- Truth Social (free, official-ish archive) ---
# Trump posts primarily on Truth Social, not X, in his second term -- to the
# point that Truth Social now sells a *paid* low-latency feed of his posts
# to trading firms (confirms these posts are genuinely market-moving).
# trumpstruth.org is a free, actively-maintained public archive with a real
# RSS feed -- far more reliable than Nitter for this specific account.
TRUTH_SOCIAL_RSS = "https://www.trumpstruth.org/feed"

# --- Forex Factory economic calendar (free, official public feed) ---
# Forex Factory itself publishes this JSON feed for third-party tools (the
# same one countless MetaTrader news indicators use) -- no login, no
# scraping. Community-documented rate limit is ~2 requests/5min; this bot
# makes 1 request per 5-minute cycle, well within that.
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# --- Notable X/Twitter accounts to track via free Nitter/RSS-Bridge sources
# (see twitter_collector.py). Usernames without @.
# NOTE: realDonaldTrump is deliberately omitted here -- his posts are
# covered more reliably by the dedicated truthsocial_collector.py above,
# since he posts primarily on Truth Social, not X, in his second term. ---
TWITTER_WATCHLIST = [
    "elonmusk",
    "SpeakerPelosi",
    "POTUS",
    "federalreserve",
    "ecb",
]

# Unofficial Nitter mirrors -- free, no signup, but public instances rotate
# and go down often. The collector tries each in order and skips silently
# if all are unreachable. Check https://github.com/zedeus/nitter/wiki/Instances
# periodically and update this list, since dead instances are common.
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
]

# Optional: URL of a self-hosted RSS-Bridge instance (free, more stable than
# public Nitter). Run with e.g.:
#   docker run -d -p 3000:80 rssbridge/rss-bridge
# then set RSSBRIDGE_URL=http://localhost:3000 in .env. Leave blank to skip.
RSSBRIDGE_URL = os.getenv("RSSBRIDGE_URL", "")

# --- Asset universe the classifier is allowed to tag ---
ASSET_UNIVERSE = {
    "forex_pairs": [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
        "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "USD/CNH",
    ],
    "metals": ["Gold (XAU/USD)", "Silver (XAG/USD)"],
    "crypto": ["BTC", "ETH", "Crypto (broad market)"],
    "stocks_sectors": [
        "Energy sector", "Defense sector", "Tech sector (broad)",
        "Individual stock (name it)", "US indices (S&P500/Nasdaq/Dow)",
    ],
}
