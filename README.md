# Macro News → Trading Sentiment Telegram Bot

Aggregates news from forex/macro outlets, geopolitics/energy sources, central
banks, and (optionally) notable Twitter/X accounts, classifies each item's
market impact using Claude, and pushes bullish/bearish alerts to a Telegram
chat or channel.

## Free social media coverage (no paid API required)

X's free API tier no longer supports reading tweets at usable volume, and
scraping x.com directly gets IP-banned fast. This project uses free
alternatives instead:

- **Trump → Truth Social, not X.** In his second term Trump posts primarily
  on Truth Social (Truth Social even sells a *paid* low-latency feed of his
  posts to trading firms, which confirms these posts are genuinely
  market-moving). `collectors/truthsocial_collector.py` pulls from
  `trumpstruth.org/feed`, a free, actively-maintained public RSS archive —
  verified live and far more reliable than Nitter for this account.
- **Musk, Pelosi, and others → Nitter (free, unofficial).** `NITTER_INSTANCES`
  in `config.py` lists free RSS mirrors for X. These are unofficial and
  individual instances go down often, so the collector tries each in order
  and skips silently (with a log warning) if all are unreachable that cycle
  — it will never crash the bot.
- **Optional, more stable free upgrade: RSS-Bridge**, a self-hostable tool
  (`docker run -d -p 3000:80 rssbridge/rss-bridge`) that generates RSS for
  X accounts more reliably than public Nitter. Set `RSSBRIDGE_URL` in `.env`
  to use it — still $0, just needs a spare machine or container to run on.
- If you ever do want paid, more complete X coverage later: the official X
  API (Basic tier, paid) or a news API that re-publishes notable tweets
  (Benzinga, Finnhub) are the ToS-compliant routes — `TWITTER_BEARER_TOKEN`
  / `NEWSAPI_KEY` / `FINNHUB_KEY` are already wired into `config.py` for
  whenever that's worth it to you.

## Data sources wired up by default (all official RSS/APIs, no scraping)

| Category | Source | Feed type | Status |
|---|---|---|---|
| Forex/macro | InvestingLive (formerly ForexLive) — News | RSS | verified live, Aug 2026 |
| Forex/macro | InvestingLive — dedicated Central Banks feed | RSS | verified live, Aug 2026 |
| Forex/macro | InvestingLive — Forex Orders | RSS | verified live, Aug 2026 |
| Forex/macro | FXStreet | RSS | not re-verified this session |
| Geopolitics/war | BBC World News | RSS | verified live, Aug 2026 — used in place of Reuters, which discontinued its free public RSS |
| Geopolitics/war | Al Jazeera | RSS | verified live, Aug 2026 |
| Energy | OilPrice.com | RSS | not re-verified this session |
| Energy | EIA (U.S. Energy Info Admin) | RSS | not re-verified this session |
| Central bank | Federal Reserve press releases | RSS | Fed's feeds page confirms this category exists; exact filename not re-verified this session |
| Central bank | ECB press releases | RSS | not re-verified this session |
| Central bank | Bank of England news | RSS | not re-verified this session |
| Central bank | Bank of Japan (English) | RSS | not re-verified this session |
| Rates/macro data | FRED (St. Louis Fed) | API (free key) | official API, stable |
| Trump posts | trumpstruth.org (Truth Social archive) | RSS | verified live, Aug 2026, confirmed valid XML |

All URLs live in `bot/config.py` — swap or add feeds freely. Any site with an
RSS feed works with zero code changes. Rows marked "not re-verified this
session" were sourced from public feed directories rather than fetched
directly — if `rss_collector.py` logs a warning for one of these on first
run, check that site's own `/rss` or `/feeds` page for the current path (site
redesigns break specific feed URLs fairly often; the collector itself is
robust to individual feeds failing, it just skips and logs).

## Setup

```bash
cd forexnews-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python -m bot.main
```

### Required keys (.env)

- `TELEGRAM_BOT_TOKEN` — from @BotFather
- `TELEGRAM_CHAT_ID` — the chat/channel ID the bot posts to
- `GEMINI_API_KEY` — free, no credit card, no expiration. Get one at
  https://aistudio.google.com/apikey. This is the default classifier
  backend (see below for why).

### Optional keys

- `FRED_API_KEY` — free at https://fred.stlouisfed.org/docs/api/api_key.html
- `ANTHROPIC_API_KEY` — only needed if you switch `CLASSIFIER_BACKEND` to
  `anthropic` (see caveat below)
- `TWITTER_BEARER_TOKEN` — if you buy X API access
- `NEWSAPI_KEY` / `FINNHUB_KEY` — for third-party tweet aggregation

## AI classifier: which backend, and what it actually costs

Set `CLASSIFIER_BACKEND` in `.env` to one of:

- **`gemini` (default, recommended)** — Google's Gemini API is the only
  major LLM provider with a genuinely permanent free tier: no credit card,
  no expiration, roughly 1,500 requests/day on the free `gemini-2.5-flash-lite`
  model. That comfortably covers this bot even polling every 5 minutes.
- **`anthropic`** — better classification nuance in some cases, but new
  accounts only get a one-time ~$5 trial credit that does **not** renew.
  At continuous polling volume this typically runs out within days, after
  which real payment is required. Use this only if you're fine paying
  afterward.
- **`ollama`** — a fully free, locally-run open-source model, no daily
  request cap ever, but requires installing Ollama on your machine and
  accepts lower classification quality. More setup than Gemini, so only
  worth it if you want zero reliance on any external API at all.

## Running this 24/7 without keeping your computer on

Honest note: the free-hosting landscape got noticeably worse through 2026.
Fly.io dropped its free tier for new signups, Railway's free tier is now
just a small monthly credit rather than real always-on runtime, and
Render's free web services sleep when idle. There is currently no clean,
genuinely-free-forever host for a continuously-polling background bot like
this one without some compromise (manual daily renewal, sleep/wake cycles,
or a small ~$5/month cost for real always-on hosting). The simplest $0
option remains running it on your own computer while it's on.

## How it works

1. `main.py` runs a loop, polling every `POLL_INTERVAL_SECONDS` (default 300).
2. Each collector (`collectors/*.py`) fetches new items and returns a common
   `NewsItem` shape.
3. `db.py` (SQLite) dedupes by URL/hash so the same story isn't reprocessed.
4. `classifier.py` sends new items to Claude with a structured prompt asking
   for: affected assets (specific FX pairs / gold / specific stocks or
   sectors / crypto), direction (bullish/bearish/neutral per asset), a
   confidence score, and a one-line rationale — returned as JSON.
5. `telegram_bot.py` formats and sends a message for any item Claude scored
   above `MIN_CONFIDENCE_TO_ALERT`.

## Important caveats (read before trading on this)

- **This is a fast-digest tool, not alpha.** News sentiment often lags or
  mismatches actual price action, especially around Trump/Musk-style posts
  where markets react in seconds, irrationally, or the move is already priced
  in by the time your bot fires.
- Nothing here is financial advice. Treat outputs as one input among many.
- LLM classification can be wrong or overconfident — the confidence score is
  Claude's self-reported estimate, not a calibrated statistical probability.
- Add your own risk controls before connecting this to any execution system.
  This project intentionally does NOT place trades.
