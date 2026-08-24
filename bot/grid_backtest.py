"""Grid/martingale-style "basket" strategy simulator -- built to test the
mechanism (like the Agent Smith / Matrix Escape product) with hard safety
caps that many commercial vendors of this style don't advertise clearly.

How a basket works, matching the mechanism such vendors describe:
- Opens a position on a simple entry trigger.
- If price moves against it by a set distance, adds another same-size
  layer at the new (worse) price -- this pulls the weighted-average entry
  price toward the current price.
- Closes the WHOLE basket once the weighted-average position is back in
  profit by a small margin, even though price never returned to the
  original entry -- this is what produces a high win rate: most baskets
  close in a small, quick profit.

Why this is fundamentally different from every other strategy in this
project: a single-entry trade with a stop-loss has a KNOWN maximum loss,
fixed in advance. A basket that keeps averaging into an adverse move does
not -- its loss is bounded only by how far price can move before the
account runs out of room. The backtest's average-case statistics (win
rate, average return) can look excellent for a long stretch while hiding
this tail risk entirely, since it usually doesn't show up until a single
large, fast move exceeds what the averaging can absorb.

MAX_LAYERS and MAX_BASKET_LOSS_PCT below are hard, non-negotiable caps
added specifically to bound that otherwise-open-ended risk. Removing them
reproduces the unbounded-risk version many commercial products use.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import ta

from . import db
from . import scalp_analysis
from . import telegram_bot
from .historical_backtest import fetch_full_history, fetch_paginated_history, WARMUP_BARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grid_backtest")


def simple_entry_trigger(rsi: float) -> str | None:
    """Deliberately simple mean-reversion entry (RSI extreme) -- the point
    of this simulation is to stress-test the basket RISK MECHANISM, not to
    find a great entry signal. A better entry wouldn't change the core
    finding: whether the hard stop is what actually bounds the loss."""
    if rsi < 30:
        return "LONG"
    if rsi > 70:
        return "SHORT"
    return None


def simulate_basket_strategy(df_15m: pd.DataFrame, atr_series: pd.Series, rsi_series: pd.Series,
                              layer_spacing_atr_mult: float = 1.0, tp_atr_mult: float = 0.5,
                              max_layers: int = 5, max_basket_loss_pct: float = 10.0,
                              layer_risk_pct: float = 0.5, account_size: float = 2000.0) -> dict:
    """Replays the basket mechanism candle-by-candle over df_15m. Returns
    a summary plus the full list of closed baskets (for inspecting the
    worst individual outcome, not just the average).

    layer_risk_pct: % of account_size committed in P&L terms per layer
    (a simplification -- treats % price move as directly proportional to
    % account P&L per layer, ignoring specific lot/margin mechanics, which
    is reasonable for testing the RISK SHAPE of the mechanism itself).

    max_basket_loss_pct: hard stop -- if the basket's total unrealized
    loss (summed across all open layers) reaches this % of account_size,
    the ENTIRE basket is force-closed immediately, regardless of the
    averaging plan. This is what prevents true unbounded ruin.
    """
    baskets_closed = []
    open_basket = None  # {'direction': str, 'entries': [price, ...], 'layer_times': [...]}

    n = len(df_15m)
    for i in range(n):
        row = df_15m.iloc[i]
        atr = atr_series.iloc[i]
        rsi = rsi_series.iloc[i]
        if pd.isna(atr) or pd.isna(rsi):
            continue
        price = row["close"]

        if open_basket is None:
            direction = simple_entry_trigger(rsi)
            if direction is not None:
                open_basket = {"direction": direction, "entries": [price], "opened_at": row["datetime"]}
            continue

        direction = open_basket["direction"]
        avg_entry = sum(open_basket["entries"]) / len(open_basket["entries"])
        n_layers = len(open_basket["entries"])

        # Check the hard stop against the WORST price touched this candle
        # (not just the close) -- more realistic, though even this can't
        # fully eliminate overshoot if a single candle's range is very
        # large during fast, gappy conditions. This is checked FIRST,
        # before any TP/averaging logic.
        worst_price_this_candle = row["low"] if direction == "LONG" else row["high"]
        if direction == "LONG":
            worst_price_move = worst_price_this_candle - avg_entry
        else:
            worst_price_move = avg_entry - worst_price_this_candle
        worst_pct_move = (worst_price_move / avg_entry) if avg_entry else 0.0
        worst_pnl_pct_of_account = worst_pct_move * layer_risk_pct * n_layers * 100

        if worst_pnl_pct_of_account <= -max_basket_loss_pct:
            baskets_closed.append({
                "direction": direction, "layers": n_layers, "avg_entry": avg_entry,
                "exit_price": worst_price_this_candle, "outcome": "HARD_STOP",
                "pnl_pct_of_account": worst_pnl_pct_of_account,
                "opened_at": open_basket["opened_at"], "closed_at": row["datetime"],
            })
            open_basket = None
            continue

        if direction == "LONG":
            unrealized_price_move = price - avg_entry
        else:
            unrealized_price_move = avg_entry - price

        # % account P&L: price move as a fraction of entry price, scaled by
        # how many layers are open (each layer contributes layer_risk_pct)
        unrealized_pct_move = (unrealized_price_move / avg_entry) if avg_entry else 0.0
        unrealized_pnl_pct_of_account = unrealized_pct_move * layer_risk_pct * n_layers * 100

        # Take-profit: whole basket closes once weighted-avg position is
        # back in profit by tp_atr_mult * ATR
        if unrealized_price_move >= tp_atr_mult * atr:
            baskets_closed.append({
                "direction": direction, "layers": n_layers, "avg_entry": avg_entry,
                "exit_price": price, "outcome": "TP",
                "pnl_pct_of_account": unrealized_pnl_pct_of_account,
                "opened_at": open_basket["opened_at"], "closed_at": row["datetime"],
            })
            open_basket = None
            continue

        # Add another layer if price has moved far enough against the
        # LAST layer's entry, and we haven't hit the hard layer cap
        last_entry = open_basket["entries"][-1]
        if direction == "LONG":
            moved_against_last = last_entry - price
        else:
            moved_against_last = price - last_entry
        if moved_against_last >= layer_spacing_atr_mult * atr and n_layers < max_layers:
            open_basket["entries"].append(price)

    # If a basket is still open at the end of the data, record it as such
    # (not closed -- we don't know its eventual outcome)
    still_open = open_basket is not None

    resolved = [b for b in baskets_closed]
    wins = sum(1 for b in resolved if b["pnl_pct_of_account"] > 0)
    hard_stops = sum(1 for b in resolved if b["outcome"] == "HARD_STOP")
    worst_single_basket = min((b["pnl_pct_of_account"] for b in resolved), default=0.0)
    total_pnl_pct = sum(b["pnl_pct_of_account"] for b in resolved)
    max_layers_used = max((b["layers"] for b in resolved), default=0)

    return {
        "baskets": resolved,
        "total_baskets": len(resolved),
        "wins": wins,
        "hard_stops": hard_stops,
        "win_rate": wins / len(resolved) if resolved else None,
        "worst_single_basket_pct": worst_single_basket,
        "total_pnl_pct_of_account": total_pnl_pct,
        "max_layers_used": max_layers_used,
        "still_open_at_end": still_open,
    }


async def run(api_symbol: str = "XAU/USD", display_symbol: str = "XAU/USD",
              deep_start_date: str | None = None, deep_end_date: str | None = None,
              max_layers: int = 5, max_basket_loss_pct: float = 10.0,
              layer_risk_pct: float = 0.5, account_size: float = 2000.0,
              layer_spacing_atr_mult: float = 1.0, tp_atr_mult: float = 0.5):
    """Fetches real 15m data (fresh -- this is NOT reused from prior
    backtests, since only derived results were persisted, not raw
    candles) and stress-tests the basket mechanism against it, with the
    hard safety caps always active."""
    logger.info("Fetching 15m data for %s...", display_symbol)
    if deep_start_date and deep_end_date:
        target_start = datetime.strptime(deep_start_date, "%Y-%m-%d").replace(tzinfo=None)
        target_end = datetime.strptime(deep_end_date, "%Y-%m-%d").replace(tzinfo=None) + timedelta(days=1)
        df = fetch_paginated_history(api_symbol, "15min", target_start, target_end)
    else:
        df = fetch_full_history(api_symbol, "15min")

    if df is None or len(df) < WARMUP_BARS:
        logger.error("Not enough 15m data for %s -- aborting", display_symbol)
        return

    logger.info("%d candles, from %s to %s", len(df), df["datetime"].min(), df["datetime"].max())

    atr_series = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    rsi_series = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    result = simulate_basket_strategy(
        df, atr_series, rsi_series,
        layer_spacing_atr_mult=layer_spacing_atr_mult, tp_atr_mult=tp_atr_mult,
        max_layers=max_layers, max_basket_loss_pct=max_basket_loss_pct,
        layer_risk_pct=layer_risk_pct, account_size=account_size,
    )

    lines = [f"*🧺 Grid/Basket Strategy Stress Test: {display_symbol}*\n"]
    lines.append(f"Period: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
    lines.append(f"Hard caps: max {max_layers} layers, max {max_basket_loss_pct:.1f}% account loss per basket")
    lines.append("")
    lines.append(f"Total baskets: {result['total_baskets']}")
    if result["total_baskets"] > 0:
        lines.append(f"Win rate: {result['win_rate']*100:.1f}% ({result['wins']}/{result['total_baskets']})")
        lines.append(f"Hit hard stop: {result['hard_stops']} times ({100*result['hard_stops']/result['total_baskets']:.1f}% of baskets)")
        lines.append(f"Max layers actually used in any basket: {result['max_layers_used']}")
        lines.append(f"*Worst single basket: {result['worst_single_basket_pct']:+.2f}% of account, in one event*")
        lines.append(f"Total P&L across all baskets: {result['total_pnl_pct_of_account']:+.2f}% of account")
    if result["still_open_at_end"]:
        lines.append("\n_Note: one basket was still open at the end of the data window -- its final outcome is unknown._")
    lines.append("")
    lines.append(
        "_This is a simplified simulation (flat layer sizing, no spread/slippage/margin mechanics) -- "
        "it exists to show the SHAPE of this strategy's risk (frequent small wins, rare large hard-stop losses), "
        "not a precise profit forecast. Without the hard stop, the worst-case loss would be unbounded._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent grid/basket stress test results")


if __name__ == "__main__":
    asyncio.run(run())
