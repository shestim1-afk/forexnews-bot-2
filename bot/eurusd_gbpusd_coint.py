"""EUR/USD - GBP/USD Statistical Arbitrage: Phase 1 (Data Audit + Cointegration Test) ONLY.

This is explicitly NOT a strategy and NOT a backtest, per the frozen
research protocol. It answers exactly one question: is there a
statistically defensible, stationary relationship between these two
pairs at all -- BEFORE any entry/exit/stop rule is designed. If this
phase fails, the hypothesis stops here, per protocol.

Method (standard Engle-Granger two-step cointegration test):
1. Fetch EUR/USD and GBP/USD 4H data over the same window already used
   throughout this project (2025-01-01 to 2025-12-31, dev+OOS combined --
   Phase 1 is about whether ANY relationship exists at all, not yet
   about splitting into dev/OOS, which only matters once a trading rule
   is being tested).
2. OLS regression: ln(EUR/USD) = alpha + beta * ln(GBP/USD) + residual,
   estimated over the FULL available window (a static beta for this
   existence test -- a rolling beta is a separate, later design choice
   for the actual trading rule, only relevant if this phase passes).
3. Augmented Dickey-Fuller (ADF) test on the residual series. If the
   residuals are NOT stationary (fail to reject the unit-root null at
   p<0.05), the pair is not cointegrated in any usable sense -- STOP,
   per the frozen falsification rule agreed before this test ran.

Nothing here is a trading rule. Nothing here touches the live XAU
forward test.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from . import telegram_bot
from .historical_backtest import fetch_paginated_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eurusd_gbpusd_coint")

SYMBOL_A = "EUR/USD"
SYMBOL_B = "GBP/USD"
FETCH_START = "2025-01-01"
FETCH_END = "2025-12-31"
ADF_SIGNIFICANCE = 0.05


def run_cointegration_test(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Aligns the two series on shared timestamps, runs the Engle-Granger
    two-step test, and returns the full result -- beta, residual series
    stats, ADF statistic, p-value, and critical values."""
    merged = pd.merge(
        df_a[["datetime", "close"]].rename(columns={"close": "close_a"}),
        df_b[["datetime", "close"]].rename(columns={"close": "close_b"}),
        on="datetime", how="inner",
    )
    n_aligned = len(merged)
    if n_aligned < 100:
        return {"n_aligned": n_aligned, "sufficient_data": False}

    log_a = np.log(merged["close_a"])
    log_b = np.log(merged["close_b"])

    X = add_constant(log_b)
    model = OLS(log_a, X).fit()
    alpha, beta = model.params.iloc[0], model.params.iloc[1]
    residuals = log_a - (alpha + beta * log_b)

    adf_result = adfuller(residuals, autolag="AIC", result_object=False)
    adf_stat, p_value, used_lag, n_obs, critical_values, _ = adf_result

    return {
        "n_aligned": n_aligned, "sufficient_data": True,
        "alpha": alpha, "beta": beta,
        "adf_statistic": adf_stat, "p_value": p_value,
        "critical_values": critical_values, "used_lag": used_lag,
        "residual_mean": residuals.mean(), "residual_std": residuals.std(),
        "is_cointegrated": p_value < ADF_SIGNIFICANCE,
    }


async def run():
    lines = [
        "*📊 EUR/USD - GBP/USD Statistical Arbitrage: Phase 1 (Data Audit + Cointegration Test)*",
        "_This is a DATA/STATISTICAL FEASIBILITY CHECK ONLY -- no trading rule, no backtest, "
        "production bot untouched. If this fails, the hypothesis stops here._\n",
    ]

    target_start = datetime.strptime(FETCH_START, "%Y-%m-%d")
    target_end = datetime.strptime(FETCH_END, "%Y-%m-%d") + timedelta(days=1)

    logger.info("Fetching 4H data for %s and %s...", SYMBOL_A, SYMBOL_B)
    df_a = fetch_paginated_history(SYMBOL_A, "4h", target_start, target_end)
    df_b = fetch_paginated_history(SYMBOL_B, "4h", target_start, target_end)

    if df_a is None or df_b is None or len(df_a) == 0 or len(df_b) == 0:
        lines.append("*RESULT: DATA INSUFFICIENT -- one or both symbols failed to fetch.*")
        await telegram_bot.send_text("\n".join(lines))
        logger.info("Sent report -- data fetch failed")
        return

    lines.append(f"{SYMBOL_A}: {len(df_a)} candles, {df_a['datetime'].min()} to {df_a['datetime'].max()}")
    lines.append(f"{SYMBOL_B}: {len(df_b)} candles, {df_b['datetime'].min()} to {df_b['datetime'].max()}")

    result = run_cointegration_test(df_a, df_b)

    if not result["sufficient_data"]:
        lines.append(f"\nAligned observations: {result['n_aligned']} -- insufficient for a meaningful ADF test.")
        lines.append("\n*RESULT: DATA INSUFFICIENT.*")
        await telegram_bot.send_text("\n".join(lines))
        logger.info("Sent report -- insufficient aligned data")
        return

    lines.append(f"\nAligned observations: {result['n_aligned']}")
    lines.append(f"OLS beta (hedge ratio): {result['beta']:.4f}")
    lines.append(f"Residual (spread) mean: {result['residual_mean']:.6f}, std: {result['residual_std']:.6f}")
    lines.append("")
    lines.append("*Engle-Granger cointegration test (Augmented Dickey-Fuller on residuals)*")
    lines.append(f"  ADF statistic: {result['adf_statistic']:.4f}")
    lines.append(f"  p-value: {result['p_value']:.4f}")
    lines.append(f"  Critical values: 1%={result['critical_values']['1%']:.3f}, 5%={result['critical_values']['5%']:.3f}, 10%={result['critical_values']['10%']:.3f}")
    lines.append(f"  Lags used: {result['used_lag']}")

    lines.append("")
    if result["is_cointegrated"]:
        lines.append(f"*RESULT: COINTEGRATED (p={result['p_value']:.4f} < {ADF_SIGNIFICANCE}) -- statistically defensible basis to proceed to Phase 2 (freeze a trading specification).*")
    else:
        lines.append(f"*RESULT: NOT COINTEGRATED (p={result['p_value']:.4f} >= {ADF_SIGNIFICANCE}) -- per the pre-declared falsification rule, STOP here. No trading rule should be designed for this pair.*")

    lines.append(
        "\n_This is Phase 1 only -- a real statistical test, not a proxy or approximation. "
        "A pass here means a trading rule is JUSTIFIED to design next, not that one would be profitable._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent cointegration test report, is_cointegrated=%s", result["is_cointegrated"])


if __name__ == "__main__":
    asyncio.run(run())
