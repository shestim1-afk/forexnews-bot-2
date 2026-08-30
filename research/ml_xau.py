"""ML Study 1 -- Orchestration entry point.

Two SEPARATE, explicitly-gated phases:
  run_development(config) -- builds features, selects model/threshold
      using ONLY development data (2025-01-01 to 2025-08-31). NEVER
      touches OOS. Freezes and reports a choice.
  run_oos(config, model_name, threshold) -- runs the EXACT frozen choice
      on untouched OOS data (2025-09-01 to 2025-12-31). Must be called
      explicitly, separately, and only after development has been
      reviewed and approved.

bot/xau_swing.py is never imported or touched by this module.
"""

import asyncio
import logging

from bot import telegram_bot
from research.ml_features import build_feature_dataset
from research.ml_validation import rows_to_dataframe, DEV_START
from research.ml_report import run_development_experiment, run_oos_experiment, MIN_TRADES_FOR_EVAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_xau")

CONFIGS = {
    "baseline": {"symbol_tag": "XAU/USD [4h-ATR]", "strategy_type": "trend", "label": "Control A (TP1=0.667R)"},
    "candidate": {"symbol_tag": "XAU/USD (R:R 1.5:1.5) [4h-ATR]", "strategy_type": "trend", "label": "Control B (TP1=1.0R)"},
}

DEV_END = "2025-09-01"
OOS_START = "2025-09-01"
OOS_END = "2026-01-01"


def _fmt_econ(econ: dict) -> str:
    if econ.get("n", 0) == 0:
        return "n=0"
    pf_s = f"{econ['profit_factor']:.2f}" if econ.get("profit_factor") is not None else "N/A"
    return f"n={econ['n']}, net R/day {econ['net_r_day']:+.3f}, avg net R {econ['avg_net_r']:+.3f}, WR {econ['win_rate']*100:.0f}%, PF {pf_s}"


async def run_development(config_key: str):
    cfg = CONFIGS[config_key]
    logger.info("Building feature dataset for %s...", cfg["label"])
    dataset = build_feature_dataset(cfg["symbol_tag"], cfg["strategy_type"], DEV_START, DEV_END)

    lines = [
        f"*🧪 ML Study 1 -- Development: {cfg['label']}*",
        "_Model/threshold selection uses ONLY development data (2025-01-01 to 2025-08-31). OOS untouched._\n",
    ]
    lines.append(f"Signals found: {dataset['n_signals']}")
    if dataset["n_signals"] == 0:
        lines.append("No signals -- aborting.")
        await telegram_bot.send_text("\n".join(lines))
        return

    lines.append(f"Usable rows (after history/timing exclusions): {dataset.get('n_usable_rows', 0)}")
    lines.append(f"Skipped for insufficient trailing history: {dataset.get('n_skipped_insufficient_history', 0)}")
    lines.append(f"Signals affected by the conservative timing rule (would have used a different, not-yet-closed candle under the original engine's inclusive rule): {dataset.get('n_affected_by_conservative_rule', 'N/A')}")
    lines.append(f"Timing violations excluded (feature-close-time >= entry, should always be 0): {dataset.get('timing_violations_excluded', 'N/A')}")
    lines.append(f"Timing audit passed: {'YES' if dataset.get('timing_audit_passed') else 'NO'}")
    lines.append("")

    if dataset["n_usable_rows"] < MIN_TRADES_FOR_EVAL:
        lines.append(f"*CLASSIFICATION: INCONCLUSIVE -- fewer than {MIN_TRADES_FOR_EVAL} usable rows.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    result = run_development_experiment(dataset["rows"], [], cfg["label"])

    lines.append(f"*Classification: {result['classification']}*")
    lines.append(f"_{result['reason']}_")
    if result.get("best_model"):
        lines.append("")
        lines.append(f"Best model: `{result['best_model']}`, threshold: {result['best_threshold']}")
        lines.append(f"Unfiltered (control): {_fmt_econ(result['unfiltered_econ'])}")
        lines.append(f"ML-filtered: {_fmt_econ(result['filtered_econ'])}")
        rb = result["random_benchmark"]
        lines.append(f"Random-filter benchmark (matched count, {rb.get('n_repeats', 0)} repeats): mean net R/day {rb['mean_net_r_day']:+.3f}" if rb.get("mean_net_r_day") is not None else "Random-filter benchmark: N/A")
        cb = result["confidence_baseline"]
        lines.append(f"Confidence-only baseline (matched retention %): {_fmt_econ(cb)}")
        conc = result["concentration"]
        lines.append(f"Concentration: top 20% = {conc['top_20pct_share']*100:.0f}% of positive R" if conc.get("top_20pct_share") is not None else "Concentration: N/A")
        perm = result["permutation"]
        lines.append(f"Permutation test: p={perm['p_value']:.3f} ({perm['n_valid_permutations']} valid permutations)" if perm.get("p_value") is not None else "Permutation test: N/A")

    # NOT wrapped in italics -- avoids nesting an underscore-containing
    # literal ("bot/xau_swing.py") inside a "_..._" span, which is
    # unreliable in Telegram's legacy Markdown parser (confirmed causing
    # a real send failure elsewhere in this project).
    lines.append(
        "\nPer protocol: development-only. OOS (2025-09-01 to 2025-12-31) is only run if this is explicitly reviewed "
        "and approved as DEVELOPMENT PROMISING. `bot/xau_swing.py` is untouched."
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent ML development report for %s: %s", config_key, result["classification"])
    return result, dataset


async def run_oos(config_key: str, model_name: str, threshold: float):
    """Explicit, separate OOS invocation -- requires the exact frozen
    model_name/threshold to be passed in, never inferred automatically,
    so this can never silently run without an explicit human decision."""
    cfg = CONFIGS[config_key]
    logger.info("Building OOS feature dataset for %s...", cfg["label"])

    dev_dataset = build_feature_dataset(cfg["symbol_tag"], cfg["strategy_type"], DEV_START, DEV_END)
    oos_dataset = build_feature_dataset(cfg["symbol_tag"], cfg["strategy_type"], OOS_START, OOS_END)

    lines = [
        f"*🧪 ML Study 1 -- OOS: {cfg['label']}*",
        # NOT italicized -- model_name may contain underscores (e.g.
        # "gradient_boosting"), and bot/xau_swing.py also does; nesting
        # either inside a "_..._" span is unreliable in Telegram's parser.
        f"Frozen model: `{model_name}`, threshold: {threshold}. No retuning. `bot/xau_swing.py` untouched.\n",
    ]

    if oos_dataset["n_signals"] == 0 or oos_dataset["n_usable_rows"] < MIN_TRADES_FOR_EVAL:
        lines.append(f"*CLASSIFICATION: INCONCLUSIVE -- {oos_dataset.get('n_usable_rows', 0)} usable OOS rows, below the {MIN_TRADES_FOR_EVAL}-trade minimum.*")
        await telegram_bot.send_text("\n".join(lines))
        return

    lines.append(f"OOS timing audit passed: {'YES' if oos_dataset.get('timing_audit_passed') else 'NO'}")
    dev_df = rows_to_dataframe(dev_dataset["rows"])
    result = run_oos_experiment(dev_df, oos_dataset["rows"], model_name, threshold)

    lines.append(f"*Classification: {result['classification']}*")
    lines.append(f"Unfiltered (control): {_fmt_econ(result['unfiltered_econ'])}")
    lines.append(f"ML-filtered: {_fmt_econ(result['filtered_econ'])}")
    rb = result["random_benchmark"]
    lines.append(f"Random-filter benchmark: mean net R/day {rb['mean_net_r_day']:+.3f}" if rb.get("mean_net_r_day") is not None else "Random-filter benchmark: N/A")

    lines.append(
        "\n_Historical OOS result only -- never 'validated' or 'forward validated' from this alone. "
        "If OOS PROMISING, the next step is a separately-approved live forward test, run alongside the "
        "unchanged existing baseline as the control._"
    )

    await telegram_bot.send_text("\n".join(lines))
    logger.info("Sent ML OOS report for %s: %s", config_key, result["classification"])
    return result


if __name__ == "__main__":
    asyncio.run(run_development("baseline"))
