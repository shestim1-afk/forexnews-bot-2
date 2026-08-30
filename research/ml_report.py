"""ML Study 1 -- Model Selection, Baselines, and Final Report.

Model + threshold selection happens ENTIRELY within development data.
This module does not touch OOS -- that is a separate, explicitly gated
function invoked only after development freezes a choice.
"""

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import ml_validation as mv
from .ml_features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_report")

MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]
MIN_TRADES_FOR_EVAL = 30
CONCENTRATION_FAILURE_THRESHOLD = 0.80


def check_concentration(net_r_values: list[float]) -> dict:
    n = len(net_r_values)
    if n == 0:
        return {"concentrated": False, "top_20pct_share": None}
    total_positive = sum(r for r in net_r_values if r > 0)
    if total_positive <= 0:
        return {"concentrated": False, "top_20pct_share": None}
    top_count = max(1, int(n * 0.2))
    top_sum = sum(sorted(net_r_values, reverse=True)[:top_count])
    share = top_sum / total_positive
    return {"concentrated": share > CONCENTRATION_FAILURE_THRESHOLD, "top_20pct_share": share}


def confidence_only_baseline(full_df: pd.DataFrame, pct_to_keep: float) -> dict:
    """Baseline C -- keep the top pct_to_keep of trades by the EXISTING
    confidence score alone, for direct comparison against ML. This tests
    whether ML adds information BEYOND confidence, not just whether
    filtering-in-general helps."""
    n_keep = max(1, int(len(full_df) * pct_to_keep))
    top_conf = full_df.nlargest(n_keep, "confidence")
    return mv.compute_economics(top_conf)


def select_best_model_and_threshold(full_df: pd.DataFrame) -> dict:
    """Runs all (model, threshold) combinations on chronological
    development folds ONLY, selecting by validation-fold net R/day
    improvement over the unfiltered control -- ties broken toward the
    simpler model per the frozen preference order."""
    results = []
    for model_name in MODELS:
        for threshold in mv.THRESHOLDS:
            fold_result = mv.run_chronological_folds(full_df, model_name, threshold)
            if fold_result["filtered"]["n"] < MIN_TRADES_FOR_EVAL:
                continue
            improvement = fold_result["filtered"]["net_r_day"] - fold_result["unfiltered"]["net_r_day"]
            results.append({
                "model": model_name, "threshold": threshold, "improvement": improvement,
                "filtered_econ": fold_result["filtered"], "unfiltered_econ": fold_result["unfiltered"],
                "combined_val_df": fold_result["combined_val_df"],
            })

    if not results:
        return {"selected": None, "all_results": results}

    model_priority = {"logistic_regression": 0, "random_forest": 1, "gradient_boosting": 2}
    results.sort(key=lambda r: (-r["improvement"], model_priority[r["model"]]))
    return {"selected": results[0], "all_results": results}


def run_development_experiment(baseline_rows: list[dict], candidate_rows: list[dict], config_label: str) -> dict:
    """Full development-phase pipeline for one XAU configuration (baseline
    OR 1.0R candidate -- run as separate experiments per the spec). NEVER
    touches OOS. Returns everything needed for the final report, plus a
    single classification: FAILED, INCONCLUSIVE, or DEVELOPMENT PROMISING."""
    if not baseline_rows or len(baseline_rows) < MIN_TRADES_FOR_EVAL:
        return {"config": config_label, "classification": "INCONCLUSIVE", "reason": f"Fewer than {MIN_TRADES_FOR_EVAL} usable signals."}

    full_df = mv.rows_to_dataframe(baseline_rows)
    selection = select_best_model_and_threshold(full_df)

    if selection["selected"] is None:
        return {"config": config_label, "classification": "FAILED", "reason": "No (model, threshold) combination retained enough trades or improved over baseline."}

    best = selection["selected"]
    val_df = best["combined_val_df"]
    filtered_val = val_df[val_df["ml_prob"] >= best["threshold"]]

    # Random-filter benchmark, matched to the ML-filtered trade count
    random_bench = mv.random_filter_benchmark(val_df, n_retain=len(filtered_val))

    # Confidence-only baseline, matched to the same retained fraction
    pct_retained = len(filtered_val) / len(val_df)
    conf_baseline = confidence_only_baseline(val_df, pct_retained)

    # Concentration check
    concentration = check_concentration(filtered_val["net_r"].tolist())

    # Permutation test
    perm = mv.permutation_test(full_df, best["model"], best["threshold"], best["improvement"])

    beats_baseline = best["improvement"] > 0
    beats_random = random_bench["mean_net_r_day"] is not None and best["filtered_econ"]["net_r_day"] > random_bench["mean_net_r_day"]
    beats_confidence = conf_baseline["net_r_day"] is not None and best["filtered_econ"]["net_r_day"] > conf_baseline["net_r_day"]
    not_concentrated = not concentration["concentrated"]
    adequate_sample = best["filtered_econ"]["n"] >= MIN_TRADES_FOR_EVAL
    significant = perm["p_value"] is not None and perm["p_value"] < 0.10

    if beats_baseline and beats_random and adequate_sample and not_concentrated and significant:
        classification = "DEVELOPMENT PROMISING"
        reason = "Improves over baseline, beats random-filter benchmark, adequate sample, not concentration-driven, and beats the permutation null. Justifies (but does not guarantee) OOS testing."
    elif not adequate_sample:
        classification = "INCONCLUSIVE"
        reason = f"Filtered sample (n={best['filtered_econ']['n']}) below the {MIN_TRADES_FOR_EVAL}-trade minimum."
    else:
        classification = "FAILED"
        failed_checks = []
        if not beats_baseline:
            failed_checks.append("does not beat unfiltered baseline")
        if not beats_random:
            failed_checks.append("does not beat random-filter benchmark")
        if not not_concentrated:
            failed_checks.append(f"concentration-driven (top 20% = {concentration['top_20pct_share']*100:.0f}% of positive R)")
        if not significant:
            failed_checks.append(f"not significant vs permutation null (p={perm['p_value']})")
        reason = "Failed: " + "; ".join(failed_checks)

    return {
        "config": config_label, "classification": classification, "reason": reason,
        "best_model": best["model"], "best_threshold": best["threshold"],
        "filtered_econ": best["filtered_econ"], "unfiltered_econ": best["unfiltered_econ"],
        "random_benchmark": random_bench, "confidence_baseline": conf_baseline,
        "concentration": concentration, "permutation": perm,
        "beats_baseline": beats_baseline, "beats_random": beats_random, "beats_confidence": beats_confidence,
        "all_results": selection["all_results"],
    }


def run_oos_experiment(full_dev_df: pd.DataFrame, oos_rows: list[dict], model_name: str, threshold: float) -> dict:
    """Runs the EXACT frozen (model, threshold) on untouched OOS data.
    Must only be called after development explicitly freezes a choice --
    this function does not make that decision itself."""
    if not oos_rows or len(oos_rows) < MIN_TRADES_FOR_EVAL:
        return {"classification": "INCONCLUSIVE", "reason": f"Fewer than {MIN_TRADES_FOR_EVAL} OOS signals."}

    oos_df = mv.rows_to_dataframe(oos_rows)
    probs = mv.fit_and_predict(full_dev_df, oos_df, model_name)
    oos_df = oos_df.copy()
    oos_df["ml_prob"] = probs
    filtered_oos = oos_df[oos_df["ml_prob"] >= threshold]

    unfiltered_econ = mv.compute_economics(oos_df)
    filtered_econ = mv.compute_economics(filtered_oos)
    random_bench = mv.random_filter_benchmark(oos_df, n_retain=len(filtered_oos))
    concentration = check_concentration(filtered_oos["net_r"].tolist())

    beats_baseline = filtered_econ["n"] > 0 and unfiltered_econ["net_r_day"] is not None and filtered_econ["net_r_day"] > unfiltered_econ["net_r_day"]
    beats_random = random_bench["mean_net_r_day"] is not None and filtered_econ["net_r_day"] is not None and filtered_econ["net_r_day"] > random_bench["mean_net_r_day"]
    adequate_sample = filtered_econ["n"] >= MIN_TRADES_FOR_EVAL
    not_concentrated = not concentration["concentrated"]
    net_positive = filtered_econ["net_r_day"] is not None and filtered_econ["net_r_day"] > 0

    if beats_baseline and beats_random and adequate_sample and not_concentrated and net_positive:
        classification = "OOS PROMISING"
    else:
        classification = "FAILED"

    return {
        "classification": classification, "unfiltered_econ": unfiltered_econ, "filtered_econ": filtered_econ,
        "random_benchmark": random_bench, "concentration": concentration,
        "beats_baseline": beats_baseline, "beats_random": beats_random,
    }
