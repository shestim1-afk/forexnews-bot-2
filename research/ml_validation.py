"""ML Study 1 -- Chronological Validation, Random-Filter Benchmark, and
Permutation Test.

NEVER random train/test split -- every fold is chronological. Model
selection (model type + threshold) happens ENTIRELY within development
folds; OOS is never touched during this phase.
"""

import logging
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.ml_features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_validation")

THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
N_RANDOM_BENCHMARK_REPEATS = 200
N_PERMUTATION_REPEATS = 200

# Frozen fold boundaries, defined BEFORE looking at any results
DEV_FOLDS = [
    {"train_end": "2025-04-01", "val_start": "2025-04-01", "val_end": "2025-06-01"},
    {"train_end": "2025-06-01", "val_start": "2025-06-01", "val_end": "2025-08-01"},
    {"train_end": "2025-08-01", "val_start": "2025-08-01", "val_end": "2025-09-01"},
]
DEV_START = "2025-01-01"


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"])
    return df.sort_values("evaluated_at").reset_index(drop=True)


def get_model(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42)
    raise ValueError(f"unknown model: {model_name}")


def fit_and_predict(train_df: pd.DataFrame, predict_df: pd.DataFrame, model_name: str,
                     feature_cols: list[str] | None = None) -> np.ndarray:
    """Fits scaler + model ONLY on train_df, applies to predict_df.
    feature_cols defaults to FEATURE_COLUMNS; overridable for the
    permutation test's identical-features-shuffled-labels scenario."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df["label"]
    X_predict = predict_df[feature_cols].fillna(X_train.median())  # impute predict-set NAs using TRAIN medians only

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    model = get_model(model_name)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_predict_scaled)[:, 1]


def compute_economics(df: pd.DataFrame) -> dict:
    """Net R/day and related stats for a (possibly filtered) trade set."""
    n = len(df)
    if n == 0:
        return {"n": 0, "net_r_day": None, "avg_net_r": None, "win_rate": None, "profit_factor": None}

    days_span = max((df["evaluated_at"].max() - df["evaluated_at"].min()).days, 1)
    trades_per_day = n / days_span
    avg_net_r = df["net_r"].mean()
    wins = (df["net_r"] > 0).sum()
    gains = df.loc[df["net_r"] > 0, "net_r"].sum()
    losses = abs(df.loc[df["net_r"] <= 0, "net_r"].sum())
    pf = gains / losses if losses > 0 else None

    return {
        "n": n, "net_r_day": trades_per_day * avg_net_r, "avg_net_r": avg_net_r,
        "win_rate": wins / n, "profit_factor": pf, "trades_per_day": trades_per_day,
    }


def run_chronological_folds(full_df: pd.DataFrame, model_name: str, threshold: float) -> dict:
    """Runs all DEV_FOLDS for one (model, threshold) combination, pooling
    the validation-fold predictions (each val fold uses a model trained
    ONLY on data strictly before it) into one combined validation-set
    economic evaluation."""
    all_val_rows = []
    for fold in DEV_FOLDS:
        train_df = full_df[(full_df["evaluated_at"] >= DEV_START) & (full_df["evaluated_at"] < fold["train_end"])]
        val_df = full_df[(full_df["evaluated_at"] >= fold["val_start"]) & (full_df["evaluated_at"] < fold["val_end"])]
        if len(train_df) < 20 or len(val_df) == 0:
            continue
        probs = fit_and_predict(train_df, val_df, model_name)
        val_df = val_df.copy()
        val_df["ml_prob"] = probs
        all_val_rows.append(val_df)

    if not all_val_rows:
        # Structurally consistent with the success case below (same keys,
        # n=0 nested where callers actually check it) -- a bug was found
        # during testing where callers checked a top-level "n" key that
        # only existed in THIS empty case, causing every successful fold
        # result to be misread as empty. Never repeat that ambiguity.
        empty_econ = {"n": 0, "net_r_day": None, "avg_net_r": None, "win_rate": None, "profit_factor": None}
        return {"unfiltered": empty_econ, "filtered": empty_econ, "pct_retained": 0, "combined_val_df": pd.DataFrame()}
    combined = pd.concat(all_val_rows, ignore_index=True)
    filtered = combined[combined["ml_prob"] >= threshold]
    unfiltered_econ = compute_economics(combined)
    filtered_econ = compute_economics(filtered)
    return {
        "unfiltered": unfiltered_econ, "filtered": filtered_econ,
        "pct_retained": 100 * filtered_econ["n"] / unfiltered_econ["n"] if unfiltered_econ["n"] > 0 else 0,
        "combined_val_df": combined,
    }


def random_filter_benchmark(full_val_df: pd.DataFrame, n_retain: int, seed: int = 7) -> dict:
    """Repeatedly samples n_retain random trades from full_val_df (NOT
    using any model), building a null distribution for 'what if you just
    picked this many trades at random'."""
    random.seed(seed)
    n_total = len(full_val_df)
    if n_retain <= 0 or n_retain > n_total:
        return {"mean_net_r_day": None, "distribution": []}

    net_r_days = []
    indices = list(range(n_total))
    for _ in range(N_RANDOM_BENCHMARK_REPEATS):
        sample_idx = random.sample(indices, n_retain)
        sample_df = full_val_df.iloc[sample_idx]
        econ = compute_economics(sample_df)
        if econ["net_r_day"] is not None:
            net_r_days.append(econ["net_r_day"])

    return {
        "mean_net_r_day": sum(net_r_days) / len(net_r_days) if net_r_days else None,
        "distribution": net_r_days,
        "n_repeats": len(net_r_days),
    }


def permutation_test(full_df: pd.DataFrame, model_name: str, threshold: float, observed_improvement: float,
                      seed: int = 11) -> dict:
    """Shuffles TRAINING labels only (preserving feature/chronological
    structure), retrains+evaluates on the SAME chronological folds,
    repeats to build a null distribution, and returns an empirical
    p-value for the observed improvement."""
    rng = random.Random(seed)
    null_improvements = []

    for i in range(N_PERMUTATION_REPEATS):
        shuffled_df = full_df.copy()
        shuffled_labels = shuffled_df["label"].values.copy()
        rng.shuffle(shuffled_labels)
        shuffled_df["label"] = shuffled_labels

        result = run_chronological_folds(shuffled_df, model_name, threshold)
        if result["filtered"]["net_r_day"] is None or result["unfiltered"]["net_r_day"] is None:
            continue
        null_improvements.append(result["filtered"]["net_r_day"] - result["unfiltered"]["net_r_day"])

    if not null_improvements:
        return {"p_value": None, "n_valid_permutations": 0}

    n_as_extreme = sum(1 for x in null_improvements if x >= observed_improvement)
    p_value = n_as_extreme / len(null_improvements)
    return {"p_value": p_value, "n_valid_permutations": len(null_improvements), "null_distribution": null_improvements}
