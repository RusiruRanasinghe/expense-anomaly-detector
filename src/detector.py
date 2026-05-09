"""
detector.py
Phase 2 — Anomaly detection models.

Runs two complementary approaches and combines their scores:
    1. Isolation Forest  – ML-based, good at multi-dimensional outliers
    2. Z-score baseline  – statistical, interpretable single-feature check

Output columns added to the DataFrame:
    if_score       – Isolation Forest anomaly score (lower = more anomalous)
    if_flag        – 1 if Isolation Forest labels as anomaly
    zscore_flag    – 1 if |amount_zscore| > threshold
    combined_flag  – 1 if flagged by either method
    risk_level     – "High" / "Medium" / "Normal"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

from features import run_pipeline as build_features, FEATURE_COLS


PROCESSED_PATH = Path("data/processed/transactions_clean.csv")
RESULTS_PATH   = Path("data/processed/anomaly_results.csv")

# ── Tunable parameters ─────────────────────────────────────────────────────────
IF_CONTAMINATION  = 0.05   # expected fraction of anomalies (5%)
IF_N_ESTIMATORS   = 100    # number of trees
IF_RANDOM_STATE   = 42
ZSCORE_THRESHOLD  = 3.0    # flag if |z| > 3  (3 std devs from account mean)


# ── 1. Isolation Forest ────────────────────────────────────────────────────────
def run_isolation_forest(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Isolation Forest on the scaled feature matrix.

    Returns:
        scores  – raw anomaly scores (more negative = more anomalous)
        flags   – binary array: 1 = anomaly, 0 = normal
    """
    print("  Training Isolation Forest …")
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    raw_scores = model.decision_function(X_scaled)   # higher = more normal
    predictions = model.predict(X_scaled)            # -1 = anomaly, 1 = normal
    flags = (predictions == -1).astype(int)

    n_flagged = flags.sum()
    print(f"  Isolation Forest flagged {n_flagged} anomalies "
          f"({n_flagged / len(flags) * 100:.1f}%)")

    return raw_scores, flags, model


# ── 2. Z-score baseline ────────────────────────────────────────────────────────
def run_zscore_baseline(df: pd.DataFrame,
                        threshold: float = ZSCORE_THRESHOLD) -> np.ndarray:
    """
    Flag transactions where the amount deviates more than `threshold`
    standard deviations from the account's own spending mean.
    Simple, interpretable, and great for explaining findings.
    """
    flags = (df["amount_zscore"].abs() > threshold).astype(int)
    n_flagged = flags.sum()
    print(f"  Z-score baseline flagged {n_flagged} anomalies "
          f"(|z| > {threshold})")
    return flags.values


# ── 3. Combine & risk-score ────────────────────────────────────────────────────
def combine_flags(df: pd.DataFrame,
                  if_scores: np.ndarray,
                  if_flags: np.ndarray,
                  zs_flags: np.ndarray) -> pd.DataFrame:
    """
    Merge model outputs back onto the original DataFrame and assign
    a human-readable risk level.

    Risk logic:
        High   – flagged by BOTH methods  (high confidence)
        Medium – flagged by ONE method    (worth reviewing)
        Normal – flagged by neither
    """
    df = df.copy()
    df["if_score"]      = if_scores
    df["if_flag"]       = if_flags
    df["zscore_flag"]   = zs_flags
    df["combined_flag"] = np.where((if_flags + zs_flags) > 0, 1, 0)

    conditions = [
        (if_flags == 1) & (zs_flags == 1),
        (if_flags == 1) | (zs_flags == 1),
    ]
    choices = ["High", "Medium"]
    df["risk_level"] = np.select(conditions, choices, default="Normal")

    counts = df["risk_level"].value_counts()
    print(f"\n  Risk breakdown:\n{counts.to_string()}")

    return df


# ── 4. Save results ────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Saved anomaly results → {path}")


# ── 5. Print top anomalies ─────────────────────────────────────────────────────
def print_top_anomalies(df: pd.DataFrame, n: int = 10) -> None:
    cols = ["transactionid", "accountid", "transactionamount",
            "transactiondate", "location", "risk_level",
            "amount_zscore", "if_score"]
    # Filter only available columns
    cols = [c for c in cols if c in df.columns]

    top = (
        df[df["risk_level"].isin(["High", "Medium"])]
        .sort_values("if_score")   # most anomalous first
        .head(n)
    )

    print(f"\n── Top {n} anomalies ─────────────────────────")
    print(top[cols].to_string(index=False))
    print("─────────────────────────────────────────────\n")


# ── Full pipeline ──────────────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    print("=" * 50)
    print("  Phase 2 — Anomaly Detection")
    print("=" * 50 + "\n")

    # Load cleaned data and build features
    df_clean = pd.read_csv(PROCESSED_PATH, parse_dates=["transactiondate"])
    features_df, X_scaled, scaler = build_features(df_clean)

    # Align index: features_df may have dropped rows
    df_clean = df_clean.loc[features_df.index].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    # Run models
    print("\n── Running models ───────────────────────────")
    if_scores, if_flags, model = run_isolation_forest(X_scaled)
    zs_flags = run_zscore_baseline(df_clean)

    # Combine
    df_results = combine_flags(df_clean, if_scores, if_flags, zs_flags)

    # Report & save
    print_top_anomalies(df_results)
    save_results(df_results)

    return df_results


if __name__ == "__main__":
    df = run_pipeline()