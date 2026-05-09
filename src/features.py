"""
features.py
Phase 2 — Feature engineering for anomaly detection.

Takes the cleaned DataFrame from data_loader.py and produces a
numeric feature matrix ready for scikit-learn models.

Feature groups:
    A. Transaction-level   – amount, duration, login attempts
    B. Time-based          – hour, day_of_week, is_weekend, is_night
    C. Account-relative    – z-score, ratio to account mean
    D. Velocity            – rolling tx count & spend per account (7-day)
    E. Encoded categoricals – transaction type, location frequency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


PROCESSED_PATH = Path("data/processed/transactions_clean.csv")
FEATURES_PATH  = Path("data/processed/features.csv")

FEATURE_COLS = [
    # A – transaction level
    "transactionamount",
    "transactionduration",
    "loginattempts",
    "accountbalance",
    # B – time
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    # C – account-relative
    "amount_zscore",
    "amount_to_mean_ratio",
    # D – velocity
    "rolling_tx_count_7d",
    "rolling_spend_7d",
    # E – categorical
    "is_debit",
    "location_freq",
]


# ── C. Extra account-relative features ────────────────────────────────────────
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_to_mean_ratio"] = np.where(
        df["acct_mean"] > 0,
        df["transactionamount"] / df["acct_mean"],
        1.0,
    )
    print("  Ratio features added: amount_to_mean_ratio")
    return df


# ── D. Velocity features ───────────────────────────────────────────────────────
def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 7-day transaction count and total spend per account.
    High velocity in a short window is a classic anomaly signal.
    """
    df = df.copy().sort_values(["accountid", "transactiondate"])
    df = df.set_index("transactiondate")

    counts = []
    spends = []

    for _, grp in df.groupby("accountid"):
        rolling = grp["transactionamount"].rolling("7D", min_periods=1)
        counts.append(rolling.count())
        spends.append(rolling.sum())

    df["rolling_tx_count_7d"] = pd.concat(counts)
    df["rolling_spend_7d"]    = pd.concat(spends)

    df = df.reset_index()
    print("  Velocity features added: rolling_tx_count_7d, rolling_spend_7d")
    return df


# ── E. Categorical encoding ────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_debit"] = (df["transactiontype"] == "Debit").astype(int)
    loc_freq = df["location"].value_counts(normalize=True)
    df["location_freq"] = df["location"].map(loc_freq).fillna(0)
    print("  Categorical features added: is_debit, location_freq")
    return df


# ── Build feature matrix ───────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame):
    """
    Run all feature steps, select FEATURE_COLS, scale with StandardScaler.

    Returns:
        features_df  – unscaled DataFrame (good for inspection & EDA)
        X_scaled     – scaled numpy array (feed directly to sklearn models)
        scaler       – fitted StandardScaler (reuse in detector.py)
    """
    df = add_ratio_features(df)
    df = add_velocity_features(df)
    df = encode_categoricals(df)

    features_df = df[FEATURE_COLS].copy()
    n_before = len(features_df)
    features_df = features_df.dropna()
    dropped = n_before - len(features_df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN in feature columns.")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    print(f"\n✅ Feature matrix: {features_df.shape[0]:,} rows × {features_df.shape[1]} features")
    print(f"   Columns: {FEATURE_COLS}\n")

    return features_df, X_scaled, scaler


# ── Save ───────────────────────────────────────────────────────────────────────
def save_features(features_df: pd.DataFrame, path: Path = FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(path, index=False)
    print(f"💾 Saved feature matrix → {path}")


# ── Pipeline entry point ───────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame = None):
    print("=" * 50)
    print("  Phase 2 — Feature Engineering")
    print("=" * 50 + "\n")

    if df is None:
        df = pd.read_csv(PROCESSED_PATH, parse_dates=["transactiondate"])
        print(f"  Loaded cleaned data: {df.shape}\n")

    features_df, X_scaled, scaler = build_feature_matrix(df)
    save_features(features_df)

    print("\n── Feature statistics ───────────────────────")
    print(features_df.describe().round(3).to_string())
    print("─────────────────────────────────────────────")

    return features_df, X_scaled, scaler


if __name__ == "__main__":
    features_df, X_scaled, scaler = run_pipeline()