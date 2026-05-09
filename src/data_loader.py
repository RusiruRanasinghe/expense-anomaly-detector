import pandas as pd
import numpy as np
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH       = Path("data/raw/bank_transactions.csv")
PROCESSED_PATH = Path("data/processed/transactions_clean.csv")


# ── 1. Load ────────────────────────────────────────────────────────────────────
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and do a quick sanity-check print."""
    df = pd.read_csv(path)

    print("── Raw data loaded ──────────────────────────")
    print(f"  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}\n")
    print(f"  Missing values:\n{df.isnull().sum()}\n")
    print(df.head(3).to_string())
    print("─────────────────────────────────────────────\n")

    return df


# ── 2. Clean ───────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, parse types, drop or fill missing values.
    Returns a cleaned DataFrame.
    """
    # 2a. Normalise column names  →  snake_case, no spaces
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    # "ip_address" becomes ip_address (already clean after above)

    # 2b. Parse datetime
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")

    # 2c. Ensure numeric columns are actually numeric
    numeric_cols = ["transactionamount", "accountbalance",
                    "transactionduration", "loginattempts"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2d. Drop rows where the core fields are missing
    core_cols = ["transactiondate", "transactionamount", "accountid"]
    before = len(df)
    df = df.dropna(subset=core_cols)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing core values.")

    # 2e. Fill remaining numeric NaNs with column medians
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"  Filled NaNs in '{col}' with median={median:.2f}")

    # 2f. Standardise TransactionType casing
    df["transactiontype"] = df["transactiontype"].str.strip().str.title()

    # 2g. Remove obvious duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["transactionid"])
    dupes = before - len(df)
    if dupes:
        print(f"  Removed {dupes} duplicate TransactionIDs.")

    # 2h. Reset index
    df = df.reset_index(drop=True)

    print(f"\n✅ Clean data: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


# ── 3. Feature engineering (date/time) ────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive useful time-based columns from transactiondate.
    These will feed directly into the anomaly model in Phase 2.
    """
    df = df.copy()
    dt = df["transactiondate"]

    df["hour"]        = dt.dt.hour                      # 0–23
    df["day_of_week"] = dt.dt.dayofweek                 # 0=Mon … 6=Sun
    df["day_name"]    = dt.dt.day_name()
    df["month"]       = dt.dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"]    = df["hour"].between(22, 6).astype(int)  # 10pm–6am

    print("  Time features added: hour, day_of_week, month, is_weekend, is_night")
    return df


# ── 4. Per-account spend statistics ───────────────────────────────────────────
def add_account_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each account, calculate its historical spend mean and std.
    A transaction far from an account's own mean is a candidate anomaly.
    This is the statistical baseline we'll compare to the ML model in Phase 2.
    """
    df = df.copy()

    stats = (
        df.groupby("accountid")["transactionamount"]
        .agg(acct_mean="mean", acct_std="std")
        .reset_index()
    )
    # Accounts with only 1 transaction have NaN std → fill with 0
    stats["acct_std"] = stats["acct_std"].fillna(0)

    df = df.merge(stats, on="accountid", how="left")

    # Z-score of this transaction relative to the account's own history
    df["amount_zscore"] = np.where(
        df["acct_std"] > 0,
        (df["transactionamount"] - df["acct_mean"]) / df["acct_std"],
        0.0
    )

    print("  Account stats added: acct_mean, acct_std, amount_zscore")
    return df


# ── 5. Save processed file ─────────────────────────────────────────────────────
def save(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Saved processed data → {path}")


# ── 6. Full pipeline ───────────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    """
    Run the complete Phase 1 pipeline:
        load → clean → time features → account stats → save
    Returns the processed DataFrame.
    """
    print("=" * 50)
    print("  Phase 1 — Data Loading & Cleaning")
    print("=" * 50 + "\n")

    df = load_raw()
    df = clean(df)
    df = add_time_features(df)
    df = add_account_stats(df)
    save(df)

    print("\n── Summary of processed data ────────────────")
    print(df[["transactionamount", "acct_mean", "acct_std",
              "amount_zscore", "hour", "is_weekend"]].describe().round(2))
    print("─────────────────────────────────────────────")

    return df


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_pipeline()