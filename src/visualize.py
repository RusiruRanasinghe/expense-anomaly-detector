"""
visualize.py
Phase 3 — Visualisation of anomaly detection results.

Charts produced:
    1. Spending over time     – line chart with anomalies highlighted
    2. Amount distribution    – histogram: normal vs anomalous
    3. Risk by location       – bar chart of flagged tx per location (top 15)
    4. Anomaly heatmap        – hour × day_of_week flag density
    5. Z-score distribution   – where flagged tx sit vs the population
    6. Feature correlation    – heatmap of feature correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path


RESULTS_PATH = Path("data/processed/anomaly_results.csv")
PLOTS_DIR = Path("plots")

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "Normal": "#4C9BE8",
    "Medium": "#F5A623",
    "High": "#E84C4C",
}

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


# ── Helpers ────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to plots directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    path = PLOTS_DIR / name

    fig.savefig(path)
    print(f"  💾 Saved → {path}")

    plt.close(fig)


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Load anomaly detection results CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path, parse_dates=["transactiondate"])

    print(f"  Loaded results: {df.shape[0]:,} rows\n")

    return df


# ── Chart 1: Spending over time ────────────────────────────────────────────────
def plot_spending_over_time(df: pd.DataFrame) -> None:
    """Daily total spend with anomalous transactions overlaid."""

    fig, ax = plt.subplots(figsize=(14, 5))

    daily = (
        df.groupby(df["transactiondate"].dt.date)["transactionamount"]
        .sum()
    )

    ax.plot(
        daily.index,
        daily.values,
        color=PALETTE["Normal"],
        linewidth=1.5,
        label="Daily spend",
        zorder=1,
    )

    for risk in ["Medium", "High"]:

        if risk not in df["risk_level"].values:
            continue

        sub = df[df["risk_level"] == risk]

        ax.scatter(
            sub["transactiondate"].dt.date,
            sub["transactionamount"],
            color=PALETTE[risk],
            s=60,
            alpha=0.8,
            zorder=2,
            label=f"{risk} risk",
        )

    ax.set_title(
        "Spending Over Time — Anomalies Highlighted",
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Amount ($)")
    ax.legend()

    _save(fig, "01_spending_over_time.png")


# ── Chart 2: Amount distribution ──────────────────────────────────────────────
def plot_amount_distribution(df: pd.DataFrame) -> None:
    """Distribution of normal vs anomalous transaction amounts."""

    fig, ax = plt.subplots(figsize=(10, 5))

    normal = df[df["combined_flag"] == 0]["transactionamount"]
    flagged = df[df["combined_flag"] == 1]["transactionamount"]

    bins = np.linspace(
        df["transactionamount"].min(),
        df["transactionamount"].quantile(0.99),
        50,
    )

    ax.hist(
        normal,
        bins=bins,
        alpha=0.6,
        color=PALETTE["Normal"],
        label="Normal",
    )

    ax.hist(
        flagged,
        bins=bins,
        alpha=0.7,
        color=PALETTE["High"],
        label="Anomalous",
    )

    ax.set_title("Transaction Amount Distribution", fontsize=14, pad=12)

    ax.set_xlabel("Transaction Amount ($)")
    ax.set_ylabel("Count")

    ax.legend()

    _save(fig, "02_amount_distribution.png")


# ── Chart 3: Risk by location ──────────────────────────────────────────────────
def plot_risk_by_location(df: pd.DataFrame, top_n: int = 15) -> None:
    """Stacked bar chart of risk levels by location."""

    if "location" not in df.columns:
        print("  ⚠️ Skipping location chart — 'location' column missing")
        return

    pivot = (
        df.groupby(["location", "risk_level"])
        .size()
        .unstack(fill_value=0)
    )

    # Only keep columns that actually exist
    risk_cols = [
        col for col in ["High", "Medium", "Normal"]
        if col in pivot.columns
    ]

    if not risk_cols:
        print("  ⚠️ No risk level columns found")
        return

    pivot = pivot[risk_cols]

    # SAFE VERSION — handles missing High/Medium columns
    severity_score = (
        pivot.get("High", 0)
        + pivot.get("Medium", 0)
    )

    top_locations = severity_score.nlargest(top_n).index

    pivot = pivot.loc[top_locations]

    fig, ax = plt.subplots(figsize=(12, 6))

    colours = [PALETTE.get(c, "#999999") for c in pivot.columns]

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colours,
        edgecolor="none",
    )

    ax.set_title(
        f"Anomalies by Location (Top {top_n})",
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel("Location")
    ax.set_ylabel("Transaction Count")

    ax.legend(title="Risk Level")

    plt.xticks(rotation=45, ha="right")

    _save(fig, "03_risk_by_location.png")


# ── Chart 4: Anomaly heatmap ───────────────────────────────────────────────────
def plot_anomaly_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of anomalies by hour and day."""

    flagged = df[df["combined_flag"] == 1].copy()

    if flagged.empty:
        print("  ⚠️ No anomalies found — skipping heatmap")
        return

    flagged["hour_bin"] = (flagged["hour"] // 3) * 3

    heat = (
        flagged.groupby(["hour_bin", "day_of_week"])
        .size()
        .unstack(fill_value=0)
    )

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Safe rename
    heat.columns = [
        day_names[int(col)] if int(col) < 7 else str(col)
        for col in heat.columns
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        heat,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.5,
        annot=True,
        fmt="d",
        cbar_kws={"label": "Flagged Transactions"},
    )

    hour_labels = [f"{int(h):02d}:00" for h in heat.index]

    ax.set_yticklabels(hour_labels, rotation=0)

    ax.set_title(
        "Anomaly Heatmap — Hour × Day of Week",
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour (3-hr bins)")

    _save(fig, "04_anomaly_heatmap.png")


# ── Chart 5: Z-score distribution ─────────────────────────────────────────────
def plot_zscore_distribution(df: pd.DataFrame) -> None:
    """KDE plot of z-score distributions."""

    if "amount_zscore" not in df.columns:
        print("  ⚠️ Skipping z-score plot — column missing")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    normal = df[df["combined_flag"] == 0]["amount_zscore"].dropna()
    anomalous = df[df["combined_flag"] == 1]["amount_zscore"].dropna()

    if len(normal) > 1:
        sns.kdeplot(
            normal,
            ax=ax,
            fill=True,
            color=PALETTE["Normal"],
            label="Normal",
            alpha=0.6,
        )

    if len(anomalous) > 1:
        sns.kdeplot(
            anomalous,
            ax=ax,
            fill=True,
            color=PALETTE["High"],
            label="Anomalous",
            alpha=0.6,
        )

    ax.axvline(
        3,
        color="grey",
        linestyle="--",
        linewidth=1,
        label="Z = ±3 threshold",
    )

    ax.axvline(
        -3,
        color="grey",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlim(-10, 10)

    ax.set_title(
        "Z-Score Distribution: Normal vs Anomalous",
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel("Amount Z-Score")
    ax.set_ylabel("Density")

    ax.legend()

    _save(fig, "05_zscore_distribution.png")


# ── Chart 6: Feature correlation ──────────────────────────────────────────────
def plot_feature_correlation(df: pd.DataFrame) -> None:
    """Correlation heatmap of numeric features."""

    numeric_cols = [
        "transactionamount",
        "transactionduration",
        "loginattempts",
        "accountbalance",
        "hour",
        "day_of_week",
        "amount_zscore",
        "rolling_tx_count_7d",
        "rolling_spend_7d",
        "combined_flag",
    ]

    available = [c for c in numeric_cols if c in df.columns]

    if len(available) < 2:
        print("  ⚠️ Not enough numeric columns for correlation plot")
        return

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(11, 9))

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12)

    plt.xticks(rotation=45, ha="right")

    _save(fig, "06_feature_correlation.png")


# ── Dashboard ──────────────────────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame) -> None:
    """Combined dashboard."""

    fig = plt.figure(figsize=(20, 14))

    fig.suptitle(
        "Expense Anomaly Detector — Summary Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
    )

    # ── 1: Spending over time ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    daily = (
        df.groupby(df["transactiondate"].dt.date)["transactionamount"]
        .sum()
    )

    ax1.plot(
        daily.index,
        daily.values,
        color=PALETTE["Normal"],
        linewidth=1.5,
    )

    for risk in ["Medium", "High"]:

        if risk not in df["risk_level"].values:
            continue

        s = df[df["risk_level"] == risk]

        ax1.scatter(
            s["transactiondate"].dt.date,
            s["transactionamount"],
            color=PALETTE[risk],
            s=40,
            alpha=0.8,
            label=f"{risk} risk",
        )

    ax1.set_title("Spending Over Time")
    ax1.legend(fontsize=8)

    # ── 2: Amount distribution ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    bins = np.linspace(
        df["transactionamount"].min(),
        df["transactionamount"].quantile(0.99),
        40,
    )

    ax2.hist(
        df[df["combined_flag"] == 0]["transactionamount"],
        bins=bins,
        alpha=0.6,
        color=PALETTE["Normal"],
        label="Normal",
    )

    ax2.hist(
        df[df["combined_flag"] == 1]["transactionamount"],
        bins=bins,
        alpha=0.7,
        color=PALETTE["High"],
        label="Anomalous",
    )

    ax2.set_title("Amount Distribution")
    ax2.legend(fontsize=8)

    # ── 3: Z-score distribution ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    if "amount_zscore" in df.columns:

        normal = df[df["combined_flag"] == 0]["amount_zscore"].dropna()
        anomalous = df[df["combined_flag"] == 1]["amount_zscore"].dropna()

        if len(normal) > 1:
            sns.kdeplot(
                normal,
                ax=ax3,
                fill=True,
                color=PALETTE["Normal"],
                alpha=0.6,
                label="Normal",
            )

        if len(anomalous) > 1:
            sns.kdeplot(
                anomalous,
                ax=ax3,
                fill=True,
                color=PALETTE["High"],
                alpha=0.6,
                label="Anomalous",
            )

        ax3.axvline(3, color="grey", linestyle="--", linewidth=1)
        ax3.axvline(-3, color="grey", linestyle="--", linewidth=1)

        ax3.set_xlim(-10, 10)

    ax3.set_title("Z-Score Distribution")
    ax3.legend(fontsize=8)

    # ── 4: Risk breakdown ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])

    counts = df["risk_level"].value_counts()

    colours = [
        PALETTE.get(risk, "#999999")
        for risk in counts.index
    ]

    ax4.pie(
        counts.values,
        labels=counts.index,
        colors=colours,
        autopct="%1.1f%%",
        startangle=140,
    )

    ax4.set_title("Risk Level Breakdown")

    _save(fig, "00_dashboard.png")


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline() -> None:

    print("=" * 50)
    print("  Phase 3 — Visualisation")
    print("=" * 50 + "\n")

    try:
        df = load_results()

        plot_spending_over_time(df)
        plot_amount_distribution(df)
        plot_risk_by_location(df)
        plot_anomaly_heatmap(df)
        plot_zscore_distribution(df)
        plot_feature_correlation(df)
        plot_dashboard(df)

        print(f"\n✅ All charts saved to ./{PLOTS_DIR}/")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()