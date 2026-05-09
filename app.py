"""
app.py
Phase 4 — Streamlit dashboard for the Expense Anomaly Detector.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from features import run_pipeline as build_features
from detector import (
    run_zscore_baseline,
    combine_flags,
    IF_CONTAMINATION,
    ZSCORE_THRESHOLD,
)

from data_loader import (
    load_raw,
    clean,
    add_time_features,
    add_account_stats,
)


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

PALETTE = {
    "Normal": "#4C9BE8",
    "Medium": "#F5A623",
    "High": "#E84C4C",
}


# ── Isolation Forest Helper ───────────────────────────────────────────────────
def run_if(X_scaled, contamination):
    """Run Isolation Forest with custom contamination."""

    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)

    flags = (preds == -1).astype(int)

    return scores, flags, model


# ── Data Pipeline ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results(file_bytes, contamination, zscore_thresh):
    """Load CSV and run anomaly detection pipeline."""

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    ) as tmp:

        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── Load & clean ─────────────────────────────────────────────────────
        df = load_raw(Path(tmp_path))
        df = clean(df)
        df = add_time_features(df)
        df = add_account_stats(df)

    finally:
        os.unlink(tmp_path)

    # ── Feature engineering ─────────────────────────────────────────────────
    features_df, X_scaled, scaler = build_features(df)

    df = df.loc[features_df.index].reset_index(drop=True)

    # ── Isolation Forest ────────────────────────────────────────────────────
    if_scores, if_flags, _ = run_if(
        X_scaled,
        contamination
    )

    # ── Z-score baseline ────────────────────────────────────────────────────
    zs_flags = run_zscore_baseline(
        df,
        threshold=zscore_thresh
    )

    # ── Combine ─────────────────────────────────────────────────────────────
    df = combine_flags(
        df,
        if_scores,
        if_flags,
        zs_flags
    )

    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("🔍 Anomaly Detector")

    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    st.markdown("### ⚙️ Detection Settings")

    contamination = st.slider(
        "Isolation Forest sensitivity",
        min_value=0.01,
        max_value=0.20,
        value=float(IF_CONTAMINATION),
        step=0.01,
        help="Expected fraction of anomalies. Higher = more flags."
    )

    zscore_thresh = st.slider(
        "Z-score threshold",
        min_value=1.0,
        max_value=5.0,
        value=float(ZSCORE_THRESHOLD),
        step=0.5,
        help="Flag transactions beyond N std devs from account mean."
    )

    run_btn = st.button(
        "▶ Run Detection",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    st.caption(
        "Bank Transaction Dataset\n"
        "kaggle.com/datasets/valakhorasani"
    )


# ── Empty State ────────────────────────────────────────────────────────────────
if uploaded is None:

    st.title("🔍 Expense Anomaly Detector")

    st.info(
        "👈 Upload your transaction CSV in the sidebar to get started."
    )

    st.markdown("""
    ### How it works

    1. **Upload** your bank transaction CSV
    2. **Adjust** sensitivity sliders
    3. **Explore** interactive anomaly charts

    ### What gets detected

    | Signal | Method |
    |--------|--------|
    | Unusual transaction amounts | Z-score analysis |
    | Multi-dimensional outliers | Isolation Forest |
    | Spending bursts | Rolling window analysis |
    | Suspicious times | Time-based features |
    """)

    st.stop()


# ── Run Pipeline ───────────────────────────────────────────────────────────────
if run_btn or "df" not in st.session_state:

    with st.spinner("Running anomaly detection..."):

        file_bytes = uploaded.read()

        df = get_results(
            file_bytes,
            contamination,
            zscore_thresh
        )

        st.session_state["df"] = df


df = st.session_state.get("df")

if df is None:
    st.warning("Click ▶ Run Detection to analyse your data.")
    st.stop()


# ── KPI Section ────────────────────────────────────────────────────────────────
st.title("🔍 Expense Anomaly Detector")

total = len(df)

high = (df["risk_level"] == "High").sum()
medium = (df["risk_level"] == "Medium").sum()
normal = (df["risk_level"] == "Normal").sum()

avg_amt = df["transactionamount"].mean()

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Transactions", f"{total:,}")

k2.metric(
    "🔴 High Risk",
    f"{high}",
    f"{high / total * 100:.1f}%"
)

k3.metric(
    "🟡 Medium Risk",
    f"{medium}",
    f"{medium / total * 100:.1f}%"
)

k4.metric(
    "🟢 Normal",
    f"{normal}",
    f"{normal / total * 100:.1f}%"
)

k5.metric(
    "Avg Amount",
    f"${avg_amt:,.2f}"
)

st.markdown("---")


# ── Chart Row 1 ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])


# ── Spending Over Time ────────────────────────────────────────────────────────
with col1:

    st.subheader("Spending Over Time")

    daily = (
        df.groupby(df["transactiondate"].dt.date)["transactionamount"]
        .sum()
        .reset_index()
    )

    daily.columns = ["date", "total"]

    fig = px.line(
        daily,
        x="date",
        y="total",
        color_discrete_sequence=[PALETTE["Normal"]],
    )

    for risk in ["Medium", "High"]:

        if risk not in df["risk_level"].values:
            continue

        sub = df[df["risk_level"] == risk]

        fig.add_trace(
            go.Scatter(
                x=sub["transactiondate"].dt.date,
                y=sub["transactionamount"],
                mode="markers",
                name=f"{risk} Risk",
                marker=dict(
                    color=PALETTE[risk],
                    size=8,
                    opacity=0.8,
                ),
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ── Risk Breakdown ────────────────────────────────────────────────────────────
with col2:

    st.subheader("Risk Breakdown")

    counts = (
        df["risk_level"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["risk", "count"]

    fig2 = px.pie(
        counts,
        values="count",
        names="risk",
        color="risk",
        color_discrete_map=PALETTE,
        hole=0.4,
    )

    fig2.update_layout(
        height=320,
        margin=dict(t=10, b=10),
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ── Chart Row 2 ────────────────────────────────────────────────────────────────
col3, col4 = st.columns(2)


# ── Amount Distribution ───────────────────────────────────────────────────────
with col3:

    st.subheader("Amount Distribution")

    fig3 = go.Figure()

    for risk, color in PALETTE.items():

        sub = df[df["risk_level"] == risk]["transactionamount"]

        if sub.empty:
            continue

        fig3.add_trace(
            go.Histogram(
                x=sub,
                name=risk,
                marker_color=color,
                opacity=0.7,
                nbinsx=40,
            )
        )

    fig3.update_layout(
        barmode="overlay",
        height=300,
        margin=dict(t=10, b=10),
        xaxis_title="Amount ($)",
        yaxis_title="Count",
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ── Heatmap ───────────────────────────────────────────────────────────────────
with col4:

    st.subheader("Anomaly Heatmap (Hour × Day)")

    flagged = df[df["combined_flag"] == 1].copy()

    if not flagged.empty:

        flagged["hour_bin"] = (
            flagged["hour"] // 3
        ) * 3

        heat = (
            flagged.groupby(["hour_bin", "day_of_week"])
            .size()
            .unstack(fill_value=0)
        )

        day_names = [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun",
        ]

        heat.columns = [
            day_names[int(i)]
            for i in heat.columns
        ]

        fig4 = px.imshow(
            heat,
            color_continuous_scale="YlOrRd",
            labels=dict(
                x="Day",
                y="Hour (3-hr)",
                color="Count",
            ),
        )

        fig4.update_layout(
            height=300,
            margin=dict(t=10, b=10),
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    else:
        st.info(
            "No anomalies flagged — try increasing sensitivity."
        )


# ── Flagged Transactions Table ────────────────────────────────────────────────
st.markdown("---")

st.subheader("🚨 Flagged Transactions")

risk_filter = st.multiselect(
    "Filter by risk level",
    ["High", "Medium"],
    default=["High", "Medium"],
)

flagged_df = (
    df[df["risk_level"].isin(risk_filter)]
    .sort_values("if_score")
)


display_cols = [
    c for c in [
        "transactionid",
        "accountid",
        "transactiondate",
        "transactionamount",
        "transactiontype",
        "location",
        "risk_level",
        "amount_zscore",
        "loginattempts",
    ]
    if c in flagged_df.columns
]


def highlight_risk(val):

    if val in PALETTE:
        return f"background-color: {PALETTE[val]}22"

    return ""


styled_df = (
    flagged_df[display_cols]
    .style
    .map(
        highlight_risk,
        subset=["risk_level"]
    )
)

st.dataframe(
    styled_df,
    use_container_width=True,
    height=350,
)


# ── Download Button ───────────────────────────────────────────────────────────
st.download_button(
    "⬇ Download flagged transactions",
    data=flagged_df[display_cols].to_csv(index=False),
    file_name="flagged_transactions.csv",
    mime="text/csv",
)