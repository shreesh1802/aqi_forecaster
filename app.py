"""
app.py  —  AQI Forecaster  |  Hybrid ARIMA-LSTM Dashboard
Run:  streamlit run app.py
"""

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sample_data import generate_sample_csv
from model_engine import run_pipeline, preprocess, TARGET_COL

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecaster",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 600; }

    /* Headers */
    h1 { color: #1e293b !important; }
    h2, h3 { color: #334155 !important; }

    /* Section dividers */
    hr { border-color: #e2e8f0; }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin-left: 8px;
    }
    .badge-good     { background:#dcfce7; color:#166534; }
    .badge-moderate { background:#fef9c3; color:#854d0e; }
    .badge-poor     { background:#fee2e2; color:#991b1b; }

    /* Spike table */
    .spike-row { background: #fff1f2; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

AQI_THRESHOLDS = [
    (0,   1.0,  "Good",         "#22c55e"),
    (1.0, 2.0,  "Fair",         "#84cc16"),
    (2.0, 4.0,  "Moderate",     "#eab308"),
    (4.0, 7.0,  "Poor",         "#f97316"),
    (7.0, 10.0, "Very Poor",    "#ef4444"),
    (10.0, 999, "Hazardous",    "#7c3aed"),
]


def co_category(val: float):
    for lo, hi, label, colour in AQI_THRESHOLDS:
        if lo <= val < hi:
            return label, colour
    return "Hazardous", "#7c3aed"


def badge_html(label, colour):
    return (
        f'<span class="badge" style="background:{colour}22; '
        f'color:{colour}; border:1px solid {colour}66">{label}</span>'
    )


def make_ts_chart(result_df: pd.DataFrame, show_arima: bool, show_lstm: bool):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df["Datetime"], y=result_df["Actual"],
        name="Actual", line=dict(color="#64748b", width=1.5), opacity=0.8,
    ))
    if show_arima:
        fig.add_trace(go.Scatter(
            x=result_df["Datetime"], y=result_df["ARIMA"],
            name="ARIMA", line=dict(color="#3b82f6", width=1, dash="dot"),
        ))
    if show_lstm:
        fig.add_trace(go.Scatter(
            x=result_df["Datetime"], y=result_df["LSTM"],
            name="LSTM", line=dict(color="#8b5cf6", width=1, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=result_df["Datetime"], y=result_df["Ensemble"],
        name="Hybrid Ensemble", line=dict(color="#f97316", width=2),
    ))

    # Spike markers
    spikes = result_df[result_df["Spike_flag"]]
    if len(spikes):
        fig.add_trace(go.Scatter(
            x=spikes["Datetime"], y=spikes["Actual"],
            mode="markers", name="Spike",
            marker=dict(color="#ef4444", size=8, symbol="x"),
        ))

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(gridcolor="#f1f5f9", title="CO concentration (mg/m³)"),
        legend=dict(orientation="h", y=-0.15, x=0),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    )
    return fig


def make_scatter_chart(result_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df["Actual"], y=result_df["Ensemble"],
        mode="markers",
        marker=dict(color="#f97316", size=4, opacity=0.5),
        name="Predictions",
    ))
    lo = float(result_df[["Actual","Ensemble"]].min().min())
    hi = float(result_df[["Actual","Ensemble"]].max().max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", line=dict(color="#94a3b8", dash="dash"),
        name="Perfect fit",
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9", title="Actual"),
        yaxis=dict(gridcolor="#f1f5f9", title="Predicted"),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    )
    return fig


def make_residual_chart(result_df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=result_df["Residual"], nbinsx=40,
        marker_color="#3b82f6", opacity=0.75,
        name="Residuals",
    ))
    fig.add_vline(x=0, line_color="#ef4444", line_dash="dash")
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Error (Actual − Predicted)"),
        yaxis=dict(gridcolor="#f1f5f9"),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    )
    return fig


def make_feature_importance_chart(df: pd.DataFrame):
    """Bar chart of Pearson correlations with target."""
    num_df = df.select_dtypes(include=[np.number])
    if TARGET_COL not in num_df.columns:
        return None
    corr = (
        num_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=True)
        .tail(10)
    )
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index,
        orientation="h",
        marker_color="#3b82f6",
    ))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="|Pearson r| with CO(GT)", gridcolor="#f1f5f9"),
        yaxis=dict(title=""),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/000000/air-pollution.png",
        width=56,
    )
    st.title("AQI Forecaster")
    st.caption("Hybrid ARIMA-LSTM · Spike Detection")
    st.divider()

    st.subheader("📂 Data source")
    use_sample = st.toggle("Use sample data (demo)", value=True)

    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader(
            "Upload AirQualityUCI-format CSV (semicolon-delimited)",
            type=["csv"],
        )
        st.caption(
            "Expected columns: `Date`, `Time`, `CO(GT)`, `NO2(GT)`, `T`, `RH`  \n"
            "-200 values treated as missing."
        )

    st.divider()
    st.subheader("📊 Chart options")
    show_arima = st.checkbox("Show ARIMA component", value=False)
    show_lstm  = st.checkbox("Show LSTM component",  value=False)

    st.divider()
    run_btn = st.button("▶  Run forecast", use_container_width=True, type="primary")

    st.divider()
    st.caption("Built by Shreesh Gupta · Thapar Institute  \nHybrid ARIMA-LSTM · AirQualityUCI dataset")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("## 🌫️ AQI Forecaster — Hybrid ARIMA-LSTM")
st.markdown(
    "Upload your pollutant data (or use the built-in sample) and get "
    "hybrid ARIMA-LSTM forecasts with spike detection."
)

if "results" not in st.session_state:
    st.session_state["results"] = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if use_sample:
        csv_bytes = generate_sample_csv()
        df_raw = pd.read_csv(io.BytesIO(csv_bytes), sep=";", decimal=",")
    elif uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded, sep=";", decimal=",")
        except Exception:
            uploaded.seek(0)
            df_raw = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV file or enable the sample data toggle.")
        st.stop()

    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    def _cb(step, total, msg):
        pct = int(step / total * 100)
        progress_bar.progress(pct, text=msg)
        status_text.caption(f"Step {step}/{total} — {msg}")

    with st.spinner(""):
        output = run_pipeline(df_raw, progress_cb=_cb)

    progress_bar.empty()
    status_text.empty()
    st.session_state["results"] = output
    st.success("✅ Forecast complete!")

# ── Display results ───────────────────────────────────────────────────────────
res = st.session_state["results"]
if res is None:
    st.info("Configure your data source in the sidebar and click **▶ Run forecast** to begin.")
    st.stop()

m       = res["metrics"]
rdf     = res["results"]
df_full = res["df_full"]

latest_val = float(rdf["Ensemble"].iloc[-1])
cat_label, cat_colour = co_category(latest_val)

# ── KPI row ──────────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("R² Score",  f"{m['R2']:.3f}",  help="Coefficient of determination")
k2.metric("MAPE",      f"{m['MAPE']:.1f}%", help="Mean Absolute Percentage Error")
k3.metric("RMSE",      f"{m['RMSE']:.3f}", help="Root Mean Squared Error")
k4.metric("MAE",       f"{m['MAE']:.3f}",  help="Mean Absolute Error")

with k5:
    st.markdown(
        f"**Latest prediction**  \n"
        f"<span style='font-size:1.8rem;font-weight:600'>{latest_val:.2f}</span> mg/m³"
        f"{badge_html(cat_label, cat_colour)}",
        unsafe_allow_html=True,
    )

# ── Model config summary ──────────────────────────────────────────────────────
st.caption(
    f"**Model config** — Best sequence: {m['best_seq']}h  ·  "
    f"ARIMA weight: {m['arima_weight']}  ·  LSTM weight: {m['lstm_weight']}"
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "🔍 Model Analysis", "⚡ Spike Detection", "📥 Download"]
)

with tab1:
    st.subheader("Actual vs Predicted CO(GT)")
    st.plotly_chart(
        make_ts_chart(rdf, show_arima, show_lstm),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Predicted vs Actual (scatter)")
        st.plotly_chart(make_scatter_chart(rdf), use_container_width=True)
    with col_b:
        st.subheader("Residual distribution")
        st.plotly_chart(make_residual_chart(rdf), use_container_width=True)
        st.caption(
            "Residuals centred near zero indicate good fit. "
            "Right tail = underpredicted extreme events."
        )

with tab2:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Feature correlations with CO(GT)")
        fig_fi = make_feature_importance_chart(df_full)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis.")

    with col_r:
        st.subheader("Model performance breakdown")
        perf = pd.DataFrame({
            "Model":  ["ARIMA only", "LSTM only", "Hybrid Ensemble"],
            "MAE":    [
                round(float(np.mean(np.abs(rdf["Actual"] - rdf["ARIMA"]))), 4),
                round(float(np.mean(np.abs(rdf["Actual"] - rdf["LSTM"]))), 4),
                m["MAE"],
            ],
            "RMSE": [
                round(float(np.sqrt(np.mean((rdf["Actual"] - rdf["ARIMA"])**2))), 4),
                round(float(np.sqrt(np.mean((rdf["Actual"] - rdf["LSTM"])**2))), 4),
                m["RMSE"],
            ],
        })
        st.dataframe(perf, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("AQI category legend")
        for lo, hi, label, col in AQI_THRESHOLDS:
            st.markdown(
                f"{badge_html(label, col)} &nbsp; {lo}–{hi} mg/m³",
                unsafe_allow_html=True,
            )

with tab3:
    spike_df = rdf[rdf["Spike_flag"]].copy()
    spike_count = len(spike_df)

    sp1, sp2 = st.columns(2)
    sp1.metric("Spikes detected", spike_count)
    sp2.metric(
        "Spike threshold",
        f"{rdf['Actual'].quantile(0.90):.2f} mg/m³",
        help="90th percentile of actual values",
    )

    if spike_count:
        st.subheader("Spike events — dates & values")
        spike_display = spike_df[["Datetime", "Actual", "Ensemble", "Residual"]].copy()
        spike_display["Actual"]   = spike_display["Actual"].round(3)
        spike_display["Ensemble"] = spike_display["Ensemble"].round(3)
        spike_display["Residual"] = spike_display["Residual"].round(3)
        st.dataframe(spike_display, hide_index=True, use_container_width=True)

        # Spike timeline
        fig_spk = go.Figure()
        fig_spk.add_trace(go.Scatter(
            x=rdf["Datetime"], y=rdf["Actual"],
            line=dict(color="#94a3b8", width=1), name="Actual", opacity=0.6,
        ))
        fig_spk.add_trace(go.Scatter(
            x=spike_df["Datetime"], y=spike_df["Actual"],
            mode="markers", marker=dict(color="#ef4444", size=9, symbol="x"),
            name="Spike",
        ))
        fig_spk.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#f1f5f9", title="CO (mg/m³)"),
            font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        )
        st.plotly_chart(fig_spk, use_container_width=True)

        st.info(
            "⚡ **Proactive emission control** — spikes flagged above represent "
            "sudden pollution surges. These timestamps correspond to events where "
            "the model predicts values near or above the 90th-percentile threshold "
            "— a signal to review refinery emission controls."
        )
    else:
        st.success("No significant spikes detected in this dataset.")

with tab4:
    st.subheader("Download results")
    csv_out = rdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download predictions CSV",
        data=csv_out,
        file_name="aqi_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Download sample input CSV")
    sample_csv = generate_sample_csv()
    st.download_button(
        "⬇ Download sample input CSV",
        data=sample_csv,
        file_name="sample_airquality.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "Use this as a template for your own data. "
        "Ensure your file uses semicolon (`;`) separators and comma decimal notation."
    )
