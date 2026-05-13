import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import run_clustering

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parcl AI – Buyer Segmentation",
    page_icon="🏠",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252836);
        border: 1px solid #2e3347;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { color: #a0aec0; font-size: 0.85rem; margin: 0 0 0.3rem 0; }
    .metric-card p  { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .section-title  { color: #e2e8f0; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Parcl AI")
    st.markdown("**Buyer segmentation controls**")
    st.divider()
    selected_k = st.slider("Number of clusters", min_value=2, max_value=8, value=4, step=1)
    st.divider()
    st.caption("Data: clients.csv + properties.csv")

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training clustering model…")
def load_dashboard_data(k: int):
    try:
        artifacts = run_clustering(k=k)
        return artifacts, None
    except Exception as e:
        return None, str(e)


artifacts, error = load_dashboard_data(selected_k)

if error:
    st.error(f"❌ Could not load data: {error}")
    st.stop()

features        = artifacts["features"]
k               = artifacts["k"]
k_range         = artifacts["k_range"]
elbow_scores    = artifacts["elbow_scores"]
silhouette_scores = artifacts["silhouette_scores"]

CLUSTER_COLORS = px.colors.qualitative.Bold

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# 🏠 Parcl AI — Buyer Segmentation Dashboard")
st.markdown(f"Segmenting **{len(features):,} clients** into **{k} clusters** using KMeans + PCA")
st.divider()

# ── KPI row ────────────────────────────────────────────────────────────────
sold_clients = features[features["total_spent"] > 0]
avg_spent    = sold_clients["total_spent"].mean() if len(sold_clients) > 0 else 0
avg_sat      = features["satisfaction_score"].mean()
loan_pct     = features["loan_applied_flag"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><h3>Total Clients</h3><p>{len(features):,}</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><h3>Avg Spend (buyers)</h3><p>${avg_spent:,.0f}</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><h3>Avg Satisfaction</h3><p>{avg_sat:.2f} / 5</p></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card"><h3>Loan Applied %</h3><p>{loan_pct:.1f}%</p></div>""", unsafe_allow_html=True)

st.divider()

# ── PCA Scatter + Cluster Breakdown ───────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<p class="section-title">🔵 Cluster Map (PCA 2D)</p>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        features,
        x="pca_x", y="pca_y",
        color=features["cluster"].astype(str),
        hover_data=["client_id", "age", "satisfaction_score", "total_spent", "num_purchases"],
        labels={"color": "Cluster", "pca_x": "PCA Component 1", "pca_y": "PCA Component 2"},
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.7,
        height=420,
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
        font=dict(color="#c8cfe0"), legend_title_text="Cluster",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.markdown('<p class="section-title">📊 Cluster Size</p>', unsafe_allow_html=True)
    cluster_counts = features["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    cluster_counts = cluster_counts.sort_values("Cluster")
    cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)

    fig_bar = px.bar(
        cluster_counts,
        x="Cluster", y="Count",
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        text="Count",
        height=420,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
        font=dict(color="#c8cfe0"), showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Elbow + Silhouette ─────────────────────────────────────────────────────
st.divider()
col_e, col_s = st.columns(2)

with col_e:
    st.markdown('<p class="section-title">📉 Elbow Curve (Inertia)</p>', unsafe_allow_html=True)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_range, y=elbow_scores,
        mode="lines+markers",
        line=dict(color="#f6ad55", width=2),
        marker=dict(size=8, color="#f6ad55"),
        name="Inertia",
    ))
    fig_elbow.add_vline(x=selected_k, line_dash="dash", line_color="#fc8181", annotation_text=f"k={selected_k}")
    fig_elbow.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
        font=dict(color="#c8cfe0"),
        xaxis_title="k", yaxis_title="Inertia",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col_s:
    st.markdown('<p class="section-title">📈 Silhouette Score</p>', unsafe_allow_html=True)
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=k_range, y=silhouette_scores,
        mode="lines+markers",
        line=dict(color="#68d391", width=2),
        marker=dict(size=8, color="#68d391"),
        name="Silhouette",
    ))
    fig_sil.add_vline(x=selected_k, line_dash="dash", line_color="#fc8181", annotation_text=f"k={selected_k}")
    fig_sil.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
        font=dict(color="#c8cfe0"),
        xaxis_title="k", yaxis_title="Silhouette Score",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
    )
    st.plotly_chart(fig_sil, use_container_width=True)

# ── Cluster Profiles ───────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">🧩 Cluster Profiles</p>', unsafe_allow_html=True)

profile_cols = ["age", "satisfaction_score", "total_spent", "avg_price", "num_purchases", "avg_area", "loan_applied_flag", "is_individual"]
profile = (
    features.groupby("cluster")[profile_cols]
    .mean()
    .round(2)
    .reset_index()
)
profile.columns = ["Cluster", "Avg Age", "Avg Satisfaction", "Avg Total Spent ($)", "Avg Unit Price ($)", "Avg Purchases", "Avg Area (sqft)", "Loan Applied %", "Individual %"]
profile["Loan Applied %"] = (profile["Loan Applied %"] * 100).round(1)
profile["Individual %"]   = (profile["Individual %"]   * 100).round(1)
profile["Cluster"] = profile["Cluster"].astype(str)

st.dataframe(
    profile.style.format({
        "Avg Total Spent ($)": "${:,.0f}",
        "Avg Unit Price ($)":  "${:,.0f}",
        "Avg Area (sqft)":     "{:,.0f}",
    }).background_gradient(cmap="Blues", subset=["Avg Total Spent ($)", "Avg Satisfaction"]),
    use_container_width=True,
    hide_index=True,
)

# ── Radar Chart ────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">🕸 Cluster Radar (Normalised Features)</p>', unsafe_allow_html=True)

radar_cols = ["Avg Age", "Avg Satisfaction", "Avg Total Spent ($)", "Avg Purchases", "Avg Area (sqft)"]
norm = profile[radar_cols].copy()
for c in radar_cols:
    rng = norm[c].max() - norm[c].min()
    norm[c] = (norm[c] - norm[c].min()) / rng if rng > 0 else 0

fig_radar = go.Figure()
for i, row in profile.iterrows():
    vals = norm.loc[i, radar_cols].tolist()
    vals += vals[:1]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=radar_cols + [radar_cols[0]],
        fill="toself",
        name=f"Cluster {row['Cluster']}",
        line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
        opacity=0.6,
    ))
fig_radar.update_layout(
    polar=dict(bgcolor="rgba(20,22,35,0.6)", radialaxis=dict(visible=True, color="#c8cfe0")),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8cfe0"),
    height=420,
    margin=dict(l=40, r=40, t=30, b=30),
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Demographic breakdown ──────────────────────────────────────────────────
st.divider()
col_g, col_p = st.columns(2)

with col_g:
    st.markdown('<p class="section-title">👥 Gender Distribution per Cluster</p>', unsafe_allow_html=True)
    gdf = features.groupby(["cluster", "gender"]).size().reset_index(name="count")
    gdf["cluster"] = gdf["cluster"].astype(str)
    fig_g = px.bar(gdf, x="cluster", y="count", color="gender", barmode="group",
                   color_discrete_sequence=CLUSTER_COLORS, height=320)
    fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
                        font=dict(color="#c8cfe0"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with col_p:
    st.markdown('<p class="section-title">🎯 Acquisition Purpose per Cluster</p>', unsafe_allow_html=True)
    pdf = features.groupby(["cluster", "acquisition_purpose"]).size().reset_index(name="count")
    pdf["cluster"] = pdf["cluster"].astype(str)
    fig_p = px.bar(pdf, x="cluster", y="count", color="acquisition_purpose", barmode="stack",
                   color_discrete_sequence=px.colors.qualitative.Pastel, height=320)
    fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
                        font=dict(color="#c8cfe0"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_p, use_container_width=True)

# ── Referral Channel ───────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">📣 Referral Channel Distribution</p>', unsafe_allow_html=True)
rdf = features.groupby(["cluster", "referral_channel"]).size().reset_index(name="count")
rdf["cluster"] = rdf["cluster"].astype(str)
fig_r = px.bar(rdf, x="referral_channel", y="count", color="cluster", barmode="group",
               color_discrete_sequence=CLUSTER_COLORS, height=320)
fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,22,35,0.6)",
                    font=dict(color="#c8cfe0"), margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_r, use_container_width=True)

# ── Raw data explorer ──────────────────────────────────────────────────────
st.divider()
with st.expander("🔍 Explore Raw Cluster Data"):
    selected_cluster = st.selectbox("Filter by cluster", options=["All"] + sorted(features["cluster"].unique().tolist()))
    display_cols = ["client_id", "client_type", "gender", "age", "country", "region",
                    "acquisition_purpose", "satisfaction_score", "loan_applied_flag",
                    "total_spent", "num_purchases", "avg_price", "cluster"]
    df_show = features if selected_cluster == "All" else features[features["cluster"] == selected_cluster]
    st.dataframe(df_show[display_cols].reset_index(drop=True), use_container_width=True)

st.divider()
st.caption("Parcl AI Buyer Segmentation · Built with Streamlit · KMeans Clustering")
