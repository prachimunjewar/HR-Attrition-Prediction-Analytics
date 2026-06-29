# ============================================================
# dashboard/app.py — AttritionIQ Streamlit Dashboard (Redesigned)
# Run: streamlit run dashboard/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="AttritionIQ — HR Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLES ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1929;
    color: #E2E8F0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060F1A !important;
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] caption {
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] h1 {
    color: #F7F9FC !important;
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* ── Selectbox in sidebar ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #0F2035 !important;
    border: 1px solid #1E3A5F !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 1400px;
}

/* ── Page title ── */
.page-header {
    display: flex;
    align-items: flex-end;
    gap: 16px;
    margin-bottom: 6px;
    padding-bottom: 20px;
    border-bottom: 1px solid #1E3A5F;
}
.page-title {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #F7F9FC;
    line-height: 1;
}
.page-title span { color: #4F46E5; }
.page-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    background: #4F46E520;
    border: 1px solid #4F46E540;
    color: #818CF8;
    padding: 4px 10px;
    border-radius: 99px;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 6px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #0F2035;
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, #4F46E5, #818CF8); }
.kpi-card.red::before   { background: linear-gradient(90deg, #F4503A, #FB923C); }
.kpi-card.amber::before { background: linear-gradient(90deg, #F59E0B, #FCD34D); }
.kpi-card.green::before { background: linear-gradient(90deg, #10B981, #34D399); }
.kpi-card.purple::before{ background: linear-gradient(90deg, #8B5CF6, #C4B5FD); }

.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #64748B;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: #F7F9FC;
    font-family: 'Inter', sans-serif;
}
.kpi-value.red    { color: #F4503A; }
.kpi-value.amber  { color: #F59E0B; }
.kpi-value.green  { color: #10B981; }
.kpi-value.purple { color: #A78BFA; }
.kpi-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-pulse {
    display: inline-block;
    width: 7px; height: 7px;
    background: #F4503A;
    border-radius: 50%;
    margin-right: 5px;
    box-shadow: 0 0 0 0 rgba(244,80,58,0.4);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(244,80,58,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(244,80,58,0); }
    100% { box-shadow: 0 0 0 0 rgba(244,80,58,0); }
}

/* ── Section Headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 4px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E3A5F;
}

/* ── Chart containers ── */
.chart-card {
    background: #0F2035;
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
}

/* ── Metric pills in sidebar ── */
.metric-pill {
    background: #0F2035;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
}
.metric-pill .mp-label { color: #64748B; }
.metric-pill .mp-value { color: #818CF8; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

/* ── Risk tier badges in table ── */
.risk-critical { color: #F4503A !important; font-weight: 700; }
.risk-high     { color: #F59E0B !important; font-weight: 600; }
.risk-medium   { color: #EAB308 !important; }
.risk-low      { color: #10B981 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #0F2035;
    border-radius: 12px;
    border: 1px solid #1E3A5F;
}
[data-testid="stDataFrame"] th {
    background: #060F1A !important;
    color: #64748B !important;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stDataFrame"] td {
    color: #CBD5E1 !important;
    font-size: 0.82rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0F2035 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #94A3B8 !important;
    font-size: 0.85rem;
}

/* ── Streamlit default element overrides ── */
.stCaption { color: #475569 !important; }
h1, h2, h3 { color: #F7F9FC !important; }

/* ── Warning / info boxes ── */
[data-testid="stAlert"] {
    background: #0F2035 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
}

/* ── Code block ── */
.stCode { border-radius: 10px; }

/* ── Sidebar model card ── */
.model-card {
    background: linear-gradient(135deg, #1E1B4B, #312E81);
    border: 1px solid #4F46E540;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0;
}
.model-card .mc-title {
    font-size: 0.72rem;
    color: #818CF8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}
.model-card .mc-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #F7F9FC;
    letter-spacing: -0.02em;
    margin: 2px 0;
}
.model-card .mc-sub { font-size: 0.72rem; color: #6D6A9C; }

/* ── Risk tier legend in sidebar ── */
.tier-row {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0;
    font-size: 0.8rem;
    color: #94A3B8;
    border-bottom: 1px solid #1E3A5F20;
}
.tier-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    risk   = pd.read_csv(f"{base}/data/employee_risk_scores.csv")
    shap_i = pd.read_csv(f"{base}/data/shap_importance.csv")
    return risk, shap_i

data_ready = os.path.exists("data/employee_risk_scores.csv")

# ── PLOTLY DARK THEME ─────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8", size=11),
    xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", tickcolor="#475569"),
    yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", tickcolor="#475569"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E3A5F"),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <h1 style="font-size:1.5rem; margin-bottom:0; color:#F7F9FC; letter-spacing:-0.03em;">
        Attrition<span style="color:#4F46E5;">IQ</span>
    </h1>
    """, unsafe_allow_html=True)
    st.caption("HR Intelligence Platform")

    st.markdown("---")
    st.markdown('<p style="font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase; color:#475569; font-weight:600;">Filters</p>', unsafe_allow_html=True)

    if data_ready:
        risk_df, shap_df = load_data()
        departments  = ["All"] + sorted(risk_df["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", departments)

        roles = ["All"] + sorted(risk_df["JobRole"].dropna().unique().tolist())
        selected_role = st.selectbox("Job Role", roles)

        tiers = ["All", "🔴 Critical", "🟠 High", "🟡 Medium", "🟢 Low"]
        selected_tier = st.selectbox("Risk Tier", tiers)

        overtime_filter = st.selectbox("OverTime", ["All", "Yes", "No"])

    st.markdown("---")
    st.markdown('<p style="font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase; color:#475569; font-weight:600;">Model Performance</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <div class="mc-title">XGBoost · Champion</div>
        <div class="mc-value">AUC 0.89</div>
        <div class="mc-sub">5-fold CV · Scale pos weight</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:10px 14px; background:#0F2035; border:1px solid #1E3A5F; border-radius:10px; margin-top:8px;">
        <p style="font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase; color:#475569; font-weight:600; margin-bottom:8px;">Risk Tiers</p>
        <div class="tier-row"><div class="tier-dot" style="background:#F4503A;"></div> Critical  ≥ 70% prob</div>
        <div class="tier-row"><div class="tier-dot" style="background:#F59E0B;"></div> High      50 – 70%</div>
        <div class="tier-row"><div class="tier-dot" style="background:#EAB308;"></div> Medium    30 – 50%</div>
        <div class="tier-row"><div class="tier-dot" style="background:#10B981;"></div> Low       < 30%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("IBM HR Analytics · 1,470 employees")
    st.caption("SHAP explainability enabled")

# ── MAIN ──────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div>
        <div class="page-badge">PREDICTIVE PEOPLE ANALYTICS</div>
        <div class="page-title">Attrition<span>IQ</span></div>
        <div class="page-sub">XGBoost + SHAP · IBM HR Analytics Dataset · 1,470 employees</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not data_ready:
    st.warning("⚠️  Run all notebooks first to generate data files.")
    st.code("""python notebooks/01_EDA.py
python notebooks/02_preprocessing.py
python notebooks/03_logistic_regression.py
python notebooks/04_xgboost_model.py
python notebooks/05_risk_scoring.py""", language="bash")
    st.stop()

risk_df, shap_df = load_data()

# Apply filters
mask = pd.Series([True] * len(risk_df))
if selected_dept   != "All": mask &= risk_df["Department"] == selected_dept
if selected_role   != "All": mask &= risk_df["JobRole"]    == selected_role
if selected_tier   != "All": mask &= risk_df["Risk_Tier"]  == selected_tier
if overtime_filter != "All": mask &= risk_df["OverTime"]   == overtime_filter
df = risk_df[mask].copy()

# ── KPI CARDS ─────────────────────────────────────────────────
total     = len(df)
critical  = (df["Risk_Tier"] == "🔴 Critical").sum()
high_risk = (df["Risk_Tier"] == "🟠 High").sum()
actual    = (df["Attrition_Actual"] == "Yes").sum()
cost_est  = critical * 50_000
attrition_rate = actual / max(total, 1) * 100
critical_pct   = critical / max(total, 1) * 100

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card blue">
        <div class="kpi-label">Total Employees</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-sub">In current view</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-label">Critical Risk</div>
        <div class="kpi-value red"><span class="kpi-pulse"></span>{critical}</div>
        <div class="kpi-sub">{critical_pct:.1f}% of workforce</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-label">High Risk</div>
        <div class="kpi-value amber">{high_risk}</div>
        <div class="kpi-sub">{high_risk/max(total,1)*100:.1f}% of workforce</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-label">Actual Attrition</div>
        <div class="kpi-value green">{actual}</div>
        <div class="kpi-sub">{attrition_rate:.1f}% actual rate</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">Est. Risk Exposure</div>
        <div class="kpi-value purple">${cost_est/1e6:.1f}M</div>
        <div class="kpi-sub">Critical × $50K avg cost</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ROW 1: Risk Distribution + Dept Risk ─────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Attrition_Prob"], nbinsx=40,
        marker=dict(color="#4F46E5", opacity=0.85,
                    line=dict(color="#818CF8", width=0.5))
    ))
    for x_val, color, label in [
        (0.70, "#F4503A", "Critical 70%"),
        (0.50, "#F59E0B", "High 50%"),
        (0.30, "#EAB308", "Med 30%"),
    ]:
        fig.add_vline(x=x_val, line_dash="dash", line_color=color, line_width=1.5,
                      annotation_text=label, annotation_font_color=color,
                      annotation_font_size=10)
    fig.update_layout(height=300, xaxis_title="Attrition Probability",
                      yaxis_title="Employees", showlegend=False, **CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">Avg Risk by Department</div>', unsafe_allow_html=True)
    dept_risk = (df.groupby("Department")
                   .agg(avg_risk=("Risk_Score", "mean"), total=("EmployeeNumber", "count"))
                   .round(1).sort_values("avg_risk").reset_index())
    fig = go.Figure(go.Bar(
        y=dept_risk["Department"], x=dept_risk["avg_risk"],
        orientation="h", text=dept_risk["avg_risk"].map("{:.1f}".format),
        textposition="outside", textfont=dict(color="#94A3B8", size=11),
        marker=dict(
            color=dept_risk["avg_risk"],
            colorscale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#F4503A"]],
            line=dict(width=0)
        )
    ))
    fig.update_layout(height=300, showlegend=False, **CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 2: SHAP + Salary Band ─────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-header">SHAP Feature Importance — Top 15</div>', unsafe_allow_html=True)
    shap_top = shap_df.head(15).sort_values("mean_shap")
    fig = go.Figure(go.Bar(
        x=shap_top["mean_shap"], y=shap_top["feature"],
        orientation="h",
        text=shap_top["mean_shap"].map("{:.3f}".format),
        textposition="outside", textfont=dict(color="#94A3B8", size=10),
        marker=dict(
            color=shap_top["mean_shap"],
            colorscale=[[0, "#312E81"], [0.5, "#4F46E5"], [1, "#F4503A"]],
            line=dict(width=0)
        )
    ))
    fig.update_layout(height=380, **CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown('<div class="section-header">Risk vs Salary Band</div>', unsafe_allow_html=True)
    df["salary_band"] = pd.cut(df["MonthlyIncome"],
                               bins=[0, 3000, 6000, 10000, 99999],
                               labels=["< $3K", "$3K–$6K", "$6K–$10K", "> $10K"])
    sal = (df.groupby("salary_band", observed=True)
             .agg(avg_risk=("Risk_Score", "mean"), count=("EmployeeNumber", "count"))
             .reset_index())
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=sal["salary_band"], y=sal["avg_risk"], name="Avg Risk Score",
        marker=dict(color=["#F4503A", "#F59E0B", "#10B981", "#4F46E5"],
                    opacity=0.9, line=dict(width=0)),
        text=sal["avg_risk"].map("{:.1f}".format),
        textposition="outside", textfont=dict(color="#94A3B8", size=10)
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=sal["salary_band"], y=sal["count"], mode="lines+markers",
        name="Headcount", line=dict(color="#A78BFA", width=2, dash="dot"),
        marker=dict(size=8, color="#A78BFA", line=dict(color="#F7F9FC", width=1))
    ), secondary_y=True)
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.08, x=0),
                      **CHART_THEME)
    fig.update_yaxes(title_text="Avg Risk Score", secondary_y=False,
                     gridcolor="#1E3A5F", tickcolor="#475569", color="#94A3B8")
    fig.update_yaxes(title_text="Headcount", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", tickcolor="#475569", color="#A78BFA")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 3: Overtime + Satisfaction ────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.markdown('<div class="section-header">Overtime vs Attrition Risk</div>', unsafe_allow_html=True)
    ot = (df.groupby("OverTime")
            .agg(avg_risk=("Risk_Score", "mean"), count=("EmployeeNumber", "count"))
            .reset_index())
    fig = go.Figure(go.Bar(
        x=ot["OverTime"], y=ot["avg_risk"],
        text=ot["avg_risk"].map("{:.1f}".format),
        textposition="outside", textfont=dict(color="#94A3B8"),
        marker=dict(
            color=["#4F46E5" if v == "No" else "#F4503A" for v in ot["OverTime"]],
            opacity=0.9, line=dict(width=0)
        )
    ))
    fig.update_layout(height=280, showlegend=False,
                      xaxis_title="Overtime Status", yaxis_title="Avg Risk Score",
                      **CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.markdown('<div class="section-header">Job Satisfaction vs Risk</div>', unsafe_allow_html=True)
    sat = df.groupby("JobSatisfaction")["Risk_Score"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sat["JobSatisfaction"], y=sat["Risk_Score"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(244,80,58,0.08)",
        line=dict(color="#F4503A", width=2.5),
        marker=dict(size=10, color="#F4503A",
                    line=dict(color="#F7F9FC", width=2))
    ))
    fig.update_layout(height=280,
                      xaxis=dict(title="Satisfaction (1 = Low, 4 = High)",
                                 tickvals=[1,2,3,4],
                                 gridcolor="#1E3A5F", tickcolor="#475569"),
                      yaxis=dict(title="Avg Risk Score", gridcolor="#1E3A5F"),
                      **CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

# ── EMPLOYEE RISK REGISTER ─────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:8px;">Employee Risk Register — Top 100</div>', unsafe_allow_html=True)

display_cols = ["EmployeeNumber", "Department", "JobRole", "Age", "MonthlyIncome",
                "OverTime", "YearsAtCompany", "JobSatisfaction",
                "Risk_Score", "Risk_Tier", "Attrition_Actual"]

def color_tier(val):
    if "Critical" in str(val): return "color: #F4503A; font-weight: 700"
    if "High"     in str(val): return "color: #F59E0B; font-weight: 600"
    if "Medium"   in str(val): return "color: #EAB308"
    if "Low"      in str(val): return "color: #10B981"
    return ""

def color_risk_score(val):
    try:
        v = float(val)
        if v >= 70: return "color: #F4503A; font-weight: 700; font-family: 'JetBrains Mono', monospace"
        if v >= 50: return "color: #F59E0B; font-weight: 600"
        if v >= 30: return "color: #EAB308"
        return "color: #10B981"
    except: return ""

st.dataframe(
    df[display_cols].sort_values("Risk_Score", ascending=False).head(100)
      .style
      .map(color_tier, subset=["Risk_Tier"])
      .map(color_risk_score, subset=["Risk_Score"]),
    use_container_width=True, hide_index=True, height=420
)

# ── MODEL COMPARISON ──────────────────────────────────────────
with st.expander("📊  Model Comparison — XGBoost vs Logistic Regression"):
    col_a, col_b = st.columns([1, 1.2])
    with col_a:
        metrics = pd.DataFrame({
            "Metric":   ["AUC-ROC", "Precision (Attrition)", "Recall (Attrition)", "F1-Score"],
            "Logistic": ["0.81",    "0.58",                  "0.71",               "0.64"],
            "XGBoost":  ["0.89",    "0.72",                  "0.78",               "0.75"],
        })
        def highlight_xgb(col):
            return ["background-color: #1E3A5F; color:#10B981; font-weight:700"
                    if col.name == "XGBoost" else "" for _ in col]
        st.dataframe(
            metrics.style.apply(highlight_xgb),
            hide_index=True, use_container_width=True
        )
    with col_b:
        st.markdown("""
        <div style="padding:16px 20px; background:#060F1A; border:1px solid #1E3A5F; border-radius:10px; font-size:0.83rem; line-height:1.8; color:#94A3B8;">
            <p style="color:#818CF8; font-weight:600; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;">Why XGBoost wins</p>
            <p>📌 Captures <strong style="color:#E2E8F0;">non-linear interactions</strong> — overtime combined with low salary is a risk multiplier, not a sum</p>
            <p>⚖️  Handles <strong style="color:#E2E8F0;">5:1 class imbalance</strong> via <code style="background:#1E3A5F; padding:1px 5px; border-radius:4px;">scale_pos_weight</code></p>
            <p>✅ Confirmed by <strong style="color:#E2E8F0;">5-fold cross-validation</strong> — no overfitting</p>
            <p>🔍 Full <strong style="color:#E2E8F0;">SHAP explainability</strong> — every prediction has a reason</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("AttritionIQ · XGBoost + SHAP + Streamlit · IBM HR Analytics Dataset")
