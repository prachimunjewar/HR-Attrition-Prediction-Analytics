# ============================================================
# dashboard/app.py — AttritionIQ Streamlit Dashboard
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
    page_title="AttritionIQ — HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f, #2563EB);
        border-radius: 12px; padding: 18px 22px; color: white;
        box-shadow: 0 4px 15px rgba(37,99,235,0.3); margin-bottom: 8px;
    }
    .kpi-red   { background: linear-gradient(135deg, #7f1d1d, #DC2626); }
    .kpi-green { background: linear-gradient(135deg, #14532d, #16A34A); }
    .kpi-amber { background: linear-gradient(135deg, #78350f, #D97706); }
    .kpi-value { font-size: 2rem; font-weight: 800; margin: 4px 0; }
    .kpi-label { font-size: 0.82rem; opacity: 0.85; letter-spacing: 0.05em; }
    .kpi-sub   { font-size: 0.85rem; margin-top: 4px; opacity: 0.9; }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #2563EB; padding-left: 12px; margin: 18px 0 10px;
    }
    div[data-testid="stSidebar"] { background: #f0f4ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    risk   = pd.read_csv(f"{base}/data/employee_risk_scores.csv")
    shap_i = pd.read_csv(f"{base}/data/shap_importance.csv")
    return risk, shap_i

data_ready = os.path.exists("data/employee_risk_scores.csv")

# SIDEBAR
with st.sidebar:
    st.title("AttritionIQ")
    st.caption("HR Attrition Prediction Dashboard")
    st.markdown("---")

    if data_ready:
        risk_df, shap_df = load_data()
        departments  = ["All"] + sorted(risk_df["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", departments)

        roles = ["All"] + sorted(risk_df["JobRole"].dropna().unique().tolist())
        selected_role = st.selectbox("Job Role", roles)

        tiers = ["All","🔴 Critical","🟠 High","🟡 Medium","🟢 Low"]
        selected_tier = st.selectbox("Risk Tier", tiers)

        overtime_filter = st.selectbox("OverTime", ["All","Yes","No"])

        st.markdown("---")
        st.success("**XGBoost** — AUC: 0.89", icon="🏆")
        st.info("SHAP explainability enabled", icon="🔍")

    st.markdown("---")
    st.caption("IBM HR Analytics Dataset · Kaggle")

# MAIN
st.title("👥 AttritionIQ — HR Attrition Prediction & Analytics")
st.caption("XGBoost + SHAP | IBM HR Analytics Dataset (1,470 employees)")

if not data_ready:
    st.warning("Run all notebooks first to generate data files.")
    st.code("""
python notebooks/01_EDA.py
python notebooks/02_preprocessing.py
python notebooks/03_logistic_regression.py
python notebooks/04_xgboost_model.py
python notebooks/05_risk_scoring.py
    """, language="bash")
    st.stop()

risk_df, shap_df = load_data()

mask = pd.Series([True] * len(risk_df))
if selected_dept   != "All": mask &= risk_df["Department"] == selected_dept
if selected_role   != "All": mask &= risk_df["JobRole"]    == selected_role
if selected_tier   != "All": mask &= risk_df["Risk_Tier"]  == selected_tier
if overtime_filter != "All": mask &= risk_df["OverTime"]   == overtime_filter

df = risk_df[mask].copy()

# KPI CARDS
st.markdown('<div class="section-header">📊 Workforce Risk Overview</div>', unsafe_allow_html=True)

total     = len(df)
critical  = (df["Risk_Tier"] == "🔴 Critical").sum()
high_risk = (df["Risk_Tier"] == "🟠 High").sum()
actual    = (df["Attrition_Actual"] == "Yes").sum()
cost_est  = critical * 50000

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f'<div class="kpi-card"><div class="kpi-label">TOTAL EMPLOYEES</div><div class="kpi-value">{total:,}</div><div class="kpi-sub">In current filter</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card kpi-red"><div class="kpi-label">🔴 CRITICAL RISK</div><div class="kpi-value">{critical}</div><div class="kpi-sub">{critical/max(total,1)*100:.1f}% of workforce</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">🟠 HIGH RISK</div><div class="kpi-value">{high_risk}</div><div class="kpi-sub">{high_risk/max(total,1)*100:.1f}% of workforce</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-label">ACTUAL ATTRITION</div><div class="kpi-value">{actual}</div><div class="kpi-sub">{actual/max(total,1)*100:.1f}% actual rate</div></div>', unsafe_allow_html=True)
c5.markdown(f'<div class="kpi-card"><div class="kpi-label">EST. RISK COST</div><div class="kpi-value">${cost_est/1e6:.1f}M</div><div class="kpi-sub">Critical x $50K avg</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ROW 1
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">🎯 Risk Score Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["Attrition_Prob"], nbinsx=40, marker_color="#2563EB", opacity=0.8))
    fig.add_vline(x=0.70, line_dash="dash", line_color="#DC2626", annotation_text="Critical (70%)")
    fig.add_vline(x=0.50, line_dash="dash", line_color="#F59E0B", annotation_text="High (50%)")
    fig.add_vline(x=0.30, line_dash="dash", line_color="#EAB308", annotation_text="Medium (30%)")
    fig.update_layout(height=320, template="plotly_white",
                      xaxis_title="Attrition Probability", yaxis_title="Employees", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">🏢 Risk by Department</div>', unsafe_allow_html=True)
    dept_risk = (df.groupby("Department").agg(avg_risk=("Risk_Score","mean"), total=("EmployeeNumber","count"))
                   .round(1).sort_values("avg_risk", ascending=True).reset_index())
    fig = px.bar(dept_risk, y="Department", x="avg_risk", orientation="h",
                 template="plotly_white", color="avg_risk",
                 color_continuous_scale="RdYlGn_r",
                 text=dept_risk["avg_risk"].map("{:.1f}".format),
                 labels={"avg_risk":"Avg Risk Score"})
    fig.update_layout(height=320, coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ROW 2
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-header">🔍 SHAP Feature Importance (Top 15)</div>', unsafe_allow_html=True)
    shap_top = shap_df.head(15).sort_values("mean_shap")
    fig = px.bar(shap_top, x="mean_shap", y="feature", orientation="h",
                 template="plotly_white", color="mean_shap",
                 color_continuous_scale="Reds",
                 labels={"mean_shap":"Mean |SHAP Value|","feature":""})
    fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown('<div class="section-header">💰 Risk by Salary Band</div>', unsafe_allow_html=True)
    df["salary_band"] = pd.cut(df["MonthlyIncome"],
                               bins=[0,3000,6000,10000,99999],
                               labels=["<$3K","$3K-$6K","$6K-$10K",">$10K"])
    sal = (df.groupby("salary_band", observed=True)
             .agg(avg_risk=("Risk_Score","mean"), count=("EmployeeNumber","count"))
             .reset_index())
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=sal["salary_band"], y=sal["avg_risk"], name="Avg Risk",
                         marker_color=["#DC2626","#F59E0B","#16A34A","#2563EB"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=sal["salary_band"], y=sal["count"], mode="lines+markers",
                             name="Employees", line=dict(color="#7C3AED", width=2.5)), secondary_y=True)
    fig.update_layout(height=380, template="plotly_white", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="Avg Risk Score", secondary_y=False)
    fig.update_yaxes(title_text="Employee Count", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ROW 3: Overtime + Satisfaction
col5, col6 = st.columns(2)

with col5:
    st.markdown('<div class="section-header">⏰ Overtime vs Risk</div>', unsafe_allow_html=True)
    ot = df.groupby("OverTime").agg(avg_risk=("Risk_Score","mean"), count=("EmployeeNumber","count")).reset_index()
    fig = px.bar(ot, x="OverTime", y="avg_risk", template="plotly_white",
                 color="OverTime", color_discrete_map={"Yes":"#DC2626","No":"#2563EB"},
                 text=ot["avg_risk"].map("{:.1f}".format),
                 labels={"avg_risk":"Avg Risk Score","OverTime":"Overtime Status"})
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.markdown('<div class="section-header">😊 Job Satisfaction vs Risk</div>', unsafe_allow_html=True)
    sat = df.groupby("JobSatisfaction")["Risk_Score"].mean().reset_index()
    fig = px.line(sat, x="JobSatisfaction", y="Risk_Score", markers=True,
                  template="plotly_white",
                  labels={"JobSatisfaction":"Satisfaction (1=Low, 4=High)","Risk_Score":"Avg Risk Score"})
    fig.update_traces(line_color="#DC2626", line_width=2.5, marker_size=10)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Employee Risk Register Table
st.markdown('<div class="section-header">🗂️ Employee Risk Register (Top 100)</div>', unsafe_allow_html=True)

display_cols = ["EmployeeNumber","Department","JobRole","Age","MonthlyIncome",
                "OverTime","YearsAtCompany","JobSatisfaction","Risk_Score","Risk_Tier","Attrition_Actual"]

def color_tier(val):
    if "Critical" in str(val): return "color: #DC2626; font-weight: bold"
    if "High"     in str(val): return "color: #F59E0B; font-weight: bold"
    if "Medium"   in str(val): return "color: #EAB308"
    if "Low"      in str(val): return "color: #16A34A"
    return ""

st.dataframe(
    df[display_cols].sort_values("Risk_Score", ascending=False).head(100)
                    .style.applymap(color_tier, subset=["Risk_Tier"]),
    use_container_width=True, hide_index=True, height=400
)

with st.expander("📊 Model Comparison — XGBoost vs Logistic Regression"):
    col_a, col_b = st.columns(2)
    with col_a:
        metrics = pd.DataFrame({
            "Metric":    ["AUC-ROC","Precision (Attrition)","Recall (Attrition)","F1-Score"],
            "Logistic":  ["0.81","0.58","0.71","0.64"],
            "XGBoost":   ["0.89","0.72","0.78","0.75"],
        })
        st.dataframe(metrics.style.set_properties(
            subset=["XGBoost"], **{"background-color":"#e8f4e8","font-weight":"bold"}
        ), hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("""
        **Why XGBoost wins:**
        - Captures non-linear interactions (overtime x low salary)
        - Handles 5:1 class imbalance via `scale_pos_weight`
        - Confirmed by 5-fold CV — not overfitting
        - SHAP gives full explainability despite complexity
        """)

st.markdown("---")
st.caption("AttritionIQ · XGBoost + SHAP + Streamlit · IBM HR Analytics Dataset")
