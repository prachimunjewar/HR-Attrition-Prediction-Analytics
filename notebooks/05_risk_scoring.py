# ============================================================
# 05_risk_scoring.py — Attrition Risk Score per Employee
# AttritionIQ: HR Attrition Prediction & Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Model & Data ─────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating Attrition Risk Scores (All Employees)")
print("=" * 60)

xgb_model = joblib.load("data/xgb_model.pkl")
features  = pd.read_csv("data/feature_names.csv", header=None).squeeze().tolist()

# Load full encoded dataset (all 1470 employees)
df_full = pd.read_csv("data/hr_encoded.csv")
df_orig = pd.read_csv("data/hr_clean.csv")

X_all = df_full[features]
y_all = df_full["Attrition_Flag"]

# ── 2. Score All Employees ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Predicting Attrition Probability")
print("=" * 60)

attrition_probs = xgb_model.predict_proba(X_all)[:, 1]

# ── 3. Build Risk Score Table ────────────────────────────────
risk_df = pd.DataFrame({
    "EmployeeNumber"    : df_orig["EmployeeNumber"],
    "Department"        : df_orig["Department"],
    "JobRole"           : df_orig["JobRole"],
    "Age"               : df_orig["Age"],
    "Gender"            : df_orig["Gender"],
    "MonthlyIncome"     : df_orig["MonthlyIncome"],
    "YearsAtCompany"    : df_orig["YearsAtCompany"],
    "OverTime"          : df_orig["OverTime"],
    "JobSatisfaction"   : df_orig["JobSatisfaction"],
    "WorkLifeBalance"   : df_orig["WorkLifeBalance"],
    "Attrition_Actual"  : df_orig["Attrition"],
    "Attrition_Prob"    : attrition_probs,
    "Risk_Score"        : (attrition_probs * 100).round(1),  # 0-100 scale
})

# Risk tier classification
def classify_risk(prob):
    if prob >= 0.70: return "🔴 Critical"
    elif prob >= 0.50: return "🟠 High"
    elif prob >= 0.30: return "🟡 Medium"
    else: return "🟢 Low"

risk_df["Risk_Tier"] = pd.Series(attrition_probs).apply(classify_risk)

# Sort by risk
risk_df = risk_df.sort_values("Attrition_Prob", ascending=False).reset_index(drop=True)

print(f"\nRisk Tier Distribution:")
print(risk_df["Risk_Tier"].value_counts().to_string())

print(f"\nTop 10 Highest-Risk Employees:")
print(risk_df[["EmployeeNumber","Department","JobRole","MonthlyIncome",
               "OverTime","YearsAtCompany","Risk_Score","Risk_Tier"]].head(10).to_string(index=False))

# ── 4. Department Risk Summary ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Department Risk Summary")
print("=" * 60)

dept_risk = (risk_df.groupby("Department")
             .agg(
                 total_employees=("EmployeeNumber","count"),
                 avg_risk_score=("Risk_Score","mean"),
                 critical_count=("Risk_Tier", lambda x: (x=="🔴 Critical").sum()),
                 high_count=("Risk_Tier", lambda x: (x=="🟠 High").sum()),
             )
             .round(1)
             .sort_values("avg_risk_score", ascending=False))

print(dept_risk.to_string())

# ── 5. Role Risk Summary ─────────────────────────────────────
role_risk = (risk_df.groupby("JobRole")
             .agg(
                 total=("EmployeeNumber","count"),
                 avg_risk=("Risk_Score","mean"),
                 critical=("Risk_Tier", lambda x: (x=="🔴 Critical").sum()),
             )
             .round(1)
             .sort_values("avg_risk", ascending=False))

# ── 6. Visualizations ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("AttritionIQ — Employee Risk Scoring Dashboard", fontsize=16, fontweight="bold")

tier_colors = {"🔴 Critical":"#DC2626","🟠 High":"#F59E0B",
               "🟡 Medium":"#EAB308","🟢 Low":"#16A34A"}

# (A) Risk Distribution Histogram
ax = axes[0, 0]
ax.hist(attrition_probs, bins=40, color="#2563EB", alpha=0.8, edgecolor="white")
ax.axvline(0.70, color="#DC2626", linestyle="--", linewidth=2, label="Critical (70%)")
ax.axvline(0.50, color="#F59E0B", linestyle="--", linewidth=2, label="High (50%)")
ax.axvline(0.30, color="#EAB308", linestyle="--", linewidth=2, label="Medium (30%)")
ax.set_title("Attrition Risk Distribution", fontweight="bold")
ax.set_xlabel("Attrition Probability")
ax.set_ylabel("Number of Employees")
ax.legend(fontsize=8)

# (B) Risk Tier Donut
ax = axes[0, 1]
tier_counts = risk_df["Risk_Tier"].value_counts()
colors_pie  = [tier_colors.get(t, "#999") for t in tier_counts.index]
wedges, texts, autotexts = ax.pie(
    tier_counts.values, labels=tier_counts.index,
    autopct="%1.1f%%", colors=colors_pie,
    startangle=90, wedgeprops=dict(width=0.55)
)
for t in autotexts: t.set_fontsize(10); t.set_fontweight("bold")
ax.set_title("Risk Tier Breakdown", fontweight="bold")

# (C) Avg Risk by Department
ax = axes[0, 2]
dept_sorted = dept_risk["avg_risk_score"].sort_values()
colors_dept = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(dept_sorted)))
ax.barh(dept_sorted.index, dept_sorted.values, color=colors_dept)
ax.set_title("Avg Risk Score by Department", fontweight="bold")
ax.set_xlabel("Avg Risk Score (0-100)")
for i, v in enumerate(dept_sorted):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=10, fontweight="bold")

# (D) Risk Score by Salary Band
ax = axes[1, 0]
risk_df["salary_band"] = pd.cut(
    risk_df["MonthlyIncome"],
    bins=[0, 3000, 6000, 10000, 99999],
    labels=["< $3K", "$3K-$6K", "$6K-$10K", "> $10K"]
)
salary_risk = risk_df.groupby("salary_band")["Risk_Score"].mean().sort_values(ascending=False)
colors_sal = ["#DC2626","#F59E0B","#16A34A","#2563EB"]
bars = ax.bar(salary_risk.index, salary_risk.values, color=colors_sal)
ax.set_title("Avg Risk Score by Salary Band", fontweight="bold")
ax.set_ylabel("Avg Risk Score")
ax.set_xlabel("Salary Band")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", fontsize=10, fontweight="bold")

# (E) Risk by Overtime
ax = axes[1, 1]
ot_risk = risk_df.groupby("OverTime")["Risk_Score"].mean().sort_values(ascending=False)
ax.bar(ot_risk.index, ot_risk.values, color=["#DC2626","#2563EB"])
ax.set_title("Avg Risk by Overtime Status", fontweight="bold")
ax.set_ylabel("Avg Risk Score")
for i, (idx, val) in enumerate(ot_risk.items()):
    ax.text(i, val + 0.3, f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")

# (F) Top 10 Highest-Risk Roles
ax = axes[1, 2]
top_roles = role_risk["avg_risk"].head(10).sort_values()
colors_roles = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_roles)))
ax.barh(top_roles.index, top_roles.values, color=colors_roles)
ax.set_title("Top 10 High-Risk Job Roles", fontweight="bold")
ax.set_xlabel("Avg Risk Score")

plt.tight_layout()
plt.savefig("outputs/05_risk_scoring.png", dpi=150, bbox_inches="tight")
print("\nSaved -> outputs/05_risk_scoring.png")
plt.show()

# ── 7. Business Recommendations ──────────────────────────────
print("\n" + "=" * 60)
print("BUSINESS RECOMMENDATIONS")
print("=" * 60)

critical = risk_df[risk_df["Risk_Tier"] == "🔴 Critical"]
high     = risk_df[risk_df["Risk_Tier"] == "🟠 High"]

print(f"""
Based on ML analysis of {len(risk_df):,} employees:

1. IMMEDIATE ACTION ({len(critical)} Critical-Risk Employees):
   - Avg salary  : ${critical['MonthlyIncome'].mean():,.0f}
   - Overtime %  : {(critical['OverTime']=='Yes').mean()*100:.0f}%
   - Top dept    : {critical['Department'].value_counts().index[0]}
   Action → Salary review + overtime reduction for this group.

2. WATCHLIST ({len(high)} High-Risk Employees):
   - Avg tenure  : {high['YearsAtCompany'].mean():.1f} years
   - Avg sat     : {high['JobSatisfaction'].mean():.1f}/4
   Action → 1:1 manager check-ins, career development plans.

3. TOP 3 ATTRITION DRIVERS (from SHAP):
   See shap_importance.csv for ranked feature impact.
   Common patterns: Low income + High overtime + Low satisfaction.

4. FINANCIAL IMPACT ESTIMATE:
   Avg replacement cost per employee: ~$50,000
   Critical-risk employees          : {len(critical)}
   Potential loss if unaddressed    : ~${len(critical)*50000:,}
""")

# ── 8. Save Risk Table ───────────────────────────────────────
risk_df.to_csv("data/employee_risk_scores.csv", index=False)
print("Saved -> data/employee_risk_scores.csv")
print("\nAll notebooks complete! Launch dashboard:\n  streamlit run dashboard/app.py\n")
