# ============================================================
# 01_EDA.py — Exploratory Data Analysis
# AttritionIQ: HR Attrition Prediction & Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sqlite3
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading IBM HR Analytics Dataset")
print("=" * 60)

df = pd.read_csv(r"data\HR_Employee_Attrition.csv")
df.columns = df.columns.str.strip()

print(f"Shape  : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Target : Attrition — {df['Attrition'].value_counts().to_dict()}")

df["Attrition_Flag"] = (df["Attrition"] == "Yes").astype(int)
attrition_rate = df["Attrition_Flag"].mean() * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.1f}%")

# ── 2. Data Quality ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Data Quality")
print("=" * 60)

print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates    : {df.duplicated().sum()}")

drop_cols = [c for c in ["EmployeeCount", "Over18", "StandardHours"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped constant columns: {drop_cols}")

# ── 3. SQLite Analytics ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: SQL Analytics")
print("=" * 60)

conn = sqlite3.connect("data/hr_attrition.db")
df.to_sql("hr_data", conn, if_exists="replace", index=False)

dept_sql = pd.read_sql("""
    SELECT Department,
           COUNT(*) AS total,
           SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS left_count,
           ROUND(SUM(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END)/COUNT(*)*100,2) AS attrition_pct
    FROM hr_data GROUP BY Department ORDER BY attrition_pct DESC
""", conn)

salary_sql = pd.read_sql("""
    SELECT CASE
               WHEN MonthlyIncome < 3000  THEN '< $3K'
               WHEN MonthlyIncome < 6000  THEN '$3K-$6K'
               WHEN MonthlyIncome < 10000 THEN '$6K-$10K'
               ELSE '> $10K'
           END AS salary_band,
           COUNT(*) AS total,
           ROUND(SUM(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END)/COUNT(*)*100,2) AS attrition_pct
    FROM hr_data GROUP BY salary_band ORDER BY attrition_pct DESC
""", conn)
conn.close()

print("Attrition by Department:\n", dept_sql.to_string(index=False))
print("\nAttrition by Salary Band:\n", salary_sql.to_string(index=False))

# ── 4. Business KPIs ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Business KPIs")
print("=" * 60)

left   = df[df["Attrition"] == "Yes"]
stayed = df[df["Attrition"] == "No"]

print(f"  Total Employees      : {len(df):,}")
print(f"  Employees Left       : {len(left):,} ({attrition_rate:.1f}%)")
print(f"  Avg Income (Left)    : ${left['MonthlyIncome'].mean():,.0f}")
print(f"  Avg Income (Stayed)  : ${stayed['MonthlyIncome'].mean():,.0f}")
print(f"  Overtime (Left)      : {(left['OverTime']=='Yes').mean()*100:.1f}%")
print(f"  Overtime (Stayed)    : {(stayed['OverTime']=='Yes').mean()*100:.1f}%")
print(f"  Avg Tenure (Left)    : {left['YearsAtCompany'].mean():.1f} years")
print(f"  Avg Tenure (Stayed)  : {stayed['YearsAtCompany'].mean():.1f} years")

# ── 5. EDA Plots ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: EDA Visualizations")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("AttritionIQ — Exploratory Data Analysis", fontsize=16, fontweight="bold")
COLORS = {"Yes": "#DC2626", "No": "#2563EB"}

# (A) Donut chart
ax = axes[0, 0]
counts = df["Attrition"].value_counts()
ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
       colors=["#2563EB","#DC2626"], startangle=90,
       wedgeprops=dict(width=0.55))
ax.set_title("Attrition Distribution", fontweight="bold")

# (B) By Department
ax = axes[0, 1]
dept_pct = df.groupby("Department")["Attrition_Flag"].mean() * 100
dept_pct.sort_values().plot(kind="barh", ax=ax, color="#2563EB")
ax.set_title("Attrition Rate by Department", fontweight="bold")
ax.set_xlabel("Attrition Rate (%)")
for i, v in enumerate(dept_pct.sort_values()):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

# (C) By Job Role
ax = axes[0, 2]
role_pct = df.groupby("JobRole")["Attrition_Flag"].mean() * 100
role_pct.sort_values().plot(kind="barh", ax=ax, color="#7C3AED")
ax.set_title("Attrition Rate by Job Role", fontweight="bold")
ax.set_xlabel("Attrition Rate (%)")

# (D) Income Distribution
ax = axes[1, 0]
for val, color in COLORS.items():
    ax.hist(df[df["Attrition"]==val]["MonthlyIncome"], bins=30, alpha=0.6, color=color, label=f"Attrition={val}")
ax.set_title("Monthly Income Distribution", fontweight="bold")
ax.set_xlabel("Monthly Income ($)")
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# (E) Overtime
ax = axes[1, 1]
ot_data = df.groupby(["OverTime","Attrition"]).size().unstack(fill_value=0)
ot_pct  = ot_data.div(ot_data.sum(axis=1), axis=0) * 100
ot_pct.plot(kind="bar", ax=ax, color=["#2563EB","#DC2626"], rot=0)
ax.set_title("Overtime vs Attrition", fontweight="bold")
ax.set_xlabel("OverTime")
ax.set_ylabel("Percentage (%)")
ax.legend(["Stayed","Left"])

# (F) Job Satisfaction
ax = axes[1, 2]
sat_pct = df.groupby("JobSatisfaction")["Attrition_Flag"].mean() * 100
colors_sat = ["#DC2626","#F59E0B","#16A34A","#2563EB"]
ax.bar(sat_pct.index, sat_pct.values, color=colors_sat)
ax.set_title("Attrition by Job Satisfaction", fontweight="bold")
ax.set_xlabel("Job Satisfaction (1=Low, 4=High)")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(["Low","Medium","High","Very High"])
for i, v in zip(sat_pct.index, sat_pct.values):
    ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=9)

# (G) Tenure
ax = axes[2, 0]
for val, color in COLORS.items():
    ax.hist(df[df["Attrition"]==val]["YearsAtCompany"], bins=20, alpha=0.6, color=color, label=f"Attrition={val}")
ax.set_title("Years at Company", fontweight="bold")
ax.set_xlabel("Years")
ax.legend()

# (H) Age
ax = axes[2, 1]
for val, color in COLORS.items():
    ax.hist(df[df["Attrition"]==val]["Age"], bins=20, alpha=0.6, color=color, label=f"Attrition={val}")
ax.set_title("Age Distribution by Attrition", fontweight="bold")
ax.set_xlabel("Age")
ax.legend()

# (I) Work-Life Balance
ax = axes[2, 2]
wlb_pct = df.groupby("WorkLifeBalance")["Attrition_Flag"].mean() * 100
ax.bar(wlb_pct.index, wlb_pct.values, color="#F59E0B")
ax.set_title("Attrition by Work-Life Balance", fontweight="bold")
ax.set_xlabel("Work-Life Balance (1=Bad, 4=Best)")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(["Bad","Good","Better","Best"])

plt.tight_layout()
plt.savefig("outputs/01_EDA_plots.png", dpi=150, bbox_inches="tight")
print("Saved -> outputs/01_EDA_plots.png")
plt.show()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/01_correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved -> outputs/01_correlation_heatmap.png")
plt.show()

df.to_csv("data/hr_clean.csv", index=False)
print("Saved -> data/hr_clean.csv")
print("\nEDA complete. Run 02_preprocessing.py next.\n")
