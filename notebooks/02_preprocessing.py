# ============================================================
# 02_preprocessing.py — Feature Engineering & Encoding
# AttritionIQ: HR Attrition Prediction & Analytics
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Cleaned Data")
print("=" * 60)

df = pd.read_csv("data/hr_clean.csv")
print(f"Shape: {df.shape}")

# ── 2. Feature Engineering ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# Income per year at company (seniority-adjusted income)
df["IncomePerYear"]     = df["MonthlyIncome"] / (df["YearsAtCompany"] + 1)

# Satisfaction composite score (average of 4 satisfaction metrics)
df["SatisfactionScore"] = (
    df["JobSatisfaction"] +
    df["EnvironmentSatisfaction"] +
    df["RelationshipSatisfaction"] +
    df["WorkLifeBalance"]
) / 4

# Tenure ratio: years in current role vs total company tenure
df["TenureRatio"]       = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)

# Years since last promotion (stagnation indicator)
df["PromotionGap"]      = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

# Salary-to-experience gap
df["SalaryExperienceGap"] = df["MonthlyIncome"] - (df["TotalWorkingYears"] * 200)

print("New features created:")
new_features = ["IncomePerYear", "SatisfactionScore", "TenureRatio",
                "PromotionGap", "SalaryExperienceGap"]
for f in new_features:
    print(f"  - {f}: mean={df[f].mean():.2f}, std={df[f].std():.2f}")

# ── 3. Encode Categoricals ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Encoding Categorical Variables")
print("=" * 60)

# Binary encoding
binary_cols = {"OverTime": {"Yes": 1, "No": 0},
               "Gender":   {"Male": 1, "Female": 0}}
for col, mapping in binary_cols.items():
    df[col] = df[col].map(mapping)
    print(f"  Binary encoded: {col}")

# Label encoding for ordinal-like columns
label_cols = ["BusinessTravel", "Department", "EducationField",
              "JobRole", "MaritalStatus"]
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"  Label encoded : {col} — classes: {list(le.classes_)}")

# Save encoders
joblib.dump(le_dict, "data/label_encoders.pkl")
print("\nSaved label encoders -> data/label_encoders.pkl")

# ── 4. Define Features & Target ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature Selection")
print("=" * 60)

# Drop non-predictive columns
drop_cols = ["Attrition", "EmployeeNumber"]
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

target = "Attrition_Flag"
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

print(f"Features : {len(features)}")
print(f"Target   : {target} — class distribution: {y.value_counts().to_dict()}")
print(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.1f}:1")

# ── 5. Train/Test Split ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train/Test Split (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Train attrition rate: {y_train.mean()*100:.1f}%")
print(f"Test  attrition rate: {y_test.mean()*100:.1f}%")

# ── 6. Scale Features ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: StandardScaler")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=features)

joblib.dump(scaler, "data/scaler.pkl")
print("Saved scaler -> data/scaler.pkl")

# ── 7. Save Processed Data ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving Processed Data")
print("=" * 60)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",   index=False)
X_train_scaled.to_csv("data/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/X_test_scaled.csv",   index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
df.to_csv("data/hr_encoded.csv",   index=False)

# Save feature list
pd.Series(features).to_csv("data/feature_names.csv", index=False, header=False)

print("Saved all train/test splits and encoded dataset.")
print("\nPreprocessing complete. Run 03_logistic_regression.py next.\n")
