# ============================================================
# 03_logistic_regression.py — Logistic Regression Model
# AttritionIQ: HR Attrition Prediction & Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

X_train = pd.read_csv("data/X_train_scaled.csv")
X_test  = pd.read_csv("data/X_test_scaled.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()
features = pd.read_csv("data/feature_names.csv", header=None).squeeze().tolist()

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 2. Handle Class Imbalance ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Class Weights (handle 5:1 imbalance)")
print("=" * 60)

classes = np.array([0, 1])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))
print(f"Class weights: {class_weight}")

# ── 3. Train Model ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training Logistic Regression")
print("=" * 60)

lr_model = LogisticRegression(
    class_weight=class_weight,
    max_iter=1000,
    C=0.5,              # regularization
    solver="lbfgs",
    random_state=42
)
lr_model.fit(X_train, y_train)
print("Model trained successfully.")

# ── 4. Predictions ───────────────────────────────────────────
y_pred      = lr_model.predict(X_test)
y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
auc_score   = roc_auc_score(y_test, y_pred_prob)

# ── 5. Evaluation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Evaluation")
print("=" * 60)

print(f"\nAUC-ROC Score: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed","Left"]))

# ── 6. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Logistic Regression — Model Evaluation", fontsize=14, fontweight="bold")

# (A) Confusion Matrix
ax = axes[0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed","Left"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix\nAUC = {auc_score:.3f}", fontweight="bold")

# (B) ROC Curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label=f"LR (AUC={auc_score:.3f})")
ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random")
ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
ax.set_title("ROC Curve", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(True, alpha=0.3)

# (C) Top Feature Coefficients
ax = axes[2]
coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": lr_model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False).head(15)

colors_coef = ["#DC2626" if v > 0 else "#2563EB" for v in coef_df["coefficient"]]
ax.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1], color=colors_coef[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 15 Feature Coefficients\n(Red=Attrition Risk, Blue=Retention)", fontweight="bold")
ax.set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig("outputs/03_logistic_regression.png", dpi=150, bbox_inches="tight")
print("\nSaved -> outputs/03_logistic_regression.png")
plt.show()

# ── 7. Save Model & Predictions ──────────────────────────────
joblib.dump(lr_model, "data/lr_model.pkl")
print("Saved -> data/lr_model.pkl")

lr_results = pd.DataFrame({
    "actual"     : y_test.values,
    "predicted"  : y_pred,
    "prob_left"  : y_pred_prob
})
lr_results.to_csv("data/lr_predictions.csv", index=False)
print("Saved -> data/lr_predictions.csv")

print(f"""
╔══════════════════════════════════════════╗
║    LOGISTIC REGRESSION SUMMARY          ║
╠══════════════════════════════════════════╣
║  AUC-ROC     : {auc_score:.4f}                  ║
║  Top Driver  : {coef_df.iloc[0]['feature']:<28}║
╚══════════════════════════════════════════╝
""")
print("LR complete. Run 04_xgboost_model.py next.\n")
