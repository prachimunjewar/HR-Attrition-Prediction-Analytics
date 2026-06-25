# ============================================================
# 04_xgboost_model.py — XGBoost + SHAP Explainability
# AttritionIQ: HR Attrition Prediction & Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("XGBOOST + SHAP EXPLAINABILITY MODEL")
print("=" * 60)

X_train = pd.read_csv("data/X_train.csv")   # unscaled for XGBoost
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()
features = pd.read_csv("data/feature_names.csv", header=None).squeeze().tolist()

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(features)}")

# ── 2. Train XGBoost ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training XGBoost Classifier")
print("=" * 60)

# scale_pos_weight handles class imbalance (ratio of negatives to positives)
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators        = 300,
    max_depth           = 4,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    scale_pos_weight    = neg_pos_ratio,   # handle imbalance
    use_label_encoder   = False,
    eval_metric         = "auc",
    random_state        = 42,
    n_jobs              = -1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("XGBoost trained successfully.")

# ── 3. Cross-Validation ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: 5-Fold Stratified Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"CV AUC Scores : {[f'{s:.3f}' for s in cv_scores]}")
print(f"Mean AUC      : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── 4. Evaluation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Evaluation on Test Set")
print("=" * 60)

y_pred      = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
auc_score   = roc_auc_score(y_test, y_pred_prob)

print(f"\nAUC-ROC Score : {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed","Left"]))

# ── 5. Standard Evaluation Plots ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("XGBoost — Model Evaluation", fontsize=14, fontweight="bold")

# (A) Confusion Matrix
ax = axes[0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed","Left"])
disp.plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title(f"Confusion Matrix\nAUC = {auc_score:.3f}", fontweight="bold")

# (B) ROC Curve
ax = axes[1]
lr_preds = pd.read_csv("data/lr_predictions.csv")
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob)
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_preds["prob_left"])
auc_lr = roc_auc_score(y_test, lr_preds["prob_left"])

ax.plot(fpr_xgb, tpr_xgb, color="#F59E0B", linewidth=2.5, label=f"XGBoost (AUC={auc_score:.3f})")
ax.plot(fpr_lr,  tpr_lr,  color="#2563EB", linewidth=2,   label=f"Log.Reg (AUC={auc_lr:.3f})", linestyle="--")
ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random")
ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.1, color="#F59E0B")
ax.set_title("ROC Curve — XGBoost vs LR", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (C) XGBoost Feature Importance
ax = axes[2]
fi = pd.DataFrame({
    "feature"   : features,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)
ax.barh(fi["feature"], fi["importance"], color="#F59E0B")
ax.set_title("XGBoost Feature Importance (Top 15)", fontweight="bold")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("outputs/04_xgboost_evaluation.png", dpi=150, bbox_inches="tight")
print("\nSaved -> outputs/04_xgboost_evaluation.png")
plt.show()

# ── 6. SHAP Explainability ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: SHAP Explainability")
print("=" * 60)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values computed for {len(X_test)} test employees.")
print(f"Top features by mean |SHAP|:")
mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
print(mean_shap.head(10).to_string())

# (A) SHAP Summary (Beeswarm)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=features,
                  max_display=15, show=False, plot_type="dot")
plt.title("SHAP Summary Plot — Global Feature Impact", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/04_shap_summary.png", dpi=150, bbox_inches="tight")
print("Saved -> outputs/04_shap_summary.png")
plt.show()

# (B) SHAP Bar Plot (mean |SHAP|)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=features,
                  max_display=15, show=False, plot_type="bar")
plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/04_shap_bar.png", dpi=150, bbox_inches="tight")
print("Saved -> outputs/04_shap_bar.png")
plt.show()

# (C) SHAP Dependence Plots for Top 3 Features
top3 = mean_shap.head(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SHAP Dependence Plots — Top 3 Drivers", fontsize=13, fontweight="bold")

for ax, feat in zip(axes, top3):
    shap.dependence_plot(feat, shap_values, X_test,
                         feature_names=features, ax=ax, show=False)
    ax.set_title(f"SHAP: {feat}", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/04_shap_dependence.png", dpi=150, bbox_inches="tight")
print("Saved -> outputs/04_shap_dependence.png")
plt.show()

# (D) Local Explanation — a single high-risk employee
print("\n" + "=" * 60)
print("STEP 6: Local Explanation (Highest-Risk Employee)")
print("=" * 60)

high_risk_idx = y_pred_prob.argmax()
print(f"Employee index: {high_risk_idx} | Predicted attrition prob: {y_pred_prob[high_risk_idx]:.2%}")

fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values      = shap_values[high_risk_idx],
        base_values = explainer.expected_value,
        data        = X_test.iloc[high_risk_idx].values,
        feature_names = features
    ),
    max_display=12,
    show=False
)
plt.title(f"Local SHAP — Highest Risk Employee\n(Attrition Prob: {y_pred_prob[high_risk_idx]:.1%})",
          fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_shap_local.png", dpi=150, bbox_inches="tight")
print("Saved -> outputs/04_shap_local.png")
plt.show()

# ── 7. Save Everything ───────────────────────────────────────
joblib.dump(xgb_model, "data/xgb_model.pkl")
np.save("data/shap_values.npy", shap_values)
pd.DataFrame({"feature": features, "mean_shap": mean_shap.values})\
  .to_csv("data/shap_importance.csv", index=False)

xgb_results = pd.DataFrame({
    "actual"    : y_test.values,
    "predicted" : y_pred,
    "prob_left" : y_pred_prob
})
xgb_results.to_csv("data/xgb_predictions.csv", index=False)

print("\nSaved: xgb_model.pkl, shap_values.npy, shap_importance.csv, xgb_predictions.csv")
print(f"""
╔══════════════════════════════════════════╗
║       XGBOOST SUMMARY                   ║
╠══════════════════════════════════════════╣
║  Test AUC    : {auc_score:.4f}                  ║
║  CV AUC      : {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})    ║
║  Top SHAP    : {top3[0]:<28}║
║  2nd SHAP    : {top3[1]:<28}║
║  3rd SHAP    : {top3[2]:<28}║
╚══════════════════════════════════════════╝
""")
print("XGBoost complete. Run 05_risk_scoring.py next.\n")
