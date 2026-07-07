"""
Phase 2a: Baselines + first models for 180-day repurchase prediction.

Methodology notes (interview defence):
- Temporal integrity is handled at DATASET level: features come from
  before the cutoff, labels from after. All customers share one cutoff,
  so a stratified random split of CUSTOMERS here does not leak time.
- Positive rate is ~1.17%, so accuracy is meaningless. Primary metric:
  PR-AUC (average precision). Secondary: ROC-AUC, F1 at tuned threshold.
- Baseline-first: no complex model earns a place unless it beats
  logistic regression. This is how real DS teams work.
- Nulls: avg_review_score missing => customer never reviewed. That is
  SIGNAL, not noise -> keep an explicit 'never_reviewed' flag and fill
  with the median (median is robust; the flag preserves the info).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, f1_score)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load + preprocess
# ---------------------------------------------------------------
df = pd.read_csv("ml_dataset.csv")

# Drop the single row with a missing payment record (data quality issue)
df = df.dropna(subset=["monetary_total"])

# 'Never reviewed' is behaviour, not missing data -> flag it
df["never_reviewed"] = df["avg_review_score"].isna().astype(int)
df["avg_review_score"] = df["avg_review_score"].fillna(
    df["avg_review_score"].median()
)

# Encode state by frequency (49k customers, 27 states; one-hot would
# explode columns for rare states; frequency encoding keeps it compact)
state_freq = df["customer_state"].value_counts(normalize=True)
df["state_freq"] = df["customer_state"].map(state_freq)

FEATURES = [
    "frequency", "monetary_total", "avg_order_value", "recency_days",
    "tenure_days", "avg_review_score", "n_reviews", "never_reviewed",
    "n_distinct_products", "n_categories", "avg_freight", "state_freq",
]
TARGET = "purchased_next_180d"

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, positives: {y_train.sum()} "
      f"({y_train.mean():.3%})")
print(f"Test:  {X_test.shape}, positives: {y_test.sum()} "
      f"({y_test.mean():.3%})\n")

# ---------------------------------------------------------------
# 2. Baseline 0: predict majority class (the floor)
# ---------------------------------------------------------------
# PR-AUC of a no-skill model equals the positive rate itself.
print(f"BASELINE 0 (majority class): PR-AUC = {y_test.mean():.4f} "
      "(the number every model must beat)\n")

# ---------------------------------------------------------------
# 3. Models (all imbalance-aware via class weights)
# ---------------------------------------------------------------
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

models = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=RANDOM_STATE), True),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=20, n_jobs=-1,
                               random_state=RANDOM_STATE), False),
    "XGBoost": (
        XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                      scale_pos_weight=pos_weight, eval_metric="aucpr",
                      random_state=RANDOM_STATE), False),
}

results = []
probs = {}
for name, (model, needs_scaling) in models.items():
    Xtr = X_train_s if needs_scaling else X_train
    Xte = X_test_s if needs_scaling else X_test
    model.fit(Xtr, y_train)
    p = model.predict_proba(Xte)[:, 1]
    probs[name] = p

    pr_auc = average_precision_score(y_test, p)
    roc = roc_auc_score(y_test, p)

    # Best-F1 threshold from PR curve (never assume 0.5 on imbalanced data)
    prec, rec, thr = precision_recall_curve(y_test, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = np.nanargmax(f1s[:-1])
    results.append({
        "model": name, "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc, 4), "best_F1": round(f1s[best], 4),
        "threshold": round(float(thr[best]), 4),
        "precision@bestF1": round(prec[best], 4),
        "recall@bestF1": round(rec[best], 4),
    })

res = pd.DataFrame(results).set_index("model")
print(res.to_string())

# ---------------------------------------------------------------
# 4. What is driving predictions? (quick look; deep dive in 2b)
# ---------------------------------------------------------------
xgb = models["XGBoost"][0]
imp = pd.Series(xgb.feature_importances_, index=FEATURES)
print("\nXGBoost feature importance (gain):")
print(imp.sort_values(ascending=False).round(4).to_string())

res.to_csv("results_phase2a.csv")
print("\nSaved: results_phase2a.csv")
