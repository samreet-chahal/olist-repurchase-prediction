"""
Phase 2c: Tuning + error analysis. This closes out the modeling phase.

1. Hyperparameter tuning: Stratified 3-fold CV, scored on PR-AUC
   (average_precision). LogReg gets a small full grid; XGBoost gets a
   randomized search. Tuning on the TRAIN set only; test set touched
   exactly once at the end.
2. Lift analysis: rank test customers by predicted probability. If we
   target the top k% with a retention offer, how many actual repeat
   buyers do we catch vs random targeting? This is the business metric.
3. Segment error analysis: where is the model relatively reliable?
   (multi-order vs single-order customers, monetary quartiles,
   review-score bands)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load v2 dataset, same preprocessing as 2b
# ---------------------------------------------------------------
df = pd.read_csv("ml_dataset_v2.csv")
df = df.dropna(subset=["monetary_total"])
df["never_reviewed"] = df["avg_review_score"].isna().astype(int)
state_freq = df["customer_state"].value_counts(normalize=True)
df["state_freq"] = df["customer_state"].map(state_freq)

TARGET = "purchased_next_180d"
DROP = ["customer_unique_id", "customer_state", TARGET]
FEATURES = [c for c in df.columns if c not in DROP]
for col in FEATURES:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---------------------------------------------------------------
# 2. Tune Logistic Regression (small grid, exhaustive)
# ---------------------------------------------------------------
logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=3000,
                               random_state=RANDOM_STATE)),
])
logreg_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"],
}
gs_lr = GridSearchCV(logreg_pipe, logreg_grid, scoring="average_precision",
                     cv=cv, n_jobs=-1)
gs_lr.fit(X_train, y_train)
print("LogReg  best params:", gs_lr.best_params_,
      f"| CV PR-AUC: {gs_lr.best_score_:.4f}")

# ---------------------------------------------------------------
# 3. Tune XGBoost (randomized search)
# ---------------------------------------------------------------
xgb_dist = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 6],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.1, 1, 10],
}
rs_xgb = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=pos_weight, eval_metric="aucpr",
                  random_state=RANDOM_STATE, n_jobs=-1),
    xgb_dist, n_iter=25, scoring="average_precision",
    cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
rs_xgb.fit(X_train, y_train)
print("XGBoost best params:", rs_xgb.best_params_,
      f"| CV PR-AUC: {rs_xgb.best_score_:.4f}")

# ---------------------------------------------------------------
# 4. Final, one-time evaluation on the held-out test set
# ---------------------------------------------------------------
print("\n=== FINAL TEST-SET SCORES (tuned) ===")
finalists = {"LogReg (tuned)": gs_lr.best_estimator_,
             "XGBoost (tuned)": rs_xgb.best_estimator_}
best_name, best_p, best_pr = None, None, -1
for name, model in finalists.items():
    p = model.predict_proba(X_test)[:, 1]
    pr = average_precision_score(y_test, p)
    roc = roc_auc_score(y_test, p)
    print(f"{name:18s} PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f}")
    if pr > best_pr:
        best_name, best_p, best_pr = name, p, pr
print(f"(untuned bests were: LogReg 0.0362, XGBoost 0.0194)")
print(f"\nChampion: {best_name}")

# ---------------------------------------------------------------
# 5. Lift analysis — the business framing
# ---------------------------------------------------------------
test = X_test.copy()
test["y"] = y_test.values
test["p"] = best_p
test = test.sort_values("p", ascending=False).reset_index(drop=True)
base_rate = test["y"].mean()

print("\n=== LIFT TABLE (target top k% by predicted score) ===")
print(f"{'top k%':>7} {'customers':>10} {'repeat buyers':>14} "
      f"{'precision':>10} {'recall':>7} {'lift':>6}")
for k in [0.01, 0.05, 0.10, 0.20]:
    n = int(len(test) * k)
    seg = test.head(n)
    prec = seg["y"].mean()
    rec = seg["y"].sum() / test["y"].sum()
    print(f"{k:>7.0%} {n:>10,} {int(seg['y'].sum()):>14} "
          f"{prec:>10.3%} {rec:>7.1%} {prec/base_rate:>5.1f}x")

# ---------------------------------------------------------------
# 6. Segment-level error analysis
# ---------------------------------------------------------------
print("\n=== SEGMENT ANALYSIS (PR-AUC within segment) ===")

def seg_pr(mask, label):
    sub_y, sub_p = test.loc[mask, "y"], test.loc[mask, "p"]
    if sub_y.sum() >= 5:  # need enough positives for a stable estimate
        print(f"{label:35s} n={mask.sum():>6,} pos_rate={sub_y.mean():.2%} "
              f"PR-AUC={average_precision_score(sub_y, sub_p):.4f}")
    else:
        print(f"{label:35s} n={mask.sum():>6,} pos_rate={sub_y.mean():.2%} "
              f"(too few positives)")

seg_pr(test["frequency"] == 1, "Single-order customers")
seg_pr(test["frequency"] >= 2, "Multi-order customers")

q = test["monetary_total"].quantile([0.25, 0.75])
seg_pr(test["monetary_total"] <= q[0.25], "Low spenders (bottom quartile)")
seg_pr(test["monetary_total"] >= q[0.75], "High spenders (top quartile)")

seg_pr(test["avg_review_score"] <= 2, "Unhappy reviewers (score <= 2)")
seg_pr(test["avg_review_score"] >= 4.5, "Happy reviewers (score >= 4.5)")

pd.DataFrame({"y": test["y"], "p": test["p"]}).to_csv(
    "final_test_predictions.csv", index=False)
print("\nSaved: final_test_predictions.csv")
