"""
Phase 2b: rebuild dataset with v2 features, rerun the SAME benchmark
as 2a. Identical evaluation protocol -> any change in scores is
attributable to the features, not the methodology. That is the point.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Build v2 dataset from SQL
# ---------------------------------------------------------------
conn = sqlite3.connect("olist.db")
with open("05_features_v2.sql") as f:
    df = pd.read_sql_query(f.read(), conn)
conn.close()

print(f"v2 dataset: {df.shape[0]:,} rows x {df.shape[1]} cols "
      f"(v1 had 15 cols)")
df.to_csv("ml_dataset_v2.csv", index=False)

# ---------------------------------------------------------------
# 2. Preprocess (same policy as 2a)
# ---------------------------------------------------------------
df = df.dropna(subset=["monetary_total"])

df["never_reviewed"] = df["avg_review_score"].isna().astype(int)

state_freq = df["customer_state"].value_counts(normalize=True)
df["state_freq"] = df["customer_state"].map(state_freq)

TARGET = "purchased_next_180d"
DROP = ["customer_unique_id", "customer_state", TARGET]
FEATURES = [c for c in df.columns if c not in DROP]

# Median-fill remaining numeric nulls (reviews, product attrs, delivery)
for col in FEATURES:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"Features: {len(FEATURES)} | Train {X_train.shape[0]:,} "
      f"| Test {X_test.shape[0]:,} | pos rate {y.mean():.3%}\n")

# ---------------------------------------------------------------
# 3. Same benchmark as 2a
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

print(f"BASELINE 0 (majority): PR-AUC = {y_test.mean():.4f}")
print("v1 scores to beat -> LogReg 0.0398 | RF 0.0311 | XGB 0.0173\n")

results = []
for name, (model, needs_scaling) in models.items():
    Xtr = X_train_s if needs_scaling else X_train
    Xte = X_test_s if needs_scaling else X_test
    model.fit(Xtr, y_train)
    p = model.predict_proba(Xte)[:, 1]

    pr_auc = average_precision_score(y_test, p)
    roc = roc_auc_score(y_test, p)
    prec, rec, thr = precision_recall_curve(y_test, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = np.nanargmax(f1s[:-1])
    results.append({
        "model": name, "PR-AUC_v2": round(pr_auc, 4),
        "ROC-AUC_v2": round(roc, 4), "best_F1": round(f1s[best], 4),
        "precision@bestF1": round(prec[best], 4),
        "recall@bestF1": round(rec[best], 4),
    })

res = pd.DataFrame(results).set_index("model")
print(res.to_string())

xgb = models["XGBoost"][0]
imp = pd.Series(xgb.feature_importances_, index=FEATURES)
print("\nTop 10 XGBoost features (v2):")
print(imp.sort_values(ascending=False).head(10).round(4).to_string())

res.to_csv("results_phase2b.csv")
print("\nSaved: ml_dataset_v2.csv, results_phase2b.csv")
