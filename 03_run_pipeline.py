"""
Step 3: Execute the SQL pipeline, sanity-check the result,
and export the ML-ready dataset for Phase 2 (modeling).
"""

import sqlite3
import pandas as pd

DB_PATH = "olist.db"
SQL_PATH = "02_features_and_labels.sql"
OUT_PATH = "ml_dataset.csv"

def main():
    conn = sqlite3.connect(DB_PATH)
    with open(SQL_PATH) as f:
        query = f.read()

    df = pd.read_sql_query(query, conn)
    conn.close()

    # ---------- Sanity checks (never skip these) ----------
    print(f"Dataset shape: {df.shape}")
    print(f"\nLabel balance:")
    print(df["purchased_next_180d"].value_counts())
    pos_rate = df["purchased_next_180d"].mean()
    print(f"Positive rate: {pos_rate:.3%}  <-- this is why accuracy is banned")

    print(f"\nNull counts:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print(f"\nFrequency distribution (orders per customer in history window):")
    print(df["frequency"].value_counts().head())

    # Leakage check: recency must never be negative
    assert (df["recency_days"] >= 0).all(), "LEAKAGE: purchase after cutoff in features!"
    print("\nLeakage check passed: no future data in feature window.")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
