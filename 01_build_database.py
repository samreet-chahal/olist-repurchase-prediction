"""
Step 1: Load all 9 Olist CSVs into SQLite.
Why SQLite: real SQL (joins, CTEs, window functions) instead of pandas merges,
and the DB file is portable — same queries work in any SQL interview setting.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = "olist.db"

TABLES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

def main():
    conn = sqlite3.connect(DB_PATH)
    for table, csv_file in TABLES.items():
        df = pd.read_csv(DATA_DIR / csv_file)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"{table:22s} {len(df):>8,} rows, {df.shape[1]} cols")

    # Indexes on join keys — makes the feature queries fast
    cur = conn.cursor()
    for stmt in [
        "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_order ON orders(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_payments_order ON order_payments(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_reviews_order ON order_reviews(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_customers_cid ON customers(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_items_product ON order_items(product_id)",
    ]:
        cur.execute(stmt)
    conn.commit()
    conn.close()
    print(f"\nDatabase built: {DB_PATH}")

if __name__ == "__main__":
    main()
