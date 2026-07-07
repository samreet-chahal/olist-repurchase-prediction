"""
Phase 3: Product recommender with REAL evaluation.

Evaluation protocol (leave-last-out):
- Take customers with >= 2 delivered orders.
- Hide their LAST purchased product (chronologically).
- Recommend K products from their EARLIER purchases.
- Metrics:
    hit@K          — was the hidden product in the top-K list?
    category-hit@K — was the hidden product's CATEGORY in the top-K
                     categories? (product space is 32k items, so exact
                     hits are brutally hard; category hits show whether
                     the recommender understands the customer at all)

Recommenders compared (baseline-first, as always):
  1. Popularity     — recommend globally best-selling products. The
                      baseline every recommender must beat.
  2. Co-purchase CF — item-item collaborative filtering: products
                      bought by the same customers are similar.
  3. Content KNN    — cosine similarity on product attributes
                      (category, price, weight, photos, desc length).
"""

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

K = 10
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load purchase history (delivered orders only)
# ---------------------------------------------------------------
conn = sqlite3.connect("olist.db")
purchases = pd.read_sql_query("""
    SELECT c.customer_unique_id,
           oi.product_id,
           p.product_category_name,
           o.order_purchase_timestamp
    FROM order_items oi
    JOIN orders o     ON o.order_id = oi.order_id
    JOIN customers c  ON c.customer_id = o.customer_id
    LEFT JOIN products p ON p.product_id = oi.product_id
    WHERE o.order_status = 'delivered'
""", conn)

products = pd.read_sql_query("""
    SELECT product_id, product_category_name,
           product_photos_qty, product_description_lenght,
           product_weight_g,
           product_length_cm * product_height_cm * product_width_cm AS volume
    FROM products
""", conn)

prices = pd.read_sql_query("""
    SELECT product_id, AVG(price) AS avg_price
    FROM order_items GROUP BY product_id
""", conn)
conn.close()

purchases["ts"] = pd.to_datetime(purchases["order_purchase_timestamp"])
purchases = purchases.sort_values("ts")

# ---------------------------------------------------------------
# 2. Build evaluation set: leave-last-out per multi-purchase customer
# ---------------------------------------------------------------
counts = purchases.groupby("customer_unique_id")["ts"].nunique()
multi = counts[counts >= 2].index
eval_purchases = purchases[purchases["customer_unique_id"].isin(multi)]

test_cases = []   # (customer, history_product_ids, hidden_product, hidden_category)
train_rows = []   # purchase rows visible to the recommenders

for cust, grp in eval_purchases.groupby("customer_unique_id"):
    last_ts = grp["ts"].max()
    hist = grp[grp["ts"] < last_ts]
    hidden = grp[grp["ts"] == last_ts]
    if len(hist) == 0:
        continue
    test_cases.append((
        cust,
        hist["product_id"].tolist(),
        set(hidden["product_id"]),
        set(hidden["product_category_name"].dropna()),
    ))
    train_rows.append(hist)

# Single-purchase customers' data is fully visible for training
single = purchases[~purchases["customer_unique_id"].isin(multi)]
train = pd.concat([single] + train_rows, ignore_index=True)

print(f"Purchases total: {len(purchases):,} | train rows: {len(train):,}")
print(f"Evaluation customers (multi-purchase): {len(test_cases):,}\n")

# ---------------------------------------------------------------
# 3. Recommender 1: Popularity baseline
# ---------------------------------------------------------------
pop = train["product_id"].value_counts()
pop_top = pop.index.tolist()

prod_cat = products.set_index("product_id")["product_category_name"].to_dict()

def rec_popularity(history, k=K):
    return [p for p in pop_top if p not in history][:k]

# ---------------------------------------------------------------
# 4. Recommender 2: Co-purchase item-item CF
# ---------------------------------------------------------------
cust_products = train.groupby("customer_unique_id")["product_id"].apply(set)
co = defaultdict(lambda: defaultdict(int))
for prods in cust_products:
    prods = list(prods)
    for i in range(len(prods)):
        for j in range(i + 1, len(prods)):
            co[prods[i]][prods[j]] += 1
            co[prods[j]][prods[i]] += 1

def rec_copurchase(history, k=K):
    scores = defaultdict(float)
    for h in history:
        for other, cnt in co.get(h, {}).items():
            if other not in history:
                scores[other] += cnt
    ranked = sorted(scores, key=scores.get, reverse=True)[:k]
    # backfill with popularity when co-purchase signal runs out
    if len(ranked) < k:
        ranked += [p for p in pop_top
                   if p not in ranked and p not in history][:k - len(ranked)]
    return ranked

# ---------------------------------------------------------------
# 5. Recommender 3: Content-based KNN on product attributes
# ---------------------------------------------------------------
feat = products.merge(prices, on="product_id", how="left")
cat_dummies = pd.get_dummies(feat["product_category_name"], prefix="cat")
Xp = pd.concat([
    feat[["product_photos_qty", "product_description_lenght",
          "product_weight_g", "volume", "avg_price"]].fillna(0),
    cat_dummies,
], axis=1)
Xp_scaled = StandardScaler().fit_transform(Xp)

knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(Xp_scaled)
pid_index = feat["product_id"].tolist()
pid_pos = {p: i for i, p in enumerate(pid_index)}

def rec_content(history, k=K):
    anchor = history[-1]                    # most recent known purchase
    if anchor not in pid_pos:
        return rec_popularity(history, k)
    _, idx = knn.kneighbors(Xp_scaled[pid_pos[anchor]].reshape(1, -1),
                            n_neighbors=k + len(history) + 1)
    recs = [pid_index[i] for i in idx[0]
            if pid_index[i] not in history][:k]
    return recs

# ---------------------------------------------------------------
# 6. Evaluate all three under the identical protocol
# ---------------------------------------------------------------
def evaluate(rec_fn, name):
    hits, cat_hits = 0, 0
    for cust, history, hidden_prods, hidden_cats in test_cases:
        recs = rec_fn(history)
        if hidden_prods & set(recs):
            hits += 1
        rec_cats = {prod_cat.get(r) for r in recs} - {None}
        if hidden_cats & rec_cats:
            cat_hits += 1
    n = len(test_cases)
    print(f"{name:22s} hit@{K}: {hits/n:.3%} ({hits}/{n})   "
          f"category-hit@{K}: {cat_hits/n:.3%}")
    return hits / n, cat_hits / n

print(f"=== EVALUATION (leave-last-out, K={K}) ===")
results = {}
results["Popularity"] = evaluate(rec_popularity, "Popularity (baseline)")
results["Co-purchase CF"] = evaluate(rec_copurchase, "Co-purchase CF")
results["Content KNN"] = evaluate(rec_content, "Content KNN")

pd.DataFrame(results, index=[f"hit@{K}", f"cat-hit@{K}"]).T.to_csv(
    "results_phase3.csv")
print("\nSaved: results_phase3.csv")
