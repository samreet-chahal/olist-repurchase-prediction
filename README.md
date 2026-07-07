# Olist Customer Repurchase Prediction & Product Recommendation

End-to-end machine learning project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100K orders, 9 relational tables, 2016–2018): a SQL feature pipeline, a repurchase-prediction model evaluated on a temporal holdout, and a product recommender with proper offline evaluation.

## Problem

Only **3.1%** of Olist customers ever place a second order. Two questions:

1. **Prediction** — given a customer's purchase history, will they buy again in the next 180 days? (So retention spend can be targeted instead of sprayed.)
2. **Recommendation** — which products should be recommended to a returning customer?

## Pipeline

```
9 raw CSVs ──> SQLite ──> SQL feature pipeline ──> ML dataset ──> models ──> evaluation
              (01)        (02, 05)                 (03, 06)      (04, 06, 07, 08)
```

| File | Purpose |
|---|---|
| `01_build_database.py` | Load 9 CSVs into SQLite with indexed join keys |
| `02_features_and_labels.sql` | v1 features: RFM + reviews, temporal split, label creation |
| `03_run_pipeline.py` | Execute SQL, sanity checks, leakage assertion, export dataset |
| `04_modeling_baselines.py` | Baselines + LogReg / RF / XGBoost with class weights |
| `05_features_v2.sql` | v2 features: delivery experience, payments, product/seller signals |
| `06_modeling_v2.py` | Identical benchmark on v2 features (controlled comparison) |
| `07_tuning_error_analysis.py` | CV hyperparameter tuning, lift table, segment error analysis |
| `08_recommender.py` | 3 recommenders evaluated leave-last-out with hit@10 |

## Key methodology decisions

- **Temporal split, not random.** Features are computed strictly *before* a cutoff date (max date − 180 days); the label is whether the customer purchased *after* it. A leakage assertion in the pipeline enforces this.
- **`customer_unique_id`, not `customer_id`.** Olist issues a new `customer_id` per order; using it would make repeat purchase invisible by construction.
- **Only `delivered` orders count** as demand, for both features and labels.
- **Accuracy is banned.** The positive rate is **1.17%**, so a model predicting "nobody repurchases" scores ~99% accuracy while being useless. Primary metric: **PR-AUC** (no-skill baseline = 0.0117).
- **Baseline-first.** Every model must beat (a) majority-class and (b) plain logistic regression to justify its complexity.

## Results — repurchase prediction

| Model | PR-AUC (v1) | PR-AUC (v2) | PR-AUC (tuned) |
|---|---|---|---|
| No-skill baseline | 0.0117 | 0.0117 | 0.0117 |
| **Logistic Regression** | 0.0398 | 0.0362 | **0.0381** |
| Random Forest | 0.0311 | 0.0286 | — |
| XGBoost | 0.0173 | 0.0194 | 0.0258 |

**The champion is tuned logistic regression** — 3.3× the no-skill baseline. XGBoost consistently underperformed: with 96% of customers having exactly one historical order, the feature space is too sparse for tree interactions, and the randomized search confirmed this by selecting the shallowest configuration available (`max_depth=2`, heavy min-child-weight).

**A documented negative result:** a second feature round (delivery lateness, payment installments, product attributes, seller signals — 10 new features) produced *no improvement* under an identical evaluation protocol. The delivery-experience hypothesis ("late delivery → won't return") was rejected by the data. Conclusion: repurchase at Olist is driven mostly by factors outside the transaction record.

### Business framing: lift over random targeting

| Target top k% by model score | Precision | Recall | Lift |
|---|---|---|---|
| 1% | 8.6% | 7.4% | **7.4×** |
| 5% | 3.9% | 16.6% | 3.3× |
| 10% | 3.1% | 26.4% | 2.6× |

Targeting a retention campaign at the model's top 1% reaches repeat buyers at **7.4× the rate of random targeting** — a usable marketing tool despite the hard prediction problem.

### Segment analysis: where the model actually works

| Segment | PR-AUC |
|---|---|
| Multi-order customers | **0.179** |
| Unhappy reviewers (≤2 stars) | 0.116 |
| High spenders (top quartile) | 0.080 |
| Single-order customers | 0.018 |

The model is ~10× more effective on customers with existing repeat behaviour. Recommended deployment: score returning customers with the model; use broad campaigns for one-time buyers.

## Results — product recommendation

Evaluation: **leave-last-out** — for each of 2,565 multi-purchase customers, hide the chronologically last purchase, recommend 10 products from earlier history, check for hits.

| Recommender | hit@10 | category-hit@10 |
|---|---|---|
| Popularity (baseline) | 2.50% | 46.2% |
| **Co-purchase item-item CF** | **5.81%** | **50.0%** |
| Content-based KNN (cosine) | 5.73% | 43.7% |

Co-purchase CF beats the popularity baseline **2.3×** on exact hits — and given a ~32,000-product catalog (random ≈ 0.03%), that is roughly **190× better than chance**. Collaborative signal beats content similarity on this data.

## Reproduce

```bash
# 1. Download the dataset from Kaggle (link above) into data/
pip install -r requirements.txt
python 01_build_database.py
python 03_run_pipeline.py
python 04_modeling_baselines.py
python 06_modeling_v2.py
python 07_tuning_error_analysis.py
python 08_recommender.py
```

## Limitations & next steps

- Absolute PR-AUC is low; the problem is intrinsically hard for one-time buyers. Text features from review comments (NLP/sentiment) are the most promising untapped signal.
- The recommender is evaluated offline only; online metrics (CTR, conversion) would differ.
- A single temporal cutoff is used; rolling-window backtesting would give more robust estimates.
