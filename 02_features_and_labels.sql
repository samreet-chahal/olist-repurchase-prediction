-- =====================================================================
-- Phase 1 core: build the ML dataset entirely in SQL.
--
-- Design decisions (interview defence notes):
-- 1. customer_unique_id, NOT customer_id — Olist assigns a fresh
--    customer_id per ORDER; customer_unique_id is the real person.
--    Using customer_id would make repeat purchase invisible (gotcha).
-- 2. Temporal split: cutoff = max purchase date - 180 days.
--    Features come ONLY from orders BEFORE the cutoff.
--    Label = did the customer purchase ON/AFTER the cutoff.
--    A random split here would leak the future into training.
-- 3. Only 'delivered' orders count as purchases for both features
--    and labels — canceled/unavailable orders are not real demand.
-- =====================================================================

-- The cutoff date (run once, hardcode result into the CTE below or
-- compute inline as done here via CROSS JOIN):
-- SELECT DATE(MAX(order_purchase_timestamp), '-180 days') FROM orders;

WITH params AS (
    SELECT DATE(MAX(order_purchase_timestamp), '-180 days') AS cutoff
    FROM orders
    WHERE order_status = 'delivered'
),

-- Every delivered order mapped to the persistent customer identity
delivered AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        DATE(o.order_purchase_timestamp) AS purchase_date,
        o.customer_id
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
),

-- ---------- FEATURE WINDOW: strictly before cutoff ----------
hist AS (
    SELECT d.*
    FROM delivered d
    CROSS JOIN params p
    WHERE d.purchase_date < p.cutoff
),

-- Order-level monetary value (sum across payment instalments)
order_value AS (
    SELECT order_id, SUM(payment_value) AS order_total
    FROM order_payments
    GROUP BY order_id
),

-- RFM + engagement features per customer, computed only on history
features AS (
    SELECT
        h.customer_unique_id,
        COUNT(DISTINCT h.order_id)                          AS frequency,
        SUM(ov.order_total)                                 AS monetary_total,
        AVG(ov.order_total)                                 AS avg_order_value,
        MIN(h.purchase_date)                                AS first_purchase,
        MAX(h.purchase_date)                                AS last_purchase,
        CAST(JULIANDAY(p.cutoff) - JULIANDAY(MAX(h.purchase_date)) AS INT)
                                                            AS recency_days,
        CAST(JULIANDAY(MAX(h.purchase_date)) - JULIANDAY(MIN(h.purchase_date)) AS INT)
                                                            AS tenure_days,
        AVG(r.review_score)                                 AS avg_review_score,
        SUM(CASE WHEN r.review_score IS NOT NULL THEN 1 ELSE 0 END)
                                                            AS n_reviews,
        COUNT(DISTINCT oi.product_id)                       AS n_distinct_products,
        COUNT(DISTINCT pr.product_category_name)            AS n_categories,
        AVG(oi.freight_value)                               AS avg_freight,
        MAX(c2.customer_state)                              AS customer_state
    FROM hist h
    CROSS JOIN params p
    LEFT JOIN order_value  ov ON ov.order_id = h.order_id
    LEFT JOIN order_reviews r ON r.order_id  = h.order_id
    LEFT JOIN order_items  oi ON oi.order_id = h.order_id
    LEFT JOIN products     pr ON pr.product_id = oi.product_id
    LEFT JOIN customers    c2 ON c2.customer_id = h.customer_id
    GROUP BY h.customer_unique_id
),

-- ---------- LABEL WINDOW: on/after cutoff ----------
future_buyers AS (
    SELECT DISTINCT d.customer_unique_id
    FROM delivered d
    CROSS JOIN params p
    WHERE d.purchase_date >= p.cutoff
)

SELECT
    f.*,
    CASE WHEN fb.customer_unique_id IS NOT NULL THEN 1 ELSE 0 END AS purchased_next_180d
FROM features f
LEFT JOIN future_buyers fb USING (customer_unique_id);
