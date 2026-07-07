-- =====================================================================
-- Phase 2b: Feature set v2.
-- Hypothesis from 2a: models are feature-starved (96% of customers
-- have 1 order, so RFM alone is near-constant). Add features that
-- describe the QUALITY of the customer's experience, not just volume:
--
--   Delivery experience: was the order late vs the promised date?
--     (late delivery -> unhappy customer -> less likely to return?)
--   Payment behaviour: installments, payment type, voucher usage
--   Product richness: photos, description length (listing quality)
--   Seller signals: distinct sellers, same-state seller (faster ship)
--
-- Same temporal rules as v1: everything computed strictly pre-cutoff.
-- =====================================================================

WITH params AS (
    SELECT DATE(MAX(order_purchase_timestamp), '-180 days') AS cutoff
    FROM orders
    WHERE order_status = 'delivered'
),

delivered AS (
    SELECT
        c.customer_unique_id,
        c.customer_state,
        c.customer_zip_code_prefix,
        o.order_id,
        DATE(o.order_purchase_timestamp)        AS purchase_date,
        DATE(o.order_delivered_customer_date)   AS delivered_date,
        DATE(o.order_estimated_delivery_date)   AS estimated_date
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
),

hist AS (
    SELECT d.* FROM delivered d
    CROSS JOIN params p
    WHERE d.purchase_date < p.cutoff
),

order_value AS (
    SELECT order_id, SUM(payment_value) AS order_total
    FROM order_payments GROUP BY order_id
),

-- Payment behaviour per order
pay AS (
    SELECT
        order_id,
        MAX(payment_installments)                                   AS max_installments,
        COUNT(DISTINCT payment_type)                                AS n_payment_types,
        MAX(CASE WHEN payment_type = 'credit_card' THEN 1 ELSE 0 END) AS used_credit_card,
        MAX(CASE WHEN payment_type = 'voucher'     THEN 1 ELSE 0 END) AS used_voucher
    FROM order_payments GROUP BY order_id
),

-- Item / product / seller signals per order
items AS (
    SELECT
        oi.order_id,
        COUNT(*)                          AS n_items,
        COUNT(DISTINCT oi.product_id)     AS n_products,
        COUNT(DISTINCT oi.seller_id)      AS n_sellers,
        AVG(oi.price)                     AS avg_item_price,
        AVG(oi.freight_value)             AS avg_freight,
        AVG(p.product_photos_qty)         AS avg_photos,
        AVG(p.product_description_lenght) AS avg_desc_len,
        AVG(p.product_weight_g)           AS avg_weight_g,
        COUNT(DISTINCT p.product_category_name) AS n_categories,
        MAX(CASE WHEN s.seller_state = c2.customer_state THEN 1 ELSE 0 END)
                                          AS any_same_state_seller
    FROM order_items oi
    LEFT JOIN products p  ON p.product_id  = oi.product_id
    LEFT JOIN sellers  s  ON s.seller_id   = oi.seller_id
    LEFT JOIN orders   o2 ON o2.order_id   = oi.order_id
    LEFT JOIN customers c2 ON c2.customer_id = o2.customer_id
    GROUP BY oi.order_id
),

features AS (
    SELECT
        h.customer_unique_id,
        -- ===== v1 core (RFM) =====
        COUNT(DISTINCT h.order_id)                       AS frequency,
        SUM(ov.order_total)                              AS monetary_total,
        AVG(ov.order_total)                              AS avg_order_value,
        CAST(JULIANDAY(p.cutoff) - JULIANDAY(MAX(h.purchase_date)) AS INT)
                                                         AS recency_days,
        CAST(JULIANDAY(MAX(h.purchase_date)) - JULIANDAY(MIN(h.purchase_date)) AS INT)
                                                         AS tenure_days,
        AVG(r.review_score)                              AS avg_review_score,
        SUM(CASE WHEN r.review_score IS NOT NULL THEN 1 ELSE 0 END)
                                                         AS n_reviews,
        MAX(h.customer_state)                            AS customer_state,
        -- ===== v2: delivery experience =====
        AVG(JULIANDAY(h.delivered_date) - JULIANDAY(h.purchase_date))
                                                         AS avg_delivery_days,
        AVG(CASE WHEN h.delivered_date > h.estimated_date THEN 1.0 ELSE 0.0 END)
                                                         AS late_delivery_rate,
        AVG(JULIANDAY(h.estimated_date) - JULIANDAY(h.delivered_date))
                                                         AS avg_days_early,
        -- ===== v2: payment behaviour =====
        AVG(py.max_installments)                         AS avg_installments,
        MAX(py.used_credit_card)                         AS ever_credit_card,
        MAX(py.used_voucher)                             AS ever_voucher,
        -- ===== v2: product / seller signals =====
        AVG(it.n_items)                                  AS avg_items_per_order,
        SUM(it.n_products)                               AS n_distinct_products,
        SUM(it.n_categories)                             AS n_categories,
        AVG(it.avg_item_price)                           AS avg_item_price,
        AVG(it.avg_freight)                              AS avg_freight,
        AVG(it.avg_photos)                               AS avg_product_photos,
        AVG(it.avg_desc_len)                             AS avg_desc_length,
        AVG(it.avg_weight_g)                             AS avg_product_weight,
        MAX(it.n_sellers)                                AS max_sellers_per_order,
        MAX(it.any_same_state_seller)                    AS ever_same_state_seller
    FROM hist h
    CROSS JOIN params p
    LEFT JOIN order_value  ov ON ov.order_id = h.order_id
    LEFT JOIN order_reviews r ON r.order_id  = h.order_id
    LEFT JOIN pay          py ON py.order_id = h.order_id
    LEFT JOIN items        it ON it.order_id = h.order_id
    GROUP BY h.customer_unique_id
),

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
