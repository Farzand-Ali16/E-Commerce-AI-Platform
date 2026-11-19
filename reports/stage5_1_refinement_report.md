# Stage 5.1 Report: Clustering Model Refinement

*E-Commerce AI Analytics & Customer Intelligence Platform*

Question: should the production segmentation use the existing **10-feature** K-Means model (A) or the **Core-5** model (B)? A third variant, **Core-4** (C = B without Preferred_Category_Share), was added only as a diagnostic to test the share concern. No Stage 5 file was changed.

Core-5 features: Total_Orders, Average_Order_Value, Days_Since_Last_Order, Average_Days_Between_Orders, Preferred_Category_Share. Same preprocessing as Stage 4 (log1p for orders and order value, sqrt for the two time features, median fill for the gap of single-order customers, scaling). The Core-5 matrix is identical to the 5 matching columns of the Stage 4 matrix (checked).

---

## 1. Core-5, K = 2 to 6

| K | Inertia | Silhouette | Calinski_Harabasz | Davies_Bouldin | Worst_cluster_silhouette | Negative_silhouette_% | Smallest_% | Seed_ARI_mean | Seed_ARI_min | Subsample_ARI_mean | Subsample_ARI_min | Inertia_drop_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 17400.0 | 0.31 | 2183.043 | 1.359 | 0.258 | 2.4 | 32.52 | 0.993 | 0.969 | 0.939 | 0.504 | - |
| 3 | 13712.747 | 0.268 | 2056.572 | 1.358 | 0.223 | 3.26 | 27.16 | 0.999 | 0.999 | 0.98 | 0.941 | 21.191 |
| 4 | 11397.214 | 0.274 | 1987.609 | 1.277 | 0.2 | 3.92 | 20.16 | 1.0 | 0.999 | 0.988 | 0.957 | 16.886 |
| 5 | 10271.427 | 0.267 | 1790.628 | 1.244 | 0.188 | 3.62 | 8.62 | 0.996 | 0.989 | 0.981 | 0.947 | 9.878 |
| 6 | 9307.54 | 0.25 | 1683.975 | 1.312 | 0.179 | 3.84 | 8.4 | 0.996 | 0.992 | 0.898 | 0.563 | 9.384 |

Cluster sizes (largest first):

| K | Sizes (n, %) |
|---|---|
| 2 | 3,374 (67.5%), 1,626 (32.5%) |
| 3 | 2,258 (45.2%), 1,384 (27.7%), 1,358 (27.2%) |
| 4 | 1,635 (32.7%), 1,193 (23.9%), 1,164 (23.3%), 1,008 (20.2%) |
| 5 | 1,632 (32.6%), 1,168 (23.4%), 1,148 (23.0%), 621 (12.4%), 431 (8.6%) |
| 6 | 1,383 (27.7%), 967 (19.3%), 847 (16.9%), 787 (15.7%), 596 (11.9%), 420 (8.4%) |

Finalist rules (same as Stage 5): smallest cluster at least 10% of customers, subsample ARI mean at least 0.9, worst subsample ARI at least 0.8.

| K | Silhouette | Smallest_% | Subsample_ARI_mean | Subsample_ARI_min | Balance_ok | Stable_ok | Finalist |
|---|---|---|---|---|---|---|---|
| 2 | 0.31 | 32.52 | 0.939 | 0.504 | True | False | False |
| 3 | 0.268 | 27.16 | 0.98 | 0.941 | True | True | True |
| 4 | 0.274 | 20.16 | 0.988 | 0.957 | True | True | True |
| 5 | 0.267 | 8.62 | 0.981 | 0.947 | False | True | False |
| 6 | 0.25 | 8.4 | 0.898 | 0.563 | False | False | False |

Finalists: **K = 3, 4**. K=4 beats K=3 on Silhouette (0.274 vs 0.268), Davies-Bouldin (1.277 vs 1.358) and stability (0.988 vs 0.980); K=3 is higher only on Calinski-Harabasz (2057 vs 1988). K=2 has the top Silhouette but one subsample dropped to ARI 0.50 (unstable). K=5 leaves a segment of 8.6% and K=6 is unstable. The elbow plot has no sharp bend (drops shrink smoothly), so it does not decide K.

## 2. Model comparison at K = 4

| index | A | B | C |
|---|---|---|---|
| Features used | 10.000 | 5.000 | 4.000 |
| Silhouette | 0.110 | 0.274 | 0.272 |
| Calinski_Harabasz | 634.5 | 1987.6 | 1891.2 |
| Davies_Bouldin | 2.283 | 1.277 | 1.244 |
| Worst_cluster_silhouette | 0.017 | 0.200 | 0.187 |
| Negative_silhouette_% | 21.4 | 3.920 | 3.560 |
| Smallest_% | 20.1 | 20.2 | 21.5 |
| Largest_% | 33.4 | 32.7 | 34.0 |
| Seed_ARI_mean | 0.998 | 1.000 | 0.997 |
| Seed_ARI_min | 0.994 | 0.999 | 0.993 |
| Subsample_ARI_mean | 0.975 | 0.988 | 0.971 |
| Subsample_ARI_min | 0.962 | 0.957 | 0.925 |
| Max_VIF | 3.301 | 3.270 | 1.679 |
| One-time cluster purity_% | 88.7 | 88.5 | 53.0 |
| Repeat customers inside one-time cluster | 114.0 | 116.0 | 506.0 |
| Sizes (n, %) | 1,668 (33.4%), 1,199 (24.0%), 1,127 (22.5%), 1,006 (20.1%) | 1,635 (32.7%), 1,193 (23.9%), 1,164 (23.3%), 1,008 (20.2%) | 1,700 (34.0%), 1,143 (22.9%), 1,081 (21.6%), 1,076 (21.5%) |

ARI between solutions: A vs B = **0.891**, B vs C = **0.594**, A vs C = 0.587. Subsample and seed stability use the same 30 random 80% subsamples and 15 seeds for every model, so the numbers can differ slightly from Stage 5, where the subsample draws were shared across K values.

Reading the table:
- **B is better than A on every quality measure.** Silhouette 0.110 to 0.274; Calinski-Harabasz 635 to 1988; Davies-Bouldin 2.28 to 1.28. In A the worst cluster has silhouette 0.017 and 21% of customers sit closer to another cluster than their own; in B the figures are 0.200 and 4%.
- **B is at least as stable as A** (subsample ARI mean 0.988 vs 0.975; worst case 0.957 vs 0.962, a negligible difference, both far above the 0.80 rule) with similar balance (smallest cluster 20.2% vs 20.1%).
- **A and B describe nearly the same segments** (ARI 0.89; only 199 customers, 4.0%, change cluster). The extra 5 features in A add noise, not information: rating, session duration, pages viewed and delivery time had eta squared near 0 in Stage 5.
- Silhouette is higher in B partly because removing noise dimensions makes distances meaningful, so the Silhouette gain is not a claim of "strong" clusters. 0.27 is still only in the weak-structure range (0.26 to 0.50).

## 3. Core-5 K=4 cluster profile

Per-customer averages. Cluster numbers sorted by mean Total_Orders (0 = most orders). Total_Spending, Average_Discount, Unique_Categories_Bought and the categorical preferences were **not** training features.

| index | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| Label | Loyal & Active | Lapsed Repeat Buyers | Slow-Cycle Repeat Buyers | One-Time Buyers |
| Customers | 1635 | 1164 | 1193 | 1008 |
| Customers_% | 32.7 | 23.28 | 23.86 | 20.16 |
| Total_Orders | 5.82 | 2.9 | 2.54 | 1.13 |
| Total_Spending | 7882.6 | 3502.83 | 2947.66 | 1286.85 |
| Average_Order_Value | 1391.38 | 1232.74 | 1196.08 | 1168.34 |
| Average_Discount | 77.22 | 71.15 | 60.8 | 54.43 |
| Average_Customer_Rating | 3.89 | 3.9 | 3.9 | 3.93 |
| Average_Session_Duration | 14.51 | 14.57 | 14.58 | 14.57 |
| Average_Pages_Viewed | 9.01 | 8.97 | 9.05 | 8.92 |
| Average_Delivery_Time | 6.51 | 6.57 | 6.35 | 6.68 |
| Days_Since_Last_Order | 50.01 | 203.74 | 79.47 | 213.15 |
| Average_Days_Between_Orders | 70.62 | 60.63 | 190.14 | 104.46 |
| Unique_Categories_Bought | 4.26 | 2.56 | 2.29 | 1.0 |
| Revenue_Share_% | 59.18 | 18.72 | 16.15 | 5.96 |
| Single_Order_% | 0.0 | 0.0 | 0.0 | 88.49 |
| Silence_vs_Usual_Gap | 0.59 | 3.15 | 0.38 | 1.45 |

| index | Label | Preferred_Category | Preferred_Device | Preferred_Payment |
|---|---|---|---|---|
| Cluster 0 | Loyal & Active | Sports (15%) | Mobile (69%) | Credit Card (52%) |
| Cluster 1 | Lapsed Repeat Buyers | Food (14%) | Mobile (62%) | Credit Card (44%) |
| Cluster 2 | Slow-Cycle Repeat Buyers | Books (15%) | Mobile (59%) | Credit Card (45%) |
| Cluster 3 | One-Time Buyers | Beauty (14%) | Mobile (55%) | Credit Card (39%) |

Eta squared of each Core-5 feature (how strongly the clusters separate it):

| index | eta_squared |
|---|---|
| Preferred_Category_Share | 0.811 |
| Total_Orders | 0.77 |
| Average_Days_Between_Orders | 0.592 |
| Days_Since_Last_Order | 0.473 |
| Average_Order_Value | 0.074 |

Proposed labels (checked against the numbers in code; they are the same four segments as Stage 5): **0: Loyal & Active**, **1: Lapsed Repeat Buyers**, **2: Slow-Cycle Repeat Buyers**, **3: One-Time Buyers**.

## 4. Does Preferred_Category_Share create a concern?

**Yes, a moderate and measurable one.** Three findings:

1. **It is mechanically tied to order count.** Share explains nothing on its own: order count alone explains **74%** of its variance (correlation with log orders -0.79). A customer with 1 order always has share 1.0, and 2 orders can only give 0.5 or 1.0. That is why share separates the 892 single-order customers so cleanly. In Core-5 it effectively counts "number of orders" a second time.
2. **It carries no category-loyalty information.** If every order's category were picked at random, the average share for each order count would be exactly what we see (table below). Real customers behave like random pickers, so share has no behavioural meaning beyond order count.

| Total_Orders | Customers | Observed_mean_share | Simulated_random_share |
|---|---|---|---|
| 1 | 892 | 1.0 | 1.0 |
| 2 | 1254 | 0.56 | 0.562 |
| 3 | 1009 | 0.443 | 0.454 |
| 4 | 575 | 0.411 | 0.413 |
| 5 | 445 | 0.389 | 0.388 |
| 6 | 299 | 0.358 | 0.363 |
| 7 | 192 | 0.338 | 0.342 |
| 8 | 141 | 0.32 | 0.324 |
| 9 | 99 | 0.31 | 0.312 |
| 10 | 94 | 0.319 | 0.303 |

3. **It causes mislabeled customers.** The "One-Time Buyers" cluster holds 1,008 customers: 892 single-order customers (all of them) plus **116 repeat customers** who happened to buy all their orders in the same category (166 such customers exist in total). Purity is 88.5%. These 116 customers are not one-time buyers; they are in the cluster by coincidence.

How much it matters: removing it (diagnostic model C) gives Silhouette 0.272, essentially the same as B (0.274), but a **different grouping** (ARI 0.59 with B). So the share is not harmless; it decides how the partition is built. Without it, the data splits by order value rather than by one-time vs repeat:

| index | Customers | Customers_% | Total_Orders | Total_Spending | Average_Order_Value | Days_Since_Last_Order | Average_Days_Between_Orders | Single_Order_% | Revenue_Share_% |
|---|---|---|---|---|---|---|---|---|---|
| C-0 | 1700 | 34.0 | 5.8 | 7407.4 | 1265.6 | 56.4 | 68.4 | 0.0 | 57.8 |
| C-1 | 1143 | 22.9 | 2.5 | 2797.5 | 1123.7 | 68.1 | 194.0 | 2.4 | 14.7 |
| C-2 | 1081 | 21.6 | 2.3 | 5063.4 | 2382.5 | 206.5 | 67.6 | 27.2 | 25.1 |
| C-3 | 1076 | 21.5 | 1.7 | 479.0 | 281.6 | 215.3 | 77.6 | 53.0 | 2.4 |

Model C's "lowest-value" and "high order value" groups come from Average_Order_Value, which section 5 shows is not a stable customer trait. So dropping the share does not simply give a cleaner version of B; it gives a different, less persistent kind of segment.

## 5. Are the features persistent customer traits? (business usefulness)

A segment is more useful for marketing if the behaviour behind it repeats over time. Checks on the raw orders:

| index | value |
|---|---|
| Order value: corr(order 1, order 2) | 0.018 |
| Order value: ICC (share of variance due to the customer) | -0.003 |
| Order value, category-adjusted: ICC | -0.001 |
| Gap: corr(gap 1, gap 2) | -0.169 |
| Order count: corr(Apr-Sep 2023, Oct 2023-Mar 2024) | 0.228 |
| Spend: corr(Apr-Sep 2023, Oct 2023-Mar 2024) | 0.096 |

- **Order value is not a customer trait:** a customer's first and second order values are uncorrelated (about 0.02) and the customer explains about 0% of the variance in order value, even after adjusting for category. A customer's Average_Order_Value mostly reflects which categories they happened to buy (and, for customers with few orders, one lucky or unlucky purchase). Segments built heavily on AOV would describe past baskets, not stable customer types.
- **Order frequency is weakly persistent:** customers' order counts in two consecutive six-month periods correlate about 0.23. This is a modest but real signal, consistent with segments built on orders and recency.
- **Spend** repeats only weakly (0.10); the gap between orders is slightly negative (-0.17), partly because the observation window is fixed (a long first gap leaves less time for the next one).
- For repeat customers only (4,108), Core-5 silhouette is K=2: 0.24, K=3: 0.26, K=4: 0.23, so the improvement over A is not only the easy one-time vs repeat split.

## 6. Decision matrix

| Criterion | A: 10 features | B: Core-5 | C: Core-4 (diagnostic) |
|---|---|---|---|
| Cluster quality (Silhouette / CH / DB) | 0.110 / 635 / 2.28 (weakest) | 0.274 / 1988 / 1.28 | 0.272 / 1891 / 1.24 |
| Stability (subsample ARI mean / worst) | 0.975 / 0.962 | 0.988 / 0.957 | 0.971 / 0.925 |
| Balance (smallest cluster) | 20.1% | 20.2% | 21.5% |
| Feature redundancy | 4 noise features, dilute distances | share overlaps with orders (R2 0.74) | none |
| Interpretability | same 4 lifecycle segments, blurry edges | same 4 lifecycle segments, clean edges | value-driven groups, no one-time group |
| Business usefulness | lifecycle, but noisy | lifecycle (frequency/recency, weakly persistent) | adds order value, which is not persistent |

## 7. Recommendation

**Use Model B (Core-5, K=4) as the production model. Retire Model A (keep its files, do not use it).**

Evidence:
1. B beats A on Silhouette, Calinski-Harabasz, Davies-Bouldin and worst-cluster quality, and matches it on stability, while producing the same four business segments (ARI 0.89). The four removed features have no separating power, so A has no advantage to offer.
2. B has the highest mean stability of the three (subsample ARI 0.988; worst case 0.957, within 0.005 of A) and is well balanced (smallest segment 20.2%).
3. B is the most interpretable: Loyal & Active, Lapsed Repeat Buyers, Slow-Cycle Repeat Buyers, One-Time Buyers, all driven mainly by order count, recency and order gap. Of the behaviours we could test, order count is the only one with meaningful persistence over time (correlation 0.23); order value has none.
4. The alternative that removes the share (C) matches B on Silhouette but is less stable (worst subsample ARI 0.925) and trades the one-time/repeat structure for groups driven by order value, which is not a stable customer trait. It is not a clear improvement, so we do not switch.

Conditions attached to this recommendation (known limitations of B):
- **Preferred_Category_Share double-counts order count and misplaces 116 repeat customers into "One-Time Buyers".** In the prediction system and dashboard, show `Is_Single_Order_Customer` next to the segment so a one-order customer is never confused with the 116 coincidence cases. Treat the segment name as "One-Time Buyers (mostly)". If exact purity matters more than the grouping, the cleaner design is a rule (Total_Orders = 1) for that segment plus clustering of repeat customers only; that would be a separate experiment.
- Silhouette 0.27 is still weak: the segments partition a continuum and are not natural groups.
- Average_Order_Value has little effect on B's grouping; revenue differences between segments come from order count.
- The data looks synthetic (random categories, flat ratings), so the stability of these findings on real data is untested. Re-run Stage 5.1 if real data replaces it.

## Files produced (nothing from Stage 5 modified)

`models/customer_segmentation_kmeans_core5.joblib`, `models/cluster_preprocessor_core5.joblib`, `models/segment_metadata_core5.json`, `data/processed/customer_segments_core5.csv`, `data/processed/cluster_profiles_core5.csv`, `data/processed/clustering_matrix_core5.csv`, `reports/stage5_1_core5_k_sweep.csv`, `reports/stage5_1_model_comparison.csv`, charts `reports/figures/25` to `29`.
