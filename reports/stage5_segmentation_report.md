# Stage 5 Report: K-Means Customer Segmentation

*E-Commerce AI Analytics & Customer Intelligence Platform*

**Data:** 5,000 customers, 10 clustering features (see Section 1). No outcome variable is predicted; this is unsupervised.
**Method:** `KMeans(init="k-means++", n_init=50, random_state=42)`, K = 2 to 10, on the standardized matrix from Stage 4.
Total_Spending, Unique_Categories_Bought, Average_Discount and the categorical preferences were **not** used for training, only to describe clusters afterwards.

---

## 1. Tested K values and training features

K tested: 2, 3, 4, 5, 6, 7, 8, 9, 10.

Training features (10): Total_Orders, Average_Order_Value, Average_Discount_Rate, Average_Customer_Rating, Average_Session_Duration, Average_Pages_Viewed, Average_Delivery_Time, Days_Since_Last_Order, Average_Days_Between_Orders, Preferred_Category_Share.

## 2. Evaluation metrics for every K

| K | Inertia | Silhouette | Calinski_Harabasz | Davies_Bouldin | Smallest_% | Largest_% | Inertia_drop_% |
|---|---|---|---|---|---|---|---|
| 2 | 42270.873 | 0.201 | 913.873 | 2.221 | 33.48 | 66.52 | - |
| 3 | 38549.786 | 0.128 | 742.12 | 2.294 | 26.92 | 45.58 | 8.803 |
| 4 | 36204.805 | 0.11 | 634.546 | 2.283 | 20.12 | 33.36 | 6.083 |
| 5 | 34511.546 | 0.117 | 560.428 | 2.201 | 7.56 | 32.72 | 4.677 |
| 6 | 33218.501 | 0.114 | 504.58 | 2.248 | 6.78 | 28.46 | 3.747 |
| 7 | 32007.789 | 0.12 | 467.781 | 2.224 | 5.52 | 28.34 | 3.645 |
| 8 | 31061.424 | 0.119 | 434.813 | 2.084 | 4.9 | 27.6 | 2.957 |
| 9 | 30185.104 | 0.098 | 409.54 | 2.101 | 4.86 | 21.2 | 2.821 |
| 10 | 29467.226 | 0.099 | 386.337 | 2.023 | 4.28 | 20.98 | 2.378 |

How to read: Inertia always falls as K grows (no clear elbow: each step drops it by a smoothly shrinking 2-9%). Silhouette and Calinski-Harabasz are higher = better; Davies-Bouldin is lower = better. Silhouette guide: above 0.50 reasonable, 0.26 to 0.50 weak, 0.25 or below means no substantial natural structure.

**Cluster sizes for every K (largest first):**

| K | Sizes (n, %) |
|---|---|
| 2 | 3,326 (66.5%), 1,674 (33.5%) |
| 3 | 2,279 (45.6%), 1,375 (27.5%), 1,346 (26.9%) |
| 4 | 1,668 (33.4%), 1,199 (24.0%), 1,127 (22.5%), 1,006 (20.1%) |
| 5 | 1,636 (32.7%), 1,155 (23.1%), 1,091 (21.8%), 740 (14.8%), 378 (7.6%) |
| 6 | 1,423 (28.5%), 854 (17.1%), 851 (17.0%), 816 (16.3%), 717 (14.3%), 339 (6.8%) |
| 7 | 1,417 (28.3%), 840 (16.8%), 817 (16.3%), 815 (16.3%), 441 (8.8%), 394 (7.9%), 276 (5.5%) |
| 8 | 1,380 (27.6%), 830 (16.6%), 810 (16.2%), 771 (15.4%), 352 (7.0%), 319 (6.4%), 293 (5.9%), 245 (4.9%) |
| 9 | 1,060 (21.2%), 898 (18.0%), 771 (15.4%), 647 (12.9%), 430 (8.6%), 333 (6.7%), 328 (6.6%), 290 (5.8%), 243 (4.9%) |
| 10 | 1,049 (21.0%), 890 (17.8%), 763 (15.3%), 650 (13.0%), 424 (8.5%), 302 (6.0%), 245 (4.9%), 235 (4.7%), 228 (4.6%), 214 (4.3%) |

## 3. Stability results

Two tests per K. (a) 15 extra random seeds, each with n_init=50, compared with the reference run. (b) 30 random 80% subsamples, re-clustered, compared with the reference labels on the same customers. ARI = 1.0 means identical groups.

| K | Seed_ARI_mean | Seed_ARI_min | Subsample_ARI_mean | Subsample_ARI_min |
|---|---|---|---|---|
| 2 | 0.997 | 0.994 | 0.952 | 0.535 |
| 3 | 0.996 | 0.992 | 0.967 | 0.934 |
| 4 | 0.998 | 0.994 | 0.974 | 0.953 |
| 5 | 0.993 | 0.98 | 0.941 | 0.852 |
| 6 | 0.985 | 0.973 | 0.821 | 0.613 |
| 7 | 0.922 | 0.866 | 0.861 | 0.591 |
| 8 | 0.925 | 0.895 | 0.745 | 0.536 |
| 9 | 0.904 | 0.799 | 0.815 | 0.705 |
| 10 | 0.904 | 0.829 | 0.752 | 0.529 |

Seed stability is high everywhere because n_init=50 almost always finds the same best solution, so the **subsample test is the stronger one**. K=2 looks stable on average but has a bad worst case; K of 6 and above lose stability.

## 4. Selected K and the evidence

Finalist rules (written before choosing): smallest cluster at least 10% of customers, subsample ARI mean at least 0.9, worst subsample ARI at least 0.8.

| K | Silhouette | Calinski_Harabasz | Davies_Bouldin | Smallest_% | Subsample_ARI_mean | Subsample_ARI_min | Balance_ok | Stable_ok | Finalist |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.201 | 913.873 | 2.221 | 33.48 | 0.952 | 0.535 | True | False | False |
| 3 | 0.128 | 742.12 | 2.294 | 26.92 | 0.967 | 0.934 | True | True | True |
| 4 | 0.11 | 634.546 | 2.283 | 20.12 | 0.974 | 0.953 | True | True | True |
| 5 | 0.117 | 560.428 | 2.201 | 7.56 | 0.941 | 0.852 | False | True | False |
| 6 | 0.114 | 504.58 | 2.248 | 6.78 | 0.821 | 0.613 | False | False | False |
| 7 | 0.12 | 467.781 | 2.224 | 5.52 | 0.861 | 0.591 | False | False | False |
| 8 | 0.119 | 434.813 | 2.084 | 4.9 | 0.745 | 0.536 | False | False | False |
| 9 | 0.098 | 409.54 | 2.101 | 4.86 | 0.815 | 0.705 | False | False | False |
| 10 | 0.099 | 386.337 | 2.023 | 4.28 | 0.752 | 0.529 | False | False | False |

Finalists: **K = 3, 4**. Comparison of the finalists:

| K | Silhouette | Calinski_Harabasz | Davies_Bouldin | Subsample_ARI_mean | Smallest_% | Most_single_order_cluster_% |
|---|---|---|---|---|---|---|
| 3 | 0.128 | 742.12 | 2.294 | 0.967 | 26.92 | 64.8 |
| 4 | 0.11 | 634.546 | 2.283 | 0.974 | 20.12 | 88.668 |

**Selected K = 4.** Evidence:
- **Metrics:** K=2 has the best Silhouette and Calinski-Harabasz, but it fails stability (worst subsample ARI 0.54) and is too coarse to act on. K=4 is not the best on Silhouette or Calinski-Harabasz (0.110 and 635); K=3 scores a little higher (0.128 and 742). The gap is small and both are in the "weak structure" range, so metrics alone cannot separate K=3 from K=4.
- **Balance:** K=4 sizes are 1,668 (33.4%), 1,199 (24.0%), 1,127 (22.5%), 1,006 (20.1%). No tiny segment.
- **Stability:** K=4 has the highest subsample stability of all K (mean ARI 0.974, worst 0.953) and near-perfect seed stability (0.998).
- **Interpretability:** moving from K=3 to K=4 gives a much purer one-time-buyer group (89% single-order customers vs 65% at K=3) and separates repeat buyers who went quiet from repeat buyers who are on their normal cycle. Those two groups need different actions.

This is a judgement call between K=3 and K=4. K=3 is the simpler, slightly higher-scoring alternative.

## 5. Cluster sizes

| Cluster | Customers | Customers_% | Label |
|---|---|---|---|
| Cluster 0 | 1668 | 33.4 | Loyal & Active |
| Cluster 1 | 1127 | 22.5 | Lapsed Repeat Buyers |
| Cluster 2 | 1199 | 24.0 | Slow-Cycle Repeat Buyers |
| Cluster 3 | 1006 | 20.1 | One-Time Buyers |

## 6. Cluster profiles (numbers first)

Values are per-customer averages. Cluster numbers are sorted by mean Total_Orders (0 = most orders). `Silence_vs_Usual_Gap` = median of (days since last order / customer's usual gap between orders), multi-order customers only; above 1 means the customer has been quiet longer than usual. Average_Days_Between_Orders is averaged over customers who have a gap (multi-order customers).

| index | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| Label | Loyal & Active | Lapsed Repeat Buyers | Slow-Cycle Repeat Buyers | One-Time Buyers |
| Customers | 1668 | 1127 | 1199 | 1006 |
| Customers_% | 33.36 | 22.54 | 23.98 | 20.12 |
| Total_Orders | 5.78 | 2.88 | 2.53 | 1.12 |
| Total_Spending | 7771.51 | 3478.04 | 3001.11 | 1290.35 |
| Average_Order_Value | 1375.42 | 1228.92 | 1215.24 | 1171.13 |
| Average_Discount | 85.14 | 64.72 | 56.16 | 53.86 |
| Average_Customer_Rating | 3.89 | 3.89 | 3.91 | 3.92 |
| Average_Session_Duration | 14.51 | 14.55 | 14.6 | 14.57 |
| Average_Pages_Viewed | 9.0 | 8.92 | 9.11 | 8.92 |
| Average_Delivery_Time | 6.55 | 6.64 | 6.24 | 6.69 |
| Days_Since_Last_Order | 51.58 | 205.3 | 80.05 | 213.91 |
| Average_Days_Between_Orders | 71.0 | 59.97 | 189.51 | 103.05 |
| Unique_Categories_Bought | 4.24 | 2.54 | 2.29 | 1.0 |
| Revenue_Share_% | 59.52 | 18.0 | 16.52 | 5.96 |
| Single_Order_% | 0.0 | 0.0 | 0.0 | 88.67 |
| Silence_vs_Usual_Gap | 0.61 | 3.23 | 0.38 | 1.51 |

| index | Label | Preferred_Category | Preferred_Device | Preferred_Payment |
|---|---|---|---|---|
| Cluster 0 | Loyal & Active | Sports (15%) | Mobile (68%) | Credit Card (52%) |
| Cluster 1 | Lapsed Repeat Buyers | Food (14%) | Mobile (61%) | Credit Card (44%) |
| Cluster 2 | Slow-Cycle Repeat Buyers | Books (14%) | Mobile (60%) | Credit Card (45%) |
| Cluster 3 | One-Time Buyers | Beauty (14%) | Mobile (55%) | Credit Card (39%) |

**Which training features separate the clusters (eta squared; 0 = clusters ignore the feature, 1 = clusters are defined by it):**

| index | eta_squared |
|---|---|
| Preferred_Category_Share | 0.81 |
| Total_Orders | 0.769 |
| Average_Days_Between_Orders | 0.59 |
| Days_Since_Last_Order | 0.468 |
| Average_Order_Value | 0.071 |
| Average_Discount_Rate | 0.043 |
| Average_Delivery_Time | 0.006 |
| Average_Pages_Viewed | 0.002 |
| Average_Session_Duration | 0.0 |
| Average_Customer_Rating | 0.0 |

## 7. Suggested business interpretation

These names were written after reading the profile above. They are proposals, not facts.

- **Cluster 0: Loyal & Active** (1,668 customers, 33.4%, 59.5% of revenue). 5.8 orders, last order 52 days ago, usual gap 71 days, average spend 7,772. Most orders, most recent, highest total spend and highest order value. The core of the business; protect and reward.
- **Cluster 1: Lapsed Repeat Buyers** (1,127 customers, 22.5%, 18.0% of revenue). 2.9 orders, last order 205 days ago, usual gap 60 days, average spend 3,478. Repeat buyers who used to order often but have been quiet for several times their usual gap. Win-back target.
- **Cluster 2: Slow-Cycle Repeat Buyers** (1,199 customers, 24.0%, 16.5% of revenue). 2.5 orders, last order 80 days ago, usual gap 190 days, average spend 3,001. Repeat buyers with long natural gaps between orders who are still within their normal cycle. Time reminders to their rhythm.
- **Cluster 3: One-Time Buyers** (1,006 customers, 20.1%, 6.0% of revenue). 1.1 orders, last order 214 days ago, usual gap 103 days, average spend 1,290. Almost all bought once. Includes genuinely lost one-time buyers and brand-new customers. Needs onboarding and second-order incentives.

**Important:** the clusters are defined almost entirely by *purchase history* (orders, recency, gap, category loyalty). Features with eta squared below 0.02 (Average_Delivery_Time, Average_Pages_Viewed, Average_Session_Duration, Average_Customer_Rating) are essentially the same in every cluster (for example average rating is 3.89 to 3.92 in all clusters). Preferred device, payment and category also barely differ between clusters, so they should not be used to describe the segments.

## 8. Limitations

1. **Weak cluster structure.** Silhouette is 0.11 at K=4 (all K are at or below 0.20), under the 0.25 line for "no substantial structure". The data looks like a continuum of customers, not separate natural groups. The segments are a useful partition for action, not a discovery of hidden types.
2. **Only part of the feature set drives the result.** Rating, session duration, pages viewed and delivery time have eta squared of about 0 (identical in every cluster), and discount rate and order value separate the clusters only slightly. The clusters are built mainly from Preferred_Category_Share, Total_Orders, Average_Days_Between_Orders and Days_Since_Last_Order; note that category share is mechanically linked to order count (one order always means a share of 1.0). Check: clustering on only the 5 core features (Total_Orders, Average_Order_Value, Days_Since_Last_Order, Average_Days_Between_Orders, Preferred_Category_Share) gives Silhouette 0.27 at K=4 (vs 0.11) and groups that match the 10-feature groups with ARI 0.89. We kept the 10 features as specified; removing the 5 weak ones is an option for the next iteration.
3. **Censoring.** The data ends on 2024-03-25. 186 of 892 single-order customers (21%) ordered in the last 90 days, so part of "One-Time Buyers" are new customers who have not had time to buy again.
4. **No ground truth.** There is no "correct" segment to check against. Validation is internal (metrics, stability) plus business sense.
5. **Snapshot only.** Recency is measured at 2024-03-26. Segments must be recomputed as time passes. No train/test split was used because there is no target; the preprocessing and model were fit on all 5,000 customers.
6. **K-Means assumptions.** It forms roughly round, similar-sized groups in the scaled space and is sensitive to scaling choices. The data also appears synthetic (flat ratings, no link between behaviour and satisfaction), so real-world data may behave differently.
7. **PCA view.** The 2D picture shows only 34% of the variance (PC1 23%, PC2 12%); clusters will look touching or overlapping, which matches the weak silhouette.

## Files produced

`models/customer_segmentation_kmeans.joblib`, `models/segment_metadata.json`, `data/processed/customer_segments.csv`, `data/processed/cluster_profiles.csv`, `reports/stage5_k_evaluation.csv`, charts `reports/figures/19` to `24`.
