"""
Stage 5: K-Means customer segmentation.

Run from project root:  python src/models/stage5_kmeans_segmentation.py

What it does (in order):
  1. Tries K = 2..10 (KMeans, k-means++, n_init=50, random_state=42) and records 4 metrics + sizes
  2. Checks stability (different random seeds + random 80% subsamples)
  3. Applies simple, written-down rules to find the finalist K values
  4. Fits the FINAL model on the 10 clustering features only
  5. Profiles the clusters (profiling can use the columns that were NOT used for training)
  6. Saves model, segments, charts and a Markdown report

The 10 training features are fixed. Total_Spending etc. are used ONLY to describe clusters afterwards.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

ROOT = Path(__file__).resolve().parents[2]
MATRIX_CSV = ROOT / "data" / "processed" / "clustering_matrix.csv"
CUSTOMER_CSV = ROOT / "data" / "processed" / "customer_features.csv"
OUT_SEGMENTS = ROOT / "data" / "processed" / "customer_segments.csv"
OUT_MODEL = ROOT / "models" / "customer_segmentation_kmeans.joblib"
OUT_META = ROOT / "models" / "segment_metadata.json"
FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports" / "stage5_segmentation_report.md"
for p in (OUT_MODEL.parent, FIG):
    p.mkdir(parents=True, exist_ok=True)

K_RANGE = range(2, 11)
SEED = 42
N_INIT = 50
EXTRA_SEEDS = list(range(1, 16))      # 15 extra random seeds for the seed-stability test
N_SUBSAMPLES, SUBSAMPLE_FRAC = 30, 0.80

# --- written-down selection rules (common conventions, set BEFORE looking at the winner) ---
MIN_CLUSTER_PCT = 10.0     # a segment smaller than 10% of customers is hard to act on
MIN_SUB_ARI_MEAN = 0.90    # ARI > 0.9 = "almost identical partitions"
MIN_SUB_ARI_WORST = 0.80   # even the worst subsample must stay reasonably close
SELECTED_K = 4             # chosen after comparing the finalists (reasons printed + in the report)

CORE5 = ["Total_Orders", "Average_Order_Value", "Days_Since_Last_Order",
         "Average_Days_Between_Orders", "Preferred_Category_Share"]


# ======================================================================================
# helpers
# ======================================================================================
def head(t):
    print("\n" + "=" * 95 + f"\n{t}\n" + "=" * 95)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [chart saved] reports/figures/{name}")


def md_table(df, index=True):
    """Tiny DataFrame -> Markdown table (so we do not need extra libraries)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.astype(object).values.tolist():
        lines.append("| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def size_text(labels, k):
    c = np.bincount(labels, minlength=k)
    order = np.argsort(-c)
    return ", ".join(f"{c[i]:,} ({c[i] / len(labels):.1%})" for i in order)


# ======================================================================================
# 1. evaluate every K
# ======================================================================================
def evaluate_all_k(X):
    rows, labels_by_k = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=SEED).fit(X)
        lab = km.labels_
        labels_by_k[k] = lab
        cnt = np.bincount(lab, minlength=k)
        rows.append({"K": k, "Inertia": km.inertia_,
                     "Silhouette": silhouette_score(X, lab),
                     "Calinski_Harabasz": calinski_harabasz_score(X, lab),
                     "Davies_Bouldin": davies_bouldin_score(X, lab),
                     "Smallest_%": cnt.min() / len(X) * 100, "Largest_%": cnt.max() / len(X) * 100,
                     "Sizes (n, %)": size_text(lab, k)})
    df = pd.DataFrame(rows).set_index("K")
    df["Inertia_drop_%"] = (-df["Inertia"].pct_change() * 100)
    return df, labels_by_k


# ======================================================================================
# 2. stability
# ======================================================================================
def stability_all_k(X, labels_by_k):
    n = len(X)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in K_RANGE:
        ref = labels_by_k[k]
        seed_ari = [adjusted_rand_score(ref, KMeans(k, init="k-means++", n_init=N_INIT, random_state=s).fit_predict(X))
                    for s in EXTRA_SEEDS]
        sub_ari = []
        for i in range(N_SUBSAMPLES):
            idx = rng.choice(n, int(SUBSAMPLE_FRAC * n), replace=False)
            lab = KMeans(k, init="k-means++", n_init=10, random_state=i).fit_predict(X[idx])
            sub_ari.append(adjusted_rand_score(ref[idx], lab))
        rows.append({"K": k, "Seed_ARI_mean": np.mean(seed_ari), "Seed_ARI_min": np.min(seed_ari),
                     "Subsample_ARI_mean": np.mean(sub_ari), "Subsample_ARI_min": np.min(sub_ari)})
        print(f"  K={k}: seeds ARI mean {np.mean(seed_ari):.3f} (min {np.min(seed_ari):.3f}) | "
              f"subsamples ARI mean {np.mean(sub_ari):.3f} (min {np.min(sub_ari):.3f})")
    return pd.DataFrame(rows).set_index("K")


# ======================================================================================
# 3. finalists
# ======================================================================================
def find_finalists(metrics, stab):
    t = metrics[["Silhouette", "Calinski_Harabasz", "Davies_Bouldin", "Smallest_%"]].join(stab)
    t["Balance_ok"] = t["Smallest_%"] >= MIN_CLUSTER_PCT
    t["Stable_ok"] = (t["Subsample_ARI_mean"] >= MIN_SUB_ARI_MEAN) & (t["Subsample_ARI_min"] >= MIN_SUB_ARI_WORST)
    t["Finalist"] = t["Balance_ok"] & t["Stable_ok"]
    return t


# ======================================================================================
# 4-5. final model + profile
# ======================================================================================
def fit_final(X, cust, k):
    """Fit the final model, then renumber clusters so Cluster 0 = most orders per customer, 1 = next, ...
    (renumbering only changes the NAMES of clusters, not who is grouped with whom)."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=SEED).fit(X)
    old = km.labels_
    orders_mean = cust["Total_Orders"].groupby(old).mean()
    order = list(orders_mean.sort_values(ascending=False).index)      # new id i  <->  old id order[i]
    remap = {o: i for i, o in enumerate(order)}
    new = np.array([remap[v] for v in old])
    km.cluster_centers_ = km.cluster_centers_[order]
    km.labels_ = new
    assert (km.predict(X) == new).all(), "renumbering broke predict()"
    return km, new


PROFILE_COLS = ["Total_Orders", "Total_Spending", "Average_Order_Value", "Average_Discount", "Average_Discount_Rate",
                "Average_Customer_Rating", "Average_Session_Duration", "Average_Pages_Viewed", "Average_Delivery_Time",
                "Days_Since_Last_Order", "Average_Days_Between_Orders", "Unique_Categories_Bought",
                "Preferred_Category_Share"]
TRAINING_COLS = {"Total_Orders", "Average_Order_Value", "Average_Discount_Rate", "Average_Customer_Rating",
                 "Average_Session_Duration", "Average_Pages_Viewed", "Average_Delivery_Time",
                 "Days_Since_Last_Order", "Average_Days_Between_Orders", "Preferred_Category_Share"}


def mode_with_share(s):
    vc = s.value_counts()
    return f"{vc.index[0]} ({vc.iloc[0] / len(s):.0%})"


def build_profile(seg):
    g = seg.groupby("Cluster")
    prof = pd.DataFrame({"Customers": g.size()})
    prof["Customers_%"] = prof["Customers"] / len(seg) * 100
    for c in PROFILE_COLS:
        prof[c] = g[c].mean()                      # Average_Days_Between_Orders: mean of customers who HAVE a gap
    prof["Revenue_Share_%"] = g["Total_Spending"].sum() / seg["Total_Spending"].sum() * 100
    prof["Single_Order_%"] = g["Is_Single_Order_Customer"].mean() * 100
    multi = seg[(seg["Total_Orders"] > 1) & (seg["Average_Days_Between_Orders"] > 0)].copy()
    multi["ratio"] = multi["Days_Since_Last_Order"] / multi["Average_Days_Between_Orders"]
    prof["Silence_vs_Usual_Gap"] = multi.groupby("Cluster")["ratio"].median()
    prof["Preferred_Category"] = g["Preferred_Product_Category"].agg(mode_with_share)
    prof["Preferred_Device"] = g["Preferred_Device_Type"].agg(mode_with_share)
    prof["Preferred_Payment"] = g["Preferred_Payment_Method"].agg(mode_with_share)
    return prof


def eta_squared(Xdf, labels):
    """Share of each feature's variance explained by the clusters (0 = clusters ignore it, 1 = clusters ARE it)."""
    out = {}
    for c in Xdf.columns:
        x = Xdf[c].to_numpy()
        ss_tot = ((x - x.mean()) ** 2).sum()
        ss_between = sum(len(x[labels == l]) * (x[labels == l].mean() - x.mean()) ** 2 for l in np.unique(labels))
        out[c] = ss_between / ss_tot
    return pd.Series(out).sort_values(ascending=False)


# Business labels are proposals, written AFTER reading the numerical profile (see section 6).
LABELS_K4 = {0: "Loyal & Active", 1: "Lapsed Repeat Buyers", 2: "Slow-Cycle Repeat Buyers", 3: "One-Time Buyers"}


def propose_labels(prof, k):
    if k != 4:
        return {i: f"Segment {i}" for i in range(k)}
    # sanity checks: the proposed names must match the numbers, otherwise stop
    assert prof["Days_Since_Last_Order"].idxmin() == 0 and prof["Total_Orders"].idxmax() == 0
    assert prof["Single_Order_%"].idxmax() == 3 and prof["Total_Orders"].idxmin() == 3
    assert prof.loc[1, "Silence_vs_Usual_Gap"] > prof.loc[2, "Silence_vs_Usual_Gap"]
    assert prof.loc[2, "Average_Days_Between_Orders"] > prof.loc[1, "Average_Days_Between_Orders"]
    return LABELS_K4


# ======================================================================================
# charts
# ======================================================================================
def chart_metrics(metrics, finalists):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    spec = [("Inertia", "1. Elbow plot (lower = tighter clusters, always falls)", "#4C78A8"),
            ("Silhouette", "2. Silhouette (higher = better; <0.25 = weak structure)", "#54A24B"),
            ("Calinski_Harabasz", "3. Calinski-Harabasz (higher = better)", "#F58518"),
            ("Davies_Bouldin", "4. Davies-Bouldin (LOWER = better)", "#E45756")]
    for a, (col, title, color) in zip(ax.ravel(), spec):
        a.plot(metrics.index, metrics[col], marker="o", color=color)
        for f in finalists:
            a.axvspan(f - .4, f + .4, color="grey", alpha=.15)
        a.axvline(SELECTED_K, color="black", ls="--", lw=1)
        a.set_title(title, fontsize=10); a.set_xlabel("K (number of clusters)"); a.set_xticks(list(K_RANGE))
    fig.suptitle(f"K-Means model selection  |  grey = finalist K, dashed line = selected K ({SELECTED_K})", y=1.01)
    save_fig("19_kmeans_metrics.png")


def chart_stability(stab, finalists):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(stab.index, stab["Subsample_ARI_mean"], marker="o", label="80% subsample: mean ARI", color="#4C78A8")
    ax.plot(stab.index, stab["Subsample_ARI_min"], marker="v", ls="--", label="80% subsample: WORST ARI", color="#E45756")
    ax.plot(stab.index, stab["Seed_ARI_mean"], marker="s", label="different seeds: mean ARI", color="#54A24B")
    ax.axhline(MIN_SUB_ARI_MEAN, color="grey", ls=":"); ax.text(10.05, MIN_SUB_ARI_MEAN, "0.90", va="center")
    for f in finalists:
        ax.axvspan(f - .4, f + .4, color="grey", alpha=.15)
    ax.set_xlabel("K"); ax.set_ylabel("Adjusted Rand Index (1 = identical groups)"); ax.set_xticks(list(K_RANGE))
    ax.set_title("Cluster stability: do customers stay in the same groups?"); ax.legend(loc="lower left", fontsize=8)
    save_fig("20_stability.png")


def chart_sizes(prof, labels):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [f"{i}: {labels[i]}" for i in prof.index]
    bars = ax.bar(names, prof["Customers"], color=sns.color_palette("Set2", len(prof)))
    for b, (n, p) in zip(bars, zip(prof["Customers"], prof["Customers_%"])):
        ax.text(b.get_x() + b.get_width() / 2, n, f"{n:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Customers"); ax.set_title("Cluster sizes"); plt.xticks(rotation=15, ha="right")
    ax.set_ylim(0, prof["Customers"].max() * 1.18)
    save_fig("21_cluster_sizes.png")


def chart_heatmap(seg, prof, labels):
    rows = PROFILE_COLS + ["Single_Order_%"]
    raw_means = prof[rows]
    overall = seg[PROFILE_COLS].mean()
    overall_std = seg[PROFILE_COLS].std()
    z = (prof[PROFILE_COLS] - overall) / overall_std
    z["Single_Order_%"] = (prof["Single_Order_%"] - seg["Is_Single_Order_Customer"].mean() * 100) / (seg["Is_Single_Order_Customer"].std() * 100)
    ylabels = [f"{c}{'' if c in TRAINING_COLS else '  (profiling only)'}" for c in rows]
    annot = raw_means.round(2).astype(object)
    plt.figure(figsize=(9.5, 8))
    sns.heatmap(z.T, annot=annot.T, fmt="", cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5,
                xticklabels=[f"{i}: {labels[i]}" for i in z.index], yticklabels=ylabels,
                cbar_kws={"label": "cluster mean vs overall mean (in std units)"}, annot_kws={"size": 8})
    plt.title("Cluster profile heatmap (numbers = real cluster means; colour = difference from average)")
    plt.xticks(rotation=15, ha="right")
    save_fig("22_cluster_profile_heatmap.png")


def chart_pca(X, km, labels_arr, labels):
    pca = PCA(n_components=2, random_state=SEED)
    P = pca.fit_transform(X)
    C = pca.transform(km.cluster_centers_)
    pal = sns.color_palette("Set2", km.n_clusters)
    plt.figure(figsize=(9, 7))
    for i in range(km.n_clusters):
        m = labels_arr == i
        plt.scatter(P[m, 0], P[m, 1], s=9, alpha=.45, color=pal[i], label=f"{i}: {labels[i]}")
        plt.scatter(C[i, 0], C[i, 1], s=260, marker="X", color=pal[i], edgecolor="black", linewidth=1.5)
    ev = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({ev[0]:.1%} of variance)"); plt.ylabel(f"PC2 ({ev[1]:.1%} of variance)")
    plt.title(f"PCA 2D view of the clusters (X = cluster centre). The 2 axes show only {ev.sum():.1%} of the information")
    plt.legend(markerscale=2.5, fontsize=9)
    save_fig("23_pca_clusters.png")
    load = pd.DataFrame(pca.components_.T, index=X.columns, columns=["PC1", "PC2"])
    return P, ev, load


def chart_sensitivity(sens, eta):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(sens.index, sens["All_10_features"], marker="o", label="all 10 features (used)")
    ax[0].plot(sens.index, sens["Core_5_features"], marker="s", label="core 5 only (check)")
    ax[0].axhline(0.25, color="grey", ls=":"); ax[0].text(2, 0.255, "0.25 = 'weak structure' line", fontsize=8)
    ax[0].set_xlabel("K"); ax[0].set_ylabel("Silhouette"); ax[0].legend(); ax[0].set_title("Silhouette: all 10 vs core 5 features")
    eta.sort_values().plot(kind="barh", ax=ax[1], color="#72B7B2")
    ax[1].set_title(f"How much of each feature do the K={SELECTED_K} clusters explain? (eta squared)")
    save_fig("24_feature_sensitivity.png")


# ======================================================================================
def main():
    X_df = pd.read_csv(MATRIX_CSV, index_col="Customer_ID")
    cust = pd.read_csv(CUSTOMER_CSV).set_index("Customer_ID")
    assert X_df.index.equals(cust.index), "customer order mismatch between the two files"
    assert X_df.shape == (5000, 10) and not X_df.isna().any().any()
    X = X_df.to_numpy()
    print("Training matrix:", X_df.shape, "| features:", list(X_df.columns))

    # ---- 1
    head("1. TESTING K = 2 ... 10")
    metrics, labels_by_k = evaluate_all_k(X)
    show = metrics.drop(columns=["Sizes (n, %)"]).round(3)
    print(show.to_string())
    print("\nCluster sizes per K (largest first):")
    for k in K_RANGE:
        print(f"  K={k:2d}: {metrics.loc[k, 'Sizes (n, %)']}")
    print("\nSilhouette guide (Kaufman & Rousseeuw): 0.71-1.0 strong | 0.51-0.70 reasonable | 0.26-0.50 weak | <=0.25 no substantial structure")

    # ---- 2
    head("2. STABILITY (15 extra seeds with n_init=50  +  30 random 80% subsamples)")
    stab = stability_all_k(X, labels_by_k)
    print("\nARI = Adjusted Rand Index: 1.0 means customers land in exactly the same groups, ~0 means random.")

    # ---- 3
    head("3. FINALIST RULES")
    print(f"Rules: smallest cluster >= {MIN_CLUSTER_PCT:.0f}% of customers | subsample ARI mean >= {MIN_SUB_ARI_MEAN} and worst >= {MIN_SUB_ARI_WORST}")
    table = find_finalists(metrics, stab)
    print(table.round(3).to_string())
    finalists = [int(k) for k in table.index[table["Finalist"]]]
    print("\nFinalists:", finalists)
    assert SELECTED_K in finalists, f"SELECTED_K={SELECTED_K} did not pass the rules; finalists are {finalists}"

    # compare the finalists on the thing that matters for the business: how pure is the 'one-time buyer' group?
    cmp_rows = []
    for k in finalists:
        lab = labels_by_k[k]
        single = cust["Is_Single_Order_Customer"].groupby(lab).mean()
        cmp_rows.append({"K": k, "Silhouette": metrics.loc[k, "Silhouette"], "Calinski_Harabasz": metrics.loc[k, "Calinski_Harabasz"],
                         "Davies_Bouldin": metrics.loc[k, "Davies_Bouldin"], "Subsample_ARI_mean": stab.loc[k, "Subsample_ARI_mean"],
                         "Smallest_%": metrics.loc[k, "Smallest_%"],
                         "Most_single_order_cluster_%": single.max() * 100})
    cmp_df = pd.DataFrame(cmp_rows).set_index("K")
    print("\nFinalist comparison:\n", cmp_df.round(3).to_string())

    # ---- 4
    head(f"4. FINAL MODEL: K = {SELECTED_K}")
    km, lab = fit_final(X, cust, SELECTED_K)
    print("Inertia:", round(km.inertia_, 1), "| Silhouette:", round(silhouette_score(X, lab), 4),
          "| CH:", round(calinski_harabasz_score(X, lab), 1), "| DB:", round(davies_bouldin_score(X, lab), 4))
    assert abs(silhouette_score(X, lab) - metrics.loc[SELECTED_K, "Silhouette"]) < 1e-9, "renumbering must not change the clustering"
    seg = cust.reset_index().copy()
    seg["Cluster"] = lab

    # ---- 5
    head("5. NUMERICAL CLUSTER PROFILE (read this BEFORE any naming)")
    prof = build_profile(seg)
    show = prof.copy()
    for c in show.select_dtypes("number").columns:
        show[c] = show[c].round(2)
    print(show.T.to_string())
    eta = eta_squared(X_df, lab)
    print("\nWhich training features do the clusters actually separate? (eta squared, 0 to 1):\n", eta.round(3).to_string())

    # ---- 6
    head("6. PROPOSED BUSINESS LABELS (based on the profile above)")
    labels = propose_labels(prof, SELECTED_K)
    seg["Segment_Label"] = seg["Cluster"].map(labels)
    for i, name in labels.items():
        p = prof.loc[i]
        print(f"  Cluster {i} = {name}: {p['Customers']:,.0f} customers ({p['Customers_%']:.1f}%), "
              f"{p['Total_Orders']:.1f} orders, last order {p['Days_Since_Last_Order']:.0f} days ago, "
              f"avg spend {p['Total_Spending']:,.0f}, {p['Revenue_Share_%']:.1f}% of revenue")
    single_recent = seg[(seg["Is_Single_Order_Customer"] == 1) & (seg["Days_Since_Last_Order"] <= 90)]
    n_single = int(seg["Is_Single_Order_Customer"].sum())
    print(f"\nCaution: {len(single_recent)} of {n_single} single-order customers ({len(single_recent) / n_single:.0%}) ordered in the last 90 days "
          "-> some 'One-Time Buyers' are simply NEW customers who have not had time to re-order.")

    # ---- sensitivity (the check promised in Stage 4)
    head("7. SENSITIVITY CHECK: are the 5 behaviour features adding anything?")
    core = X_df[CORE5].to_numpy()
    sens_rows, core_labels = [], {}
    for k in K_RANGE:
        l = KMeans(k, init="k-means++", n_init=N_INIT, random_state=SEED).fit_predict(core)
        core_labels[k] = l
        sens_rows.append({"K": k, "All_10_features": metrics.loc[k, "Silhouette"], "Core_5_features": silhouette_score(core, l)})
    sens = pd.DataFrame(sens_rows).set_index("K")
    print(sens.round(3).to_string())
    ari_core = adjusted_rand_score(lab, core_labels[SELECTED_K])
    print(f"\nARI between the K={SELECTED_K} groups from 10 features and from the core 5: {ari_core:.3f}")

    # ---- PCA + charts
    head("8. CHARTS")
    chart_metrics(metrics, finalists)
    chart_stability(stab, finalists)
    chart_sizes(prof, labels)
    chart_heatmap(seg, prof, labels)
    P, ev, load = chart_pca(X_df, km, lab, labels)
    chart_sensitivity(sens, eta)
    print(f"  PCA: PC1 {ev[0]:.1%} + PC2 {ev[1]:.1%} = {ev.sum():.1%} of the variance")
    print("  PCA loadings (what each axis is made of):\n", load.round(2).to_string())
    seg["PC1"], seg["PC2"] = P[:, 0], P[:, 1]

    # ---- save
    head("9. SAVE")
    joblib.dump(km, OUT_MODEL)
    reloaded = joblib.load(OUT_MODEL)
    assert (reloaded.predict(X) == lab).all()
    seg.to_csv(OUT_SEGMENTS, index=False)
    assert seg["Customer_ID"].is_unique and len(seg) == 5000
    meta = {"selected_k": SELECTED_K, "features": list(X_df.columns), "random_state": SEED, "n_init": N_INIT,
            "cluster_labels": {str(k): v for k, v in labels.items()},
            "preprocessor": "models/cluster_preprocessor.joblib",
            "note": "Cluster ids are numbered by mean Total_Orders (0 = most orders). Labels are proposed names and can be edited."}
    OUT_META.write_text(json.dumps(meta, indent=2))
    prof.round(3).to_csv(ROOT / "data" / "processed" / "cluster_profiles.csv")
    metrics.round(4).join(stab.round(4)).to_csv(ROOT / "reports" / "stage5_k_evaluation.csv")
    print(f"  {OUT_MODEL.relative_to(ROOT)} (reload test OK)\n  {OUT_SEGMENTS.relative_to(ROOT)} {seg.shape}\n  {OUT_META.relative_to(ROOT)}")

    write_report(metrics, stab, table, finalists, cmp_df, prof, labels, eta, sens, ari_core, ev, load, single_recent, n_single)
    print(f"  {REPORT.relative_to(ROOT)}")
    head("STAGE 5 DONE - no API / website started")


# ======================================================================================
# report
# ======================================================================================
def write_report(metrics, stab, table, finalists, cmp_df, prof, labels, eta, sens, ari_core, ev, load, single_recent, n_single):
    k = SELECTED_K
    m = metrics.drop(columns=["Sizes (n, %)"]).round(3)
    sizes = metrics[["Sizes (n, %)"]]
    stab_t = stab.round(3)
    fin_t = table[["Silhouette", "Calinski_Harabasz", "Davies_Bouldin", "Smallest_%", "Subsample_ARI_mean",
                   "Subsample_ARI_min", "Balance_ok", "Stable_ok", "Finalist"]].round(3)

    prof_main = prof.copy()
    prof_main.insert(0, "Label", [labels[i] for i in prof_main.index])
    num_cols = ["Customers", "Customers_%", "Total_Orders", "Total_Spending", "Average_Order_Value", "Average_Discount",
                "Average_Customer_Rating", "Average_Session_Duration", "Average_Pages_Viewed", "Average_Delivery_Time",
                "Days_Since_Last_Order", "Average_Days_Between_Orders", "Unique_Categories_Bought",
                "Revenue_Share_%", "Single_Order_%", "Silence_vs_Usual_Gap"]
    pt = prof_main[["Label"] + num_cols].round(2).T
    pt.columns = [f"Cluster {c}" for c in pt.columns]
    cat_t = prof_main[["Label", "Preferred_Category", "Preferred_Device", "Preferred_Payment"]].copy()
    cat_t.index = [f"Cluster {c}" for c in cat_t.index]

    p = prof
    top_eta = eta.round(3)
    weak = [c for c in eta.index if eta[c] < 0.02]
    s = []
    s.append(f"""# Stage 5 Report: K-Means Customer Segmentation

*E-Commerce AI Analytics & Customer Intelligence Platform*

**Data:** 5,000 customers, 10 clustering features (see Section 1). No outcome variable is predicted; this is unsupervised.
**Method:** `KMeans(init="k-means++", n_init=50, random_state=42)`, K = 2 to 10, on the standardized matrix from Stage 4.
Total_Spending, Unique_Categories_Bought, Average_Discount and the categorical preferences were **not** used for training, only to describe clusters afterwards.

---

## 1. Tested K values and training features

K tested: 2, 3, 4, 5, 6, 7, 8, 9, 10.

Training features (10): Total_Orders, Average_Order_Value, Average_Discount_Rate, Average_Customer_Rating, Average_Session_Duration, Average_Pages_Viewed, Average_Delivery_Time, Days_Since_Last_Order, Average_Days_Between_Orders, Preferred_Category_Share.

## 2. Evaluation metrics for every K

{md_table(m)}

How to read: Inertia always falls as K grows (no clear elbow: each step drops it by a smoothly shrinking 2-9%). Silhouette and Calinski-Harabasz are higher = better; Davies-Bouldin is lower = better. Silhouette guide: above 0.50 reasonable, 0.26 to 0.50 weak, 0.25 or below means no substantial natural structure.

**Cluster sizes for every K (largest first):**

{md_table(sizes)}

## 3. Stability results

Two tests per K. (a) 15 extra random seeds, each with n_init=50, compared with the reference run. (b) 30 random 80% subsamples, re-clustered, compared with the reference labels on the same customers. ARI = 1.0 means identical groups.

{md_table(stab_t)}

Seed stability is high everywhere because n_init=50 almost always finds the same best solution, so the **subsample test is the stronger one**. K=2 looks stable on average but has a bad worst case; K of 6 and above lose stability.

## 4. Selected K and the evidence

Finalist rules (written before choosing): smallest cluster at least {MIN_CLUSTER_PCT:.0f}% of customers, subsample ARI mean at least {MIN_SUB_ARI_MEAN}, worst subsample ARI at least {MIN_SUB_ARI_WORST}.

{md_table(fin_t)}

Finalists: **K = {', '.join(str(f) for f in finalists)}**. Comparison of the finalists:

{md_table(cmp_df.round(3))}

**Selected K = {k}.** Evidence:
- **Metrics:** K=2 has the best Silhouette and Calinski-Harabasz, but it fails stability (worst subsample ARI {stab.loc[2, 'Subsample_ARI_min']:.2f}) and is too coarse to act on. K={k} is not the best on Silhouette or Calinski-Harabasz ({metrics.loc[k, 'Silhouette']:.3f} and {metrics.loc[k, 'Calinski_Harabasz']:.0f}); K=3 scores a little higher ({metrics.loc[3, 'Silhouette']:.3f} and {metrics.loc[3, 'Calinski_Harabasz']:.0f}). The gap is small and both are in the "weak structure" range, so metrics alone cannot separate K=3 from K=4.
- **Balance:** K={k} sizes are {metrics.loc[k, 'Sizes (n, %)']}. No tiny segment.
- **Stability:** K={k} has the highest subsample stability of all K (mean ARI {stab.loc[k, 'Subsample_ARI_mean']:.3f}, worst {stab.loc[k, 'Subsample_ARI_min']:.3f}) and near-perfect seed stability ({stab.loc[k, 'Seed_ARI_mean']:.3f}).
- **Interpretability:** moving from K=3 to K=4 gives a much purer one-time-buyer group ({cmp_df.loc[4, 'Most_single_order_cluster_%']:.0f}% single-order customers vs {cmp_df.loc[3, 'Most_single_order_cluster_%']:.0f}% at K=3) and separates repeat buyers who went quiet from repeat buyers who are on their normal cycle. Those two groups need different actions.

This is a judgement call between K=3 and K=4. K=3 is the simpler, slightly higher-scoring alternative.

## 5. Cluster sizes

{md_table(prof[['Customers', 'Customers_%']].round(1).assign(Label=[labels[i] for i in prof.index]).rename(index=lambda i: f'Cluster {i}'))}

## 6. Cluster profiles (numbers first)

Values are per-customer averages. Cluster numbers are sorted by mean Total_Orders (0 = most orders). `Silence_vs_Usual_Gap` = median of (days since last order / customer's usual gap between orders), multi-order customers only; above 1 means the customer has been quiet longer than usual. Average_Days_Between_Orders is averaged over customers who have a gap (multi-order customers).

{md_table(pt)}

{md_table(cat_t)}

**Which training features separate the clusters (eta squared; 0 = clusters ignore the feature, 1 = clusters are defined by it):**

{md_table(top_eta.rename('eta_squared').to_frame())}

## 7. Suggested business interpretation

These names were written after reading the profile above. They are proposals, not facts.

""")
    notes = {
        0: "Most orders, most recent, highest total spend and highest order value. The core of the business; protect and reward.",
        1: "Repeat buyers who used to order often but have been quiet for several times their usual gap. Win-back target.",
        2: "Repeat buyers with long natural gaps between orders who are still within their normal cycle. Time reminders to their rhythm.",
        3: "Almost all bought once. Includes genuinely lost one-time buyers and brand-new customers. Needs onboarding and second-order incentives.",
    }
    for i in prof.index:
        pr = prof.loc[i]
        s.append(f"- **Cluster {i}: {labels[i]}** ({pr['Customers']:,.0f} customers, {pr['Customers_%']:.1f}%, {pr['Revenue_Share_%']:.1f}% of revenue). "
                 f"{pr['Total_Orders']:.1f} orders, last order {pr['Days_Since_Last_Order']:.0f} days ago, usual gap {pr['Average_Days_Between_Orders']:.0f} days, "
                 f"average spend {pr['Total_Spending']:,.0f}. {notes.get(i, '') if SELECTED_K == 4 else ''}\n")
    s.append(f"""
**Important:** the clusters are defined almost entirely by *purchase history* (orders, recency, gap, category loyalty). Features with eta squared below 0.02 ({', '.join(weak)}) are essentially the same in every cluster (for example average rating is {p['Average_Customer_Rating'].min():.2f} to {p['Average_Customer_Rating'].max():.2f} in all clusters). Preferred device, payment and category also barely differ between clusters, so they should not be used to describe the segments.

## 8. Limitations

1. **Weak cluster structure.** Silhouette is {metrics.loc[k, 'Silhouette']:.2f} at K={k} (all K are at or below {metrics['Silhouette'].max():.2f}), under the 0.25 line for "no substantial structure". The data looks like a continuum of customers, not separate natural groups. The segments are a useful partition for action, not a discovery of hidden types.
2. **Only part of the feature set drives the result.** Rating, session duration, pages viewed and delivery time have eta squared of about 0 (identical in every cluster), and discount rate and order value separate the clusters only slightly. The clusters are built mainly from Preferred_Category_Share, Total_Orders, Average_Days_Between_Orders and Days_Since_Last_Order; note that category share is mechanically linked to order count (one order always means a share of 1.0). Check: clustering on only the 5 core features ({', '.join(CORE5)}) gives Silhouette {sens.loc[k, 'Core_5_features']:.2f} at K={k} (vs {sens.loc[k, 'All_10_features']:.2f}) and groups that match the 10-feature groups with ARI {ari_core:.2f}. We kept the 10 features as specified; removing the 5 weak ones is an option for the next iteration.
3. **Censoring.** The data ends on 2024-03-25. {len(single_recent)} of {n_single} single-order customers ({len(single_recent) / n_single:.0%}) ordered in the last 90 days, so part of "One-Time Buyers" are new customers who have not had time to buy again.
4. **No ground truth.** There is no "correct" segment to check against. Validation is internal (metrics, stability) plus business sense.
5. **Snapshot only.** Recency is measured at 2024-03-26. Segments must be recomputed as time passes. No train/test split was used because there is no target; the preprocessing and model were fit on all 5,000 customers.
6. **K-Means assumptions.** It forms roughly round, similar-sized groups in the scaled space and is sensitive to scaling choices. The data also appears synthetic (flat ratings, no link between behaviour and satisfaction), so real-world data may behave differently.
7. **PCA view.** The 2D picture shows only {ev.sum():.0%} of the variance (PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%}); clusters will look touching or overlapping, which matches the weak silhouette.

## Files produced

`models/customer_segmentation_kmeans.joblib`, `models/segment_metadata.json`, `data/processed/customer_segments.csv`, `data/processed/cluster_profiles.csv`, `reports/stage5_k_evaluation.csv`, charts `reports/figures/19` to `24`.
""")
    REPORT.write_text("".join(s))


if __name__ == "__main__":
    main()
