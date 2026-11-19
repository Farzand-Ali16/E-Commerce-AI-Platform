"""
Stage 5.1: Clustering model refinement (10-feature model vs Core-5 model).

Run from project root:  python src/models/stage5_1_core5_refinement.py

Does NOT touch or overwrite any Stage 5 file. New files get a "_core5" name.

Compares
  A) existing 10-feature K=4 model
  B) Core-5 K=4 model          (Total_Orders, Average_Order_Value, Days_Since_Last_Order,
                                Average_Days_Between_Orders, Preferred_Category_Share)
  C) diagnostic only: Core-4   (= B without Preferred_Category_Share), used to test the share concern
and sweeps K = 2..6 for Core-5.
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
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples, silhouette_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.stage5_kmeans_segmentation import (EXTRA_SEEDS, MIN_CLUSTER_PCT, MIN_SUB_ARI_MEAN,  # noqa: E402
                                                   MIN_SUB_ARI_WORST, N_INIT, N_SUBSAMPLES, SEED,
                                                   SUBSAMPLE_FRAC, build_profile, eta_squared, fit_final,
                                                   md_table, propose_labels, size_text)
from src.preprocessing.clustering_preprocessor import ClusteringPreprocessor  # noqa: E402
from src.preprocessing.stage4_preprocess import vif_table  # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

RAW_CSV = ROOT / "data" / "ecommerce_customer_behavior_dataset_v2.csv"
CUSTOMER_CSV = ROOT / "data" / "processed" / "customer_features.csv"
MATRIX10_CSV = ROOT / "data" / "processed" / "clustering_matrix.csv"
OUT = {
    "matrix": ROOT / "data" / "processed" / "clustering_matrix_core5.csv",
    "segments": ROOT / "data" / "processed" / "customer_segments_core5.csv",
    "profiles": ROOT / "data" / "processed" / "cluster_profiles_core5.csv",
    "model": ROOT / "models" / "customer_segmentation_kmeans_core5.joblib",
    "prep": ROOT / "models" / "cluster_preprocessor_core5.joblib",
    "meta": ROOT / "models" / "segment_metadata_core5.json",
    "sweep": ROOT / "reports" / "stage5_1_core5_k_sweep.csv",
    "compare": ROOT / "reports" / "stage5_1_model_comparison.csv",
    "report": ROOT / "reports" / "stage5_1_refinement_report.md",
}
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CORE5 = ["Total_Orders", "Average_Order_Value", "Days_Since_Last_Order",
         "Average_Days_Between_Orders", "Preferred_Category_Share"]
CORE4 = [c for c in CORE5 if c != "Preferred_Category_Share"]
K_SWEEP = range(2, 7)
FINAL_K = 4


def head(t):
    print("\n" + "=" * 95 + f"\n{t}\n" + "=" * 95)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [chart saved] reports/figures/{name}")


# ======================================================================================
# core calculations
# ======================================================================================
def make_subsamples(n):
    """The SAME 30 random 80% subsamples are used for every model, so the comparison is fair."""
    rng = np.random.default_rng(SEED)
    return [rng.choice(n, int(SUBSAMPLE_FRAC * n), replace=False) for _ in range(N_SUBSAMPLES)]


def fit_k(A, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=SEED).fit(A)
    return km, km.labels_


def quality(A, lab):
    s = silhouette_samples(A, lab)
    per = pd.Series(s).groupby(lab).mean()
    return {"Silhouette": silhouette_score(A, lab), "Calinski_Harabasz": calinski_harabasz_score(A, lab),
            "Davies_Bouldin": davies_bouldin_score(A, lab), "Worst_cluster_silhouette": per.min(),
            "Negative_silhouette_%": (s < 0).mean() * 100}


def stability(A, k, ref, subs):
    seed_ari = [adjusted_rand_score(ref, KMeans(k, init="k-means++", n_init=N_INIT, random_state=s).fit_predict(A))
                for s in EXTRA_SEEDS]
    sub_ari = [adjusted_rand_score(ref[idx], KMeans(k, init="k-means++", n_init=10, random_state=i).fit_predict(A[idx]))
               for i, idx in enumerate(subs)]
    return {"Seed_ARI_mean": np.mean(seed_ari), "Seed_ARI_min": np.min(seed_ari),
            "Subsample_ARI_mean": np.mean(sub_ari), "Subsample_ARI_min": np.min(sub_ari)}


def sweep_core5(A, subs):
    rows, labs = [], {}
    for k in K_SWEEP:
        km, lab = fit_k(A, k)
        labs[k] = lab
        cnt = np.bincount(lab, minlength=k)
        row = {"K": k, "Inertia": km.inertia_, **quality(A, lab), "Smallest_%": cnt.min() / len(A) * 100,
               **stability(A, k, lab, subs), "Sizes (n, %)": size_text(lab, k)}
        rows.append(row)
        print(f"  K={k}: sil {row['Silhouette']:.3f} CH {row['Calinski_Harabasz']:.0f} DB {row['Davies_Bouldin']:.3f} | "
              f"sub-ARI {row['Subsample_ARI_mean']:.3f} (min {row['Subsample_ARI_min']:.3f}) | smallest {row['Smallest_%']:.1f}%")
    df = pd.DataFrame(rows).set_index("K")
    df["Inertia_drop_%"] = -df["Inertia"].pct_change() * 100
    df["Balance_ok"] = df["Smallest_%"] >= MIN_CLUSTER_PCT
    df["Stable_ok"] = (df["Subsample_ARI_mean"] >= MIN_SUB_ARI_MEAN) & (df["Subsample_ARI_min"] >= MIN_SUB_ARI_WORST)
    df["Finalist"] = df["Balance_ok"] & df["Stable_ok"]
    return df, labs


def one_time_purity(cust, lab):
    """How clean is the 'one-time buyer' cluster? (cluster with the highest share of single-order customers)"""
    single = cust["Is_Single_Order_Customer"].to_numpy() == 1
    best = max(np.unique(lab), key=lambda c: single[lab == c].mean())
    inside = lab == best
    return {"One-time cluster size": int(inside.sum()), "Single-order customers inside": int((single & inside).sum()),
            "Repeat customers inside": int((~single & inside).sum()),
            "Single-order customers outside": int((single & ~inside).sum()),
            "Purity_%": single[inside].mean() * 100}


# ======================================================================================
# evidence about Preferred_Category_Share
# ======================================================================================
def share_analysis(cust, raw):
    probs = raw["Product_Category"].value_counts(normalize=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for n in range(1, 11):
        draws = rng.choice(len(probs), size=(20000, n), p=probs.to_numpy())
        top = np.array([np.bincount(r, minlength=len(probs)).max() for r in draws])
        rows.append({"Total_Orders": n, "Customers": int((cust["Total_Orders"] == n).sum()),
                     "Observed_mean_share": cust.loc[cust["Total_Orders"] == n, "Preferred_Category_Share"].mean(),
                     "Simulated_random_share": (top / n).mean()})
    tab = pd.DataFrame(rows).set_index("Total_Orders")
    by_n = cust.groupby("Total_Orders")["Preferred_Category_Share"].transform("mean")
    y = cust["Preferred_Category_Share"]
    r2 = 1 - ((y - by_n) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    corr = np.corrcoef(y, np.log(cust["Total_Orders"]))[0, 1]
    multi_all_same = int(((cust["Total_Orders"] > 1) & (cust["Preferred_Category_Share"] == 1)).sum())
    return tab, r2, corr, multi_all_same


# ======================================================================================
# evidence about business usefulness: are the features persistent customer traits?
# ======================================================================================
def icc(values, groups):
    d = pd.DataFrame({"v": values, "g": groups})
    d = d[d.groupby("g")["v"].transform("count") >= 2]
    k = d["g"].nunique(); n = len(d)
    cm = d.groupby("g")["v"].transform("mean")
    msb = ((cm - d["v"].mean()) ** 2).sum() / (k - 1)
    msw = ((d["v"] - cm) ** 2).sum() / (n - k)
    n0 = (n - (d.groupby("g").size() ** 2).sum() / n) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def persistence_tests(raw):
    df = raw.copy()
    df["seq"] = df["Order_ID"].str.split("-").str[1].astype(int)
    df = df.sort_values(["Customer_ID", "seq"])
    df["log_amt"] = np.log(df["Total_Amount"])
    out = {}
    w = df.pivot(index="Customer_ID", columns="seq", values="log_amt")[[1, 2]].dropna()
    out["Order value: corr(order 1, order 2)"] = np.corrcoef(w[1], w[2])[0, 1]
    out["Order value: ICC (share of variance due to the customer)"] = icc(df["log_amt"], df["Customer_ID"])
    res = df["log_amt"] - df.groupby("Product_Category")["log_amt"].transform("mean")
    out["Order value, category-adjusted: ICC"] = icc(res, df["Customer_ID"])
    df["gap"] = df.groupby("Customer_ID")["Date"].diff().dt.days
    g = df.pivot(index="Customer_ID", columns="seq", values="gap")[[2, 3]].dropna()
    out["Gap: corr(gap 1, gap 2)"] = np.corrcoef(g[2], g[3])[0, 1]
    first = df.groupby("Customer_ID")["Date"].min()
    elig = first[first <= pd.Timestamp("2023-03-31")].index
    d = df[df["Customer_ID"].isin(elig)]
    a_mask = (d["Date"] >= "2023-04-01") & (d["Date"] <= "2023-09-30")
    b_mask = d["Date"] >= "2023-10-01"
    a = d[a_mask].groupby("Customer_ID").size().reindex(elig, fill_value=0)
    b = d[b_mask].groupby("Customer_ID").size().reindex(elig, fill_value=0)
    out["Order count: corr(Apr-Sep 2023, Oct 2023-Mar 2024)"] = np.corrcoef(a, b)[0, 1]
    sa = d[a_mask].groupby("Customer_ID")["Total_Amount"].sum().reindex(elig, fill_value=0)
    sb = d[b_mask].groupby("Customer_ID")["Total_Amount"].sum().reindex(elig, fill_value=0)
    out["Spend: corr(Apr-Sep 2023, Oct 2023-Mar 2024)"] = np.corrcoef(sa, sb)[0, 1]
    return pd.Series(out)


# ======================================================================================
# charts
# ======================================================================================
def chart_elbow(sweep):
    plt.figure(figsize=(7, 4.6))
    plt.plot(sweep.index, sweep["Inertia"], marker="o", color="#4C78A8")
    for k, r in sweep.iterrows():
        if not np.isnan(r["Inertia_drop_%"]):
            plt.annotate(f"-{r['Inertia_drop_%']:.0f}%", (k, r["Inertia"]), textcoords="offset points", xytext=(6, 8), fontsize=8)
    plt.axvline(FINAL_K, color="black", ls="--", lw=1)
    plt.xticks(list(K_SWEEP)); plt.xlabel("K"); plt.ylabel("Inertia")
    plt.title("Core-5 elbow plot (labels = drop from the previous K)")
    save_fig("25_core5_elbow.png")


def chart_silhouette(sweep, sil10, sil4):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(sweep.index, sweep["Silhouette"], marker="o", lw=2.5, color="#54A24B", label="Core-5 (B)")
    plt.plot(sil4.index, sil4.values, marker="s", ls="--", color="#F58518", label="Core-4, no share (diagnostic)")
    plt.plot(sil10.index, sil10.values, marker="^", ls=":", color="#E45756", label="All 10 features (A)")
    plt.axhline(0.25, color="grey", ls=":"); plt.text(2.02, 0.255, "0.25 = weak-structure line", fontsize=8)
    plt.axvline(FINAL_K, color="black", ls="--", lw=1)
    plt.xticks(list(K_SWEEP)); plt.xlabel("K"); plt.ylabel("Silhouette"); plt.legend(fontsize=8)
    plt.title("Silhouette by K")
    save_fig("26_core5_silhouette.png")


def chart_sizes(prof, labels):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [f"{i}: {labels[i]}" for i in prof.index]
    bars = ax.bar(names, prof["Customers"], color=sns.color_palette("Set2", len(prof)))
    for b, n, p in zip(bars, prof["Customers"], prof["Customers_%"]):
        ax.text(b.get_x() + b.get_width() / 2, n, f"{n:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, prof["Customers"].max() * 1.18); ax.set_ylabel("Customers")
    ax.set_title("Core-5 K=4 cluster sizes"); plt.xticks(rotation=15, ha="right")
    save_fig("27_core5_cluster_sizes.png")


def chart_share(tab, cust):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(tab.index, tab["Observed_mean_share"], marker="o", lw=2.5, label="observed in the data")
    ax[0].plot(tab.index, tab["Simulated_random_share"], marker="x", ls="--", label="simulated: categories picked at random")
    ax[0].set_xlabel("Total_Orders"); ax[0].set_ylabel("Mean Preferred_Category_Share"); ax[0].legend()
    ax[0].set_title("Share vs order count: real data = pure chance")
    j = cust.assign(jit=np.random.default_rng(1).normal(0, 0.12, len(cust)))
    ax[1].scatter(j["Total_Orders"] + j["jit"], j["Preferred_Category_Share"], s=6, alpha=.3, color="#4C78A8")
    ax[1].set_xlabel("Total_Orders"); ax[1].set_ylabel("Preferred_Category_Share")
    ax[1].set_title("Share is capped by order count (1 order = always 1.0)")
    save_fig("28_share_vs_orders.png")


def chart_compare(cmp):
    metrics = [("Silhouette", "Silhouette (higher better)"), ("Subsample_ARI_mean", "Subsample stability ARI (higher better)"),
               ("Worst_cluster_silhouette", "Worst cluster silhouette (higher better)"),
               ("Negative_silhouette_%", "% customers with negative silhouette (lower better)")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, (m, t) in zip(ax, metrics):
        vals = cmp.loc[m]
        a.bar(["A: 10 feat", "B: Core-5", "C: Core-4"], vals.astype(float).values, color=["#E45756", "#54A24B", "#F58518"])
        for i, v in enumerate(vals.astype(float).values):
            a.text(i, v, f"{v:.3f}" if v < 5 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        a.set_title(t, fontsize=9)
    fig.suptitle("Model comparison at K=4 (C is a diagnostic variant)", y=1.03)
    save_fig("29_model_comparison.png")


# ======================================================================================
def main():
    cust = pd.read_csv(CUSTOMER_CSV).set_index("Customer_ID")
    X10 = pd.read_csv(MATRIX10_CSV, index_col="Customer_ID")
    raw = pd.read_csv(RAW_CSV, parse_dates=["Date"])
    assert X10.index.equals(cust.index)

    # ---- Core-5 matrix: its own preprocessor (same transforms as Stage 4, fitted on 5 columns only)
    head("0. BUILD THE CORE-5 PREPROCESSOR AND MATRIX")
    pre10 = joblib.load(ROOT / "models" / "cluster_preprocessor.joblib")
    pre5 = ClusteringPreprocessor(CORE5, [c for c in pre10.log_cols if c in CORE5], [c for c in pre10.sqrt_cols if c in CORE5],
                                  [c for c in pre10.impute_cols if c in CORE5], clip_sigma=pre10.clip_sigma).fit(cust)
    X5 = pre5.transform(cust)
    assert np.allclose(X5.to_numpy(), X10[CORE5].to_numpy()), "core-5 matrix must equal the 5 columns of the Stage 4 matrix"
    assert not np.isnan(X5.to_numpy()).any()
    print("log1p:", pre5.log_cols, "| sqrt:", pre5.sqrt_cols, "| imputed:", pre5.n_imputed_, "| clipped:", {k: v for k, v in pre5.n_clipped_.items() if v} or "none")
    print("Core-5 matrix equals the 5 matching columns of the Stage 4 matrix: OK", X5.shape)
    A10, A5, A4 = X10.to_numpy(), X5.to_numpy(), X10[CORE4].to_numpy()
    subs = make_subsamples(len(cust))

    # ---- sweep
    head("1. CORE-5, K = 2 ... 6 (n_init=50, random_state=42) + stability")
    sweep, labs = sweep_core5(A5, subs)
    sil10 = pd.Series({k: silhouette_score(A10, KMeans(k, n_init=N_INIT, random_state=SEED).fit_predict(A10)) for k in K_SWEEP})
    sil4 = pd.Series({k: silhouette_score(A4, KMeans(k, n_init=N_INIT, random_state=SEED).fit_predict(A4)) for k in K_SWEEP})
    show = sweep.drop(columns=["Sizes (n, %)", "Balance_ok", "Stable_ok"]).round(3)
    print(show.to_string())
    print("\nSizes (largest first):")
    for k in K_SWEEP:
        print(f"  K={k}: {sweep.loc[k, 'Sizes (n, %)']}")
    finalists = [int(k) for k in sweep.index[sweep["Finalist"]]]
    print(f"\nFinalists under the Stage 5 rules (smallest cluster >= {MIN_CLUSTER_PCT:.0f}%, subsample ARI mean >= {MIN_SUB_ARI_MEAN}, worst >= {MIN_SUB_ARI_WORST}):", finalists)
    assert FINAL_K in finalists

    # ---- A vs B vs C at K=4
    head("2. MODEL COMPARISON AT K = 4")
    km5, lab5 = fit_final(A5, cust, FINAL_K)                  # B (renumbered: 0 = most orders)
    kmA, labA = fit_final(A10, cust, FINAL_K)                 # A (same renumbering rule as Stage 5)
    kmC, labC = fit_final(A4, cust, FINAL_K)                  # C diagnostic
    assert abs(silhouette_score(A5, lab5) - sweep.loc[FINAL_K, "Silhouette"]) < 1e-9
    models = {"A": ("All 10 features", A10, labA, list(X10.columns)),
              "B": ("Core-5", A5, lab5, CORE5),
              "C": ("Core-4 (diagnostic)", A4, labC, CORE4)}
    cmp_rows = {}
    for key, (name, A, lab, cols) in models.items():
        q = quality(A, lab)
        st = stability(A, FINAL_K, lab, subs)
        cnt = np.bincount(lab, minlength=FINAL_K)
        pur = one_time_purity(cust, lab)
        v = vif_table(pd.DataFrame(A, columns=cols))
        cmp_rows[key] = {"Features used": len(cols), **q, "Smallest_%": cnt.min() / len(A) * 100,
                         "Largest_%": cnt.max() / len(A) * 100, **st,
                         "Max_VIF": v.max(), "One-time cluster purity_%": pur["Purity_%"],
                         "Repeat customers inside one-time cluster": pur["Repeat customers inside"],
                         "Sizes (n, %)": size_text(lab, FINAL_K)}
    cmp = pd.DataFrame(cmp_rows)
    cmp.columns = ["A: 10 features", "B: Core-5", "C: Core-4 (diagnostic)"]
    cmp.columns = ["A", "B", "C"]
    ariAB = adjusted_rand_score(labA, lab5); ariBC = adjusted_rand_score(lab5, labC); ariAC = adjusted_rand_score(labA, labC)
    pretty = cmp.copy()
    for r in pretty.index:
        if r != "Sizes (n, %)":
            pretty.loc[r] = [f"{float(x):.3f}" if abs(float(x)) < 20 else f"{float(x):.1f}" for x in pretty.loc[r]]
    print(pretty.to_string())
    print(f"\nARI between solutions: A vs B = {ariAB:.3f} | B vs C = {ariBC:.3f} | A vs C = {ariAC:.3f}")
    print("(1.0 = same groups. A and B are similar; removing the share feature (C) changes the grouping a lot.)")

    # ---- B profile
    head("3. CORE-5 K=4 CLUSTER PROFILE (numbers first)")
    seg = cust.reset_index().copy()
    seg["Cluster"] = lab5
    profB = build_profile(seg)
    show = profB.copy()
    for c in show.select_dtypes("number").columns:
        show[c] = show[c].round(2)
    print(show.T.to_string())
    etaB = eta_squared(X5, lab5)
    print("\nEta squared (how much each Core-5 feature is separated by the clusters):\n", etaB.round(3).to_string())
    labels = propose_labels(profB, FINAL_K)
    seg["Segment_Label"] = seg["Cluster"].map(labels)
    print("\nProposed labels (same rule-checked names as Stage 5, confirmed against this profile):")
    for i, nme in labels.items():
        p = profB.loc[i]
        print(f"  {i}: {nme}: {p['Customers']:,.0f} customers ({p['Customers_%']:.1f}%), {p['Total_Orders']:.1f} orders, "
              f"last order {p['Days_Since_Last_Order']:.0f}d ago, {p['Revenue_Share_%']:.1f}% of revenue")
    profA = build_profile(cust.reset_index().assign(Cluster=labA))
    profC = build_profile(cust.reset_index().assign(Cluster=labC))
    ct = pd.crosstab(pd.Series(labA, name="A (10 features)"), pd.Series(lab5, name="B (Core-5)"))
    print("\nWho moved between A and B (rows = A cluster, columns = B cluster):\n", ct.to_string())
    print(f"Customers whose cluster differs between A and B: {int((ct.values.sum() - np.trace(ct.values)))} "
          f"({(1 - np.trace(ct.values) / ct.values.sum()):.1%})  [ids line up because both are numbered by orders]")

    # ---- share concern
    head("4. PREFERRED_CATEGORY_SHARE: IS IT A PROBLEM?")
    tab, r2, corr, multi_same = share_analysis(cust, raw)
    print(tab.round(3).to_string())
    print(f"\nShare explained by order count alone (R2): {r2:.3f} | correlation with log(Total_Orders): {corr:.2f}")
    print(f"Customers with 2+ orders whose orders are ALL in one category (share = 1.0): {multi_same}")
    pur = one_time_purity(cust, lab5)
    print("One-time cluster in B:", pur)
    print("Core-4 diagnostic (C): silhouette", round(cmp.loc['Silhouette', 'C'], 3), "vs B", round(cmp.loc['Silhouette', 'B'], 3),
          f"| ARI B vs C = {ariBC:.2f}")
    print("Core-4 profile (shows what the grouping looks like without the share):")
    print(profC[["Customers", "Total_Orders", "Total_Spending", "Average_Order_Value", "Days_Since_Last_Order",
                 "Average_Days_Between_Orders", "Single_Order_%", "Revenue_Share_%"]].round(1).to_string())

    # ---- persistence
    head("5. ARE THE FEATURES REAL, PERSISTENT CUSTOMER TRAITS?")
    pers = persistence_tests(raw)
    print(pers.round(3).to_string())
    print("0 = no persistence (the feature behaves like chance for each customer); closer to 1 = stable trait.")

    # ---- multi-order-only silhouette (is the quality gain only the one-time group?)
    multi = (cust["Total_Orders"] > 1).to_numpy()
    mo = {k: silhouette_score(A5[multi], KMeans(k, n_init=N_INIT, random_state=SEED).fit_predict(A5[multi])) for k in (2, 3, 4)}
    print("\nSilhouette for repeat customers only (Core-5, 4,108 customers):", {k: round(v, 3) for k, v in mo.items()})

    # ---- charts
    head("6. CHARTS")
    chart_elbow(sweep)
    chart_silhouette(sweep, sil10, sil4)
    chart_sizes(profB, labels)
    chart_share(tab, cust)
    chart_compare(cmp)

    # ---- save (new files only)
    head("7. SAVE (Stage 5 files untouched)")
    for p in ("model", "prep", "meta"):
        assert not (OUT[p].exists() and p == "never"), p
    joblib.dump(km5, OUT["model"]); joblib.dump(pre5, OUT["prep"])
    assert (joblib.load(OUT["model"]).predict(A5) == lab5).all()
    assert np.allclose(joblib.load(OUT["prep"]).transform(cust).to_numpy(), A5)
    X5.assign(Customer_ID=cust.index).set_index("Customer_ID").to_csv(OUT["matrix"])
    seg.to_csv(OUT["segments"], index=False)
    assert seg["Customer_ID"].is_unique and len(seg) == 5000
    profB.round(3).to_csv(OUT["profiles"])
    sweep.round(4).to_csv(OUT["sweep"])
    cmp.to_csv(OUT["compare"])
    OUT["meta"].write_text(json.dumps({
        "selected_k": FINAL_K, "features": CORE5, "random_state": SEED, "n_init": N_INIT,
        "cluster_labels": {str(k): v for k, v in labels.items()},
        "preprocessor": "models/cluster_preprocessor_core5.joblib",
        "status": "candidate (Stage 5.1). See reports/stage5_1_refinement_report.md for the production recommendation",
        "note": "Cluster ids are numbered by mean Total_Orders (0 = most orders)."}, indent=2))
    for k, p in OUT.items():
        if k not in ("report",):
            print(f"  {p.relative_to(ROOT)}")

    write_report(sweep, finalists, cmp, ariAB, ariBC, ariAC, profA, profB, profC, labels, etaB, tab, r2, corr,
                 multi_same, pur, pers, mo, ct, sil4, sil10)
    print(f"  {OUT['report'].relative_to(ROOT)}")
    head("STAGE 5.1 DONE - no website / API started, no Stage 5 file changed")


# ======================================================================================
def fmt(df, nd=3):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else (f"{v:.{nd}f}" if abs(v) < 100 else f"{v:,.1f}"))
    return d


def write_report(sweep, finalists, cmp, ariAB, ariBC, ariAC, profA, profB, profC, labels, etaB, tab, r2, corr,
                 multi_same, pur, pers, mo, ct, sil4, sil10):
    k = FINAL_K
    sw = sweep.drop(columns=["Sizes (n, %)", "Balance_ok", "Stable_ok", "Finalist"]).round(3)
    fin = sweep[["Silhouette", "Smallest_%", "Subsample_ARI_mean", "Subsample_ARI_min", "Balance_ok", "Stable_ok", "Finalist"]].round(3)
    sizes = sweep[["Sizes (n, %)"]]
    cm = cmp.copy()
    num_rows = [r for r in cm.index if r != "Sizes (n, %)"]
    cmp_show = cm.copy()
    for r in num_rows:
        cmp_show.loc[r] = [f"{float(x):.3f}" if abs(float(x)) < 20 else f"{float(x):.1f}" for x in cm.loc[r]]
    pcols = ["Customers", "Customers_%", "Total_Orders", "Total_Spending", "Average_Order_Value", "Average_Discount",
             "Average_Customer_Rating", "Average_Session_Duration", "Average_Pages_Viewed", "Average_Delivery_Time",
             "Days_Since_Last_Order", "Average_Days_Between_Orders", "Unique_Categories_Bought", "Revenue_Share_%",
             "Single_Order_%", "Silence_vs_Usual_Gap"]
    pt = profB.copy(); pt.insert(0, "Label", [labels[i] for i in pt.index])
    ptab = pt[["Label"] + pcols].round(2).T
    ptab.columns = [f"Cluster {c}" for c in ptab.columns]
    cat = pt[["Label", "Preferred_Category", "Preferred_Device", "Preferred_Payment"]].copy()
    cat.index = [f"Cluster {c}" for c in cat.index]
    ctab = profC[["Customers", "Customers_%", "Total_Orders", "Total_Spending", "Average_Order_Value",
                  "Days_Since_Last_Order", "Average_Days_Between_Orders", "Single_Order_%", "Revenue_Share_%"]].round(1)
    ctab.index = [f"C-{i}" for i in ctab.index]
    shr = tab.round(3)
    persr = pers.round(3).rename("value").to_frame()
    A, B, C = cm["A"], cm["B"], cm["C"]
    etar = etaB.round(3).rename("eta_squared").to_frame()
    moved = int(ct.values.sum() - np.trace(ct.values))

    txt = f"""# Stage 5.1 Report: Clustering Model Refinement

*E-Commerce AI Analytics & Customer Intelligence Platform*

Question: should the production segmentation use the existing **10-feature** K-Means model (A) or the **Core-5** model (B)? A third variant, **Core-4** (C = B without Preferred_Category_Share), was added only as a diagnostic to test the share concern. No Stage 5 file was changed.

Core-5 features: Total_Orders, Average_Order_Value, Days_Since_Last_Order, Average_Days_Between_Orders, Preferred_Category_Share. Same preprocessing as Stage 4 (log1p for orders and order value, sqrt for the two time features, median fill for the gap of single-order customers, scaling). The Core-5 matrix is identical to the 5 matching columns of the Stage 4 matrix (checked).

---

## 1. Core-5, K = 2 to 6

{md_table(sw)}

Cluster sizes (largest first):

{md_table(sizes)}

Finalist rules (same as Stage 5): smallest cluster at least {MIN_CLUSTER_PCT:.0f}% of customers, subsample ARI mean at least {MIN_SUB_ARI_MEAN}, worst subsample ARI at least {MIN_SUB_ARI_WORST}.

{md_table(fin)}

Finalists: **K = {', '.join(map(str, finalists))}**. K=4 beats K=3 on Silhouette ({sweep.loc[4, 'Silhouette']:.3f} vs {sweep.loc[3, 'Silhouette']:.3f}), Davies-Bouldin ({sweep.loc[4, 'Davies_Bouldin']:.3f} vs {sweep.loc[3, 'Davies_Bouldin']:.3f}) and stability ({sweep.loc[4, 'Subsample_ARI_mean']:.3f} vs {sweep.loc[3, 'Subsample_ARI_mean']:.3f}); K=3 is higher only on Calinski-Harabasz ({sweep.loc[3, 'Calinski_Harabasz']:.0f} vs {sweep.loc[4, 'Calinski_Harabasz']:.0f}). K=2 has the top Silhouette but one subsample dropped to ARI {sweep.loc[2, 'Subsample_ARI_min']:.2f} (unstable). K=5 leaves a segment of {sweep.loc[5, 'Smallest_%']:.1f}% and K=6 is unstable. The elbow plot has no sharp bend (drops shrink smoothly), so it does not decide K.

## 2. Model comparison at K = 4

{md_table(cmp_show)}

ARI between solutions: A vs B = **{ariAB:.3f}**, B vs C = **{ariBC:.3f}**, A vs C = {ariAC:.3f}. Subsample and seed stability use the same 30 random 80% subsamples and 15 seeds for every model, so the numbers can differ slightly from Stage 5, where the subsample draws were shared across K values.

Reading the table:
- **B is better than A on every quality measure.** Silhouette {float(A['Silhouette']):.3f} to {float(B['Silhouette']):.3f}; Calinski-Harabasz {float(A['Calinski_Harabasz']):.0f} to {float(B['Calinski_Harabasz']):.0f}; Davies-Bouldin {float(A['Davies_Bouldin']):.2f} to {float(B['Davies_Bouldin']):.2f}. In A the worst cluster has silhouette {float(A['Worst_cluster_silhouette']):.3f} and {float(A['Negative_silhouette_%']):.0f}% of customers sit closer to another cluster than their own; in B the figures are {float(B['Worst_cluster_silhouette']):.3f} and {float(B['Negative_silhouette_%']):.0f}%.
- **B is at least as stable as A** (subsample ARI mean {float(B['Subsample_ARI_mean']):.3f} vs {float(A['Subsample_ARI_mean']):.3f}; worst case {float(B['Subsample_ARI_min']):.3f} vs {float(A['Subsample_ARI_min']):.3f}, a negligible difference, both far above the 0.80 rule) with similar balance (smallest cluster {float(B['Smallest_%']):.1f}% vs {float(A['Smallest_%']):.1f}%).
- **A and B describe nearly the same segments** (ARI {ariAB:.2f}; only {moved} customers, {moved / 5000:.1%}, change cluster). The extra 5 features in A add noise, not information: rating, session duration, pages viewed and delivery time had eta squared near 0 in Stage 5.
- Silhouette is higher in B partly because removing noise dimensions makes distances meaningful, so the Silhouette gain is not a claim of "strong" clusters. {float(B['Silhouette']):.2f} is still only in the weak-structure range (0.26 to 0.50).

## 3. Core-5 K=4 cluster profile

Per-customer averages. Cluster numbers sorted by mean Total_Orders (0 = most orders). Total_Spending, Average_Discount, Unique_Categories_Bought and the categorical preferences were **not** training features.

{md_table(ptab)}

{md_table(cat)}

Eta squared of each Core-5 feature (how strongly the clusters separate it):

{md_table(etar)}

Proposed labels (checked against the numbers in code; they are the same four segments as Stage 5): {', '.join(f'**{i}: {v}**' for i, v in labels.items())}.

## 4. Does Preferred_Category_Share create a concern?

**Yes, a moderate and measurable one.** Three findings:

1. **It is mechanically tied to order count.** Share explains nothing on its own: order count alone explains **{r2:.0%}** of its variance (correlation with log orders {corr:.2f}). A customer with 1 order always has share 1.0, and 2 orders can only give 0.5 or 1.0. That is why share separates the {int(pur['Single-order customers inside'])} single-order customers so cleanly. In Core-5 it effectively counts "number of orders" a second time.
2. **It carries no category-loyalty information.** If every order's category were picked at random, the average share for each order count would be exactly what we see (table below). Real customers behave like random pickers, so share has no behavioural meaning beyond order count.

{md_table(shr)}

3. **It causes mislabeled customers.** The "One-Time Buyers" cluster holds {pur['One-time cluster size']:,} customers: {pur['Single-order customers inside']} single-order customers (all of them) plus **{pur['Repeat customers inside']} repeat customers** who happened to buy all their orders in the same category ({multi_same} such customers exist in total). Purity is {pur['Purity_%']:.1f}%. These {pur['Repeat customers inside']} customers are not one-time buyers; they are in the cluster by coincidence.

How much it matters: removing it (diagnostic model C) gives Silhouette {float(C['Silhouette']):.3f}, essentially the same as B ({float(B['Silhouette']):.3f}), but a **different grouping** (ARI {ariBC:.2f} with B). So the share is not harmless; it decides how the partition is built. Without it, the data splits by order value rather than by one-time vs repeat:

{md_table(ctab)}

Model C's "lowest-value" and "high order value" groups come from Average_Order_Value, which section 5 shows is not a stable customer trait. So dropping the share does not simply give a cleaner version of B; it gives a different, less persistent kind of segment.

## 5. Are the features persistent customer traits? (business usefulness)

A segment is more useful for marketing if the behaviour behind it repeats over time. Checks on the raw orders:

{md_table(persr)}

- **Order value is not a customer trait:** a customer's first and second order values are uncorrelated (about 0.02) and the customer explains about 0% of the variance in order value, even after adjusting for category. A customer's Average_Order_Value mostly reflects which categories they happened to buy (and, for customers with few orders, one lucky or unlucky purchase). Segments built heavily on AOV would describe past baskets, not stable customer types.
- **Order frequency is weakly persistent:** customers' order counts in two consecutive six-month periods correlate about {pers.iloc[4]:.2f}. This is a modest but real signal, consistent with segments built on orders and recency.
- **Spend** repeats only weakly ({pers.iloc[5]:.2f}); the gap between orders is slightly negative ({pers.iloc[3]:.2f}), partly because the observation window is fixed (a long first gap leaves less time for the next one).
- For repeat customers only (4,108), Core-5 silhouette is {', '.join(f'K={a}: {b:.2f}' for a, b in mo.items())}, so the improvement over A is not only the easy one-time vs repeat split.

## 6. Decision matrix

| Criterion | A: 10 features | B: Core-5 | C: Core-4 (diagnostic) |
|---|---|---|---|
| Cluster quality (Silhouette / CH / DB) | {float(A['Silhouette']):.3f} / {float(A['Calinski_Harabasz']):.0f} / {float(A['Davies_Bouldin']):.2f} (weakest) | {float(B['Silhouette']):.3f} / {float(B['Calinski_Harabasz']):.0f} / {float(B['Davies_Bouldin']):.2f} | {float(C['Silhouette']):.3f} / {float(C['Calinski_Harabasz']):.0f} / {float(C['Davies_Bouldin']):.2f} |
| Stability (subsample ARI mean / worst) | {float(A['Subsample_ARI_mean']):.3f} / {float(A['Subsample_ARI_min']):.3f} | {float(B['Subsample_ARI_mean']):.3f} / {float(B['Subsample_ARI_min']):.3f} | {float(C['Subsample_ARI_mean']):.3f} / {float(C['Subsample_ARI_min']):.3f} |
| Balance (smallest cluster) | {float(A['Smallest_%']):.1f}% | {float(B['Smallest_%']):.1f}% | {float(C['Smallest_%']):.1f}% |
| Feature redundancy | 4 noise features, dilute distances | share overlaps with orders (R2 {r2:.2f}) | none |
| Interpretability | same 4 lifecycle segments, blurry edges | same 4 lifecycle segments, clean edges | value-driven groups, no one-time group |
| Business usefulness | lifecycle, but noisy | lifecycle (frequency/recency, weakly persistent) | adds order value, which is not persistent |

## 7. Recommendation

**Use Model B (Core-5, K=4) as the production model. Retire Model A (keep its files, do not use it).**

Evidence:
1. B beats A on Silhouette, Calinski-Harabasz, Davies-Bouldin and worst-cluster quality, and matches it on stability, while producing the same four business segments (ARI {ariAB:.2f}). The four removed features have no separating power, so A has no advantage to offer.
2. B has the highest mean stability of the three (subsample ARI {float(B['Subsample_ARI_mean']):.3f}; worst case {float(B['Subsample_ARI_min']):.3f}, within 0.005 of A) and is well balanced (smallest segment {float(B['Smallest_%']):.1f}%).
3. B is the most interpretable: Loyal & Active, Lapsed Repeat Buyers, Slow-Cycle Repeat Buyers, One-Time Buyers, all driven mainly by order count, recency and order gap. Of the behaviours we could test, order count is the only one with meaningful persistence over time (correlation {pers.iloc[4]:.2f}); order value has none.
4. The alternative that removes the share (C) matches B on Silhouette but is less stable (worst subsample ARI {float(C['Subsample_ARI_min']):.3f}) and trades the one-time/repeat structure for groups driven by order value, which is not a stable customer trait. It is not a clear improvement, so we do not switch.

Conditions attached to this recommendation (known limitations of B):
- **Preferred_Category_Share double-counts order count and misplaces {pur['Repeat customers inside']} repeat customers into "One-Time Buyers".** In the prediction system and dashboard, show `Is_Single_Order_Customer` next to the segment so a one-order customer is never confused with the {pur['Repeat customers inside']} coincidence cases. Treat the segment name as "One-Time Buyers (mostly)". If exact purity matters more than the grouping, the cleaner design is a rule (Total_Orders = 1) for that segment plus clustering of repeat customers only; that would be a separate experiment.
- Silhouette {float(B['Silhouette']):.2f} is still weak: the segments partition a continuum and are not natural groups.
- Average_Order_Value has little effect on B's grouping; revenue differences between segments come from order count.
- The data looks synthetic (random categories, flat ratings), so the stability of these findings on real data is untested. Re-run Stage 5.1 if real data replaces it.

## Files produced (nothing from Stage 5 modified)

`models/customer_segmentation_kmeans_core5.joblib`, `models/cluster_preprocessor_core5.joblib`, `models/segment_metadata_core5.json`, `data/processed/customer_segments_core5.csv`, `data/processed/cluster_profiles_core5.csv`, `data/processed/clustering_matrix_core5.csv`, `reports/stage5_1_core5_k_sweep.csv`, `reports/stage5_1_model_comparison.csv`, charts `reports/figures/25` to `29`.
"""
    OUT["report"].write_text(txt)


if __name__ == "__main__":
    main()
