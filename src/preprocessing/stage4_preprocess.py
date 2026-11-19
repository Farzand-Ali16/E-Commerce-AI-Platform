"""
Stage 4: Clustering feature selection + preprocessing (NO K-Means here).

Run from project root:  python src/preprocessing/stage4_preprocess.py
Outputs:
  models/cluster_preprocessor.joblib        (fitted preprocessing object)
  data/processed/clustering_matrix.csv      (5000 x final features, scaled)
  reports/figures/15_*.png ... 18_*.png
"""
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing.clustering_preprocessor import ClusteringPreprocessor  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid")

IN_CSV = ROOT / "data" / "processed" / "customer_features.csv"
OUT_MATRIX = ROOT / "data" / "processed" / "clustering_matrix.csv"
OUT_MODEL = ROOT / "models" / "cluster_preprocessor.joblib"
FIG = ROOT / "reports" / "figures"
OUT_MODEL.parent.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Starting list given for Stage 4 (12 features)
START_FEATURES = ["Total_Orders", "Total_Spending", "Average_Order_Value", "Average_Discount_Rate",
                  "Average_Customer_Rating", "Average_Session_Duration", "Average_Pages_Viewed",
                  "Average_Delivery_Time", "Days_Since_Last_Order", "Average_Days_Between_Orders",
                  "Unique_Categories_Bought", "Preferred_Category_Share"]
# Features removed after the redundancy check (reasons are printed in section 2)
DROPPED = {"Total_Spending": "log(Total_Spending) = log(Total_Orders) + log(Average_Order_Value) EXACTLY",
           "Unique_Categories_Bought": "correlation 0.92 with Total_Orders (just counts orders again)"}
FINAL_FEATURES = [f for f in START_FEATURES if f not in DROPPED]
GAP = "Average_Days_Between_Orders"

SKEW_LIMIT = 0.75      # only consider a transform if |skew| is above this
MIN_GAIN = 0.30        # ...and only keep it if it lowers |skew| by at least this much


def head(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()
    print(f"  [chart saved] reports/figures/{name}")


def vif_table(Z):
    """Variance Inflation Factor: how well can ONE feature be predicted from the others?
    1 = independent, >5 = overlapping, >10 = badly redundant."""
    out = {}
    for k in Z.columns:
        others = [c for c in Z.columns if c != k]
        X = np.c_[np.ones(len(Z)), Z[others].to_numpy()]
        y = Z[k].to_numpy()
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[k] = 1 / (1 - r2) if r2 < 1 - 1e-12 else np.inf
    return pd.Series(out)


def choose_transforms(df, features):
    """Data-driven choice: compare skewness for none / log1p / sqrt for each feature."""
    rows, log_cols, sqrt_cols = [], [], []
    for c in features:
        s = df[c].dropna()
        raw, lg, sq = s.skew(), np.log1p(s).skew(), np.sqrt(s).skew()
        choice = "none"
        if abs(raw) > SKEW_LIMIT:
            best, name = min([(abs(lg), "log1p"), (abs(sq), "sqrt")])
            if abs(raw) - best >= MIN_GAIN:
                choice = name
        (log_cols if choice == "log1p" else sqrt_cols if choice == "sqrt" else []).append(c)
        rows.append((c, round(raw, 2), round(lg, 2), round(sq, 2), choice))
    table = pd.DataFrame(rows, columns=["feature", "skew_raw", "skew_log1p", "skew_sqrt", "chosen"])
    return table, log_cols, sqrt_cols


# ---------------------------------------------------------------------------
def section1_missing(df):
    head("1. MISSING VALUES: Average_Days_Between_Orders")
    miss = df[GAP].isna()
    print(f"Missing: {miss.sum()} of {len(df)} ({miss.mean():.1%})")
    print("Total_Orders of the missing customers:", sorted(df.loc[miss, "Total_Orders"].unique()),
          "-> all single-order customers (a gap needs 2 orders)")
    print("Missing values in the other 11 start features:", int(df[START_FEATURES].drop(columns=GAP).isna().sum().sum()))
    med = df[GAP].median()
    print(f"\nMedian gap of the {(~miss).sum()} multi-order customers = {med:.1f} days (mean {df[GAP].mean():.1f})")
    ind_corr = np.corrcoef(miss.astype(int), np.log1p(df["Total_Orders"]))[0, 1]
    print(f"Correlation of a 'missing indicator' with log(Total_Orders): {ind_corr:.2f}")
    print("-> The indicator would be the SAME fact as 'Total_Orders == 1' (already a feature). "
          "Adding it would give 'one-time buyer' two votes in K-Means.")
    print("\nSTRATEGY: median imputation, NO extra indicator. Single-order customers get the typical gap (neutral),")
    print("and are separated from others by Total_Orders and Days_Since_Last_Order. Is_Single_Order_Customer stays in the")
    print("table for cluster PROFILING (it is not used to build clusters).")
    print("Why not 0? 0 days would say 'orders every day' = the most loyal customer, which is false.")
    return med


def section2_redundancy(df):
    head("2. REDUNDANCY CHECK ON THE 12 STARTING FEATURES (after transforms)")
    table, log_cols, sqrt_cols = choose_transforms(df, START_FEATURES)
    pre12 = ClusteringPreprocessor(START_FEATURES, log_cols, sqrt_cols, (GAP,)).fit(df)
    Z12 = pre12.transform(df)
    exact = np.abs(np.log(df["Total_Spending"]) - (np.log(df["Total_Orders"]) + np.log(df["Average_Order_Value"]))).max()
    print(f"max |log(Spend) - (log(Orders) + log(AOV))| = {exact:.1e}  -> exactly zero: Total_Spending is a combination of two other features")
    corr = Z12.corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "corr"]
    print("\nPairs with |corr| >= 0.6:\n", pairs[pairs["corr"].abs() >= 0.6].sort_values("corr", key=abs, ascending=False).round(2).to_string(index=False))
    v12 = vif_table(Z12)
    print("\nVIF with all 12 features (1 = fine, >10 = badly redundant):\n", v12.round(1).to_string())
    kept = vif_table(Z12[FINAL_FEATURES])
    print("\nVIF after dropping the 2 redundant features:\n", kept.round(1).to_string())
    print("\nDROPPED:")
    for k, v in DROPPED.items():
        print(f"  - {k}: {v}")
    return v12, kept


def section3_transforms(df):
    head("3. WHICH FEATURES GET TRANSFORMED? (skewness test, not blind)")
    table, log_cols, sqrt_cols = choose_transforms(df, FINAL_FEATURES)
    print(f"Rule: transform only if |skew| > {SKEW_LIMIT} AND the best of log1p/sqrt lowers |skew| by >= {MIN_GAIN}.\n")
    print(table.to_string(index=False))
    print("\nlog1p :", log_cols, "\nsqrt  :", sqrt_cols)
    print("none  :", [c for c in FINAL_FEATURES if c not in log_cols + sqrt_cols])
    print("\nNotes: Average_Customer_Rating is skewed to the LEFT (most ratings near 4-5); log/sqrt would make it worse, so it stays.")
    print("       Average_Discount_Rate: 30% of customers have exactly 0 discount, so a spike at 0 remains after sqrt.")
    return log_cols, sqrt_cols


def section4_pipeline(df, log_cols, sqrt_cols):
    head("4. FIT PREPROCESSOR (impute -> transform -> clip extremes -> StandardScaler)")
    pre = ClusteringPreprocessor(FINAL_FEATURES, log_cols, sqrt_cols, (GAP,), clip_sigma=4.0).fit(df)
    print("Imputed:", pre.n_imputed_, "| median used:", {k: round(v, 1) for k, v in pre.medians_.items()})
    # how extreme are values before clipping?
    Xt = ClusteringPreprocessor(FINAL_FEATURES, log_cols, sqrt_cols, (GAP,), clip_sigma=1e9).fit(df).transform_before_scaling(df)
    z = (Xt - Xt.mean()) / Xt.std(ddof=0)
    print("\nMax |z| after transforms, before clipping:\n", z.abs().max().round(2).to_string())
    print("\nValues with |z| > 3.5:", {k: int(v) for k, v in (z.abs() > 3.5).sum().items() if v})
    print("Values clipped at +/-4 standard deviations:", {k: v for k, v in pre.n_clipped_.items() if v} or "none")
    print("\nOutlier policy: customers are NOT deleted (high spenders are real VIPs). Transforms already shrink the tails;")
    print("clipping at 4 std only stops 1-2 freak values from dragging a cluster centre.")
    X = pre.transform(df)
    return pre, X, Xt


def section5_verify(df, X):
    head("5. VERIFY THE FINAL MATRIX")
    arr = X.to_numpy()
    print("Shape:", X.shape, "| NaN count:", int(np.isnan(arr).sum()), "| Inf count:", int(np.isinf(arr).sum()))
    assert X.shape == (len(df), len(FINAL_FEATURES)) and not np.isnan(arr).any() and not np.isinf(arr).any()
    assert np.allclose(X.mean(), 0, atol=1e-8) and np.allclose(X.std(ddof=0), 1, atol=1e-8)
    print("OK: no NaN / no infinite values, every column has mean 0 and std 1")
    print("\nFINAL FEATURE LIST (%d):" % len(FINAL_FEATURES))
    for i, c in enumerate(FINAL_FEATURES, 1):
        print(f"  {i:2d}. {c}")
    print("\nSummary statistics after preprocessing:\n", X.describe().T.round(3).to_string())
    print("\nSkewness after preprocessing:\n", X.skew().round(2).to_string())
    return X


def charts(df, pre, X, Xt, v12, vkept, med):
    head("6. CHARTS")
    # 15: missing-value strategy
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.histplot(df[GAP].dropna(), bins=40, ax=ax[0], color="#4C78A8")
    ax[0].axvline(med, color="red", ls="--", label=f"median = {med:.0f}"); ax[0].legend()
    ax[0].set_title("Observed gap (4,108 multi-order customers)")
    filled = df[GAP].fillna(med)
    sns.histplot(filled, bins=40, ax=ax[1], color="#E45756"); ax[1].set_title("After median fill: spike of 892 single-order customers")
    sns.histplot(X[GAP], bins=40, ax=ax[2], color="#54A24B"); ax[2].set_title("Final (sqrt + scaled)")
    save("15_missing_value_strategy.png")

    # 16: before / after
    cols = [c for c in FINAL_FEATURES if c in set(pre.log_cols) | set(pre.sqrt_cols)]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 6.5))
    for i, c in enumerate(cols):
        raw = df[c].fillna(med) if c == GAP else df[c]
        sns.histplot(raw, bins=35, ax=axes[0, i], color="#E45756"); axes[0, i].set_title(f"{c}\nraw (skew {raw.skew():.2f})", fontsize=9)
        sns.histplot(X[c], bins=35, ax=axes[1, i], color="#54A24B")
        how = "log1p" if c in pre.log_cols else "sqrt"
        axes[1, i].set_title(f"{how} + scaled (skew {X[c].skew():.2f})", fontsize=9)
        axes[0, i].set_xlabel(""); axes[1, i].set_xlabel("")
    fig.suptitle("Before (top) vs after (bottom) for every transformed feature", y=1.01)
    save("16_before_after_distributions.png")

    # 17: VIF + final correlation
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    vv = pd.concat([v12.rename("12 starting features"), vkept.rename("10 final features")], axis=1).sort_values("12 starting features")
    vv.clip(upper=60).plot(kind="barh", ax=ax[0], color=["#E45756", "#54A24B"])
    ax[0].axvline(10, color="black", ls="--"); ax[0].set_title("VIF (bars cut at 60; Total_Spending was 1,783). Dashed line = 10 = too redundant")
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1], annot_kws={"size": 7}, cbar=False)
    ax[1].set_title("Correlation of the final 10 features")
    save("17_vif_and_final_correlation.png")

    # 18: final boxplots
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=X, orient="h", color="#9ECAE9", fliersize=2)
    plt.title("Final scaled features: all centered on 0, comparable scale, no extreme spikes")
    save("18_final_boxplots.png")


def main():
    df = pd.read_csv(IN_CSV)
    assert len(df) == 5000
    med = section1_missing(df)
    v12, vkept = section2_redundancy(df)
    log_cols, sqrt_cols = section3_transforms(df)
    pre, X, Xt = section4_pipeline(df, log_cols, sqrt_cols)
    section5_verify(df, X)
    charts(df, pre, X, Xt, v12, vkept, med)

    head("7. SAVE")
    joblib.dump(pre, OUT_MODEL)
    X.assign(Customer_ID=df["Customer_ID"].values).set_index("Customer_ID").to_csv(OUT_MATRIX)
    # reload test: saved object must give the same result
    again = joblib.load(OUT_MODEL).transform(df)
    assert np.allclose(again.to_numpy(), X.to_numpy())
    print(f"Saved {OUT_MODEL.relative_to(ROOT)} (reload test OK) and {OUT_MATRIX.relative_to(ROOT)}")
    head("STAGE 4 DONE - K-Means NOT trained")


if __name__ == "__main__":
    main()
