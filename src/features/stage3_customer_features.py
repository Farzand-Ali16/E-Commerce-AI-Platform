"""
Stage 3: Customer-level feature engineering (ORDER-level -> ONE ROW PER CUSTOMER).
NO clustering / NO model training here.

Run from project root:  python src/features/stage3_customer_features.py
Outputs:
  data/processed/customer_features.csv   (the customer table, raw values, NaN kept)
  reports/figures/10_*.png ... 14_*.png   (distribution / outlier / correlation charts)
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "ecommerce_customer_behavior_dataset_v2.csv"
OUT_CSV = ROOT / "data" / "processed" / "customer_features.csv"
FIG = ROOT / "reports" / "figures"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Feature engineering (reusable function - the backend will reuse this later)
# ----------------------------------------------------------------------------
def most_common_with_tiebreak(df, col):
    """Most frequent value per customer. If tied, the MOST RECENT one wins
    (so the result is always the same every time we run the code)."""
    counts = df.groupby(["Customer_ID", col]).agg(n=("Date", "size"), last=("Date", "max")).reset_index()
    counts = counts.sort_values(["Customer_ID", "n", "last"], ascending=[True, False, False])
    top = counts.drop_duplicates("Customer_ID").set_index("Customer_ID")
    return top[col], top["n"]


def build_customer_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Turn the order table into one row per customer.
    reference_date = the 'today' used for recency. We use the day AFTER the last
    order in the data (NOT the real today), so results are reproducible."""
    df = df.sort_values(["Customer_ID", "Date"]).copy()
    if reference_date is None:
        reference_date = df["Date"].max() + pd.Timedelta(days=1)
    df["Gross_Amount"] = df["Unit_Price"] * df["Quantity"]          # price before discount

    g = df.groupby("Customer_ID")
    cust = g.agg(
        Total_Orders=("Order_ID", "count"),
        Total_Spending=("Total_Amount", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Average_Quantity=("Quantity", "mean"),
        Average_Discount=("Discount_Amount", "mean"),
        Total_Discount=("Discount_Amount", "sum"),
        Total_Gross=("Gross_Amount", "sum"),
        Average_Customer_Rating=("Customer_Rating", "mean"),
        Average_Session_Duration=("Session_Duration_Minutes", "mean"),
        Average_Pages_Viewed=("Pages_Viewed", "mean"),
        Average_Delivery_Time=("Delivery_Time_Days", "mean"),
        First_Order_Date=("Date", "min"),
        Last_Order_Date=("Date", "max"),
        Age=("Age", "first"),            # constant per customer (checked in Stage 1)
        Gender=("Gender", "first"),
        City=("City", "first"),
    )
    cust["Average_Order_Value"] = cust["Total_Spending"] / cust["Total_Orders"]
    cust["Average_Discount_Rate"] = cust["Total_Discount"] / cust["Total_Gross"]   # % of price saved
    cust["Days_Since_Last_Order"] = (reference_date - cust["Last_Order_Date"]).dt.days   # Recency
    cust["Customer_Tenure_Days"] = (cust["Last_Order_Date"] - cust["First_Order_Date"]).dt.days
    # mean gap between orders = (last - first) / (orders - 1). Undefined (NaN) for 1-order customers.
    cust["Average_Days_Between_Orders"] = np.where(
        cust["Total_Orders"] > 1, cust["Customer_Tenure_Days"] / (cust["Total_Orders"] - 1), np.nan)
    cust["Is_Single_Order_Customer"] = (cust["Total_Orders"] == 1).astype(int)

    # Preferred (most frequent) category / device / payment + how dominant it is
    for col, name in [("Product_Category", "Preferred_Product_Category"),
                      ("Device_Type", "Preferred_Device_Type"),
                      ("Payment_Method", "Preferred_Payment_Method")]:
        top, n = most_common_with_tiebreak(df, col)
        cust[name] = top
        if col == "Product_Category":
            cust["Preferred_Category_Share"] = n / cust["Total_Orders"]
    cust["Unique_Categories_Bought"] = g["Product_Category"].nunique()

    cust = cust.drop(columns=["Total_Discount", "Total_Gross"])   # helper columns only
    order = ["Total_Orders", "Total_Spending", "Average_Order_Value", "Total_Quantity", "Average_Quantity",
             "Average_Discount", "Average_Discount_Rate", "Average_Customer_Rating", "Average_Session_Duration",
             "Average_Pages_Viewed", "Average_Delivery_Time", "Days_Since_Last_Order", "Average_Days_Between_Orders",
             "Customer_Tenure_Days", "Is_Single_Order_Customer", "Unique_Categories_Bought",
             "Preferred_Product_Category", "Preferred_Category_Share", "Preferred_Device_Type",
             "Preferred_Payment_Method", "Age", "Gender", "City", "First_Order_Date", "Last_Order_Date"]
    return cust[order].reset_index()


# ----------------------------------------------------------------------------
# 2. Checks and charts
# ----------------------------------------------------------------------------
def head(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()
    print(f"  [chart saved] reports/figures/{name}")


NUM_FEATURES = ["Total_Orders", "Total_Spending", "Average_Order_Value", "Total_Quantity", "Average_Quantity",
                "Average_Discount", "Average_Discount_Rate", "Average_Customer_Rating", "Average_Session_Duration",
                "Average_Pages_Viewed", "Average_Delivery_Time", "Days_Since_Last_Order",
                "Average_Days_Between_Orders", "Customer_Tenure_Days"]


def verify_against_raw(raw, cust):
    head("0. SANITY CHECKS (does the customer table match the raw data?)")
    assert len(cust) == raw["Customer_ID"].nunique() == 5000
    assert cust["Customer_ID"].is_unique
    assert cust["Total_Orders"].sum() == len(raw)
    assert np.isclose(cust["Total_Spending"].sum(), raw["Total_Amount"].sum())
    assert np.isclose(cust["Total_Quantity"].sum(), raw["Quantity"].sum())
    print("OK: 5,000 unique customers | orders add up to", f"{len(raw):,}",
          "| revenue adds up to", f"{raw['Total_Amount'].sum():,.2f}")


def show_table(cust):
    head("1-2. CUSTOMER-LEVEL DATAFRAME + SHAPE")
    print("Shape:", cust.shape, "(rows = customers, columns = features)\n")
    print(cust.head(8).round(2).to_string(index=False))
    print("\nColumn types:\n", cust.dtypes.to_string())


def check_missing(cust):
    head("3. MISSING VALUES")
    miss = cust.isna().sum()
    print(miss[miss > 0].to_string() if (miss > 0).any() else "none")
    n = cust["Average_Days_Between_Orders"].isna().sum()
    print(f"\nAverage_Days_Between_Orders is NaN for {n} customers = exactly the single-order customers "
          f"({cust['Is_Single_Order_Customer'].sum()}).")
    print("This is a STRUCTURAL missing value (a gap needs 2 orders). We do NOT fill it with 0 (0 would mean 'orders every day').")


def check_distributions(cust):
    head("4. DISTRIBUTIONS (summary + skewness)")
    desc = cust[NUM_FEATURES].describe(percentiles=[.01, .25, .5, .75, .99]).T.round(2)
    desc["skew"] = cust[NUM_FEATURES].skew().round(2)
    print(desc.to_string())

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for ax, c in zip(axes.ravel(), NUM_FEATURES):
        sns.histplot(cust[c].dropna(), bins=40, ax=ax, color="#4C78A8")
        ax.set_title(f"{c} (skew {cust[c].skew():.1f})", fontsize=10); ax.set_xlabel("")
    for ax in axes.ravel()[len(NUM_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Customer feature distributions (raw)", y=1.01, fontsize=14)
    save("10_customer_feature_distributions.png")

    skewed = ["Total_Spending", "Average_Order_Value", "Average_Discount", "Total_Quantity"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for i, c in enumerate(skewed):
        sns.histplot(cust[c], bins=40, ax=axes[0, i], color="#E45756"); axes[0, i].set_title(f"{c} - raw")
        sns.histplot(np.log1p(cust[c]), bins=40, ax=axes[1, i], color="#54A24B"); axes[1, i].set_title(f"log(1 + {c})")
    fig.suptitle("Skewed features: raw (top) vs log-transformed (bottom)", y=1.01)
    save("11_log_transform_preview.png")

    for c in ["Preferred_Product_Category", "Preferred_Device_Type", "Preferred_Payment_Method", "Gender", "City"]:
        print(f"\n{c}:\n", cust[c].value_counts().to_string())
    print("\nPreferred_Category_Share (1.0 = bought only that category):\n", cust["Preferred_Category_Share"].describe().round(2).to_string())


def check_outliers(cust):
    head("5. EXTREME OUTLIERS (IQR x 3 rule and top-1%)")
    rows = []
    for c in NUM_FEATURES:
        s = cust[c].dropna()
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        extreme = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()
        rows.append((c, extreme, f"{extreme / len(s):.1%}", round(s.median(), 1), round(s.quantile(.99), 1), round(s.max(), 1)))
    print(pd.DataFrame(rows, columns=["feature", "extreme_count", "share", "median", "p99", "max"]).to_string(index=False))

    print("\nTop 8 customers by Total_Spending:")
    print(cust.nlargest(8, "Total_Spending")[["Customer_ID", "Total_Orders", "Total_Spending", "Average_Order_Value",
                                               "Preferred_Product_Category"]].round(1).to_string(index=False))
    share = cust.nlargest(int(len(cust) * .01), "Total_Spending")["Total_Spending"].sum() / cust["Total_Spending"].sum()
    print(f"\nTop 1% of customers (50 people) bring {share:.1%} of all revenue.")
    share10 = cust.nlargest(int(len(cust) * .10), "Total_Spending")["Total_Spending"].sum() / cust["Total_Spending"].sum()
    print(f"Top 10% of customers bring {share10:.1%} of all revenue.")

    cols = ["Total_Spending", "Average_Order_Value", "Total_Orders", "Average_Discount",
            "Days_Since_Last_Order", "Average_Days_Between_Orders", "Average_Delivery_Time", "Average_Session_Duration"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(y=cust[c], ax=ax, color="#9ECAE9"); ax.set_title(c)
    fig.suptitle("Boxplots: dots above the whiskers are outliers", y=1.01)
    save("12_outlier_boxplots.png")

    top_n = cust.nlargest(int(len(cust) * .01), "Total_Spending")["Preferred_Product_Category"].value_counts()
    print("\nPreferred category of the top-1% spenders:\n", top_n.to_string())


def check_relationships(cust):
    head("6. CORRELATIONS (redundancy check before clustering)")
    corr = cust[NUM_FEATURES].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
    plt.title("Customer feature correlation")
    save("13_customer_correlation.png")
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "corr"]
    strong = pairs[pairs["corr"].abs() >= 0.7].sort_values("corr", key=abs, ascending=False)
    print("Pairs with |correlation| >= 0.7 (they carry nearly the SAME information):\n",
          strong.round(2).to_string(index=False))

    sns.scatterplot(data=cust, x="Total_Orders", y="Total_Spending", hue="Preferred_Product_Category",
                    palette="tab10", alpha=0.6, s=18)
    plt.yscale("log"); plt.title("Spending vs number of orders (log scale)")
    plt.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    save("14_orders_vs_spending.png")

    print("\nMean Average_Order_Value by Preferred_Product_Category:")
    print(cust.groupby("Preferred_Product_Category")["Average_Order_Value"].mean().round(0).sort_values().to_string())


def main():
    raw = pd.read_csv(RAW, parse_dates=["Date"])
    cust = build_customer_features(raw)
    verify_against_raw(raw, cust)
    show_table(cust)
    check_missing(cust)
    check_distributions(cust)
    check_outliers(cust)
    check_relationships(cust)
    cust.to_csv(OUT_CSV, index=False)
    head(f"STAGE 3 DONE - saved {OUT_CSV.relative_to(ROOT)} {cust.shape}")


if __name__ == "__main__":
    main()
