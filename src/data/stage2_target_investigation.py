"""
Stage 2: Target Investigation + EDA (NO final model training).
Run from project root:  python src/data/stage2_target_investigation.py
Saves charts to reports/figures/ and prints a findings summary.

Only one tiny "diagnostic" model is used (a sanity check to see if ANY signal
exists). It is NOT a project model and is never reported as performance.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ecommerce_customer_behavior_dataset_v2.csv"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

CATS = ["Gender", "City", "Product_Category", "Payment_Method", "Device_Type"]
NUMS = ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
        "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"]


def head(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()
    print(f"  [chart saved] reports/figures/{name}")


def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    # Order_ID looks like ORD_000001-2 -> the "-2" is the customer's 2nd order
    df["Order_Seq"] = df["Order_ID"].str.split("-").str[1].astype(int)
    df = df.sort_values(["Customer_ID", "Date", "Order_Seq"]).reset_index(drop=True)
    df["Orders_Total"] = df.groupby("Customer_ID")["Order_ID"].transform("count")
    df["Is_Returning_Customer"] = df["Is_Returning_Customer"].astype(int)
    return df


def q1_class_distribution(df):
    head("Q1. CLASS DISTRIBUTION of Is_Returning_Customer")
    vc = df["Is_Returning_Customer"].value_counts()
    print(vc.to_string(), "\n", (vc / len(df)).round(3).to_string())
    ax = sns.countplot(x="Is_Returning_Customer", data=df, palette="Blues")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}\n({p.get_height()/len(df):.1%})",
                    (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    ax.set_title("Q1: Class distribution (0 = new, 1 = returning)")
    save("01_class_distribution.png")


def q2_orders_per_customer(df):
    head("Q2. ORDERS PER CUSTOMER")
    per = df.groupby("Customer_ID").size()
    print(per.describe().round(2).to_string())
    print("Customers with exactly 1 order:", (per == 1).sum(), f"({(per == 1).mean():.1%})")
    per.value_counts().sort_index().plot(kind="bar", color="#4C78A8")
    plt.title("Q2: How many orders does each customer have?")
    plt.xlabel("Orders per customer"); plt.ylabel("Customers")
    save("02_orders_per_customer.png")


def q3_constant_per_customer(df):
    head("Q3. IS THE TARGET CONSTANT PER CUSTOMER?")
    g = df.groupby("Customer_ID")["Is_Returning_Customer"].nunique()
    print("Customers with ONE flag value :", (g == 1).sum())
    print("Customers whose flag CHANGES  :", (g > 1).sum())
    by_seq = df.groupby("Order_Seq")["Is_Returning_Customer"].agg(["mean", "count"]).round(3)
    print("\nShare of Is_Returning=1 by order number:\n", by_seq.to_string())
    ax = by_seq["mean"].plot(kind="bar", color="#F58518")
    ax.set_ylim(0, 1.1)
    plt.title("Q3: Is_Returning=1 rate by order number\n(order #2+ is ALWAYS 1 -> flag is derived from order history)")
    plt.xlabel("Order number of the customer"); plt.ylabel("Share flagged returning")
    save("03_flag_by_order_number.png")


def q4_features_vs_target(df):
    head("Q4. TARGET vs FEATURES (first orders only, because later orders are always 1)")
    f = df[df["Order_Seq"] == 1]
    print("First-order rows:", len(f), "| flag=1 share:", f["Is_Returning_Customer"].mean().round(3))
    print("\nCorrelation of numeric features with the flag (first orders):")
    print(f[NUMS + ["Is_Returning_Customer"]].corr()["Is_Returning_Customer"].drop("Is_Returning_Customer").round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), CATS):
        r = f.groupby(c)["Is_Returning_Customer"].mean().sort_values()
        r.plot(kind="barh", ax=ax, color="#54A24B")
        ax.axvline(f["Is_Returning_Customer"].mean(), color="red", ls="--")
        ax.set_title(f"{c}"); ax.set_xlim(0, 1)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Q4: Returning rate by category (first orders). Red line = overall average. Bars all look the same -> no signal", y=1.02)
    save("04_flag_vs_categories.png")

    # Diagnostic: can ANY model beat random on the first-order flag? (AUC 0.5 = random guessing)
    X = pd.get_dummies(f[NUMS + CATS])
    auc = cross_val_score(GradientBoostingClassifier(random_state=42), X, f["Is_Returning_Customer"],
                          cv=5, scoring="roc_auc").mean()
    print(f"\nDIAGNOSTIC ONLY - 5-fold CV ROC-AUC predicting first-order flag: {auc:.3f}  (0.5 = random)")
    return f


def q5_date_patterns(df):
    head("Q5. DATE / TIME PATTERNS")
    df = df.assign(Month=df["Date"].dt.to_period("M").astype(str))
    m = df.groupby("Month").agg(Orders=("Order_ID", "count"), Revenue=("Total_Amount", "sum"),
                                Flag_Rate=("Is_Returning_Customer", "mean")).round(3)
    print(m.to_string())
    print("\nDate range:", df["Date"].min().date(), "->", df["Date"].max().date())
    print("Day-of-week order counts:\n", df["Date"].dt.day_name().value_counts().to_string())

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    m["Orders"].plot(kind="bar", ax=ax[0], color="#4C78A8"); ax[0].set_title("Orders per month")
    m["Flag_Rate"].plot(marker="o", ax=ax[1], color="#E45756"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Is_Returning rate per month (rises because later orders are always 1)")
    plt.setp(ax[0].get_xticklabels(), rotation=60); plt.setp(ax[1].get_xticklabels(), rotation=60)
    save("05_monthly_patterns.png")

    d = df.copy()
    d["Gap_Days"] = d.groupby("Customer_ID")["Date"].diff().dt.days
    print("\nDays between consecutive orders:\n", d["Gap_Days"].describe().round(1).to_string())
    sns.histplot(d["Gap_Days"].dropna(), bins=40, color="#72B7B2")
    plt.title("Days between a customer's consecutive orders"); plt.xlabel("Days")
    save("06_gap_between_orders.png")


def q6_censoring_and_repeat(df):
    head("Q6. ALTERNATIVE TARGET: 'REPEAT WITHIN 90 DAYS' (needs fair observation window)")
    end = df["Date"].max()
    f = df[df["Order_Seq"] == 1].copy()
    second = df[df["Order_Seq"] == 2].set_index("Customer_ID")["Date"]
    f["Gap2"] = (f["Customer_ID"].map(second) - f["Date"]).dt.days
    f["FirstMonth"] = f["Date"].dt.to_period("M").astype(str)

    naive = (f["Orders_Total"] > 1).groupby(f["FirstMonth"]).mean().round(2)
    print("Naive 'ever ordered again' rate by month of FIRST order (biased by cut-off date):\n", naive.to_string())
    naive.plot(kind="bar", color="#B279A2")
    plt.title("Q6: Naive repeat rate falls for late joiners just because data ends 2024-03-25\n(= censoring, a trap!)")
    plt.ylabel("Share who ordered again"); save("07_censoring_trap.png")

    elig = f[(end - f["Date"]).dt.days >= 90].copy()
    elig["Repeat_90d"] = (elig["Gap2"] <= 90).astype(int)
    print(f"\nFair version: customers with >=90 days of follow-up = {len(elig):,}")
    print("Repeat within 90 days rate:", elig["Repeat_90d"].mean().round(3))
    X = pd.get_dummies(elig[NUMS + CATS])
    auc = cross_val_score(GradientBoostingClassifier(random_state=42), X, elig["Repeat_90d"],
                          cv=5, scoring="roc_auc").mean()
    print(f"DIAGNOSTIC ONLY - 5-fold CV ROC-AUC (first-order features -> repeat in 90d): {auc:.3f}  (0.5 = random)")


def q7_leakage_and_ids(df):
    head("Q7-Q9. LEAKAGE, IDs, DATE FEATURES")
    print("Order_ID unique:", df["Order_ID"].nunique(), "| Customer_ID unique:", df["Customer_ID"].nunique())
    print("Order_ID suffix = order number -> it directly encodes the answer; Customer_ID is a label with no meaning.")
    calc = df["Unit_Price"] * df["Quantity"] - df["Discount_Amount"]
    print("Total_Amount == Price*Qty - Discount:", np.allclose(calc, df["Total_Amount"], atol=0.02))
    num = df[NUMS].corr()
    plt.figure(figsize=(9, 7)); sns.heatmap(num, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric correlation (only price/quantity/total are related; rating & behaviour are ~0)")
    save("08_correlation_heatmap.png")


def q10_other_signals(df):
    head("Q10. WHICH COLUMNS HAVE REAL STRUCTURE? (for alternative problems)")
    print("Mean Total_Amount by category:\n", df.groupby("Product_Category")["Total_Amount"].mean().round(0).sort_values().to_string())
    print("\nMean rating by category (flat):\n", df.groupby("Product_Category")["Customer_Rating"].mean().round(3).to_string())
    order = df.groupby("Product_Category")["Total_Amount"].median().sort_values().index
    sns.boxplot(data=df, x="Product_Category", y="Total_Amount", order=order, showfliers=False, color="#9ECAE9")
    plt.xticks(rotation=30); plt.title("Order value by category: strong structure (good for regression)")
    save("09_value_by_category.png")


def main():
    df = load()
    q1_class_distribution(df)
    q2_orders_per_customer(df)
    q3_constant_per_customer(df)
    q4_features_vs_target(df)
    q5_date_patterns(df)
    q6_censoring_and_repeat(df)
    q7_leakage_and_ids(df)
    q10_other_signals(df)
    head("STAGE 2 DONE")


if __name__ == "__main__":
    main()
