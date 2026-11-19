"""
Stage 1: Dataset Inspection (NO ML here).
Run from project root:  python src/data/stage1_inspect.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ecommerce_customer_behavior_dataset_v2.csv"


def section(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    return pd.read_csv(path, parse_dates=["Date"])


def basic_info(df):
    section("1. SHAPE, COLUMNS, DTYPES")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print("\nColumns:", list(df.columns))
    print("\nData types:\n", df.dtypes)


def peek(df):
    section("2. FIRST 5 / LAST 5 ROWS")
    print(df.head(), "\n")
    print(df.tail())


def quality(df):
    section("3. MISSING VALUES & DUPLICATES")
    print("Missing values per column:\n", df.isna().sum())
    print("\nFully duplicated rows:", df.duplicated().sum())
    print("Duplicate Order_IDs   :", df["Order_ID"].duplicated().sum())


def categoricals(df):
    section("4. CATEGORICAL COLUMNS")
    for c in ["Gender", "City", "Product_Category", "Payment_Method", "Device_Type", "Is_Returning_Customer"]:
        print(f"\n{c} ({df[c].nunique()} unique)")
        print(df[c].value_counts(dropna=False).to_string())


def numerics(df):
    section("5. NUMERICAL STATISTICS")
    num = df.select_dtypes("number")
    print(num.describe().T)
    print("\nSkewness:\n", num.skew().round(2))


def outliers(df):
    section("6. OUTLIERS (IQR rule)")
    for c in ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
              "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days"]:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
        print(f"{c:26s} outliers: {n:5d} ({n/len(df):.1%})")


def business_checks(df):
    section("7. BUSINESS LOGIC & LEAKAGE CHECKS")
    calc = df["Unit_Price"] * df["Quantity"] - df["Discount_Amount"]
    print("Total = Price*Qty - Discount for all rows? ->", np.allclose(calc, df["Total_Amount"], atol=0.02))
    print("Date range:", df["Date"].min().date(), "->", df["Date"].max().date())

    print("\nTarget distributions:")
    print("Customer_Rating:\n", df["Customer_Rating"].value_counts(normalize=True).sort_index().round(3).to_string())
    print("Is_Returning_Customer:\n", df["Is_Returning_Customer"].value_counts(normalize=True).round(3).to_string())

    per = df.groupby("Customer_ID").agg(orders=("Order_ID", "count"),
                                        ret_nunique=("Is_Returning_Customer", "nunique"),
                                        age_nunique=("Age", "nunique"),
                                        city_nunique=("City", "nunique"),
                                        gender_nunique=("Gender", "nunique"))
    print("\nOrders per customer:\n", per["orders"].describe().round(2).to_string())
    print("Customers with 1 order:", (per["orders"] == 1).sum())
    print("Customers whose Is_Returning flag changes across orders:", (per["ret_nunique"] > 1).sum())
    orders_map = df["Customer_ID"].map(per["orders"])
    print("Mean orders per customer by Is_Returning flag:\n",
          orders_map.groupby(df["Is_Returning_Customer"]).mean().round(2).to_string())
    print("Customers with >1 city / age / gender:",
          (per.city_nunique > 1).sum(), (per.age_nunique > 1).sum(), (per.gender_nunique > 1).sum())

    num = df.select_dtypes("number").assign(Returning=df["Is_Returning_Customer"].astype(int))
    print("\nCorrelation with Customer_Rating:")
    print(num.corr()["Customer_Rating"].drop("Customer_Rating").round(3).sort_values().to_string())
    print("\nCorrelation with Is_Returning_Customer:")
    print(num.corr()["Returning"].drop("Returning").round(3).sort_values().to_string())

    print("\nMean rating by Delivery_Time_Days:")
    print(df.groupby("Delivery_Time_Days")["Customer_Rating"].mean().round(2).to_string())
    print("\nMean rating by Product_Category:")
    print(df.groupby("Product_Category")["Customer_Rating"].mean().round(3).to_string())
    print("\nTotal_Amount by Product_Category:")
    print(df.groupby("Product_Category")["Total_Amount"].agg(["mean", "median", "count"]).round(1).to_string())


def main():
    df = load_data()
    basic_info(df)
    peek(df)
    quality(df)
    categoricals(df)
    numerics(df)
    outliers(df)
    business_checks(df)
    section("STAGE 1 DONE")


if __name__ == "__main__":
    main()
