"""
Tests for the Stage 7 dashboard backend (/api/dashboard).

Every expected number is calculated independently with pandas from the raw CSV files,
so the tests check the API against the real dataset (nothing is hard-coded except the known labels).

Run:  pytest tests -v
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app  # noqa: E402
from backend.routes.dashboard import get_service  # noqa: E402
from backend.services.dashboard_service import DashboardFilters, DashboardService  # noqa: E402
from src.features.stage3_customer_features import build_customer_features  # noqa: E402

DATA = ROOT / "data"
MODEL_FILES = [ROOT / "models" / "customer_segmentation_kmeans_core5.joblib",
               ROOT / "models" / "cluster_preprocessor_core5.joblib",
               ROOT / "models" / "segment_metadata_core5.json"]
HASHES_BEFORE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL_FILES}

LABELS = {0: "Loyal & Active", 1: "Lapsed Repeat Buyers", 2: "Slow-Cycle Repeat Buyers",
          3: "One-Time / Mostly One-Time Customers"}

ALL_ENDPOINTS = ["summary", "sales-trend", "category-performance", "city-performance", "payment-analysis",
                 "device-analysis", "customer-analysis", "segments", "segment/0"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:           # start-up loads the datasets once
        yield c


@pytest.fixture(scope="module")
def raw():
    df = pd.read_csv(DATA / "ecommerce_customer_behavior_dataset_v2.csv", parse_dates=["Date"])
    seg = pd.read_csv(DATA / "processed" / "customer_segments_core5.csv", usecols=["Customer_ID", "Cluster"])
    return df.merge(seg, on="Customer_ID", how="left")


def get(client, path, **params):
    return client.get(f"/api/dashboard/{path}", params={k: v for k, v in params.items() if v is not None})


# ----------------------------------------------------------------------------- 1. summary
def test_summary_matches_dataset(client, raw):
    r = get(client, "summary")
    assert r.status_code == 200
    b = r.json()
    k = b["kpis"]
    assert b["has_data"] is True
    assert k["total_revenue"] == pytest.approx(raw["Total_Amount"].sum(), abs=0.01)
    assert k["total_orders"] == len(raw)
    assert k["total_customers"] == raw["Customer_ID"].nunique() == 5000
    assert k["average_order_value"] == pytest.approx(raw["Total_Amount"].mean(), abs=0.01)
    assert k["average_customer_rating"] == pytest.approx(raw["Customer_Rating"].mean(), abs=0.01)
    assert k["average_quantity_per_order"] == pytest.approx(raw["Quantity"].mean(), abs=0.01)
    assert b["date_range"] == {"start": str(raw["Date"].min().date()), "end": str(raw["Date"].max().date())}
    assert b["filters"]["category"] is None


# ----------------------------------------------------------------------------- 2. sales trend
def test_sales_trend_monthly(client, raw):
    b = get(client, "sales-trend").json()
    pts = b["points"]
    assert b["granularity"] == "month"
    assert [p["period"] for p in pts] == sorted(p["period"] for p in pts)
    assert pts[0]["period"] == "2023-01" and pts[-1]["period"] == "2024-03" and len(pts) == 15
    assert sum(p["orders"] for p in pts) == len(raw)
    assert sum(p["revenue"] for p in pts) == pytest.approx(raw["Total_Amount"].sum(), abs=0.5)
    m = raw.groupby(raw["Date"].dt.to_period("M"))["Total_Amount"].sum()
    for p in pts:
        assert p["revenue"] == pytest.approx(m[pd.Period(p["period"])], abs=0.01)
    assert pts[-1]["is_partial"] is True          # data ends on 25 March
    assert pts[0]["is_partial"] is False and pts[5]["is_partial"] is False


@pytest.mark.parametrize("granularity", ["day", "week", "month"])
def test_sales_trend_granularities_add_up(client, raw, granularity):
    b = get(client, "sales-trend", granularity=granularity).json()
    assert b["granularity"] == granularity
    assert sum(p["orders"] for p in b["points"]) == len(raw)
    assert sum(p["revenue"] for p in b["points"]) == pytest.approx(raw["Total_Amount"].sum(), abs=1.0)


def test_sales_trend_rejects_bad_granularity(client):
    r = get(client, "sales-trend", granularity="year")
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert r.json()["details"][0]["field"] == "granularity"


# ----------------------------------------------------------------------------- 3. category and other breakdowns
def test_category_performance(client, raw):
    b = get(client, "category-performance").json()
    rows = b["categories"]
    assert len(rows) == raw["Product_Category"].nunique() == 8
    assert [r["revenue"] for r in rows] == sorted((r["revenue"] for r in rows), reverse=True)
    g = raw.groupby("Product_Category").agg(rev=("Total_Amount", "sum"), n=("Order_ID", "count"))
    for r in rows:
        assert r["revenue"] == pytest.approx(g.loc[r["category"], "rev"], abs=0.01)
        assert r["orders"] == g.loc[r["category"], "n"]
        assert r["average_order_value"] == pytest.approx(g.loc[r["category"], "rev"] / g.loc[r["category"], "n"], abs=0.01)
    assert sum(r["revenue_share_pct"] for r in rows) == pytest.approx(100, abs=0.1)
    assert sum(r["orders_share_pct"] for r in rows) == pytest.approx(100, abs=0.1)


@pytest.mark.parametrize("path,list_key,name_key,col", [
    ("city-performance", "cities", "city", "City"),
    ("payment-analysis", "payment_methods", "payment_method", "Payment_Method"),
    ("device-analysis", "devices", "device_type", "Device_Type")])
def test_other_breakdowns(client, raw, path, list_key, name_key, col):
    rows = get(client, path).json()[list_key]
    g = raw.groupby(col)["Total_Amount"].sum()
    assert {r[name_key] for r in rows} == set(g.index)
    assert sum(r["revenue"] for r in rows) == pytest.approx(raw["Total_Amount"].sum(), abs=0.5)
    assert sum(r["orders"] for r in rows) == len(raw)
    for r in rows:
        assert r["revenue"] == pytest.approx(g[r[name_key]], abs=0.01)


# ----------------------------------------------------------------------------- 4. customers
def test_customer_analysis(client, raw):
    b = get(client, "customer-analysis").json()
    per = raw.groupby("Customer_ID").agg(n=("Order_ID", "count"), s=("Total_Amount", "sum"))
    assert b["total_customers"] == 5000
    assert b["single_order_customers"] == int((per["n"] == 1).sum())
    assert b["single_order_customers"] + b["repeat_customers"] == 5000
    assert sum(x["customers"] for x in b["orders_per_customer"]) == 5000
    got = {x["orders"]: x["customers"] for x in b["orders_per_customer"]}
    assert got == per["n"].value_counts().to_dict()
    assert b["spending"]["average"] == pytest.approx(per["s"].mean(), abs=0.01)
    assert b["spending"]["median"] == pytest.approx(per["s"].median(), abs=0.01)
    assert sum(x["customers"] for x in b["spending"]["distribution"]) == 5000
    assert sum(x["orders"] for x in b["rating"]["distribution"]) == len(raw)
    assert [x["rating"] for x in b["rating"]["distribution"]] == [1, 2, 3, 4, 5]
    assert b["rating"]["average"] == pytest.approx(raw["Customer_Rating"].mean(), abs=0.01)
    assert sum(x["customers"] for x in b["recency"]["distribution"]) == 5000
    ref = raw["Date"].max() + pd.Timedelta(days=1)
    rec = (ref - raw.groupby("Customer_ID")["Date"].max()).dt.days
    assert b["recency"]["average_days"] == pytest.approx(rec.mean(), abs=0.01)
    assert b["as_of_date"] == str(ref.date())
    nvr = {g["group"]: g for g in b["new_vs_returning"]["groups"]}
    assert nvr["New customer orders"]["orders"] == int((~raw["Is_Returning_Customer"]).sum())
    assert nvr["Returning customer orders"]["orders"] == int(raw["Is_Returning_Customer"].sum())
    assert sum(g["orders"] for g in nvr.values()) == len(raw)
    assert sum(m["new_orders"] + m["returning_orders"] for m in b["new_vs_returning"]["monthly"]) == len(raw)
    assert b["new_vs_returning"]["note"]


# ----------------------------------------------------------------------------- 5. segments
def test_segments_use_saved_production_result(client, raw):
    b = get(client, "segments").json()
    rows = b["segments"]
    assert [r["cluster"] for r in rows] == [0, 1, 2, 3]
    assert {r["cluster"]: r["segment"] for r in rows} == LABELS          # approved labels, unchanged
    assert b["total_customers"] == 5000 and sum(r["customers"] for r in rows) == 5000
    assert sum(r["customers_pct"] for r in rows) == pytest.approx(100, abs=0.1)
    assert sum(r["revenue_share_pct"] for r in rows) == pytest.approx(100, abs=0.1)
    # same sizes as the Stage 5.1 segmentation file
    seg = pd.read_csv(DATA / "processed" / "customer_segments_core5.csv")
    assert {r["cluster"]: r["customers"] for r in rows} == seg["Cluster"].value_counts().to_dict()
    prof = pd.read_csv(DATA / "processed" / "cluster_profiles_core5.csv", index_col=0)
    by_cluster = raw.groupby("Cluster").agg(rev=("Total_Amount", "sum"), n=("Order_ID", "count"))
    for r in rows:
        c = r["cluster"]
        assert r["average_orders"] == pytest.approx(prof.loc[c, "Total_Orders"], abs=0.01)
        assert r["average_recency_days"] == pytest.approx(prof.loc[c, "Days_Since_Last_Order"], abs=0.01)
        assert r["average_days_between_orders"] == pytest.approx(prof.loc[c, "Average_Days_Between_Orders"], abs=0.01)
        assert r["revenue_share_pct"] == pytest.approx(prof.loc[c, "Revenue_Share_%"], abs=0.01)
        assert r["average_order_value"] == pytest.approx(by_cluster.loc[c, "rev"] / by_cluster.loc[c, "n"], abs=0.01)
        for key, col in (("preferred_category", "Preferred_Category"), ("preferred_device", "Preferred_Device"),
                         ("preferred_payment", "Preferred_Payment")):
            assert r[key]["name"] == prof.loc[c, col].rsplit(" (", 1)[0]


@pytest.mark.parametrize("cluster_id", [0, 1, 2, 3])
def test_segment_detail(client, raw, cluster_id):
    r = get(client, f"segment/{cluster_id}")
    assert r.status_code == 200
    b = r.json()
    sub = raw[raw["Cluster"] == cluster_id]
    assert b["segment"]["cluster"] == cluster_id and b["segment"]["segment"] == LABELS[cluster_id]
    assert b["segment"]["customers"] == sub["Customer_ID"].nunique()
    assert b["segment"]["revenue"] == pytest.approx(sub["Total_Amount"].sum(), abs=0.01)
    assert sum(x["revenue"] for x in b["category_mix"]) == pytest.approx(sub["Total_Amount"].sum(), abs=0.5)
    assert sum(x["orders"] for x in b["device_mix"]) == len(sub)
    assert sum(x["customers"] for x in b["orders_per_customer"]) == b["segment"]["customers"]
    assert sum(x["orders"] for x in b["monthly_trend"]) == len(sub)
    assert b["suggested_actions"]


# ----------------------------------------------------------------------------- 6. invalid cluster
@pytest.mark.parametrize("bad", [4, 7, -1, 99])
def test_unknown_cluster_returns_404(client, bad):
    r = get(client, f"segment/{bad}")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found" and "Valid ids: 0, 1, 2, 3" in r.json()["message"]


def test_non_integer_cluster_returns_422(client):
    r = get(client, "segment/abc")
    assert r.status_code == 422 and r.json()["details"][0]["field"] == "cluster_id"


# ----------------------------------------------------------------------------- 7. filtering
def test_filter_by_category(client, raw):
    sub = raw[raw["Product_Category"] == "Electronics"]
    b = get(client, "summary", category="Electronics").json()
    assert b["kpis"]["total_orders"] == len(sub)
    assert b["kpis"]["total_revenue"] == pytest.approx(sub["Total_Amount"].sum(), abs=0.01)
    assert b["kpis"]["total_customers"] == sub["Customer_ID"].nunique()
    assert b["filters"]["category"] == "Electronics"
    rows = get(client, "category-performance", category="Electronics").json()["categories"]
    assert [r["category"] for r in rows] == ["Electronics"] and rows[0]["revenue_share_pct"] == 100


def test_filters_combine_and_ignore_case(client, raw):
    sub = raw[(raw["City"] == "Istanbul") & (raw["Device_Type"] == "Mobile") & (raw["Payment_Method"] == "Credit Card")
              & (raw["Date"] >= "2023-06-01") & (raw["Date"] <= "2023-12-31")]
    b = get(client, "summary", city="istanbul", device_type="MOBILE", payment_method="credit card",
            start_date="2023-06-01", end_date="2023-12-31").json()
    assert b["kpis"]["total_orders"] == len(sub) > 0
    assert b["kpis"]["total_revenue"] == pytest.approx(sub["Total_Amount"].sum(), abs=0.01)
    f = b["filters"]
    assert (f["city"], f["device_type"], f["payment_method"]) == ("Istanbul", "Mobile", "Credit Card")


def test_filter_by_segment_and_date_end_inclusive(client, raw):
    sub = raw[(raw["Cluster"] == 0) & (raw["Date"] <= "2023-03-31")]
    b = get(client, "summary", segment=0, end_date="2023-03-31").json()
    assert b["kpis"]["total_orders"] == len(sub)
    assert b["filters"]["segment_label"] == LABELS[0]
    cust = get(client, "customer-analysis", segment=0, end_date="2023-03-31").json()
    assert cust["total_customers"] == sub["Customer_ID"].nunique()
    assert cust["as_of_date"] == "2023-04-01"          # recency is measured from the end of the selected range


def test_filtered_customer_numbers_follow_the_filter(client, raw):
    sub = raw[raw["Product_Category"] == "Books"]
    b = get(client, "customer-analysis", category="Books").json()
    per = sub.groupby("Customer_ID")["Total_Amount"].agg(["count", "sum"])
    assert b["total_customers"] == len(per)
    assert {x["orders"]: x["customers"] for x in b["orders_per_customer"]} == per["count"].value_counts().to_dict()
    assert b["spending"]["average"] == pytest.approx(per["sum"].mean(), abs=0.01)


def test_segments_endpoint_respects_other_filters(client, raw):
    sub = raw[raw["Payment_Method"] == "Cash on Delivery"]
    b = get(client, "segments", payment_method="Cash on Delivery").json()
    assert b["total_customers"] == sub["Customer_ID"].nunique()
    assert sum(r["customers"] for r in b["segments"]) == b["total_customers"]
    assert b["total_revenue"] == pytest.approx(sub["Total_Amount"].sum(), abs=0.01)
    assert [r["cluster"] for r in b["segments"]] == [0, 1, 2, 3]


def test_filtering_with_everything_selected_equals_no_filter(client, raw):
    """The filtered code path (rebuilds customer table with the Stage 3 function) must agree with the pre-computed one."""
    full = dict(start_date=str(raw["Date"].min().date()), end_date=str(raw["Date"].max().date()))
    for path in ("customer-analysis", "segments"):
        a, b = get(client, path).json(), get(client, path, **full).json()
        a.pop("filters"), b.pop("filters")
        assert a == b


def test_invalid_filter_values_return_helpful_422(client):
    cases = [({"category": "Nope"}, "category", "Allowed values"), ({"city": "Atlantis"}, "city", "Istanbul"),
             ({"payment_method": "Bitcoin"}, "payment_method", "Credit Card"),
             ({"device_type": "Fridge"}, "device_type", "Mobile"),
             ({"segment": 9}, "segment", "Loyal & Active"),
             ({"start_date": "not-a-date"}, "start_date", None),
             ({"start_date": "2024-02-01", "end_date": "2023-01-01"}, "start_date", "must not be after")]
    for params, field, text in cases:
        r = get(client, "summary", **params)
        assert r.status_code == 422, params
        body = r.json()
        assert body["error"] == "validation_error"
        d = next(x for x in body["details"] if x["field"] == field)
        if text:
            assert text in d["message"]


def test_multiple_invalid_filters_reported_together(client):
    r = get(client, "summary", category="Nope", city="Atlantis", segment=9)
    assert {d["field"] for d in r.json()["details"]} == {"category", "city", "segment"}


def test_every_endpoint_validates_filters(client):
    for path in ALL_ENDPOINTS:
        assert get(client, path, category="Nope").status_code == 422, path


# ----------------------------------------------------------------------------- 8. empty results
@pytest.mark.parametrize("path", ALL_ENDPOINTS)
def test_empty_filter_result_is_a_valid_response(client, path):
    r = get(client, path, start_date="2030-01-01")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["has_data"] is False
    if path == "summary":
        assert b["kpis"]["total_orders"] == 0 and b["kpis"]["total_revenue"] == 0
        assert b["kpis"]["average_order_value"] is None and b["date_range"] is None
    elif path == "sales-trend":
        assert b["points"] == []
    elif path == "category-performance":
        assert b["categories"] == []
    elif path == "city-performance":
        assert b["cities"] == []
    elif path == "payment-analysis":
        assert b["payment_methods"] == []
    elif path == "device-analysis":
        assert b["devices"] == []
    elif path == "customer-analysis":
        assert b["total_customers"] == 0 and b["orders_per_customer"] == [] and b["spending"]["average"] is None
        assert b["recency"]["distribution"] == []
    elif path == "segments":
        assert [r["customers"] for r in b["segments"]] == [0, 0, 0, 0]
        assert all(r["average_orders"] is None for r in b["segments"])
    elif path == "segment/0":
        assert b["segment"]["customers"] == 0 and b["category_mix"] == [] and b["monthly_trend"] == []


def test_combination_that_matches_nothing_is_valid(client, raw):
    r = get(client, "summary", start_date="2023-01-01", end_date="2023-01-01", city="Konya", category="Toys",
            payment_method="Bank Transfer", device_type="Tablet", segment=3)
    expected = raw[(raw["Date"] == "2023-01-01") & (raw["City"] == "Konya") & (raw["Product_Category"] == "Toys")
                   & (raw["Payment_Method"] == "Bank Transfer") & (raw["Device_Type"] == "Tablet") & (raw["Cluster"] == 3)]
    assert r.status_code == 200 and r.json()["kpis"]["total_orders"] == len(expected)


# ----------------------------------------------------------------------------- data loading and robustness
def test_filters_and_health_endpoints(client, raw):
    f = client.get("/api/dashboard/filters").json()
    assert f["categories"] == sorted(raw["Product_Category"].unique())
    assert f["cities"] == sorted(raw["City"].unique())
    assert [s["label"] for s in f["segments"]] == list(LABELS.values())
    assert f["date_min"] == "2023-01-01" and f["date_max"] == "2024-03-25"
    h = client.get("/api/dashboard/health").json()
    assert h["status"] == "ok" and h["orders"] == len(raw) and h["customers"] == 5000 and h["errors"] == []


def test_data_is_not_reloaded_per_request(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("CSV was read during a request")
    monkeypatch.setattr(pd, "read_csv", boom)
    for path in ALL_ENDPOINTS:
        assert get(client, path).status_code == 200
    assert get(client, "customer-analysis", city="Ankara").status_code == 200


def test_filtered_customer_tables_are_cached(client):
    before = client.get("/api/dashboard/health").json()["cached_filter_views"]
    get(client, "customer-analysis", city="Bursa", device_type="Tablet")
    mid = client.get("/api/dashboard/health").json()["cached_filter_views"]
    get(client, "segments", city="Bursa", device_type="Tablet")
    after = client.get("/api/dashboard/health").json()["cached_filter_views"]
    assert mid == before + 1 and after == mid


def test_repeated_requests_give_identical_results(client):
    a = get(client, "customer-analysis", category="Food").json()
    b = get(client, "customer-analysis", category="Food").json()
    assert a == b


def test_service_works_on_a_different_dataset(tmp_path, raw):
    """Proves the numbers come from the files, not from hard-coded values."""
    keep = sorted(raw["Customer_ID"].unique())[:300]
    orders = raw[raw["Customer_ID"].isin(keep)].drop(columns=["Cluster"])
    orders.to_csv(tmp_path / "orders.csv", index=False)
    build_customer_features(orders).to_csv(tmp_path / "customers.csv", index=False)
    seg = pd.read_csv(DATA / "processed" / "customer_segments_core5.csv")
    seg[seg["Customer_ID"].isin(keep)].to_csv(tmp_path / "segments.csv", index=False)
    svc = DashboardService(tmp_path / "orders.csv", tmp_path / "customers.csv", tmp_path / "segments.csv").load()
    assert svc.is_ready
    s = svc.summary(DashboardFilters())["kpis"]
    assert s["total_orders"] == len(orders) and s["total_customers"] == 300
    assert s["total_revenue"] == pytest.approx(orders["Total_Amount"].sum(), abs=0.01)
    assert sum(r["customers"] for r in svc.segments(DashboardFilters())["segments"]) == 300


def test_inconsistent_customer_file_is_rejected(tmp_path, raw):
    orders = raw.drop(columns=["Cluster"])
    orders.to_csv(tmp_path / "orders.csv", index=False)
    cust = pd.read_csv(DATA / "processed" / "customer_features.csv")
    cust.loc[0, "Total_Spending"] += 1000
    cust.to_csv(tmp_path / "customers.csv", index=False)
    svc = DashboardService(tmp_path / "orders.csv", tmp_path / "customers.csv",
                           DATA / "processed" / "customer_segments_core5.csv").load()
    assert not svc.is_ready and any("do not match" in e for e in svc.health()["errors"])


def test_dashboard_returns_503_when_data_is_missing(tmp_path):
    broken = DashboardService(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv").load()
    assert not broken.is_ready
    app.dependency_overrides[get_service] = lambda: broken
    try:
        with TestClient(app) as c:
            assert c.get("/api/dashboard/health").status_code == 503
            for path in ["summary", "segments", "filters", "segment/0", "summary?category=Toys"]:
                r = c.get(f"/api/dashboard/{path}")
                assert r.status_code == 503, path
                assert r.json()["error"] == "data_unavailable"
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------------- existing features keep working
def test_segmentation_api_still_works(client):
    h = client.get("/api/segmentation/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    r = client.post("/api/segmentation/predict", json={
        "Total_Orders": 6, "Average_Order_Value": 1400.0, "Days_Since_Last_Order": 45,
        "Average_Days_Between_Orders": 70, "Preferred_Category_Share": 0.33})
    assert r.status_code == 200 and r.json()["segment"] == "Loyal & Active"
    bad = client.post("/api/segmentation/predict", json={"Total_Orders": 0})
    assert bad.status_code == 422 and bad.json()["error"] == "validation_error"


def test_model_files_were_not_modified():
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL_FILES}
    assert after == HASHES_BEFORE
