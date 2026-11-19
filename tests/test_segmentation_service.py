"""
Tests for the Stage 6 segmentation service and API.

Run all tests:        python -m pytest tests -v
Run demo + tests:     python tests/test_segmentation_service.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app  # noqa: E402
from backend.routes.segmentation import get_service  # noqa: E402
from backend.schemas.segmentation import SegmentationRequest  # noqa: E402
from backend.services.segmentation_service import (FEATURES, SEGMENT_CATALOG, TRAINING_RANGE,  # noqa: E402
                                                   ModelNotLoadedError, SegmentationService)

ARTIFACTS = [ROOT / "models" / "customer_segmentation_kmeans_core5.joblib",
             ROOT / "models" / "cluster_preprocessor_core5.joblib",
             ROOT / "models" / "segment_metadata_core5.json"]
HASHES_BEFORE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ARTIFACTS}

# Four realistic customers, written from the segment profiles (not copied from the training data).
CUSTOMERS = {
    "Loyal / high-frequency": (
        dict(Total_Orders=6, Average_Order_Value=1400.0, Days_Since_Last_Order=45,
             Average_Days_Between_Orders=70, Preferred_Category_Share=0.33), 0, "Loyal & Active"),
    "Lapsed repeat": (
        dict(Total_Orders=3, Average_Order_Value=1250.0, Days_Since_Last_Order=210,
             Average_Days_Between_Orders=60, Preferred_Category_Share=0.33), 1, "Lapsed Repeat Buyers"),
    "Slow-cycle repeat": (
        dict(Total_Orders=2, Average_Order_Value=1200.0, Days_Since_Last_Order=80,
             Average_Days_Between_Orders=190, Preferred_Category_Share=0.5), 2, "Slow-Cycle Repeat Buyers"),
    "One-time": (
        dict(Total_Orders=1, Average_Order_Value=1150.0, Days_Since_Last_Order=215,
             Average_Days_Between_Orders=None, Preferred_Category_Share=1.0), 3,
        "One-Time / Mostly One-Time Customers"),
}


@pytest.fixture(scope="module")
def service():
    return SegmentationService().load()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # `with` runs the start-up (loads the model once)
        yield c


# ----------------------------------------------------------------------------- service
def test_service_loads_everything(service):
    h = service.health()
    assert h["status"] == "ok"
    assert h["model_loaded"] and h["preprocessor_loaded"] and h["metadata_loaded"] and h["consistency_checked"]
    assert h["n_clusters"] == 4 and h["features"] == FEATURES
    assert h["segments"] == {"0": "Loyal & Active", "1": "Lapsed Repeat Buyers", "2": "Slow-Cycle Repeat Buyers",
                             "3": "One-Time / Mostly One-Time Customers"}


@pytest.mark.parametrize("name", list(CUSTOMERS))
def test_realistic_customers(service, name):
    features, cluster, segment = CUSTOMERS[name]
    r = service.predict(features)
    assert r["cluster"] == cluster
    assert r["segment"] == segment
    assert r["description"] and r["suggested_actions"]
    assert set(r["input_features"]) == set(FEATURES)
    assert r["assignment"]["strength"] in {"clear", "moderate", "borderline"}
    assert r["assignment"]["closest_alternative"]["cluster"] != cluster
    assert r["is_single_order_customer"] == (features["Total_Orders"] == 1)


def test_same_category_repeat_customer_gets_caveat(service):
    """Known limitation of the model: repeat customers whose orders are all in one category fall into cluster 3."""
    r = service.predict(dict(Total_Orders=2, Average_Order_Value=1200.0, Days_Since_Last_Order=200,
                             Average_Days_Between_Orders=90, Preferred_Category_Share=1.0))
    assert r["cluster"] == 3 and not r["is_single_order_customer"]
    assert any("more than one order" in w for w in r["warnings"])


def test_out_of_range_input_warns(service):
    r = service.predict(dict(Total_Orders=30, Average_Order_Value=90000.0, Days_Since_Last_Order=900,
                             Average_Days_Between_Orders=10, Preferred_Category_Share=0.2))
    assert sum("outside the range" in w for w in r["warnings"]) == 3


def test_reproduces_all_saved_training_segments(service):
    """The service (saved preprocessor + saved model) must give the same cluster as the Stage 5.1 file
    for all 5,000 customers. This proves the preprocessing is identical to training."""
    cust = pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")
    seg = pd.read_csv(ROOT / "data" / "processed" / "customer_segments_core5.csv")
    assert (cust["Customer_ID"].values == seg["Customer_ID"].values).all()
    pred = service.predict_clusters(cust[FEATURES])
    assert (pred == seg["Cluster"].values).all()


def test_every_real_customer_passes_request_validation():
    """The input rules must not reject real customers."""
    cust = pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")
    for rec in cust[FEATURES].to_dict("records"):
        gap = rec["Average_Days_Between_Orders"]
        SegmentationRequest(Total_Orders=int(rec["Total_Orders"]), Average_Order_Value=rec["Average_Order_Value"],
                            Days_Since_Last_Order=rec["Days_Since_Last_Order"],
                            Average_Days_Between_Orders=None if pd.isna(gap) else gap,
                            Preferred_Category_Share=round(rec["Preferred_Category_Share"], 2))


def test_catalog_matches_cluster_profiles_and_training_range():
    prof = pd.read_csv(ROOT / "data" / "processed" / "cluster_profiles_core5.csv", index_col=0)
    for i, info in SEGMENT_CATALOG.items():
        t, p = info["typical"], prof.loc[i]
        assert abs(t["customers_pct"] - p["Customers_%"]) < 0.1
        assert abs(t["avg_orders"] - p["Total_Orders"]) < 0.01
        assert abs(t["avg_order_value"] - p["Average_Order_Value"]) < 1
        assert abs(t["avg_days_since_last_order"] - p["Days_Since_Last_Order"]) < 0.1
        assert abs(t["avg_days_between_orders"] - p["Average_Days_Between_Orders"]) < 0.1
        assert abs(t["revenue_share_pct"] - p["Revenue_Share_%"]) < 0.1
    cust = pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")
    for f, (lo, hi) in TRAINING_RANGE.items():
        assert abs(cust[f].min() - lo) < 0.01 and abs(cust[f].max() - hi) < 0.01


def test_missing_files_are_reported_not_crashing(tmp_path):
    broken = SegmentationService(model_path=tmp_path / "nope.joblib", preprocessor_path=tmp_path / "nope2.joblib",
                                 metadata_path=tmp_path / "nope.json").load()
    h = broken.health()
    assert h["status"] == "unavailable" and not h["model_loaded"] and len(h["errors"]) == 3
    with pytest.raises(ModelNotLoadedError):
        broken.predict(CUSTOMERS["One-time"][0])


def test_corrupted_file_is_reported(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a model")
    s = SegmentationService(model_path=bad).load()
    h = s.health()
    assert h["status"] == "unavailable" and not h["model_loaded"]
    assert any("could not be loaded" in e for e in h["errors"])


# ----------------------------------------------------------------------------- API
def test_api_health(client):
    r = client.get("/api/segmentation/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and body["preprocessor_loaded"] and body["metadata_loaded"]


@pytest.mark.parametrize("name", list(CUSTOMERS))
def test_api_predict(client, name):
    features, cluster, segment = CUSTOMERS[name]
    r = client.post("/api/segmentation/predict", json=features)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cluster"] == cluster and body["segment"] == segment
    for key in ("description", "input_features", "assignment", "segment_profile", "suggested_actions", "warnings"):
        assert key in body
    assert body["input_features"]["Total_Orders"] == features["Total_Orders"]


def test_api_single_order_gap_can_be_omitted(client):
    payload = {k: v for k, v in CUSTOMERS["One-time"][0].items() if k != "Average_Days_Between_Orders"}
    r = client.post("/api/segmentation/predict", json=payload)
    assert r.status_code == 200 and r.json()["cluster"] == 3


GOOD = dict(Total_Orders=3, Average_Order_Value=1250.0, Days_Since_Last_Order=210,
            Average_Days_Between_Orders=60, Preferred_Category_Share=0.33)


@pytest.mark.parametrize("change,field", [
    ({"Total_Orders": 0}, "Total_Orders"),
    ({"Total_Orders": 2.5}, "Total_Orders"),
    ({"Total_Orders": "many"}, "Total_Orders"),
    ({"Average_Order_Value": -5}, "Average_Order_Value"),
    ({"Average_Order_Value": 0}, "Average_Order_Value"),
    ({"Days_Since_Last_Order": -1}, "Days_Since_Last_Order"),
    ({"Preferred_Category_Share": 1.5}, "Preferred_Category_Share"),
    ({"Preferred_Category_Share": 0}, "Preferred_Category_Share"),
    ({"Preferred_Category_Share": 0.5}, "request"),                    # impossible with 3 orders
    ({"Average_Days_Between_Orders": None}, "request"),                # needed when orders > 1
    ({"Total_Orders": 1, "Preferred_Category_Share": 0.5, "Average_Days_Between_Orders": None}, "request"),
    ({"Total_Orders": 1, "Preferred_Category_Share": 1.0}, "request"),  # gap given for a 1-order customer
    ({"Unexpected_Field": 1}, "Unexpected_Field"),
])
def test_api_rejects_invalid_input(client, change, field):
    payload = {**GOOD, **change}
    r = client.post("/api/segmentation/predict", json=payload)
    assert r.status_code == 422, (change, r.text)
    body = r.json()
    assert body["error"] == "validation_error"
    assert any(d["field"] == field for d in body["details"]), body


def test_api_rejects_missing_field(client):
    payload = {k: v for k, v in GOOD.items() if k != "Days_Since_Last_Order"}
    r = client.post("/api/segmentation/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "Days_Since_Last_Order"


def test_api_rejects_bad_json(client):
    r = client.post("/api/segmentation/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_api_returns_503_when_model_unavailable(tmp_path):
    broken = SegmentationService(model_path=tmp_path / "x.joblib").load()
    app.dependency_overrides[get_service] = lambda: broken
    try:
        with TestClient(app) as c:
            assert c.get("/api/segmentation/health").status_code == 503
            r = c.post("/api/segmentation/predict", json=GOOD)
            assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
    finally:
        app.dependency_overrides.clear()


def test_model_files_were_not_modified():
    """Nothing in the service or API writes to the trained files."""
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ARTIFACTS}
    assert after == HASHES_BEFORE


# ----------------------------------------------------------------------------- demo
def demo():
    s = SegmentationService().load()
    print("\nHEALTH:", {k: v for k, v in s.health().items() if k in ("status", "model_loaded", "preprocessor_loaded", "metadata_loaded")})
    print("\n" + "=" * 100)
    print(f"{'Customer':<26}{'Orders':>7}{'AOV':>8}{'Recency':>9}{'Gap':>6}{'Share':>7}  ->  Cluster  Segment (assignment)")
    print("=" * 100)
    extra = {"Repeat, all in 1 category": dict(Total_Orders=2, Average_Order_Value=1200.0, Days_Since_Last_Order=200,
                                                Average_Days_Between_Orders=90, Preferred_Category_Share=1.0),
             "Brand-new customer": dict(Total_Orders=1, Average_Order_Value=900.0, Days_Since_Last_Order=5,
                                        Average_Days_Between_Orders=None, Preferred_Category_Share=1.0)}
    rows = {k: v[0] for k, v in CUSTOMERS.items()} | extra
    for name, f in rows.items():
        r = s.predict(f)
        gap = "-" if f["Average_Days_Between_Orders"] is None else f"{f['Average_Days_Between_Orders']:.0f}"
        print(f"{name:<26}{f['Total_Orders']:>7}{f['Average_Order_Value']:>8.0f}{f['Days_Since_Last_Order']:>9.0f}{gap:>6}"
              f"{f['Preferred_Category_Share']:>7.2f}  ->  {r['cluster']:^7}  {r['segment']} "
              f"({r['assignment']['strength']}, margin {r['assignment']['margin']:.2f}, next: {r['assignment']['closest_alternative']['segment']})")
        for w in r["warnings"]:
            print(f"{'':<26}   ! {w}")
    print("=" * 100)


if __name__ == "__main__":
    demo()
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
