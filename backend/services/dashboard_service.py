"""
Dashboard data service.

- Loads the three CSV files ONCE (at API start-up) and keeps them in memory.
- Every request only filters the in-memory tables and aggregates. No file is re-read, no model is trained,
  and K-Means is never run: segments come from the saved production result (customer_segments_core5.csv).
- Customer-level numbers for a filtered view are built with the SAME Stage 3 function used for training
  (src/features/stage3_customer_features.build_customer_features), so definitions never drift.

Filter semantics
  Filters select ORDERS (date, category, city, payment method, device) or CUSTOMERS (segment).
  Every number on the page is then calculated from the selected orders. For example "orders per customer"
  means "number of selected orders per customer", and recency is measured from the end of the selected
  date range (or from the end of the data if no end date is chosen).
"""
import logging
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.segmentation_service import SEGMENT_CATALOG  # noqa: E402  (approved segment labels)
from src.features.stage3_customer_features import build_customer_features  # noqa: E402  (reused, not copied)

logger = logging.getLogger("dashboard_service")

DEFAULT_ORDERS_PATH = ROOT / "data" / "ecommerce_customer_behavior_dataset_v2.csv"
DEFAULT_CUSTOMERS_PATH = ROOT / "data" / "processed" / "customer_features.csv"
DEFAULT_SEGMENTS_PATH = ROOT / "data" / "processed" / "customer_segments_core5.csv"

GRANULARITIES = ("day", "week", "month")
CACHE_SIZE = 128

REQUIRED_ORDER_COLS = ["Order_ID", "Customer_ID", "Date", "Product_Category", "City", "Payment_Method", "Device_Type",
                       "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount", "Customer_Rating",
                       "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Age", "Gender",
                       "Is_Returning_Customer"]

SPENDING_EDGES = [0, 500, 1000, 2500, 5000, 10000, 20000, np.inf]
SPENDING_LABELS = ["< 500", "500 - 1,000", "1,000 - 2,500", "2,500 - 5,000", "5,000 - 10,000",
                   "10,000 - 20,000", "20,000+"]
RECENCY_EDGES = [0, 31, 61, 91, 181, 271, 366, np.inf]
RECENCY_LABELS = ["0 - 30 days", "31 - 60 days", "61 - 90 days", "91 - 180 days", "181 - 270 days",
                  "271 - 365 days", "366+ days"]

NEW_VS_RETURNING_NOTE = (
    "Based on the Is_Returning_Customer field exactly as recorded in the dataset (descriptive only, not used by any "
    "model). In this dataset every order after a customer's first is flagged as returning, and about 60% of first "
    "orders are flagged as returning too, so 'new customer' orders understate first-time purchases.")


# ----------------------------------------------------------------------------- errors
class DashboardError(Exception):
    """Base class for dashboard service errors."""


class DashboardNotReadyError(DashboardError):
    """The datasets are not loaded (missing/corrupt files)."""


class InvalidFilterError(DashboardError):
    """One or more filter values are not valid. `errors` is a list of {"field", "message"}."""

    def __init__(self, errors: List[Dict[str, str]]):
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


class SegmentNotFoundError(DashboardError):
    def __init__(self, cluster_id: int):
        super().__init__(f"Unknown segment (cluster) id {cluster_id}. Valid ids: "
                         f"{', '.join(str(i) for i in sorted(SEGMENT_CATALOG))}.")
        self.cluster_id = cluster_id


# ----------------------------------------------------------------------------- filters
@dataclass(frozen=True)
class DashboardFilters:
    """Validated filters. Frozen, so it can be used as a cache key."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    category: Optional[str] = None
    city: Optional[str] = None
    payment_method: Optional[str] = None
    device_type: Optional[str] = None
    segment: Optional[int] = None

    @property
    def is_empty(self) -> bool:
        return all(v is None for v in (self.start_date, self.end_date, self.category, self.city,
                                       self.payment_method, self.device_type, self.segment))


# ----------------------------------------------------------------------------- helpers
def _num(x, nd: int = 2) -> Optional[float]:
    """Python float rounded for JSON, or None for missing values."""
    if x is None or pd.isna(x):
        return None
    return round(float(x), nd)


def _int(x) -> int:
    return int(x) if x is not None and not pd.isna(x) else 0


def _pct(part, whole, nd: int = 2) -> float:
    return round(float(part) / float(whole) * 100, nd) if whole else 0.0


class _LRU:
    """Small thread-safe LRU cache for customer tables (one entry per distinct filter combination)."""

    def __init__(self, size: int):
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


# ----------------------------------------------------------------------------- service
class DashboardService:
    def __init__(self, orders_path: Path = DEFAULT_ORDERS_PATH, customers_path: Path = DEFAULT_CUSTOMERS_PATH,
                 segments_path: Path = DEFAULT_SEGMENTS_PATH):
        self.orders_path, self.customers_path, self.segments_path = map(Path, (orders_path, customers_path, segments_path))
        self._reset()

    def _reset(self):
        self._orders: Optional[pd.DataFrame] = None
        self._customers: Optional[pd.DataFrame] = None
        self._cluster_by_customer: Optional[pd.Series] = None
        self._min_date = self._max_date = None
        self._options: Dict[str, Any] = {}
        self._lookup: Dict[str, Dict[str, str]] = {}
        self._cache = _LRU(CACHE_SIZE)
        self._errors: List[str] = []
        self._notes: List[str] = []
        self._ready = False

    # ---- loading ------------------------------------------------------------------------------
    def load(self) -> "DashboardService":
        """Read and validate the datasets once. Never raises: problems are stored and shown by health()."""
        self._reset()
        try:
            self._load_and_validate()
        except Exception as exc:  # missing file, bad columns, inconsistent data ...
            self._errors.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Dashboard data failed to load")
            self._orders = None
            self._ready = False
        if self._ready:
            logger.info("Dashboard data ready: %s orders, %s customers", f"{len(self._orders):,}", f"{len(self._customers):,}")
        return self

    def _load_and_validate(self):
        for p in (self.orders_path, self.customers_path, self.segments_path):
            if not p.exists():
                raise FileNotFoundError(f"data file not found: {p.name}")

        orders = pd.read_csv(self.orders_path, parse_dates=["Date"])
        missing = [c for c in REQUIRED_ORDER_COLS if c not in orders.columns]
        if missing:
            raise ValueError(f"orders file is missing columns: {missing}")
        customers = pd.read_csv(self.customers_path, parse_dates=["First_Order_Date", "Last_Order_Date"])
        segments = pd.read_csv(self.segments_path, usecols=["Customer_ID", "Cluster", "Segment_Label"])

        if segments["Customer_ID"].duplicated().any() or customers["Customer_ID"].duplicated().any():
            raise ValueError("duplicate Customer_ID in a customer-level file")
        unknown_clusters = sorted(set(segments["Cluster"]) - set(SEGMENT_CATALOG))
        if unknown_clusters:
            raise ValueError(f"segments file contains cluster ids not in the label catalog: {unknown_clusters}")
        no_segment = set(orders["Customer_ID"]) - set(segments["Customer_ID"])
        if no_segment:
            raise ValueError(f"{len(no_segment)} customers in the orders have no segment")

        # The customer-level file must agree with the order-level data.
        agg = orders.groupby("Customer_ID").agg(n=("Order_ID", "count"), s=("Total_Amount", "sum"))
        chk = customers.set_index("Customer_ID").join(agg, how="outer")
        if chk[["n", "s", "Total_Orders"]].isna().any().any():
            raise ValueError("customer_features.csv and the orders do not contain the same customers")
        if not ((chk["Total_Orders"] == chk["n"]).all() and np.allclose(chk["Total_Spending"], chk["s"])):
            raise ValueError("customer_features.csv totals do not match the order data")

        cluster_map = segments.set_index("Customer_ID")["Cluster"].astype(int)
        orders["Cluster"] = orders["Customer_ID"].map(cluster_map).astype(int)
        customers["Cluster"] = customers["Customer_ID"].map(cluster_map).astype(int)
        orders = orders.sort_values(["Date", "Order_ID"]).reset_index(drop=True)

        # The segments file still carries the Stage 5.1 name for cluster 3. The approved names live in the catalog.
        file_labels = segments.groupby("Cluster")["Segment_Label"].first().to_dict()
        for cid, info in SEGMENT_CATALOG.items():
            if file_labels.get(cid) != info["label"]:
                self._notes.append(f"Cluster {cid}: segments file label '{file_labels.get(cid)}' is replaced by the "
                                   f"approved label '{info['label']}'.")

        self._orders, self._customers, self._cluster_by_customer = orders, customers, cluster_map
        self._min_date, self._max_date = orders["Date"].min(), orders["Date"].max()
        cats = {"category": "Product_Category", "city": "City", "payment_method": "Payment_Method",
                "device_type": "Device_Type"}
        self._options = {k: sorted(orders[c].unique().tolist()) for k, c in cats.items()}
        self._lookup = {k: {v.lower(): v for v in vals} for k, vals in self._options.items()}
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _require_ready(self):
        if not self._ready:
            raise DashboardNotReadyError("Dashboard data is not available: " + "; ".join(self._errors or ["not loaded"]))

    def health(self) -> Dict[str, Any]:
        r = self._ready
        return {"status": "ok" if r else "unavailable", "data_loaded": r,
                "orders": len(self._orders) if r else 0, "customers": len(self._customers) if r else 0,
                "segments": len(SEGMENT_CATALOG) if r else 0,
                "date_min": self._min_date.date() if r else None, "date_max": self._max_date.date() if r else None,
                "cached_filter_views": len(self._cache), "errors": list(self._errors), "notes": list(self._notes)}

    # ---- filters ------------------------------------------------------------------------------
    def build_filters(self, start_date=None, end_date=None, category=None, city=None, payment_method=None,
                      device_type=None, segment=None) -> DashboardFilters:
        """Validate raw query values and return DashboardFilters. Raises InvalidFilterError with all problems."""
        self._require_ready()
        errors: List[Dict[str, str]] = []
        clean: Dict[str, Optional[str]] = {}
        for field, value in (("category", category), ("city", city), ("payment_method", payment_method),
                             ("device_type", device_type)):
            if value is None or not str(value).strip():
                clean[field] = None
                continue
            canon = self._lookup[field].get(str(value).strip().lower())
            if canon is None:
                errors.append({"field": field, "message": f"Unknown {field} '{value}'. Allowed values: "
                                                          f"{', '.join(self._options[field])}."})
            clean[field] = canon
        if segment is not None and segment not in SEGMENT_CATALOG:
            allowed = ", ".join(f"{i} ({v['label']})" for i, v in SEGMENT_CATALOG.items())
            errors.append({"field": "segment", "message": f"Unknown segment {segment}. Allowed values: {allowed}."})
        if start_date and end_date and start_date > end_date:
            errors.append({"field": "start_date", "message": "start_date must not be after end_date."})
        if errors:
            raise InvalidFilterError(errors)
        return DashboardFilters(start_date, end_date, clean["category"], clean["city"], clean["payment_method"],
                                clean["device_type"], segment)

    def filters_dict(self, f: DashboardFilters) -> Dict[str, Any]:
        return {"start_date": f.start_date, "end_date": f.end_date, "category": f.category, "city": f.city,
                "payment_method": f.payment_method, "device_type": f.device_type, "segment": f.segment,
                "segment_label": SEGMENT_CATALOG[f.segment]["label"] if f.segment is not None else None}

    def filter_options(self) -> Dict[str, Any]:
        self._require_ready()
        return {"categories": self._options["category"], "cities": self._options["city"],
                "payment_methods": self._options["payment_method"], "device_types": self._options["device_type"],
                "segments": [{"cluster": i, "label": v["label"]} for i, v in sorted(SEGMENT_CATALOG.items())],
                "granularities": list(GRANULARITIES),
                "date_min": self._min_date.date(), "date_max": self._max_date.date()}

    # ---- internal data access -----------------------------------------------------------------
    def _apply(self, f: DashboardFilters) -> pd.DataFrame:
        df = self._orders
        if f.start_date:
            df = df[df["Date"] >= pd.Timestamp(f.start_date)]
        if f.end_date:
            df = df[df["Date"] < pd.Timestamp(f.end_date) + pd.Timedelta(days=1)]
        if f.category:
            df = df[df["Product_Category"] == f.category]
        if f.city:
            df = df[df["City"] == f.city]
        if f.payment_method:
            df = df[df["Payment_Method"] == f.payment_method]
        if f.device_type:
            df = df[df["Device_Type"] == f.device_type]
        if f.segment is not None:
            df = df[df["Cluster"] == f.segment]
        return df

    def _as_of(self, f: DashboardFilters) -> pd.Timestamp:
        """Date that recency is measured from: the day after the end of the selected range (never after the data end)."""
        end = self._max_date
        if f.end_date:
            end = min(pd.Timestamp(f.end_date), self._max_date)
        return end + pd.Timedelta(days=1)

    def _customer_table(self, f: DashboardFilters, orders: pd.DataFrame) -> pd.DataFrame:
        """One row per customer who has at least one selected order (Stage 3 definitions) + their segment."""
        if orders.empty:
            return self._customers.iloc[0:0]
        if f.is_empty:
            return self._customers                      # pre-computed Stage 3 table
        key = f
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        ct = build_customer_features(orders, reference_date=self._as_of(f))
        ct["Cluster"] = ct["Customer_ID"].map(self._cluster_by_customer).astype(int)
        self._cache.put(key, ct)
        return ct

    def _view(self, f: DashboardFilters) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._require_ready()
        orders = self._apply(f)
        return orders, self._customer_table(f, orders)

    # ---- generic builders ---------------------------------------------------------------------
    def _trend(self, df: pd.DataFrame, granularity: str, f: DashboardFilters) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        freq = {"day": "D", "week": "W", "month": "M"}[granularity]
        period = df["Date"].dt.to_period(freq)
        g = df.groupby(period).agg(revenue=("Total_Amount", "sum"), orders=("Order_ID", "count"))
        g = g.reindex(pd.period_range(g.index.min(), g.index.max(), freq=freq), fill_value=0)
        eff_start = max(f.start_date, self._min_date.date()) if f.start_date else self._min_date.date()
        eff_end = min(f.end_date, self._max_date.date()) if f.end_date else self._max_date.date()
        points = []
        for p, row in g.iterrows():
            start, end = p.start_time.date(), p.end_time.date()
            label = str(p) if granularity == "month" else start.isoformat()
            orders = int(row["orders"])
            points.append({"period": label, "period_start": start, "revenue": _num(row["revenue"]), "orders": orders,
                           "average_order_value": _num(row["revenue"] / orders) if orders else None,
                           "is_partial": granularity != "day" and (start < eff_start or end > eff_end)})
        return points

    @staticmethod
    def _breakdown(df: pd.DataFrame, col: str, key: str) -> List[Dict[str, Any]]:
        """Revenue/orders per value of `col`, largest revenue first."""
        if df.empty:
            return []
        total_rev, total_orders = df["Total_Amount"].sum(), len(df)
        g = df.groupby(col).agg(revenue=("Total_Amount", "sum"), orders=("Order_ID", "count"),
                                quantity=("Quantity", "sum"), customers=("Customer_ID", "nunique"),
                                rating=("Customer_Rating", "mean"), session=("Session_Duration_Minutes", "mean"),
                                pages=("Pages_Viewed", "mean"))
        g = g.reset_index().sort_values(["revenue", col], ascending=[False, True])
        rows = []
        for r in g.itertuples(index=False):
            d = r._asdict()
            rows.append({key: d[col], "revenue": _num(d["revenue"]), "orders": int(d["orders"]),
                         "quantity": int(d["quantity"]), "customers": int(d["customers"]),
                         "average_order_value": _num(d["revenue"] / d["orders"]),
                         "revenue_share_pct": _pct(d["revenue"], total_rev), "orders_share_pct": _pct(d["orders"], total_orders),
                         "average_rating": _num(d["rating"]), "average_session_minutes": _num(d["session"]),
                         "average_pages_viewed": _num(d["pages"])})
        return rows

    @staticmethod
    def _bucket_counts(values: pd.Series, edges, labels) -> List[Dict[str, Any]]:
        cut = pd.cut(values, bins=edges, labels=labels, right=False)
        counts = cut.value_counts(sort=False).reindex(labels, fill_value=0)
        total = int(counts.sum())
        return [{"label": lab, "customers": int(counts[lab]), "percentage": _pct(counts[lab], total)} for lab in labels]

    @staticmethod
    def _orders_per_customer(ct: pd.DataFrame) -> List[Dict[str, Any]]:
        if ct.empty:
            return []
        vc = ct["Total_Orders"].value_counts().sort_index()
        return [{"orders": int(n), "customers": int(c), "percentage": _pct(c, len(ct))} for n, c in vc.items()]

    @staticmethod
    def _preference(series: pd.Series) -> Optional[Dict[str, Any]]:
        if series.empty:
            return None
        vc = series.value_counts()
        return {"name": str(vc.index[0]), "share_pct": _pct(vc.iloc[0], len(series))}

    def _segment_row(self, cid: int, ct: pd.DataFrame, total_customers: int, total_revenue: float) -> Dict[str, Any]:
        s = ct[ct["Cluster"] == cid]
        n = len(s)
        revenue, orders = (float(s["Total_Spending"].sum()), int(s["Total_Orders"].sum())) if n else (0.0, 0)
        return {
            "cluster": cid, "segment": SEGMENT_CATALOG[cid]["label"], "description": SEGMENT_CATALOG[cid]["description"],
            "customers": n, "customers_pct": _pct(n, total_customers),
            "revenue": _num(revenue), "revenue_share_pct": _pct(revenue, total_revenue), "orders": orders,
            "average_orders": _num(orders / n) if n else None,
            "average_order_value": _num(revenue / orders) if orders else None,
            "average_customer_spending": _num(revenue / n) if n else None,
            "average_recency_days": _num(s["Days_Since_Last_Order"].mean()) if n else None,
            "average_days_between_orders": _num(s["Average_Days_Between_Orders"].mean()) if n else None,
            "single_order_pct": _pct(s["Is_Single_Order_Customer"].sum(), n) if n else 0.0,
            "preferred_category": self._preference(s["Preferred_Product_Category"]),
            "preferred_device": self._preference(s["Preferred_Device_Type"]),
            "preferred_payment": self._preference(s["Preferred_Payment_Method"]),
        }

    # ---- public analytics ---------------------------------------------------------------------
    def summary(self, f: DashboardFilters) -> Dict[str, Any]:
        df, ct = self._view(f)
        has = not df.empty
        revenue = float(df["Total_Amount"].sum())
        return {
            "has_data": has, "filters": self.filters_dict(f),
            "date_range": {"start": df["Date"].min().date(), "end": df["Date"].max().date()} if has else None,
            "kpis": {"total_revenue": _num(revenue), "total_orders": len(df), "total_customers": df["Customer_ID"].nunique(),
                     "average_order_value": _num(revenue / len(df)) if has else None,
                     "average_customer_rating": _num(df["Customer_Rating"].mean()) if has else None,
                     "average_quantity_per_order": _num(df["Quantity"].mean()) if has else None},
        }

    def sales_trend(self, f: DashboardFilters, granularity: str = "month") -> Dict[str, Any]:
        df, _ = self._view(f)
        return {"has_data": not df.empty, "filters": self.filters_dict(f), "granularity": granularity,
                "points": self._trend(df, granularity, f)}

    def category_performance(self, f: DashboardFilters) -> Dict[str, Any]:
        df, _ = self._view(f)
        return {"has_data": not df.empty, "filters": self.filters_dict(f),
                "categories": self._breakdown(df, "Product_Category", "category")}

    def city_performance(self, f: DashboardFilters) -> Dict[str, Any]:
        df, _ = self._view(f)
        return {"has_data": not df.empty, "filters": self.filters_dict(f), "cities": self._breakdown(df, "City", "city")}

    def payment_analysis(self, f: DashboardFilters) -> Dict[str, Any]:
        df, _ = self._view(f)
        return {"has_data": not df.empty, "filters": self.filters_dict(f),
                "payment_methods": self._breakdown(df, "Payment_Method", "payment_method")}

    def device_analysis(self, f: DashboardFilters) -> Dict[str, Any]:
        df, _ = self._view(f)
        return {"has_data": not df.empty, "filters": self.filters_dict(f),
                "devices": self._breakdown(df, "Device_Type", "device_type")}

    def customer_analysis(self, f: DashboardFilters) -> Dict[str, Any]:
        df, ct = self._view(f)
        has = not df.empty
        n = len(ct)
        ratings = df["Customer_Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        rating_total = int(ratings.sum())

        flag = df.groupby("Is_Returning_Customer").agg(orders=("Order_ID", "count"), revenue=("Total_Amount", "sum"))
        groups = []
        for is_ret, name in ((False, "New customer orders"), (True, "Returning customer orders")):
            o = int(flag["orders"].get(is_ret, 0)); r = float(flag["revenue"].get(is_ret, 0.0))
            groups.append({"group": name, "orders": o, "orders_pct": _pct(o, len(df)), "revenue": _num(r),
                           "revenue_pct": _pct(r, df["Total_Amount"].sum()), "average_order_value": _num(r / o) if o else None})
        monthly = []
        if has:
            per = df["Date"].dt.to_period("M")
            m = df.groupby([per, "Is_Returning_Customer"]).size().unstack(fill_value=0)
            m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"), fill_value=0)
            for p, row in m.iterrows():
                monthly.append({"period": str(p), "new_orders": int(row.get(False, 0)), "returning_orders": int(row.get(True, 0))})

        single = int((ct["Total_Orders"] == 1).sum()) if n else 0
        spend = ct["Total_Spending"] if n else pd.Series(dtype=float)
        return {
            "has_data": has, "filters": self.filters_dict(f), "as_of_date": self._as_of(f).date() if has else None,
            "total_customers": n, "single_order_customers": single, "repeat_customers": n - single,
            "single_order_pct": _pct(single, n), "repeat_pct": _pct(n - single, n) if n else 0.0,
            "orders_per_customer": self._orders_per_customer(ct),
            "spending": {"average": _num(spend.mean()) if n else None, "median": _num(spend.median()) if n else None,
                         "maximum": _num(spend.max()) if n else None,
                         "distribution": self._bucket_counts(spend, SPENDING_EDGES, SPENDING_LABELS) if n else []},
            "rating": {"average": _num(df["Customer_Rating"].mean()) if has else None, "total_ratings": rating_total,
                       "distribution": [{"rating": int(r), "orders": int(c), "percentage": _pct(c, rating_total)}
                                        for r, c in ratings.items()] if has else []},
            "recency": {"average_days": _num(ct["Days_Since_Last_Order"].mean()) if n else None,
                        "median_days": _num(ct["Days_Since_Last_Order"].median()) if n else None,
                        "distribution": self._bucket_counts(ct["Days_Since_Last_Order"], RECENCY_EDGES, RECENCY_LABELS) if n else []},
            "new_vs_returning": {"note": NEW_VS_RETURNING_NOTE, "groups": groups if has else [], "monthly": monthly},
        }

    def segments(self, f: DashboardFilters) -> Dict[str, Any]:
        """All four segments (the segment filter does not apply here; use /segment/{id} for one segment)."""
        f = replace(f, segment=None)
        df, ct = self._view(f)
        total_customers, total_revenue = len(ct), float(ct["Total_Spending"].sum()) if len(ct) else 0.0
        rows = [self._segment_row(cid, ct, total_customers, total_revenue) for cid in sorted(SEGMENT_CATALOG)]
        return {"has_data": not df.empty, "filters": self.filters_dict(f),
                "as_of_date": self._as_of(f).date() if not df.empty else None,
                "total_customers": total_customers, "total_revenue": _num(total_revenue), "segments": rows}

    def segment_detail(self, cluster_id: int, f: DashboardFilters) -> Dict[str, Any]:
        self._require_ready()
        if cluster_id not in SEGMENT_CATALOG:
            raise SegmentNotFoundError(cluster_id)
        f = replace(f, segment=cluster_id)
        df, ct = self._view(f)
        # share figures inside a detail page are relative to the whole filtered customer base, so use the other segments too
        base_df, base_ct = self._view(replace(f, segment=None))
        row = self._segment_row(cluster_id, base_ct, len(base_ct), float(base_ct["Total_Spending"].sum()) if len(base_ct) else 0.0)
        info = SEGMENT_CATALOG[cluster_id]

        def mix(col, key):
            return [{"name": r[key], "revenue": r["revenue"], "orders": r["orders"], "revenue_share_pct": r["revenue_share_pct"]}
                    for r in self._breakdown(df, col, key)]

        return {
            "has_data": not df.empty, "filters": self.filters_dict(f), "as_of_date": self._as_of(f).date() if not df.empty else None,
            "segment": row, "suggested_actions": list(info["suggested_actions"]),
            "category_mix": mix("Product_Category", "category"), "device_mix": mix("Device_Type", "device_type"),
            "payment_mix": mix("Payment_Method", "payment_method"),
            "orders_per_customer": self._orders_per_customer(ct),
            "recency_distribution": self._bucket_counts(ct["Days_Since_Last_Order"], RECENCY_EDGES, RECENCY_LABELS) if len(ct) else [],
            "monthly_trend": self._trend(df, "month", f),
        }


@lru_cache(maxsize=1)
def get_dashboard_service() -> DashboardService:
    """One shared service for the whole app (loaded once at API start-up)."""
    return DashboardService()
