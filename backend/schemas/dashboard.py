"""
Response schemas for /api/dashboard.

Every response contains `has_data` (False when the filters select no orders) and `filters`
(the filters that were applied, with canonical spelling), so the frontend can show an empty state.
Lists are ready for charting libraries: one object per bar / point / slice.
"""
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel


class AppliedFilters(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    category: Optional[str] = None
    city: Optional[str] = None
    payment_method: Optional[str] = None
    device_type: Optional[str] = None
    segment: Optional[int] = None
    segment_label: Optional[str] = None


class DashboardResponse(BaseModel):
    has_data: bool
    filters: AppliedFilters


# ----------------------------------------------------------------------------- filters / health
class SegmentOption(BaseModel):
    cluster: int
    label: str


class FilterOptionsResponse(BaseModel):
    categories: List[str]
    cities: List[str]
    payment_methods: List[str]
    device_types: List[str]
    segments: List[SegmentOption]
    granularities: List[str]
    date_min: date
    date_max: date


class DashboardHealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    data_loaded: bool
    orders: int
    customers: int
    segments: int
    date_min: Optional[date] = None
    date_max: Optional[date] = None
    cached_filter_views: int = 0
    errors: List[str] = []
    notes: List[str] = []


# ----------------------------------------------------------------------------- summary
class DateRange(BaseModel):
    start: date
    end: date


class Kpis(BaseModel):
    total_revenue: float
    total_orders: int
    total_customers: int
    average_order_value: Optional[float] = None
    average_customer_rating: Optional[float] = None
    average_quantity_per_order: Optional[float] = None


class SummaryResponse(DashboardResponse):
    date_range: Optional[DateRange] = None
    kpis: Kpis


# ----------------------------------------------------------------------------- sales
class TrendPoint(BaseModel):
    period: str                         # "2023-01" (month) or the first day of the week / the day (ISO date)
    period_start: date
    revenue: float
    orders: int
    average_order_value: Optional[float] = None
    is_partial: bool = False            # True when the data covers only part of this period


class SalesTrendResponse(DashboardResponse):
    granularity: Literal["day", "week", "month"]
    points: List[TrendPoint]


class BreakdownRow(BaseModel):
    """Shared numbers for category / city / payment / device rows."""
    revenue: float
    orders: int
    quantity: int
    customers: int
    average_order_value: Optional[float] = None
    revenue_share_pct: float
    orders_share_pct: float
    average_rating: Optional[float] = None
    average_session_minutes: Optional[float] = None
    average_pages_viewed: Optional[float] = None


class CategoryRow(BreakdownRow):
    category: str


class CityRow(BreakdownRow):
    city: str


class PaymentRow(BreakdownRow):
    payment_method: str


class DeviceRow(BreakdownRow):
    device_type: str


class CategoryPerformanceResponse(DashboardResponse):
    categories: List[CategoryRow]


class CityPerformanceResponse(DashboardResponse):
    cities: List[CityRow]


class PaymentAnalysisResponse(DashboardResponse):
    payment_methods: List[PaymentRow]


class DeviceAnalysisResponse(DashboardResponse):
    devices: List[DeviceRow]


# ----------------------------------------------------------------------------- customers
class OrderCountBucket(BaseModel):
    orders: int
    customers: int
    percentage: float


class LabelBucket(BaseModel):
    label: str
    customers: int
    percentage: float


class SpendingAnalysis(BaseModel):
    average: Optional[float] = None
    median: Optional[float] = None
    maximum: Optional[float] = None
    distribution: List[LabelBucket]


class RatingBucket(BaseModel):
    rating: int
    orders: int
    percentage: float


class RatingAnalysis(BaseModel):
    average: Optional[float] = None
    total_ratings: int
    distribution: List[RatingBucket]


class RecencyAnalysis(BaseModel):
    average_days: Optional[float] = None
    median_days: Optional[float] = None
    distribution: List[LabelBucket]


class NewVsReturningGroup(BaseModel):
    group: str
    orders: int
    orders_pct: float
    revenue: float
    revenue_pct: float
    average_order_value: Optional[float] = None


class NewVsReturningMonth(BaseModel):
    period: str
    new_orders: int
    returning_orders: int


class NewVsReturning(BaseModel):
    note: str
    groups: List[NewVsReturningGroup]
    monthly: List[NewVsReturningMonth]


class CustomerAnalysisResponse(DashboardResponse):
    as_of_date: Optional[date] = None
    total_customers: int
    single_order_customers: int
    repeat_customers: int
    single_order_pct: float
    repeat_pct: float
    orders_per_customer: List[OrderCountBucket]
    spending: SpendingAnalysis
    rating: RatingAnalysis
    recency: RecencyAnalysis
    new_vs_returning: NewVsReturning


# ----------------------------------------------------------------------------- segments
class Preference(BaseModel):
    name: str
    share_pct: float                    # share of the segment's customers whose favourite is this value


class SegmentRow(BaseModel):
    cluster: int
    segment: str
    description: str
    customers: int
    customers_pct: float
    revenue: float
    revenue_share_pct: float
    orders: int
    average_orders: Optional[float] = None
    average_order_value: Optional[float] = None
    average_customer_spending: Optional[float] = None
    average_recency_days: Optional[float] = None
    average_days_between_orders: Optional[float] = None      # only customers with 2+ orders have a gap
    single_order_pct: float
    preferred_category: Optional[Preference] = None
    preferred_device: Optional[Preference] = None
    preferred_payment: Optional[Preference] = None


class SegmentsResponse(DashboardResponse):
    as_of_date: Optional[date] = None
    total_customers: int
    total_revenue: float
    segments: List[SegmentRow]


class MixRow(BaseModel):
    name: str
    revenue: float
    orders: int
    revenue_share_pct: float


class SegmentDetailResponse(DashboardResponse):
    as_of_date: Optional[date] = None
    segment: SegmentRow
    suggested_actions: List[str]
    category_mix: List[MixRow]
    device_mix: List[MixRow]
    payment_mix: List[MixRow]
    orders_per_customer: List[OrderCountBucket]
    recency_distribution: List[LabelBucket]
    monthly_trend: List[TrendPoint]
