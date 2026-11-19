"""
Dashboard endpoints (all read-only GET requests under /api/dashboard).

Common filters (all optional, combine freely):
  start_date, end_date  inclusive, YYYY-MM-DD
  category, city, payment_method, device_type   exact value (case-insensitive); see GET /api/dashboard/filters
  segment               customer segment id (0-3)

The segment endpoints (/segments and /segment/{id}) do not take `segment`, because they already split by segment.
"""
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from backend.schemas.dashboard import (CategoryPerformanceResponse, CityPerformanceResponse, CustomerAnalysisResponse,
                                       DashboardHealthResponse, DeviceAnalysisResponse, FilterOptionsResponse,
                                       PaymentAnalysisResponse, SalesTrendResponse, SegmentDetailResponse,
                                       SegmentsResponse, SummaryResponse)
from backend.services.dashboard_service import DashboardFilters, DashboardService, get_dashboard_service

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

ERROR_RESPONSES = {422: {"description": "Invalid filter value"}, 503: {"description": "Dashboard data not available"}}


def get_service() -> DashboardService:
    """Dependency (tests can replace it)."""
    return get_dashboard_service()


def common_filters(
    start_date: Optional[date] = Query(None, description="First day to include (YYYY-MM-DD)."),
    end_date: Optional[date] = Query(None, description="Last day to include (YYYY-MM-DD)."),
    category: Optional[str] = Query(None, description="Product category, e.g. Electronics."),
    city: Optional[str] = Query(None, description="City, e.g. Istanbul."),
    payment_method: Optional[str] = Query(None, description="Payment method, e.g. Credit Card."),
    device_type: Optional[str] = Query(None, description="Device type: Desktop, Mobile or Tablet."),
    segment: Optional[int] = Query(None, description="Customer segment (cluster) id, 0 to 3."),
    service: DashboardService = Depends(get_service),
) -> DashboardFilters:
    return service.build_filters(start_date, end_date, category, city, payment_method, device_type, segment)


def filters_without_segment(
    start_date: Optional[date] = Query(None, description="First day to include (YYYY-MM-DD)."),
    end_date: Optional[date] = Query(None, description="Last day to include (YYYY-MM-DD)."),
    category: Optional[str] = Query(None, description="Product category, e.g. Electronics."),
    city: Optional[str] = Query(None, description="City, e.g. Istanbul."),
    payment_method: Optional[str] = Query(None, description="Payment method, e.g. Credit Card."),
    device_type: Optional[str] = Query(None, description="Device type: Desktop, Mobile or Tablet."),
    service: DashboardService = Depends(get_service),
) -> DashboardFilters:
    return service.build_filters(start_date, end_date, category, city, payment_method, device_type, None)


@router.get("/health", response_model=DashboardHealthResponse, summary="Dashboard data status")
def dashboard_health(service: DashboardService = Depends(get_service)):
    h = service.health()
    if h["status"] != "ok":
        return JSONResponse(status_code=503, content=jsonable_encoder(h))
    return h


@router.get("/filters", response_model=FilterOptionsResponse, responses=ERROR_RESPONSES,
            summary="Values available for the filters (for dropdowns)")
def filter_options(service: DashboardService = Depends(get_service)):
    return service.filter_options()


@router.get("/summary", response_model=SummaryResponse, responses=ERROR_RESPONSES,
            summary="Headline KPIs")
def summary(f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.summary(f)


@router.get("/sales-trend", response_model=SalesTrendResponse, responses=ERROR_RESPONSES,
            summary="Revenue and orders over time")
def sales_trend(granularity: Literal["day", "week", "month"] = Query("month", description="Size of each time bucket."),
                f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.sales_trend(f, granularity)


@router.get("/category-performance", response_model=CategoryPerformanceResponse, responses=ERROR_RESPONSES,
            summary="Revenue, orders and average order value by product category")
def category_performance(f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.category_performance(f)


@router.get("/city-performance", response_model=CityPerformanceResponse, responses=ERROR_RESPONSES,
            summary="Revenue and orders by city")
def city_performance(f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.city_performance(f)


@router.get("/payment-analysis", response_model=PaymentAnalysisResponse, responses=ERROR_RESPONSES,
            summary="Revenue and orders by payment method")
def payment_analysis(f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.payment_analysis(f)


@router.get("/device-analysis", response_model=DeviceAnalysisResponse, responses=ERROR_RESPONSES,
            summary="Revenue, orders and engagement by device type")
def device_analysis(f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.device_analysis(f)


@router.get("/customer-analysis", response_model=CustomerAnalysisResponse, responses=ERROR_RESPONSES,
            summary="Orders per customer, spending, ratings, recency and new vs returning orders")
def customer_analysis(f: DashboardFilters = Depends(common_filters), service: DashboardService = Depends(get_service)):
    return service.customer_analysis(f)


@router.get("/segments", response_model=SegmentsResponse, responses=ERROR_RESPONSES,
            summary="All four customer segments side by side (saved production K-Means result)")
def segments(f: DashboardFilters = Depends(filters_without_segment), service: DashboardService = Depends(get_service)):
    return service.segments(f)


@router.get("/segment/{cluster_id}", response_model=SegmentDetailResponse,
            responses={**ERROR_RESPONSES, 404: {"description": "Unknown cluster id"}},
            summary="Detail page data for one segment")
def segment_detail(cluster_id: int = Path(..., description="Segment (cluster) id, 0 to 3."),
                   f: DashboardFilters = Depends(filters_without_segment),
                   service: DashboardService = Depends(get_service)):
    return service.segment_detail(cluster_id, f)
