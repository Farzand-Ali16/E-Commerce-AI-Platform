"""
FastAPI application.

Start from the project root:
    uvicorn backend.app:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.dashboard import router as dashboard_router
from backend.routes.segmentation import router as segmentation_router
from backend.services.dashboard_service import (DashboardError, DashboardNotReadyError, InvalidFilterError,
                                                SegmentNotFoundError, get_dashboard_service)
from backend.services.segmentation_service import SegmentationError, get_segmentation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved model files and the dashboard datasets ONCE when the server starts (never during a request)."""
    get_segmentation_service().load()
    get_dashboard_service().load()
    yield


app = FastAPI(
    title="E-Commerce AI Analytics & Customer Intelligence Platform",
    description="Backend API. Stage 6: customer segment prediction (K-Means, 4 segments). "
                "Stage 7: dashboard analytics.",
    version="0.2.0",
    lifespan=lifespan,
)

# Allows the React/Vite dev server (built in a later stage) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(segmentation_router)
app.include_router(dashboard_router)


# ----------------------------------------------------------------------------- error handling
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Clean 422 body: which field is wrong and why."""
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path")]
        details.append({"field": ".".join(loc) or "request",
                        "message": str(err.get("msg", "Invalid value")).removeprefix("Value error, ")})
    return JSONResponse(status_code=422, content={"error": "validation_error",
                                                  "message": "The request data is not valid.",
                                                  "details": details})


@app.exception_handler(InvalidFilterError)
async def invalid_filter_handler(request: Request, exc: InvalidFilterError):
    """Same 422 shape as other validation errors, e.g. an unknown category or start_date after end_date."""
    return JSONResponse(status_code=422, content={"error": "validation_error",
                                                  "message": "One or more filters are not valid.",
                                                  "details": exc.errors})


@app.exception_handler(SegmentNotFoundError)
async def segment_not_found_handler(request: Request, exc: SegmentNotFoundError):
    return JSONResponse(status_code=404, content={"error": "not_found", "message": str(exc)})


@app.exception_handler(DashboardNotReadyError)
async def dashboard_not_ready_handler(request: Request, exc: DashboardNotReadyError):
    logger.error("Dashboard request while data is unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"error": "data_unavailable", "message": str(exc)})


@app.exception_handler(DashboardError)
async def dashboard_error_handler(request: Request, exc: DashboardError):
    logger.error("Unhandled dashboard error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "dashboard_error",
                                                  "message": "The dashboard service failed."})


@app.exception_handler(SegmentationError)
async def segmentation_error_handler(request: Request, exc: SegmentationError):
    logger.error("Unhandled segmentation error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "segmentation_error",
                                                  "message": "The segmentation service failed."})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error",
                                                  "message": "Something went wrong on the server."})


@app.get("/", tags=["Info"], summary="API info")
def root():
    return {"name": app.title, "version": app.version, "docs": "/docs",
            "endpoints": ["POST /api/segmentation/predict", "GET /api/segmentation/health",
                          "GET /api/dashboard/summary", "GET /api/dashboard/sales-trend",
                          "GET /api/dashboard/category-performance", "GET /api/dashboard/city-performance",
                          "GET /api/dashboard/payment-analysis", "GET /api/dashboard/device-analysis",
                          "GET /api/dashboard/customer-analysis", "GET /api/dashboard/segments",
                          "GET /api/dashboard/segment/{cluster_id}", "GET /api/dashboard/filters",
                          "GET /api/dashboard/health"]}
