"""
Segmentation endpoints:
  POST /api/segmentation/predict
  GET  /api/segmentation/health
"""
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.schemas.segmentation import (ErrorResponse, HealthResponse, SegmentationRequest,
                                          SegmentationResponse)
from backend.services.segmentation_service import (ModelNotLoadedError, PredictionError,
                                                   SegmentationService, get_segmentation_service)

logger = logging.getLogger("segmentation_routes")

router = APIRouter(prefix="/api/segmentation", tags=["Customer segmentation"])


def get_service() -> SegmentationService:
    """Dependency (tests can replace it with a different service)."""
    return get_segmentation_service()


@router.post(
    "/predict",
    response_model=SegmentationResponse,
    summary="Predict the segment of one customer",
    responses={422: {"model": ErrorResponse, "description": "Invalid input"},
               500: {"model": ErrorResponse, "description": "Prediction failed"},
               503: {"model": ErrorResponse, "description": "Model not available"}},
)
def predict_segment(payload: SegmentationRequest, service: SegmentationService = Depends(get_service)):
    """Uses the saved preprocessing pipeline and the saved K-Means model. Nothing is trained during the request."""
    try:
        result = service.predict(payload.model_dump())
    except ModelNotLoadedError as exc:
        logger.error("Predict called but model unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"error": "model_unavailable", "message": str(exc)})
    except PredictionError as exc:
        logger.error("Prediction error: %s", exc)
        return JSONResponse(status_code=500, content={"error": "prediction_failed",
                                                      "message": "The prediction could not be completed."})
    return result


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check that the model, preprocessor and metadata are loaded",
    responses={503: {"model": HealthResponse, "description": "Something is not loaded"}},
)
def segmentation_health(service: SegmentationService = Depends(get_service)):
    health = service.health()
    if health["status"] != "ok":
        return JSONResponse(status_code=503, content=health)
    return health
