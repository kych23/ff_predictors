from __future__ import annotations

from fastapi import APIRouter

from api.schemas import PredictRequest, PredictResponse
from api.services.prediction_service import predict_points


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return predict_points(req)


