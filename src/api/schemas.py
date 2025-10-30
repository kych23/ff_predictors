from __future__ import annotations

from pydantic import BaseModel

class PredictRequest(BaseModel):
    player_id: str
    season: int
    week: int
    opponent_team: str


class PredictResponse(BaseModel):
    y_pred: float
    y_std: float | None = None
    model_version: str
    ci_lower: float | None = None
    ci_upper: float | None = None


