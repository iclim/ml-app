from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four iris features: sepal_length, sepal_width, petal_length, petal_width",
        examples=[5.1, 3.5, 1.4, 0.2],
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted iris species")
    prediction_id: int = Field(..., description="Numeric prediction (0, 1, or 2)")
    confidence: float = Field(..., description="Model confidence score")


class BatchPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch prediction",
        examples=[[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]],
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_count: int
    model_type: str
