from pydantic import BaseModel, Field
from typing import List


class IrisRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four iris features: sepal_length, sepal_width, petal_length, petal_width",
        examples=[5.1, 3.5, 1.4, 0.2],
    )


class DiabetesRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="10 diabetes features: age, sex, bmi, bp average blood pressure, 6 blood test measurements",
    )


class IrisResponse(BaseModel):
    prediction: str = Field(..., description="Predicted iris species")
    prediction_id: int = Field(..., description="Numeric prediction (0, 1, or 2)")
    confidence: float = Field(..., description="Model confidence score")


class DiabetesResponse(BaseModel):
    prediction: float = Field(..., description="Predicted diabetes species")
    ## todo: once a confidence or uncertainty measure is implemented handle that field here too.


class BatchIrisPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch iris prediction",
        examples=[[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]],
    )


class BatchDiabetesPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch diabetes prediction",
        examples=[
            [
                0.03807591,
                0.05068012,
                0.06169621,
                0.02187239,
                -0.0442235,
                -0.03482076,
                -0.04340085,
                -0.00259226,
                0.01990749,
                -0.01764613,
            ],
            [
                -0.00188202,
                -0.04464164,
                -0.05147406,
                -0.02632753,
                -0.00844872,
                -0.01916334,
                0.07441156,
                -0.03949338,
                -0.06833155,
                -0.09220405,
            ],
        ],
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[IrisResponse | DiabetesResponse]
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_count: int
    model_type: str
