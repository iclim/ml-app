from pydantic import BaseModel, Field, validator
from typing import List, Optional


class IrisRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four iris features: sepal_length, sepal_width, petal_length, petal_width",
    )


class DiabetesRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Ten diabetes features: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6",
    )


class BatchIrisPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch iris prediction",
        examples=[[[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]],
    )

    @validator("samples")
    def validate_sample_features(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 4:
                raise ValueError(
                    f"Sample {i} must have exactly 4 features, got {len(sample)}"
                )
        return v


class BatchDiabetesPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch diabetes prediction",
        examples=[
            [
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
            ]
        ],
    )

    @validator("samples")
    def validate_sample_features(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 10:
                raise ValueError(
                    f"Sample {i} must have exactly 10 features, got {len(sample)}"
                )
        return v


class IrisResponse(BaseModel):
    prediction: str = Field(..., description="Predicted iris species")
    prediction_id: int = Field(..., description="Numeric prediction (0, 1, or 2)")
    confidence: float = Field(..., description="Model confidence score")


class DiabetesResponse(BaseModel):
    prediction: float = Field(..., description="Predicted diabetes progression")
    confidence: Optional[float] = Field(None, description="Model confidence score")


class BatchIrisResponse(BaseModel):
    predictions: List[IrisResponse]
    batch_size: int


class BatchDiabetesResponse(BaseModel):
    predictions: List[DiabetesResponse]
    batch_size: int


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    feature_count: int = Field(..., description="Number of features expected")
    model_type: str = Field(..., description="Type and name of the model")
