from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from app.ml.model import ml_model

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    try:
        prediction, prediction_id, confidence = ml_model.predict(request.features)

        return PredictionResponse(
            prediction=prediction,
            prediction_id=prediction_id,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        results = ml_model.predict_batch(request.samples)

        predictions = [
            PredictionResponse(
                prediction=pred,
                prediction_id=pred_id,
                confidence=conf
            )
            for pred, pred_id, conf in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = ml_model.get_model_info()

    return HealthResponse(
        status="healthy" if ml_model.is_loaded else "unhealthy",
        model_loaded=ml_model.is_loaded,
        feature_count=model_info.get('n_features', 0),
        model_type=model_info.get('model_type', 'unknown')
    )


@router.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    if not ml_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ml_model.get_model_info()