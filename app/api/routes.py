from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    IrisRequest,
    DiabetesRequest,
    IrisResponse,
    DiabetesResponse,
    BatchIrisPredictionRequest,
    BatchDiabetesPredictionRequest,
    BatchIrisResponse,
    BatchDiabetesResponse,
    HealthResponse,
)
from app.ml.model import model_registry, ModelOptions

router = APIRouter()


@router.post("/iris/predict", response_model=IrisResponse)
async def predict_iris(request: IrisRequest):
    """Make a single iris prediction"""
    try:
        ml_model = model_registry.get_model(ModelOptions.iris)
        prediction, prediction_id, confidence = ml_model.predict(request.features)

        return IrisResponse(
            prediction=prediction, prediction_id=prediction_id, confidence=confidence
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/diabetes/predict", response_model=DiabetesResponse)
async def predict_diabetes(request: DiabetesRequest):
    """Make a single diabetes prediction"""
    try:
        ml_model = model_registry.get_model(ModelOptions.diabetes)
        prediction, prediction_id, confidence = ml_model.predict(request.features)

        return DiabetesResponse(
            prediction=float(prediction),  # Ensure it's a float for regression
            confidence=confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/iris/predict/batch", response_model=BatchIrisResponse)
async def predict_iris_batch(request: BatchIrisPredictionRequest):
    """Make batch iris predictions"""
    try:
        ml_model = model_registry.get_model(ModelOptions.iris)
        results = ml_model.predict_batch(request.samples)

        predictions = [
            IrisResponse(prediction=pred, prediction_id=pred_id, confidence=conf)
            for pred, pred_id, conf in results
        ]

        return BatchIrisResponse(predictions=predictions, batch_size=len(predictions))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/diabetes/predict/batch", response_model=BatchDiabetesResponse)
async def predict_diabetes_batch(request: BatchDiabetesPredictionRequest):
    """Make batch diabetes predictions"""
    try:
        ml_model = model_registry.get_model(ModelOptions.diabetes)
        results = ml_model.predict_batch(request.samples)

        predictions = [
            DiabetesResponse(prediction=float(pred), confidence=conf)
            for pred, pred_id, conf in results
        ]

        return BatchDiabetesResponse(
            predictions=predictions, batch_size=len(predictions)
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/{model_option}/health", response_model=HealthResponse)
async def health_check(model_option: ModelOptions):
    """Health check endpoint for any model"""
    try:
        ml_model = model_registry.get_model(model_option)
        model_info = ml_model.get_model_info()

        return HealthResponse(
            status="healthy" if ml_model.is_loaded else "unhealthy",
            model_loaded=ml_model.is_loaded,
            feature_count=model_info.get("n_features", 0),
            model_type=f"{model_option.value}_{model_info.get('model_type', 'unknown')}",
        )
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Model {model_option.value} not found"
        )


@router.get("/{model_option}/info")
async def get_model_info(model_option: ModelOptions):
    """Get detailed model information"""
    try:
        ml_model = model_registry.get_model(model_option)

        if not ml_model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return ml_model.get_model_info()
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Model {model_option.value} not found"
        )


@router.get("/")
async def list_models():
    """List all available models"""
    return {
        "available_models": [option.value for option in ModelOptions],
        "endpoints": {
            "iris": {
                "predict": "/iris/predict",
                "batch_predict": "/iris/predict/batch",
                "health": "/iris/health",
                "info": "/iris/info",
            },
            "diabetes": {
                "predict": "/diabetes/predict",
                "batch_predict": "/diabetes/predict/batch",
                "health": "/diabetes/health",
                "info": "/diabetes/info",
            },
        },
    }
