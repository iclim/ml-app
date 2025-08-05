from fastapi import FastAPI
from app.api.routes import router
from app.ml.model import model_registry, ModelOptions

app = FastAPI(
    title="ML Model Serving API",
    description="A FastAPI application for serving machine learning models",
    version="1.0.0",
)


# Load the model on startup
@app.on_event("startup")
async def startup_event():
    model_registry.get_model(ModelOptions.iris).load_model()
    model_registry.get_model(ModelOptions.diabetes).load_model()


# Include the API routes
app.include_router(router, prefix="/api/v1", tags=["predictions"])


@app.get("/")
async def root():
    return {"message": "ML Model Serving API", "version": "1.0.0"}
