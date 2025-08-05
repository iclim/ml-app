# ML Model Serving API

A FastAPI-based REST API for serving machine learning models, supporting both classification and regression tasks.

## 🎯 Project Purpose

This project is a **learning exercise** to gain hands-on experience with FastAPI and production ML deployment patterns. It demonstrates core concepts including:

- RESTful API design with FastAPI
- Pydantic models for request/response validation
- ML model serving and management
- Error handling and API documentation
- Project structure for scalable ML services

## 🚀 Features

- **Multi-model support**: Serve both classification (Iris) and regression (Diabetes) models
- **Batch predictions**: Process multiple samples in a single request
- **Automatic validation**: Input validation with clear error messages
- **Health checks**: Monitor model status and metadata
- **Interactive documentation**: Auto-generated API docs with Swagger UI
- **Type safety**: Full type hints and Pydantic validation

## 📋 API Endpoints

### Root
- `GET /api/v1/` - List all available models and endpoints

### Iris Classification Model
- `POST /api/v1/iris/predict` - Single iris classification
- `POST /api/v1/iris/predict/batch` - Batch iris classification
- `GET /api/v1/iris/health` - Iris model health check
- `GET /api/v1/iris/info` - Iris model information

### Diabetes Regression Model
- `POST /api/v1/diabetes/predict` - Single diabetes progression prediction
- `POST /api/v1/diabetes/predict/batch` - Batch diabetes predictions
- `GET /api/v1/diabetes/health` - Diabetes model health check
- `GET /api/v1/diabetes/info` - Diabetes model information

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train and save models**
   ```bash
   python scripts/train_model.py
   ```

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## 📁 Project Structure

```
ml-app/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API endpoints
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py            # ML model classes
│   │   │──saved_models/        # *.pkl model files
│   └── models/
│       ├── __init__.py
│       └── schemas.py          # Pydantic models
├── scripts/
│   ├── train_model.py          # Train available models
│   └── test_manually.py        # Manual testing script
├── requirements.txt
└── README.md
```

## 🧪 Testing

### Manual Testing
Run the comprehensive test script:
```bash
python scripts/test_manually.py
```

### Example Requests

**Iris Classification:**
```bash
curl -X POST "http://localhost:8000/api/v1/iris/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Diabetes Regression:**
```bash
curl -X POST "http://localhost:8000/api/v1/diabetes/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.020, -0.018]}'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/api/v1/iris/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"samples": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}'
```

## 📊 Model Details

### Iris Classification
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Algorithm**: Random Forest Classifier
- **Output**: Species name, prediction ID, confidence score

### Diabetes Regression
- **Features**: 10 (age, sex, BMI, blood pressure, 6 serum measurements)
- **Target**: Disease progression (continuous value)
- **Algorithm**: Linear Regression
- **Output**: Predicted progression value

## 🚦 Response Examples

### Successful Iris Prediction
```json
{
  "prediction": "setosa",
  "prediction_id": 0,
  "confidence": 1.0
}
```

### Successful Diabetes Prediction
```json
{
  "prediction": 142.5,
}
```

### Error Response
```json
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "features"],
      "msg": "List should have at least 4 items after validation, not 2",
      "input": [5.1, 3.5]
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)

### Model Configuration
Models are automatically loaded on startup. Configuration is stored in the metadata files alongside each model.

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure model files exist in `app/ml/`
   - Check file permissions
   - Verify metadata format matches expected structure

2. **Validation errors**
   - Check feature count (4 for iris, 10 for diabetes)
   - Ensure all features are numeric
   - Verify request JSON format

3. **Import errors**
   - Activate virtual environment
   - Install all requirements
   - Check Python version compatibility

## 📚 Learning Resources

This project demonstrates several FastAPI concepts:

- **Path parameters**: `/{model_option}/predict`
- **Request/Response models**: Pydantic schemas
- **Dependency injection**: Model registry pattern
- **Error handling**: HTTP exceptions
- **Startup events**: Model loading
- **API documentation**: Automatic OpenAPI generation

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Please feel free to:

- Report bugs
- Suggest improvements
- Share alternative approaches
- Add new model types

## 📄 License

This project is for educational purposes. Feel free to use it as a reference for your own FastAPI learning journey.

---

**Built with FastAPI** 🚀 | **Learning ML Deployment** 📚 | **Serving Models at Scale** ⚡