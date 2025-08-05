"""
Manual testing script for the ML API
Run this while your server is running to test endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_root():
    """Test root endpoint to see available models"""
    print("🏠 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_health_endpoints():
    """Test health endpoints for both models"""
    print("🔍 Testing health endpoints...")

    models = ["iris", "diabetes"]
    for model in models:
        print(f"  Testing {model} health...")
        response = requests.get(f"{BASE_URL}/{model}/health")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"  Error: {response.text}")
        print()


def test_iris_prediction():
    """Test iris single prediction"""
    print("🌸 Testing iris single prediction...")
    data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Should be setosa
    response = requests.post(f"{BASE_URL}/iris/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request: {data}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


def test_diabetes_prediction():
    """Test diabetes single prediction"""
    print("🩺 Testing diabetes single prediction...")
    # Example diabetes features (10 features)
    data = {"features": [0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.020, -0.018]}
    response = requests.post(f"{BASE_URL}/diabetes/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request: {data}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


def test_iris_batch_prediction():
    """Test iris batch prediction"""
    print("🌸📊 Testing iris batch prediction...")
    data = {
        "samples": [
            [5.1, 3.5, 1.4, 0.2],  # Should be setosa
            [6.2, 2.9, 4.3, 1.3],  # Should be versicolor
            [7.3, 2.9, 6.3, 1.8]   # Should be virginica
        ]
    }
    response = requests.post(f"{BASE_URL}/iris/predict/batch", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(data, indent=2)}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


def test_diabetes_batch_prediction():
    """Test diabetes batch prediction"""
    print("🩺📊 Testing diabetes batch prediction...")
    data = {
        "samples": [
            [0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.020, -0.018],
            [-0.002, -0.045, -0.051, -0.026, -0.008, -0.019, 0.074, -0.039, -0.068, -0.092]
        ]
    }
    response = requests.post(f"{BASE_URL}/diabetes/predict/batch", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(data, indent=2)}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


def test_model_info():
    """Test model info endpoints"""
    print("ℹ️  Testing model info endpoints...")

    models = ["iris", "diabetes"]
    for model in models:
        print(f"  Testing {model} info...")
        response = requests.get(f"{BASE_URL}/{model}/info")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"  Error: {response.text}")
        print()


def test_error_cases():
    """Test error handling"""
    print("❌ Testing error cases...")

    # Test wrong number of features for iris
    print("  Testing iris with wrong number of features...")
    data = {"features": [5.1, 3.5]}  # Only 2 features instead of 4
    response = requests.post(f"{BASE_URL}/iris/predict", json=data)
    print(f"  Status: {response.status_code}")
    print(f"  Error response: {response.text}")
    print()

    # Test wrong number of features for diabetes
    print("  Testing diabetes with wrong number of features...")
    data = {"features": [0.038, 0.051, 0.062]}  # Only 3 features instead of 10
    response = requests.post(f"{BASE_URL}/diabetes/predict", json=data)
    print(f"  Status: {response.status_code}")
    print(f"  Error response: {response.text}")
    print()

    # Test non-existent model
    print("  Testing non-existent model...")
    response = requests.get(f"{BASE_URL}/nonexistent/health")
    print(f"  Status: {response.status_code}")
    print(f"  Error response: {response.text}")
    print()


if __name__ == "__main__":
    print("🚀 Starting ML API tests...\n")

    try:
        test_root()
        test_health_endpoints()
        test_iris_prediction()
        test_diabetes_prediction()
        test_iris_batch_prediction()
        test_diabetes_batch_prediction()
        test_model_info()
        test_error_cases()
        print("✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")