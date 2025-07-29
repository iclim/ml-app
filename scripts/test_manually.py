"""
Manual testing script for the ML API
Run this while your server is running to test endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_single_prediction():
    """Test single prediction"""
    print("🎯 Testing single prediction...")
    data = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request: {data}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_batch_prediction():
    """Test batch prediction"""
    print("📊 Testing batch prediction...")
    data = {"samples": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print(f"Status: {response.status_code}")
    print(f"Request: {data}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Test model info endpoint"""
    print("ℹ️  Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


if __name__ == "__main__":
    print("🚀 Starting API tests...\n")

    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        print("✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   uvicorn app.main:app --reload")