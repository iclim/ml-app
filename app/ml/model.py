import joblib
import numpy as np
from typing import List, Tuple
import os


class MLModel:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.is_loaded = False

    def load_model(self):
        """Load the trained model and metadata"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
            metadata_path = os.path.join(os.path.dirname(__file__), "model_metadata.pkl")

            self.model = joblib.load(model_path)
            self.metadata = joblib.load(metadata_path)
            self.is_loaded = True
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False

    def predict(self, features: List[float]) -> Tuple[str, int, float]:
        """Make a single prediction"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        # Convert to numpy array and reshape for single prediction
        X = np.array(features).reshape(1, -1)

        # Get prediction and probability
        prediction_id = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(max(probabilities))

        # Get species name
        prediction_name = self.metadata['target_names'][prediction_id]

        return prediction_name, int(prediction_id), confidence

    def predict_batch(self, samples: List[List[float]]) -> List[Tuple[str, int, float]]:
        """Make batch predictions"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        # Convert to numpy array
        X = np.array(samples)

        # Get predictions and probabilities
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        for i, (pred_id, probs) in enumerate(zip(predictions, probabilities)):
            prediction_name = self.metadata['target_names'][pred_id]
            confidence = float(max(probs))
            results.append((prediction_name, int(pred_id), confidence))

        return results

    def get_model_info(self) -> dict:
        """Get model metadata"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}

        return {
            "feature_names": self.metadata['feature_names'],
            "target_names": self.metadata['target_names'],
            "n_features": self.metadata['n_features'],
            "model_type": type(self.model).__name__
        }


# Create global model instance
ml_model = MLModel()