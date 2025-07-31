import joblib
import numpy as np
from typing import List, Tuple
import os


class MLModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.model_type = None
        self.model = None
        self.metadata = None
        self.is_loaded = False

    def load_model(self):
        """Load the trained model and metadata"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), f"{self.dataset_name}_model.pkl")
            metadata_path = os.path.join(os.path.dirname(__file__), f"{self.dataset_name}_metadata.pkl")

            self.model = joblib.load(model_path)
            self.metadata = joblib.load(metadata_path)
            self.model_type = self.metadata['model_type']
            self.is_loaded = True
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False

    def predict(self, features: List[float]) -> Tuple[str, int, float] | float:
        """Make a single prediction"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        # Convert to numpy array and reshape for single prediction
        X = np.array(features).reshape(1, -1)

        if self.model_type == "classification":
            # Get prediction and probability
            prediction_id = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(max(probabilities))

            # Get species name
            prediction_name = self.metadata['target_names'][prediction_id]

            return prediction_name, int(prediction_id), confidence
        elif self.model_type == "regression":
            prediction = self.model.predict(X)[0]
            ## todo: justify a confidence measurement and implement

            return prediction
        else:
            raise ValueError("Invalid model type")

    def predict_batch(self, samples: List[List[float]]) -> List[Tuple[str, int, float]] | List[float]:
        """Make batch predictions"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        # Convert to numpy array
        X = np.array(samples)

        if self.model_type == "classification":
            # Get predictions and probabilities
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            results = []
            for i, (pred_id, probs) in enumerate(zip(predictions, probabilities)):
                prediction_name = self.metadata['target_names'][pred_id]
                confidence = float(max(probs))
                results.append((prediction_name, int(pred_id), confidence))
            return results
        elif self.model_type == "regression":
            predictions = self.model.predict(X)
            return predictions
        else:
            raise ValueError("Invalid model type")

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
