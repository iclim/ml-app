import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


def train_and_save_model():
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # Save the model
    os.makedirs("app/ml", exist_ok=True)
    joblib.dump(model, "app/ml/trained_model.pkl")

    # Save feature names and target names for later use
    model_metadata = {
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "n_features": len(iris.feature_names),
    }
    joblib.dump(model_metadata, "app/ml/model_metadata.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    train_and_save_model()
