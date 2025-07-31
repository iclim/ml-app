import joblib
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import os


def train_iris_model():
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
    joblib.dump(model, "app/ml/saved_models/iris_model.pkl")

    # Save feature names and target names for later use
    model_metadata = {
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "n_features": len(iris.feature_names),
    }
    joblib.dump(model_metadata, "app/ml/saved_models/iris_metadata.pkl")

    print("Iris model saved successfully!")


def train_diabetes_model():
    diabetes = load_diabetes()

    X, y = diabetes.data, diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    y_diff = y_test - predictions

    print("Model Performance:")
    print(f"R² Score: {r2_score(y_test, predictions)}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")
    print(f"Mean Residuals: {np.mean(y_diff)}")
    print(f"Std Residuals: {np.std(y_diff)}")

    joblib.dump(model, "app/ml/saved_models/diabetes_model.pkl")

    model_metadata = {
        "feature_names": diabetes.feature_names,
        "target": "disease_progression",
        "n_features": len(diabetes.feature_names),
    }
    joblib.dump(model_metadata, "app/ml/saved_models/diabetes_metadata.pkl")

    print("Diabetes model saved successfully!")




if __name__ == "__main__":
    os.makedirs("app/ml/saved_models", exist_ok=True)
    train_iris_model()
    train_diabetes_model()
