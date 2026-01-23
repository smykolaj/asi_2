import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def prepare_data(test_size, random_state):
    """
    Splits the Iris dataset into training and test sets.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_single_model(X_train, y_train, X_test, y_test, parameters, model_name):
    """
    Trains a single model, logs to MLflow, and returns evaluation data.
    """
    # Instantiate model based on name
    if model_name == "RandomForest":
        model = RandomForestClassifier(**parameters)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(**parameters)
    elif model_name == "SVM":
        model = SVC(**parameters)
    elif model_name == "KNN":
        model = KNeighborsClassifier(**parameters)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # MLflow tracking
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log Params & Metrics
        mlflow.log_params(parameters)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))

        # Confusion Matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
        cm_path = f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Classification Report
        report = classification_report(y_test, y_pred)
        report_path = f"{model_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log Model
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="IrisModel"
        )

        # Clean up local artifacts if needed to avoid cluttering workspace (optional)
        if os.path.exists(cm_path):
            os.remove(cm_path)
        if os.path.exists(report_path):
            os.remove(report_path)

        return {
            "model_name": model_name,
            "model": model,
            "f1_macro": f1,
            "accuracy": acc,
            "run_id": run.info.run_id,
            "metrics": {"accuracy": acc, "f1_macro": f1},
        }


def select_best_model(*evaluations):
    """
    Compares models based on F1-macro score.
    """
    best_eval = max(evaluations, key=lambda x: x["f1_macro"])
    return best_eval


def save_final_assets(best_eval):
    """
    Saves the best model and metadata to the app folder.
    """
    if not os.path.exists("app"):
        os.makedirs("app")

    # Save model
    joblib.dump(best_eval["model"], "app/model.joblib")

    # Save meta
    meta = {
        "best_model": best_eval["model_name"],
        "metrics": best_eval["metrics"],
        "mlflow_run_id": best_eval["run_id"],
        "version": "v1.0.0",  # Hardcoded as in original, or could be parameterized
    }
    with open("app/model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Best Model: {best_eval['model_name']} saved to app/ folder.")
