import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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

mlflow.set_tracking_uri("file:mlruns")
EXPERIMENT_NAME = "iris-model-zoo"
REGISTERED_MODEL_NAME = "IrisModel"
VERSION = "v1.0.0"


def train_model():
    if not os.path.exists("app"):
        os.makedirs("app")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=15),
        "LogisticRegression": LogisticRegression(max_iter=20),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=2),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_f1 = -1
    best_run_id = None
    best_model_name = None
    best_metrics = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            joblib_path = f"{name}_model.joblib"
            joblib.dump(model, joblib_path)
            mlflow.log_artifact(joblib_path)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric(
                "precision", precision_score(y_test, y_pred, average="macro")
            )
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))
            mlflow.set_tag("version", VERSION)

            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            with open("classification_report.txt", "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact("classification_report.txt")

            mlflow.sklearn.log_model(model, registered_model_name=REGISTERED_MODEL_NAME)

            if f1 > best_f1:
                best_f1 = f1
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = name
                best_metrics = {"accuracy": acc, "f1_macro": f1}

    joblib.dump(models[best_model_name], "app/model.joblib")
    meta = {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "mlflow_run_id": best_run_id,
        "version": VERSION,
    }
    with open("app/model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"✅ Best Model: {best_model_name} saved to app/ folder.")


if __name__ == "__main__":
    train_model()
