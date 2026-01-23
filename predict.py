import os

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

filename = "model.joblib"
target = ["setosa", "versicolor", "virginica"]


def train_model():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    joblib.dump(model, filename)


def get_model():
    if not os.path.exists(filename):
        train_model()
    return joblib.load("model.joblib")


def make_prediction(params):
    prediction = get_model().predict(params)
    flower = target[prediction[0]]
    return flower
