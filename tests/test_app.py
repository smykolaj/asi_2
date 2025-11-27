import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from predict import make_prediction

def test_make_prediction_setosa():
    params = [[5.1, 3.5, 1.4, 0.2]]

    result = make_prediction(params)

    assert isinstance(result, str)
    assert result == "setosa"


def test_make_prediction_versicolor():
    params = [[7.0, 3.2, 4.7, 1.4]]

    result = make_prediction(params)

    assert isinstance(result, str)
    assert result == "versicolor"


def test_make_prediction_virginica():
    params = [[6.3, 3.3, 6.0, 2.5]]

    result = make_prediction(params)

    assert isinstance(result, str)
    assert result == "virginica"