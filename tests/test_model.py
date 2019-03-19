
import pytest
from requests.exceptions import HTTPError
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegressionCV

from blazee import model

from .conftest import client


def test_serialize_invalid_model(client):
    with pytest.raises(TypeError):
        model._serialize_model(object())


def test_serialize_custom_model(client):
    class CustomModel(BaseEstimator):
        def fit(self, X):
            pass

        def predict(self, X):
            return [x + 1 for x in X]

    custom = CustomModel()
    with pytest.raises(TypeError):
        model._serialize_model(custom)


def test_serialize_untrained_model():
    clf = LogisticRegressionCV()

    with pytest.raises(AttributeError):
        model._serialize_model(clf)
