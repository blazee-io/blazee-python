import pytest
from sklearn.linear_model import LogisticRegressionCV

from blazee.client import Client

from .conftest import client


def test_deploy_invalid_model(client):
    with pytest.raises(TypeError):
        client.deploy(object())


def test_deploy_sklearn_model(client):
    clf = LogisticRegressionCV()

    client.deploy(clf)
