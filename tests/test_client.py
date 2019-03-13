import pytest
from requests.exceptions import HTTPError
from sklearn.linear_model import LogisticRegressionCV

from blazee.client import Client

from .conftest import client


def test_deploy_invalid_model(client):
    with pytest.raises(TypeError):
        client.deploy_model(object())


def test_deploy_sklearn_model(client):
    clf = LogisticRegressionCV()

    client.deploy_model(clf)


def test_invalid_api_key():
    client = Client('InvalidApiKey')
    clf = LogisticRegressionCV()

    with pytest.raises(HTTPError):
        client.deploy_model(clf)
