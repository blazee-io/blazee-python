import uuid

import pytest
import responses
from requests.exceptions import HTTPError

from .conftest import client, model_resp


@responses.activate
def test_all_models_error(client):
    responses.add(responses.GET, 'http://test/models',
                  json=[], status=500)

    with pytest.raises(HTTPError):
        client.all_models()


@responses.activate
def test_all_models_empty(client):
    responses.add(responses.GET, 'http://test/models',
                  json=[], status=200)

    assert client.all_models() == []


@responses.activate
def test_all_models(client, model_resp):
    responses.add(responses.GET, 'http://test/models',
                  json=[
                      model_resp(with_default=False),
                      model_resp(with_default=True)
                  ], status=200)

    models = client.all_models()
    assert len(models) == 2
    assert models[0].default_version == None
    assert models[1].default_version != None


def test_get_model_invalid_id(client):
    with pytest.raises(ValueError):
        client.get_model('my-id')


@responses.activate
def test_get_model_error(client):
    model_id = str(uuid.uuid4())
    responses.add(responses.GET, f'http://test/models/{model_id}',
                  status=500)
    with pytest.raises(HTTPError):
        client.get_model(model_id)


@responses.activate
def test_get_model_missing(client):
    model_id = str(uuid.uuid4())
    responses.add(responses.GET, f'http://test/models/{model_id}',
                  json={
                      "error": {
                          "code": "MODEL_NOT_FOUND",
                          "message": "Model does not exist",
                          "details": []
                      }
                  }, status=404)
    with pytest.raises(HTTPError):
        client.get_model(model_id)


@responses.activate
def test_get_model(client, model_resp):
    model_id = str(uuid.uuid4())
    responses.add(responses.GET, f'http://test/models/{model_id}',
                  json=model_resp(True), status=200)

    model = client.get_model(model_id)
    assert model.default_version != None
