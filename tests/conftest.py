import datetime
import uuid

import pytest

from blazee.client import Client


@pytest.fixture
def client():
    return Client(api_key='TEST_KEY', host='http://test')


@pytest.fixture
def model_resp(model_version_resp):
    def factory(with_default=False):
        resp = {
            "id": str(uuid.uuid4()),
            "name": str(uuid.uuid4()),
            "default_version": None,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }
        if with_default:
            resp['default_version'] = model_version_resp(True)
        return resp
    return factory


@pytest.fixture
def model_version_resp():
    def factory(deployed=True):
        return {
            "id": str(uuid.uuid4()),
            "name": "v1",
            "type": "sklearn",
            "meta": {
                "requirements": {
                    "scikit-learn": "0.20.3"
                }
            },
            "deployed": deployed,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat()
        }
    return factory
