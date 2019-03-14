import pytest

from blazee.client import Client


@pytest.fixture
def client():
    return Client()
