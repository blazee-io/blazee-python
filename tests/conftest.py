import pytest

from blazee.client import Client

TEST_API_KEY = 'TestApiKey'


@pytest.fixture
def client():
    return Client(TEST_API_KEY)
