import pytest

from blazee.client import Client

# TEST_API_KEY = 'TestApiKey'
TEST_API_KEY = '8ce0a16c-b517-45c8-8155-6c5c137e25c0'


@pytest.fixture
def client():
    return Client(TEST_API_KEY)
