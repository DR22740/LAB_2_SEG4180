import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_no_auth(client):
    response = client.post('/predict', json={"text": "Hello"})
    assert response.status_code == 401