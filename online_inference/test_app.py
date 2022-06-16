from fastapi.testclient import TestClient
from app import app
import pytest

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_main(client):
    response = client.get("/")
    assert 200 == response.status_code
    assert response.json() == 'Andreev Danil Homework2'


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict(client):
    request_data = [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0]
    request_features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                        'fbs', 'restecg', 'thalach', 'exang',
                        'oldpeak', 'slope', 'ca', 'thal']

    response = client.get(
        "/predict",
        json={"data": [request_data], "features": request_features},
    )
    assert response.status_code == 200
    assert response.json()[0] == {'condition': 0}