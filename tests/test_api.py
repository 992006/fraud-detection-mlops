from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_placeholder():
    payload = {
        "amount": 149.50,
        "time": 45231,
        "features": {
            "V1": -1.21,
            "V2": 0.83,
        },
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "fraud_probability" in response.json()
    assert "prediction" in response.json()