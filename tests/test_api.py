from fastapi.testclient import TestClient
from api.main import app

FULL_PAYLOAD = {
    "Time": 1453.0,
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378, "V5": -0.338,
    "V6": 0.462, "V7": 0.239, "V8": 0.098, "V9": 0.363, "V10": 0.155,
    "V11": -0.213, "V12": -0.035, "V13": -0.143, "V14": -0.112, "V15": -0.214,
    "V16": 0.141, "V17": -0.069, "V18": 0.059, "V19": -0.042, "V20": -0.015,
    "V21": -0.053, "V22": -0.132, "V23": -0.028, "V24": 0.112, "V25": 0.045,
    "V26": 0.088, "V27": 0.012, "V28": 0.005,
    "Amount": 149.50,
}


def test_root():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json=FULL_PAYLOAD)
        # 200 when model artifacts exist (local), 503 when they don't (CI server)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            body = response.json()
            assert "fraud_probability" in body
            assert "prediction" in body
            assert "threshold_used" in body