from fastapi.testclient import TestClient

from flightbridge.api import create_app
from flightbridge.service import FlightBridge


def test_health_readiness_validation_and_analysis():
    with TestClient(create_app(FlightBridge())) as client:
        assert client.get("/health").json()["model_ready"] is False
        assert client.get("/ready").status_code == 503
        assert client.post("/analyze", json={"message": ""}).status_code == 422
        assert client.post("/analyze", json={"message": " "}).status_code == 422
        assert client.post("/analyze", json={"message": "a" * 4001}).status_code == 422
        assert (
            client.post("/analyze", json={"message": "I want a refund"}).json()["facts"]["intent"] == "REFUND"
        )
        body = {
            "facts": {
                "airline": "A3",
                "airline_type": "LEGACY",
                "alternative_provided": "YES",
                "delay_minutes": 30,
            },
            "confirmed": True,
        }
        assert client.post("/predict", json=body).status_code == 503
        body["facts"]["sc_type"] = "MA SC"
        assert client.post("/predict", json=body).status_code == 422
