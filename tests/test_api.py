"""Tests for FastAPI endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from greenedge.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestDecisionEndpoint:
    """Test /decision endpoint."""

    def test_decision_valid_input(self, client):
        response = client.post(
            "/decision",
            json={"obs": [0.3, 0.5, 0.2, 0.25, 0.8, 0.4]}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "action" in data
        assert data["action"] in [0, 1, 2]
        assert "action_label" in data
        assert data["action_label"] in ["edge-a", "edge-b", "cloud"]
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "fallback_used" in data
        assert "predicted_kpis" in data

    def test_decision_invalid_obs_length(self, client):
        response = client.post(
            "/decision",
            json={"obs": [0.3, 0.5, 0.2]}  # Too short
        )
        assert response.status_code == 422  # Validation error

    def test_decision_empty_obs(self, client):
        response = client.post(
            "/decision",
            json={"obs": []}
        )
        assert response.status_code == 422

    def test_decision_kpis_structure(self, client):
        response = client.post(
            "/decision",
            json={"obs": [0.3, 0.5, 0.2, 0.25, 0.8, 0.4]}
        )
        assert response.status_code == 200
        kpis = response.json()["predicted_kpis"]
        
        assert "latency_ms" in kpis
        assert "energy_per_mbps" in kpis
        assert "sla_violation" in kpis
        assert kpis["latency_ms"] > 0
        assert kpis["energy_per_mbps"] > 0
        assert kpis["sla_violation"] in [0, 1]

    def test_decision_edge_cases(self, client):
        # All zeros
        response = client.post(
            "/decision",
            json={"obs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        )
        assert response.status_code == 200

        # All ones
        response = client.post(
            "/decision",
            json={"obs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
        )
        assert response.status_code == 200


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
