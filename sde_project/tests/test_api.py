from fastapi.testclient import TestClient
from sde_project.api import app
import pytest

client = TestClient(app)

def test_health_check():
    """Verify the health endpoint returns 200 and expected status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_metadata_endpoint():
    """Verify metadata endpoint returns 200 (even if empty in test env)."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "countries" in data
    assert "regions" in data

def test_predict_valid_input():
    """Verify prediction endpoint handles valid input correctly."""
    payload = {
        "iyear": 2017,
        "imonth": 1,
        "iday": 1,
        "country": 4, # Afghanistan (ID 4 is common in GTD)
        "region": 6,  # South Asia
        "attacktype1_txt": "Bombing/Explosion",
        "targtype1_txt": "Military",
        "weaptype1_txt": "Explosives"
    }
    response = client.post("/predict", json=payload)
    
    # We expect a 200 even if model isn't loaded (dummy mode) or is loaded
    assert response.status_code == 200
    result = response.json()
    assert "predicted_fatalities" in result
    assert "status" in result
    # Fatalities should be a number >= 0
    assert result["predicted_fatalities"] >= 0

def test_predict_invalid_input():
    """Verify that missing fields cause a validation error (422)."""
    payload = {
        "iyear": 2017
        # Missing other required fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
