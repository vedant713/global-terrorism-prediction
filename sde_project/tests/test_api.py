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

def test_history_endpoint():
    """Verify history endpoint returns valid structure."""
    # Country ID 4 is Afghanistan
    response = client.get("/history", params={"country_id": 4})
    assert response.status_code == 200
    data = response.json()
    # Check if we get either data or a valid empty message
    if "years" in data:
        assert isinstance(data["years"], list)
        assert isinstance(data["counts"], list)
    else:
        assert "message" in data

@patch("sde_project.api.df_data")
def test_similar_incidents(mock_df):
    """Verify similar incidents endpoint returns valid structure."""
    # Mock the dataframe to ensure data exists for the test
    import pandas as pd
    mock_data = pd.DataFrame({
        'region': [6, 6],
        'attacktype1_txt': ['Bombing/Explosion', 'Bombing/Explosion'],
        'iyear': [2020, 2021],
        'latitude': [34.0, 35.0],
        'longitude': [65.0, 66.0],
        'city': ['Kabul', 'Kabul'],
        'nkill': [1, 2],
        'summary': ['Incident 1', 'Incident 2']
    })
    
    # We need to set the global df_data in api module
    from sde_project import api
    api.df_data = mock_data
    
    response = client.get("/similar", params={"region": 6, "attack_type": "Bombing/Explosion"})
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert isinstance(data["incidents"], list)
    if len(data["incidents"]) > 0:
        first = data["incidents"][0]
        assert "latitude" in first
        assert "longitude" in first

import os
from unittest.mock import patch

def test_genai_advisory_mock():
    """Verify GenAI endpoint returns mock response when no key present (or mocked to be missing)."""
    # Force GEMINI_API_KEY to be None for this test to ensure fallback works
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        payload = {
            "country": "TestLand",
            "year": "2025",
            "summary_text": "A test incident occurred."
        }
        response = client.post("/genai/advisory", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "advisory" in data
        assert "Mock" in data["source"]
