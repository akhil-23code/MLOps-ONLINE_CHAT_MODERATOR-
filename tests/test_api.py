from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    """Test the health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction_neither():
    """Test a clear 'Neither' case"""
    response = client.post("/predict", json={"text": "I love sunshine and kittens."})
    assert response.status_code == 200
    assert response.json()["label"] == "Neither"

def test_prediction_offensive():
    """Test an 'Offensive' case (using mild input)"""
    # Use a word you know your model flags as offensive
    response = client.post("/predict", json={"text": "you are so stupid"})
    assert response.status_code == 200
    assert response.json()["label"] == "Offensive"

def test_empty_input():
    """Ensure the API handles empty strings gracefully"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400 # Or 422 depending on your validation

# Add this to the bottom of tests/test_api.py for monitoring report test
import os

def test_monitoring_report_exists():
    """Verify that the monitoring script produces an output"""
    assert os.path.exists("reports/monitoring_report.html")