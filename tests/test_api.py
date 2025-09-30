from fastapi.testclient import TestClient

from api.main import app


def test_health():
	client = TestClient(app)
	resp = client.get("/health")
	assert resp.status_code == 200
	assert resp.json() == {"status": "ok"}


def test_predict(tmp_model_path):
	client = TestClient(app)
	payload = {
		"subject": "Billing issue",
		"description": "Charged twice this month",
	}
	resp = client.post("/predict", json=payload)
	assert resp.status_code == 200
	data = resp.json()
	assert "category" in data
	assert isinstance(data["category"], str)

