import os
from pathlib import Path
import pytest
import joblib

from src.model import build_pipeline


@pytest.fixture(scope="session")
def tmp_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
	"""Train a tiny model and persist it for API tests."""
	tmp_dir = tmp_path_factory.mktemp("models")
	model_path = Path(tmp_dir) / "model.joblib"

	texts = [
		"App crash on login when 2FA enabled",
		"Export to CSV would be helpful",
		"VPN not connecting on corporate network",
		"Charged twice for monthly subscription",
		"Reset password link expired quickly",
	]
	labels = [
		"Bug Report",
		"Feature Request",
		"Technical Issue",
		"Billing Inquiry",
		"Account Management",
	]

	pipeline = build_pipeline()
	pipeline.fit(texts, labels)
	joblib.dump({"pipeline": pipeline, "labels": sorted(set(labels))}, model_path)
	return str(model_path)


@pytest.fixture(autouse=True)
def set_model_env(tmp_model_path: str, monkeypatch: pytest.MonkeyPatch):
	"""Point API to the temp model by default."""
	monkeypatch.setenv("MODEL_PATH", tmp_model_path)
	yield

