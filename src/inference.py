from typing import Dict

import joblib


class TicketClassifier:
	"""Lightweight wrapper around the trained sklearn Pipeline."""

	def __init__(self, model_path: str) -> None:
		obj = joblib.load(model_path)
		self.pipeline = obj["pipeline"]
		self.labels = obj.get("labels")

	def predict(self, subject: str, description: str) -> Dict[str, str]:
		text = f"{subject or ''} {description or ''}".strip()
		pred = self.pipeline.predict([text])[0]
		return {"category": str(pred)}

