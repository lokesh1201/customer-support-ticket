from fastapi import FastAPI
from pydantic import BaseModel
import os

from src.inference import TicketClassifier


class TicketInput(BaseModel):
	subject: str
	description: str


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
classifier = TicketClassifier(MODEL_PATH)

app = FastAPI(title="Ticket Auto-Triage API")


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/predict")
def predict(item: TicketInput) -> dict:
	return classifier.predict(item.subject, item.description)

