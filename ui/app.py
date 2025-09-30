import sys
from pathlib import Path
import os
import streamlit as st

# Ensure project root is on sys.path so that `src` can be imported when running Streamlit
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from src.inference import TicketClassifier

# Must be the first Streamlit command in the script
st.set_page_config(page_title="Ticket Auto‑Triage", page_icon="🎫", layout="centered")


ROUTING_HINTS = {
	"Bug Report": "Engineering",
	"Feature Request": "Product Management",
	"Technical Issue": "Technical Support",
	"Billing Inquiry": "Billing/Finance",
	"Account Management": "Customer Success",
}


@st.cache_resource
def load_classifier(model_path: str) -> TicketClassifier:
	"""Load and cache the classifier so it persists across reruns."""
	return TicketClassifier(model_path)


def render_header() -> None:
	st.title("Ticket Auto‑Triage")
	st.caption("Classify support tickets into categories and route them quickly.")


def render_sidebar(model_path: str) -> None:
	st.sidebar.header("Settings")
	st.sidebar.write("Using model:")
	st.sidebar.code(model_path, language="text")
	with st.sidebar.expander("Routing hints"):
		for category, team in ROUTING_HINTS.items():
			st.write(f"- {category} → {team}")


def render_form(classifier: TicketClassifier) -> None:
	with st.form("ticket_form"):
		subject = st.text_input("Subject", placeholder="e.g., Billing issue")
		description = st.text_area(
			"Description",
			height=160,
			placeholder="e.g., Charged twice this month despite plan upgrade",
		)
		submitted = st.form_submit_button("Classify")

	if submitted:
		if not subject and not description:
			st.warning("Please enter a subject or description.")
			return
		result = classifier.predict(subject, description)
		predicted_category = result["category"]
		st.success(f"Predicted category: {predicted_category}")
		routing_team = ROUTING_HINTS.get(predicted_category)
		if routing_team:
			st.info(f"Routing hint: {routing_team}")


def main() -> None:
	render_header()
	model_path = os.getenv("MODEL_PATH", "models/model.joblib")
	classifier = load_classifier(model_path)
	render_sidebar(model_path)
	render_form(classifier)
	st.caption("Tip: Set environment variable MODEL_PATH to switch models.")


if __name__ == "__main__":
	main()

