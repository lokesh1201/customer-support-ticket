Customer Support Ticket Auto‑Triage

Overview
- Automates the first step of support handling by classifying incoming tickets into: Bug Report, Feature Request, Technical Issue, Billing Inquiry, Account Management.
- Includes training pipeline, evaluation (Accuracy, Precision, Recall, F1), latency measurement, REST API for real‑time classification, and a simple Streamlit UI.

Live Demo
- Streamlit app: [customer-support-ticket.streamlit.app](https://customer-support-ticket.streamlit.app/)

Tech Stack
- Python 3.8+ (tested on 3.12)
- Libraries: scikit‑learn, pandas, numpy, FastAPI, Uvicorn, Streamlit, pytest

File Structure
```
CST/
├─ api/
│  └─ main.py                 # FastAPI app (/health, /predict)
├─ data/
│  └─ tickets_sample.csv      # Example dataset (replace with your own)
├─ models/
│  └─ model.joblib            # Trained model artifact (generated)
│  └─ metrics.json            # Evaluation report (generated)
├─ scripts/
│  ├─ measure_latency.py      # p50/p90 latency measurement
│  └─ __init__.py
├─ src/
│  ├─ data.py                 # Dataset loading, text field, robust split
│  ├─ model.py                # TF‑IDF + LinearSVC pipeline factory
│  ├─ train.py                # Train and save model
│  ├─ evaluate.py             # Evaluate and write metrics
│  └─ inference.py            # Light inference wrapper
├─ tests/
│  ├─ conftest.py             # Temp model fixture + env setup
│  ├─ test_api.py             # API smoke tests
│  ├─ test_data.py            # Data utils tests
│  └─ test_model.py           # Pipeline trains/predicts
├─ ui/
│  └─ app.py                  # Streamlit UI for manual testing
├─ README.md
├─ requirements.txt
└─ .gitignore
```

Dataset Format
- CSV columns: `Ticket_ID, Subject, Description, Category, Priority, Timestamp`
- The model uses free text = `Subject + " " + Description`.

Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1           # PowerShell
pip install -r requirements.txt
```

Training
```bash
python -m src.train --data_path data/your_tickets.csv --model_path models/model.joblib --test_size 0.2 --random_state 42
```

Evaluation (saves metrics.json)
```bash
python -m src.evaluate --data_path data/your_tickets.csv --model_path models/model.joblib --report_path models/metrics.json
```

Latency (p50/p90)
```bash
python -m scripts.measure_latency --model_path models/model.joblib \
  --subject "App crashes" \
  --description "Crashes on login when 2FA enabled" \
  --num_runs 200
```

Run API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# POST http://localhost:8000/predict
# {
#   "subject": "Billing issue",
#   "description": "Charged twice this month"
# }
```

Streamlit UI (optional)
```bash
streamlit run ui/app.py
# Optionally select a different model file
$env:MODEL_PATH="models/model.joblib"
```

Tests
```bash
python -m pytest -q
```

Model
- Baseline: TF‑IDF + LinearSVC (fast, strong for text classification)
- Tunables: `clf__C`, `tfidf__max_features`, `tfidf__ngram_range`

Results (example)
- Accuracy (hold‑out split): ~0.95 (sample data)
- Full‑dataset accuracy: ~0.99 (for demonstration only)
- Latency: p50 ≈ 1.22 ms, p90 ≈ 5.87 ms (local CPU)

Submission (per assessment)
- Push to a Git repository with clear, concise commits.
- Email `support@leadmasters.ai` with subject: `AI/ML Assessment 3 Support Ticket Auto‑Triage 3 [Your Full Name]`.
- Ensure `models/model.joblib` and `models/metrics.json` are present and accessible.

Maintainer
- Lokesh Manchala (+91 8639242091).

