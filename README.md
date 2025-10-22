# AI-Enabled Medical Diagnosis Assistant

> Educational demo for preliminary symptom-based disease prediction. Uses synthetic data, an ensemble ML model, and explainability. **Not a medical device.**

## Tech
- FastAPI (backend API)
- scikit-learn Stacking + Voting ensemble
- SHAP KernelExplainer for local feature attributions
- Static frontend (HTML/CSS/JS) deployable to GitHub Pages

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/synthetic_generator.py
python -m src.train_model
uvicorn app.main:app --reload --port 8000