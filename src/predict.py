import numpy as np
from src.utils import MODELS_DIR, load

MODEL = load(MODELS_DIR / "ensemble_model.pkl")
SCALER = load(MODELS_DIR / "scaler.pkl")
FEATURES = load(MODELS_DIR / "features.joblib")


def predict_one(symptom_dict):
    # symptom_dict: {feature: 0..3}
    x = np.array([[symptom_dict.get(f, 0) for f in FEATURES]])
    x = SCALER.transform(x)
    proba = MODEL.predict_proba(x)[0]
    classes = MODEL.classes_
    top_idx = proba.argmax()
    return {
        "predicted": str(classes[top_idx]),
        "confidence": float(proba[top_idx]),
        "proba": {str(c): float(p) for c, p in zip(classes, proba)}
    }


def features_list():
    return FEATURES