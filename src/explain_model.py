import shap
import numpy as np
from src.utils import MODELS_DIR, load

MODEL = load(MODELS_DIR / "ensemble_model.pkl")
SCALER = load(MODELS_DIR / "scaler.pkl")
FEATURES = load(MODELS_DIR / "features.joblib")

# KernelExplainer works broadly; TreeExplainer may fail on stacked meta, so use Kernel
explainer = None


def ensure_explainer(background_size=200):
    global explainer
    if explainer is None:
        import numpy as np
        # synthetic background: zeros and small randoms
        bg = np.clip(np.random.normal(0, 0.5, size=(background_size, len(FEATURES))), -2, 2)
        explainer = shap.KernelExplainer(lambda X: MODEL.predict_proba(SCALER.transform(X)), bg)
    return explainer


def explain_instance(symptom_dict, top_k=8):
    ensure_explainer()
    x = np.array([[symptom_dict.get(f, 0) for f in FEATURES]])
    shap_vals = explainer.shap_values(x, nsamples=200)

    # pick class with highest model probability
    proba = MODEL.predict_proba(SCALER.transform(x))[0]
    cls_idx = int(proba.argmax())
    contribs = shap_vals[cls_idx][0]

    pairs = list(zip(FEATURES, contribs))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    top = [{"feature": f, "impact": float(v)} for f, v in pairs[:top_k]]
    return {"class_index": cls_idx, "top_contributors": top}