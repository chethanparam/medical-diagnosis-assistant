from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

MODELS_DIR.mkdir(exist_ok=True, parents=True)


def save(obj, path):
    import joblib
    joblib.dump(obj, path)


def load(path):
    return joblib.load(path)