import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_preprocessing import load_data, split_scale
from src.utils import MODELS_DIR, save


def build_model():
    lr = LogisticRegression(max_iter=1000, n_jobs=None)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    # Stacking base -> meta
    stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True
    )

    # Voting on top of stacking and RF (diversity)
    ensemble = VotingClassifier(
        estimators=[("stack", stack), ("rf2", RandomForestClassifier(n_estimators=300, random_state=7))],
        voting="soft"
    )
    return ensemble


def train_and_eval():
    X, y = load_data()
    Xtr, Xte, ytr, yte = split_scale(X, y)
    model = build_model()
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)

    acc = accuracy_score(yte, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(yte, preds))

    save(model, MODELS_DIR / "ensemble_model.pkl")
    print("Saved model → models/ensemble_model.pkl")


if __name__ == "__main__":
    train_and_eval()