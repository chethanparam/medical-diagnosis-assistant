import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import DATA_DIR, MODELS_DIR, save

FEATURES_CACHE = MODELS_DIR / "features.joblib"


def load_data():
    df = pd.read_csv(DATA_DIR / "synthetic_symptoms.csv")
    X = df.drop(columns=["disease"])  # numeric 0..3
    y = df["disease"].astype("category")
    return X, y


def split_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save(scaler, MODELS_DIR / "scaler.pkl")
    save(list(X.columns), FEATURES_CACHE)

    return X_train_scaled, X_test_scaled, y_train, y_test