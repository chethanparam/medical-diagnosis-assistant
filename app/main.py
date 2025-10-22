from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
from src.predict import predict_one, features_list
from src.explain_model import explain_instance

app = FastAPI(title="AI Medical Diagnosis Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomInput(BaseModel):
    symptoms: Dict[str, int] = Field(..., description="Map of symptom→intensity (0..3)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"features": features_list(), "scale": [0,1,2,3]}


@app.post("/predict")
def predict(payload: SymptomInput):
    res = predict_one(payload.symptoms)
    return res


@app.post("/explain")
def explain(payload: SymptomInput):
    exp = explain_instance(payload.symptoms)
    return exp