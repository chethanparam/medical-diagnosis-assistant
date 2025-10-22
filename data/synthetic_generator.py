import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

SYMPTOMS = [
    "fever","cough","sore_throat","runny_nose","headache","fatigue","nausea","vomiting",
    "diarrhea","stomach_pain","shortness_of_breath","chest_pain","body_ache","chills","sweating",
    "loss_of_smell","loss_of_taste","rash","itching","joint_pain","back_pain","ear_pain","eye_redness",
    "sensitivity_to_light","dizziness","confusion","weight_loss","weight_gain","anxiety","depression"
]

DISEASES = [
    "Common Cold","Influenza","COVID-19","Food Poisoning","Gastritis","Migraine","Allergic Rhinitis",
    "Dermatitis","Conjunctivitis","Asthma","Pneumonia","GERD","Anxiety Disorder","Depression",
    "Irritable Bowel Syndrome"
]

# Rule patterns to make data realistic-ish: each disease has a symptom-weight signature
SIGNATURES = {
    "Common Cold": {"cough":2, "runny_nose":3, "sore_throat":2, "fever":1},
    "Influenza": {"fever":3, "body_ache":3, "chills":2, "fatigue":2, "cough":2},
    "COVID-19": {"fever":2, "cough":2, "loss_of_smell":3, "loss_of_taste":3, "shortness_of_breath":2},
    "Food Poisoning": {"nausea":3, "vomiting":3, "diarrhea":3, "stomach_pain":2, "fever":1},
    "Gastritis": {"stomach_pain":3, "nausea":2, "vomiting":1},
    "Migraine": {"headache":3, "sensitivity_to_light":2, "nausea":1},
    "Allergic Rhinitis": {"runny_nose":3, "itching":2, "eye_redness":1, "sore_throat":1},
    "Dermatitis": {"rash":3, "itching":3, "sweating":1},
    "Conjunctivitis": {"eye_redness":3, "itching":2},
    "Asthma": {"shortness_of_breath":3, "cough":2, "chest_pain":2},
    "Pneumonia": {"fever":2, "cough":2, "shortness_of_breath":2, "chest_pain":2},
    "GERD": {"stomach_pain":2, "nausea":1, "cough":1},
    "Anxiety Disorder": {"anxiety":3, "sweating":1, "dizziness":1},
    "Depression": {"depression":3, "fatigue":2, "weight_gain":1, "weight_loss":1},
    "Irritable Bowel Syndrome": {"stomach_pain":2, "diarrhea":2, "nausea":1}
}

def generate(n_rows=10000, noise=0.15):
    rows = []
    diseases = rng.choice(DISEASES, size=n_rows)
    for d in diseases:
        row = {}
        for s in SYMPTOMS:
            base = SIGNATURES.get(d, {}).get(s, 0)
            # symptom intensity 0..3 with noise
            val = base + (rng.random()<noise) * rng.integers(-1, 2)
            val = max(0, min(3, int(val)))
            row[s] = val
        row["disease"] = d
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    out = Path(__file__).parent / "synthetic_symptoms.csv"
    df = generate(12000, noise=0.18)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with shape {df.shape}")