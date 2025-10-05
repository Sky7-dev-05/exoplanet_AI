import pickle
import pandas as pd
import json
import os

# Load trained model and scaler
with open("exoplanet_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURES = [
    "koi_score",
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_sma",
    "koi_teq",
    "koi_model_snr"
]

CLASSES = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
def read_csv_auto(file_path):
    """Lit un CSV en détectant automatiquement le séparateur ; ou ,"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(file_path, sep=sep)
        return df
    except Exception as e:
        raise ValueError(f"Erreur lecture CSV: {e}")

def predict_from_dataframe(df: pd.DataFrame):
    # Check that all columns are present
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing features: {missing_cols}")

    # Scale
    X_scaled = scaler.transform(df[FEATURES])

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    results = []
    for pred, proba in zip(predictions, probabilities):
        results.append({
            "predicted_class": CLASSES[pred],
            "probabilities": {
                "FALSE POSITIVE": round(proba[0], 3),
                "CANDIDATE": round(proba[1], 3),
                "CONFIRMED": round(proba[2], 3)
            }
        })
    return results

def predict_from_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and JSON are allowed.")

    return predict_from_dataframe(df)

# ---------------- Test ----------------
if __name__ == "__main__":
    # Exemple CSV ou JSON
    file_path = "./exoplanete_data.csv"  # ou sample_inputs.json
    results = predict_from_file(file_path)
    print(json.dumps(results, indent=4))
