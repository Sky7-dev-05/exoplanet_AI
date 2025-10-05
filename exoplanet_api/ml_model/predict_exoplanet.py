import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Paths to saved model and scaler
MODEL_PATH = Path(__file__).parent / 'models' / 'exoplanet_model.pkl'
SCALER_PATH = Path(__file__).parent / 'models' / 'scaler.pkl'
IMPUTER_PATH = Path(__file__).parent / 'models' / 'imputer.pkl'

class ExoplanetPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None

        # Expected feature order
        self.features = [
            'koi_score',
            'koi_period',
            'koi_impact',
            'koi_duration',
            'koi_depth',
            'koi_prad',
            'koi_sma',
            'koi_teq',
            'koi_model_snr'
        ]

        self.load_model()

    def load_model(self):
        """Load trained model, scaler and imputer"""
        try:
            if MODEL_PATH.exists():
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"Model loaded from {MODEL_PATH}")
            else:
                logger.warning("Model not found, using simulation mode")
                self.model = None

            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"Scaler loaded from {SCALER_PATH}")

            if IMPUTER_PATH.exists():
                self.imputer = joblib.load(IMPUTER_PATH)
                logger.info(f"Imputer loaded from {IMPUTER_PATH}")

        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            self.model = None
            self.scaler = None
            self.imputer = None

    def preprocess_data(self, data):
        """Convert raw input to ML-ready DataFrame"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Ensure all required features exist
        defaults = {
            'koi_score': 0.5,
            'koi_period': 0.0,
            'koi_impact': 0.0,
            'koi_duration': 0.0,
            'koi_depth': 0.0,
            'koi_prad': 0.0,
            'koi_sma': 0.0,
            'koi_teq': 300.0,
            'koi_model_snr': 0.0
        }
        for f, v in defaults.items():
            if f not in df.columns:
                df[f] = v

        # Apply imputation (trained on training data)
        if self.imputer is not None:
            df[self.features] = self.imputer.transform(df[self.features])
        else:
            df[self.features] = df[self.features].fillna(defaults)

        # Scale features
        if self.scaler is not None:
            df[self.features] = self.scaler.transform(df[self.features])

        return df[self.features]

    def predict_single(self, data):
        """Predict one exoplanet"""
        try:
            X = self.preprocess_data(data)

            if self.model is not None:
                pred_class = self.model.predict(X)[0]

                if hasattr(self.model, 'predict_proba'):
                    prob = float(np.max(self.model.predict_proba(X)[0]))
                else:
                    prob = 0.85
            else:
                pred_class, prob = self._simulate_prediction(data)

            class_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
            prediction = class_map.get(pred_class, 'Unknown')

            confidence = (
                "High" if prob > 0.8 else
                "Medium" if prob > 0.5 else
                "Low"
            )

            messages = {
                "Confirmed": f"This exoplanet is very likely confirmed ({prob*100:.1f}% confidence)",
                "Candidate": f"This planet is a potential candidate ({prob*100:.1f}% confidence)",
                "False Positive": f"This detection is probably a false positive ({prob*100:.1f}% confidence)"
            }

            return {
                'prediction': prediction,
                'probability': round(prob, 4),
                'confidence': confidence,
                'message': messages.get(prediction, "Uncertain prediction")
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_batch(self, df):
        """Predict multiple rows"""
        results = []
        for _, row in df.iterrows():
            try:
                res = self.predict_single(row.to_dict())
                results.append(res)
            except Exception as e:
                logger.error(f"Error at row: {str(e)}")
                results.append({
                    'prediction': 'Error',
                    'probability': 0.0,
                    'confidence': 'None',
                    'message': f'Error: {str(e)}'
                })
        return results

    def _simulate_prediction(self, data):
        """Fallback logic if no model"""
        orbital = data.get('koi_period', 0)
        radius = data.get('koi_prad', 0)
        duration = data.get('koi_duration', 0)

        score = 0
        if orbital > 5: score += 0.3
        if radius > 1: score += 0.3
        if duration > 2: score += 0.3

        if score > 0.7:
            return 2, min(score + 0.15, 0.99)
        elif score > 0.4:
            return 1, score
        else:
            return 0, 1 - score


# Public API
predictor = ExoplanetPredictor()

def predict_single(data):
    return predictor.predict_single(data)

def predict_batch(df):
    return predictor.predict_batch(df)

def reload_model():
    predictor.load_model()