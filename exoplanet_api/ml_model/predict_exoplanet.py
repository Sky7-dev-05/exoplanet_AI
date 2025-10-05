import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / 'models' / 'exoplanet_model.pkl'
SCALER_PATH = Path(__file__).parent / 'models' / 'scaler.pkl'


class ExoplanetPredictor:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        
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

        self.target = 'koi_disposition'
        self.load_model()
    
    def load_model(self):
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
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.scaler = None
    
    def preprocess_data(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        if 'koi_teq' not in df.columns:
            df['koi_teq'] = 300
            
        X = df[self.features].copy()

        X = X.fillna({
            'koi_score': 0.0,
            'koi_period': 0.0,
            'koi_impact': 0.0,
            'koi_duration': 0.0,
            'koi_depth': 0.0,
            'koi_prad': 0.0,
            'koi_sma': 0.0,
            'koi_teq': 300.0,
            'koi_model_snr': 0.0
        })

        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X
    
    def predict_single(self, data):
        """Makes a prediction for a single planet"""
        try:
            X = self.preprocess_data(data)
            
            if self.model is not None:
                prediction_class = self.model.predict(X)[0]
                
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)[0]
                    probability = float(np.max(probabilities))
                else:
                    probability = 0.85
                
                class_mapping = {
                    0: "False Positive",
                    1: "Candidate",
                    2: "Confirmed"
                }
                prediction = class_mapping.get(prediction_class, "Unknown")
            
            else:
                prediction, probability = self._simulate_prediction(data)
            
            if probability > 0.8:
                confidence = "High"
            elif probability > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            messages = {
                "Confirmed": f"This exoplanet is very likely confirmed ({probability*100:.1f}% confidence)",
                "Candidate": f"This planet is a potential candidate ({probability*100:.1f}% confidence)",
                "False Positive": f"This detection is probably a false positive ({probability*100:.1f}% confidence)"
            }
            message = messages.get(prediction, "Uncertain prediction")
            
            return {
                'prediction': prediction,
                'probability': round(probability, 4),
                'confidence': confidence,
                'message': message
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, dataframe):
        """Makes predictions for multiple planets"""
        results = []
        
        for idx, row in dataframe.iterrows():
            try:
                result = self.predict_single(row.to_dict())
                results.append(result)
            except Exception as e:
                logger.error(f"Error at row {idx}: {str(e)}")
                results.append({
                    'prediction': 'Error',
                    'probability': 0.0,
                    'confidence': 'None',
                    'message': f'Error: {str(e)}'
                })
        
        return results
    
    def _simulate_prediction(self, data):
        """Temporary simulation logic for testing without ML model"""
        orbital = data.get('koi_period', 0)
        radius = data.get('koi_prad', 0)
        duration = data.get('koi_duration', 0)

        score = 0
        if orbital > 5:
            score += 0.3
        if radius > 1:
            score += 0.3
        if duration > 2:
            score += 0.3
        
        import random
        score += random.uniform(0, 0.2)
        
        if score > 0.7:
            return "Confirmed", min(score + 0.15, 0.99)
        elif score > 0.4:
            return "Candidate", score
        else:
            return "False Positive", 1 - score


predictor = ExoplanetPredictor()


def predict_single(data):
    """Public interface for single prediction"""
    return predictor.predict_single(data)


def predict_batch(dataframe):
    """Public interface for batch prediction"""
    return predictor.predict_batch(dataframe)


def reload_model():
    """Reloads the model from disk"""
    predictor.load_model()


if __name__ == "__main__":
    test_data = {
        'koi_score': 0.95,
        'koi_period': 3.52,
        'koi_impact': 0.1,
        'koi_duration': 2.5,
        'koi_depth': 500,
        'koi_prad': 1.2,
        'koi_sma': 0.05,
        'koi_teq': 580,
        'koi_model_snr': 10.0
    }
    
    result = predict_single(test_data)
    print("Prediction test:")
    print(result)