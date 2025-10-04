"""
Module de prédiction d'exoplanètes - À COMPLÉTER PAR POWELL

Ce fichier est un TEMPLATE pour Powell.
Il doit implémenter les fonctions predict_single() et predict_batch()
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Chemin vers le modèle sauvegardé
MODEL_PATH = Path(__file__).parent / 'models' / 'exoplanet_model.pkl'
SCALER_PATH = Path(__file__).parent / 'models' / 'scaler.pkl'


class ExoplanetPredictor:
    """
    Classe pour encapsuler la logique de prédiction
    """
    
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

        
        self.target = 'koi_disposition'  # facultatif, pour info
        self.load_model()
    
    def load_model(self):
        """
        Charge le modèle ML et le scaler depuis les fichiers .pkl
        
        🚨 POWELL : Remplace cette fonction avec ton vrai modèle
        """
        try:
            if MODEL_PATH.exists():
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"Modèle chargé depuis {MODEL_PATH}")
            else:
                logger.warning("Modèle non trouvé, utilisation du mode simulation")
                self.model = None
            
            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"Scaler chargé depuis {SCALER_PATH}")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
            self.model = None
            self.scaler = None
    
    def preprocess_data(self, data):
        """
        Prétraitement des données avant prédiction
        
        Args:
            data (dict ou DataFrame): Données d'entrée
        
        Returns:
            np.array: Données prétraitées
        
        🚨 POWELL : Ajoute ici ton preprocessing (normalisation, etc.)
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Si la colonne koi_teq manque, mettre une valeur par défaut
        if 'koi_teq' not in df.columns:
            df['koi_teq'] = 300  # température moyenne par défaut
            
            
            
            # Sélectionner les features dans le bon ordre (DataFrame, pas ndarray)
        X = df[self.features].copy()

        # Remplir les valeurs manquantes avec des valeurs sûres
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

        # Normaliser si un scaler existe (passer un DataFrame garde les noms)
        if self.scaler is not None:
            X = self.scaler.transform(X)

# X est maintenant un ndarray prêt pour la prédiction

            
        
        return X
    
    def predict_single(self, data):
        """
        Fait une prédiction pour une seule planète
        
        Args:
            data (dict): {
                'orbital_period': float,
                'transit_duration': float,
                'planetary_radius': float,
                'star_temperature': float (optionnel)
            }
        
        Returns:
            dict: {
                'prediction': str,      # "Confirmed", "Candidate", ou "False Positive"
                'probability': float,   # Entre 0 et 1
                'confidence': str,      # "High", "Medium", ou "Low"
                'message': str
            }
        
        🚨 POWELL : Remplace la simulation par ton vrai modèle
        """
        try:
            # Prétraiter les données
            X = self.preprocess_data(data)
            
            # Si le modèle existe, utiliser la vraie prédiction
            if self.model is not None:
                # Prédiction
                prediction_class = self.model.predict(X)[0]
                
                # Probabilité (si le modèle supporte predict_proba)
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)[0]
                    probability = float(np.max(probabilities))
                else:
                    probability = 0.85  # Valeur par défaut
                
                # Mapper les classes
                class_mapping = {
                    0: "False Positive",
                    1: "Candidate",
                    2: "Confirmed"
                }
                prediction = class_mapping.get(prediction_class, "Unknown")
            
            else:
                # 🚨 MODE SIMULATION (à supprimer quand le vrai modèle est prêt)
                prediction, probability = self._simulate_prediction(data)
            
            # Calculer le niveau de confiance
            if probability > 0.8:
                confidence = "High"
            elif probability > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Message personnalisé
            messages = {
                "Confirmed": f"Cette exoplanète est très probablement confirmée ({probability*100:.1f}% de confiance)",
                "Candidate": f"Cette planète est un candidat potentiel ({probability*100:.1f}% de confiance)",
                "False Positive": f"Cette détection est probablement un faux positif ({probability*100:.1f}% de confiance)"
            }
            message = messages.get(prediction, "Prédiction incertaine")
            
            return {
                'prediction': prediction,
                'probability': round(probability, 4),
                'confidence': confidence,
                'message': message
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction : {str(e)}")
            raise
    
    def predict_batch(self, dataframe):
        """
        Fait des prédictions pour plusieurs planètes
        
        Args:
            dataframe (pd.DataFrame): DataFrame avec colonnes requises
        
        Returns:
            list[dict]: Liste de résultats de prédiction
        
        🚨 POWELL : Optimise cette fonction pour le batch processing
        """
        results = []
        
        for idx, row in dataframe.iterrows():
            try:
                result = self.predict_single(row.to_dict())
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur ligne {idx} : {str(e)}")
                results.append({
                    'prediction': 'Error',
                    'probability': 0.0,
                    'confidence': 'None',
                    'message': f'Erreur : {str(e)}'
                })
        
        return results
    
    def _simulate_prediction(self, data):
        """
        🚨 SIMULATION TEMPORAIRE - À SUPPRIMER
        Logique simple pour tester l'API sans modèle ML
        """
        orbital = data.get('koi_period', 0)
        radius = data.get('koi_prad', 0)
        duration = data.get('koi_duration', 0)

        # Logique simplifiée basée sur des seuils
        score = 0
        if orbital > 5:
            score += 0.3
        if radius > 1:
            score += 0.3
        if duration > 2:
            score += 0.3
        
        # Ajouter un peu de randomness
        import random
        score += random.uniform(0, 0.2)
        
        if score > 0.7:
            return "Confirmed", min(score + 0.15, 0.99)
        elif score > 0.4:
            return "Candidate", score
        else:
            return "False Positive", 1 - score


# ========================================
# FONCTIONS PUBLIQUES (utilisées par l'API)
# ========================================

# Créer une instance globale du prédictor
predictor = ExoplanetPredictor()


def predict_single(data):
    """
    Interface publique pour prédiction unique
    
    Args:
        data (dict): Données de la planète
    
    Returns:
        dict: Résultat de la prédiction
    """
    return predictor.predict_single(data)


def predict_batch(dataframe):
    """
    Interface publique pour prédiction batch
    
    Args:
        dataframe (pd.DataFrame): Données de plusieurs planètes
    
    Returns:
        list[dict]: Liste de résultats
    """
    return predictor.predict_batch(dataframe)


def reload_model():
    """
    Recharge le modèle depuis le disque
    Utile après un retrain
    """
    predictor.load_model()


# ========================================
# TEST DU MODULE (pour debugging)
# ========================================

if __name__ == "__main__":
    # Test simple avec les nouvelles features Powell
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
    print("Test de prédiction :")
    print(result)
