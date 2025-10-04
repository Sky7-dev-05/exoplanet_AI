"""
Routes de l'API predictions
"""
from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    # Endpoint principal de prédiction
    path('predict/', views.predict_exoplanet, name='predict'),
    
    # Prédiction en batch (CSV)
    path('predict-batch/', views.predict_batch_endpoint, name='predict-batch'),
    
    # Informations sur le modèle ML
    path('model-info/', views.model_info, name='model-info'),
    
    # Historique des prédictions
    path('history/', views.prediction_history, name='history'),
    
    # Statistiques globales
    path('stats/', views.statistics, name='stats'),
    
    # Ré-entraînement du modèle (admin only)
    path('retrain/', views.retrain_model, name='retrain'),
]



