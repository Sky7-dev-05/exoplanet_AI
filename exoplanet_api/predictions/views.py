"""
Views principales de l'API - Tes endpoints Nahine !
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from django.utils import timezone
import pandas as pd
import logging

from .models import Prediction, ModelInfo
from .serializers import (
    PredictionInputSerializer,
    PredictionOutputSerializer,
    PredictionHistorySerializer,
    ModelInfoSerializer,
    CSVUploadSerializer,
    BatchPredictionOutputSerializer
)

# Importer le module ML de Powell
from ml_model.predict_exoplanet import predict_single, predict_batch

logger = logging.getLogger(__name__)


# ========================================
# 🎯 ENDPOINT 1 : POST /api/predict
# ========================================
@api_view(['POST'])
@permission_classes([AllowAny])
def predict_exoplanet(request):
    """
    Prédit si les données correspondent à une exoplanète
    
    Input (JSON):
    {
        "koi_score": 0.95,
        "koi_period": 3.52,
        "koi_impact": 0.1,
        "koi_duration": 2.5,
        "koi_depth": 500,
        "koi_prad": 1.2,
        "koi_sma": 0.05,
        "koi_teq": 580,
        "koi_model_snr": 10.0
    }
    
    Output:
    {
        "prediction": "Confirmed",
        "probability": 0.92,
        "confidence": "High",
        "message": "Cette planète est très probablement confirmée"
    }
    """
    # Valider les données d'entrée
    serializer = PredictionInputSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {"error": "Données invalides", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    data = serializer.validated_data
    
    try:
        # 🔥 Appeler le module ML de Powell
        result = predict_single(data)
        
        # Sauvegarder dans l'historique
        Prediction.objects.create(
            koi_score=data.get('koi_score'),
            koi_period=data['koi_period'],
            koi_impact=data.get('koi_impact'),
            koi_duration=data['koi_duration'],
            koi_depth=data.get('koi_depth'),
            koi_prad=data['koi_prad'],
            koi_sma=data.get('koi_sma'),
            koi_teq=data.get('koi_teq'),
            koi_model_snr=data.get('koi_model_snr'),
            prediction=result['prediction'],
            probability=result['probability'],
            ip_address=get_client_ip(request)
        )
        
        # Mettre à jour le compteur du modèle
        update_model_prediction_count()
        
        return Response(result, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return Response(
            {"error": "Erreur lors de la prédiction", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ========================================
# 🎯 ENDPOINT 2 : POST /api/predict-batch
# ========================================
@api_view(['POST'])
@permission_classes([AllowAny])
def predict_batch_endpoint(request):
    """
    Prédit plusieurs planètes à partir d'un fichier CSV
    
    Input: Fichier CSV avec colonnes:
    - koi_score
    - koi_period (requis)
    - koi_impact
    - koi_duration (requis)
    - koi_depth
    - koi_prad (requis)
    - koi_sma
    - koi_teq (optionnel)
    - koi_model_snr
    
    Output:
    {
        "total_predictions": 10,
        "predictions": [...],
        "summary": {
            "Confirmed": 6,
            "Candidate": 2,
            "False Positive": 2
        }
    }
    """
    serializer = CSVUploadSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    csv_file = request.FILES['file']
    
    try:
        # Lire le CSV
        df = pd.read_csv(csv_file)
        
        # Vérifier les colonnes requises
        required_cols = ['koi_period', 'koi_duration', 'koi_prad']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return Response(
                {"error": f"Colonnes manquantes : {', '.join(missing_cols)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Ajouter les colonnes optionnelles avec valeurs par défaut si manquantes
        optional_defaults = {
            'koi_score': 0.5,
            'koi_impact': 0.0,
            'koi_depth': 0.0,
            'koi_sma': 0.0,
            'koi_teq': 300.0,
            'koi_model_snr': 0.0
        }
        
        for col, default_val in optional_defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        # 🔥 Appeler le module ML de Powell pour batch
        results = predict_batch(df)
        
        # Calculer le résumé
        summary = {"Confirmed": 0, "Candidate": 0, "False Positive": 0}
        
        # Sauvegarder chaque prédiction en DB
        for idx, result in enumerate(results):
            row = df.iloc[idx]
            
            # Sauvegarder en base de données
            Prediction.objects.create(
                koi_score=float(row.get('koi_score', 0.5)),
                koi_period=float(row['koi_period']),
                koi_impact=float(row.get('koi_impact', 0.0)),
                koi_duration=float(row['koi_duration']),
                koi_depth=float(row.get('koi_depth', 0.0)),
                koi_prad=float(row['koi_prad']),
                koi_sma=float(row.get('koi_sma', 0.0)),
                koi_teq=float(row.get('koi_teq', 300.0)),
                koi_model_snr=float(row.get('koi_model_snr', 0.0)),
                prediction=result['prediction'],
                probability=result['probability'],
                ip_address=get_client_ip(request)
            )
            
            # Mettre à jour le résumé
            pred_class = result['prediction']
            summary[pred_class] = summary.get(pred_class, 0) + 1
        
        # Mettre à jour le compteur du modèle
        model = ModelInfo.objects.filter(is_active=True).first()
        if model:
            model.total_predictions += len(results)
            model.save()
        
        output = {
            "total_predictions": len(results),
            "predictions": results,
            "summary": summary
        }
        
        return Response(output, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Erreur lors du batch : {str(e)}")
        return Response(
            {"error": "Erreur lors du traitement du CSV", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ========================================
# 🎯 ENDPOINT 3 : GET /api/model-info
# ========================================
@api_view(['GET'])
@permission_classes([AllowAny])
def model_info(request):
    """
    Retourne les informations sur le modèle ML actuel

    Output:
    {
        "version": "1.0",
        "accuracy": 0.95,
        "f1_score": 0.93,
        "trained_on": "2025-09-30T10:00:00Z",
        "features_list": ["orbital_period", "transit_duration", ...],
        "total_predictions": 1247,
        "is_active": true
    }
    """
    try:
        # Récupérer le modèle actif ou créer un modèle par défaut
        model, created = ModelInfo.objects.get_or_create(
            is_active=True,
            defaults={
                "version": "1.0",
                "accuracy": 0.95,
                "f1_score": 0.93,
                "trained_on": timezone.now(),
                "features_used": "orbital_period,transit_duration,planetary_radius,star_temperature",
                "total_predictions": 0
            }
        )

        serializer = ModelInfoSerializer(model)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Erreur model-info : {str(e)}")
        return Response(
            {"error": "Erreur lors de la récupération des infos", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ========================================
# 🎯 ENDPOINT 4 : GET /api/history
# ========================================
@api_view(['GET'])
@permission_classes([AllowAny])
def prediction_history(request):
    """
    Retourne l'historique des prédictions (paginé)
    """
    predictions = Prediction.objects.all()[:50]  # Limiter à 50 dernières
    serializer = PredictionHistorySerializer(predictions, many=True)
    
    return Response({
        "count": predictions.count(),
        "results": serializer.data
    }, status=status.HTTP_200_OK)


# ========================================
# 🎯 ENDPOINT 5 : GET /api/stats
# ========================================
@api_view(['GET'])
@permission_classes([AllowAny])
def statistics(request):
    """
    Retourne des statistiques globales
    """
    total = Prediction.objects.count()
    confirmed = Prediction.objects.filter(prediction__icontains="Confirmed").count()
    candidate = Prediction.objects.filter(prediction__icontains="Candidate").count()
    false_pos = Prediction.objects.filter(prediction__icontains="False").count()
    
    return Response({
        "total_predictions": total,
        "confirmed": confirmed,
        "candidate": candidate,
        "false_positive": false_pos,
        "confirmed_percentage": round((confirmed / total * 100) if total > 0 else 0, 2)
    }, status=status.HTTP_200_OK)


# ========================================
# 🔒 ENDPOINT 6 : POST /api/retrain (ADMIN ONLY)
# ========================================
@api_view(['POST'])
@permission_classes([IsAdminUser])
def retrain_model(request):
    """
    Ré-entraîne le modèle avec de nouvelles données
    ⚠️ Réservé aux admins uniquement
    """
    # TODO : Implémenter la logique de retrain avec Powell
    return Response(
        {"message": "Retrain en cours... (fonctionnalité à implémenter)"},
        status=status.HTTP_200_OK
    )


# ========================================
# 🛠️ FONCTIONS UTILITAIRES
# ========================================

def get_client_ip(request):
    """Récupère l'IP du client"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def update_model_prediction_count():
    """Incrémente le compteur de prédictions du modèle actif"""
    model = ModelInfo.objects.filter(is_active=True).first()
    if model:
        model.total_predictions += 1
        model.save()