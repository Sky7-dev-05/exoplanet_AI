"""
Main API Views - Your endpoints Nahine! (Version corrigée)
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from django.utils import timezone
import pandas as pd
import numpy as np
import logging
from io import StringIO

from .models import Prediction, ModelInfo
from .serializers import (
    PredictionInputSerializer,
    PredictionOutputSerializer,
    PredictionHistorySerializer,
    ModelInfoSerializer,
    CSVUploadSerializer,
    BatchPredictionOutputSerializer
)

from ml_model.predict_exoplanet import predict_single, predict_batch

logger = logging.getLogger(__name__)


def safe_float_conversion(value, default=0.0):
    """
    Convertit une valeur en float de manière sécurisée
    Gère None, NaN, chaînes vides, etc.
    """
    if pd.isna(value) or value is None or value == '':
        return default
    try:
        float_val = float(value)
        # Vérifier si c'est NaN après conversion
        if np.isnan(float_val):
            return default
        return float_val
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to float, using default {default}")
        return default


@api_view(['POST'])
@permission_classes([AllowAny])
def predict_exoplanet(request):
    """
    Predicts if the data corresponds to an exoplanet
    
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
    """
    serializer = PredictionInputSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {"error": "Invalid data", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    data = serializer.validated_data
    
    try:
        result = predict_single(data)
        
        Prediction.objects.create(
            koi_score=data.get('koi_score', 0.5),
            koi_period=data['koi_period'],
            koi_impact=data.get('koi_impact', 0.0),
            koi_duration=data['koi_duration'],
            koi_depth=data.get('koi_depth', 0.0),
            koi_prad=data['koi_prad'],
            koi_sma=data.get('koi_sma', 0.0),
            koi_teq=data.get('koi_teq', 300.0),
            koi_model_snr=data.get('koi_model_snr', 0.0),
            prediction=result['prediction'],
            probability=result['probability'],
            ip_address=get_client_ip(request)
        )
        
        update_model_prediction_count()
        
        return Response(result, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return Response(
            {"error": "Error during prediction", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def predict_batch_endpoint(request):
    """
    Predicts multiple planets from a CSV file
    
    Input: CSV file with columns (comma OR semicolon separated):
    - koi_score (optional, default: 0.5)
    - koi_period (REQUIRED)
    - koi_impact (optional, default: 0.0)
    - koi_duration (REQUIRED)
    - koi_depth (optional, default: 0.0)
    - koi_prad (REQUIRED)
    - koi_sma (optional, default: 0.0)
    - koi_teq (optional, default: 300.0)
    - koi_model_snr (optional, default: 0.0)
    """
    serializer = CSVUploadSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    csv_file = request.FILES['file']
    
    try:
        # Décoder le fichier en string
        decoded_file = csv_file.read().decode('utf-8')
        csv_string = StringIO(decoded_file)
        
        # Détecter automatiquement le séparateur
        df = pd.read_csv(csv_string, sep=None, engine='python')
        
        # Nettoyer les noms de colonnes (enlever espaces)
        df.columns = df.columns.str.strip()
        
        logger.info(f"CSV columns found: {list(df.columns)}")
        logger.info(f"CSV shape: {df.shape}")
        
        # Colonnes OBLIGATOIRES
        required_cols = ['koi_period', 'koi_duration', 'koi_prad']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return Response(
                {
                    "error": f"Missing required columns: {', '.join(missing_cols)}",
                    "found_columns": list(df.columns),
                    "hint": "Required: koi_period, koi_duration, koi_prad"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Valeurs par défaut pour colonnes optionnelles
        optional_defaults = {
            'koi_score': 0.5,
            'koi_impact': 0.0,
            'koi_depth': 0.0,
            'koi_sma': 0.0,
            'koi_teq': 300.0,
            'koi_model_snr': 0.0
        }
        
        # Ajouter les colonnes manquantes avec valeurs par défaut
        for col, default_val in optional_defaults.items():
            if col not in df.columns:
                df[col] = default_val
                logger.info(f"Added missing column '{col}' with default {default_val}")
        
        # ÉTAPE CRITIQUE : Remplacer toutes les valeurs NaN/None/vides
        # Ordre d'opérations important !
        
        # 1. Remplacer les chaînes vides et espaces par NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # 2. Pour les colonnes REQUISES, vérifier qu'elles n'ont pas de NaN
        for col in required_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                return Response(
                    {
                        "error": f"Column '{col}' has {null_count} missing values",
                        "details": f"Required column '{col}' cannot have empty values",
                        "hint": "Check your CSV file for empty cells in required columns"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # 3. Remplacer les NaN dans les colonnes optionnelles
        df = df.fillna(optional_defaults)
        
        # 4. S'assurer que toutes les valeurs sont numériques
        all_cols = list(optional_defaults.keys()) + required_cols
        for col in all_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Vérification finale : aucun NaN après conversion
        df = df.fillna(optional_defaults)
        
        # Vérifier qu'il reste des lignes valides
        if len(df) == 0:
            return Response(
                {"error": "No valid rows found in CSV after cleaning"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Processing {len(df)} valid rows")
        
        # Prédictions par batch
        results = predict_batch(df)
        
        summary = {"Confirmed": 0, "Candidate": 0, "False Positive": 0}
        saved_count = 0
        errors = []
        
        # Sauvegarder chaque prédiction
        for idx, result in enumerate(results):
            try:
                row = df.iloc[idx]
                
                # Conversion sécurisée de chaque valeur
                prediction_data = {
                    'koi_score': safe_float_conversion(row['koi_score'], 0.5),
                    'koi_period': safe_float_conversion(row['koi_period'], 0.0),
                    'koi_impact': safe_float_conversion(row['koi_impact'], 0.0),
                    'koi_duration': safe_float_conversion(row['koi_duration'], 0.0),
                    'koi_depth': safe_float_conversion(row['koi_depth'], 0.0),
                    'koi_prad': safe_float_conversion(row['koi_prad'], 0.0),
                    'koi_sma': safe_float_conversion(row['koi_sma'], 0.0),
                    'koi_teq': safe_float_conversion(row['koi_teq'], 300.0),
                    'koi_model_snr': safe_float_conversion(row['koi_model_snr'], 0.0),
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'ip_address': get_client_ip(request)
                }
                
                # Log pour debug
                logger.debug(f"Row {idx}: koi_prad={prediction_data['koi_prad']}")
                
                Prediction.objects.create(**prediction_data)
                saved_count += 1
                
                pred_class = result['prediction']
                summary[pred_class] = summary.get(pred_class, 0) + 1
                
            except Exception as row_error:
                error_msg = f"Row {idx + 1}: {str(row_error)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Mettre à jour le compteur du modèle
        model = ModelInfo.objects.filter(is_active=True).first()
        if model:
            model.total_predictions += saved_count
            model.save()
        
        output = {
            "total_predictions": len(results),
            "saved_predictions": saved_count,
            "predictions": results,
            "summary": summary
        }
        
        if errors:
            output["warnings"] = errors
            output["note"] = f"{len(errors)} rows had issues but were skipped"
        
        return Response(output, status=status.HTTP_200_OK)
    
    except UnicodeDecodeError:
        return Response(
            {"error": "Unable to decode file. Make sure it's a valid UTF-8 CSV"},
            status=status.HTTP_400_BAD_REQUEST
        )
    except pd.errors.EmptyDataError:
        return Response(
            {"error": "The CSV file is empty"},
            status=status.HTTP_400_BAD_REQUEST
        )
    except pd.errors.ParserError as e:
        return Response(
            {"error": "Unable to parse CSV file", "details": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}", exc_info=True)
        return Response(
            {"error": "Error processing CSV", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def model_info(request):
    """Returns information about the current ML model"""
    try:
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
        logger.error(f"Error in model-info: {str(e)}")
        return Response(
            {"error": "Error retrieving info", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def prediction_history(request):
    """Returns prediction history (paginated)"""
    predictions = Prediction.objects.all().order_by('-timestamp')[:50]
    serializer = PredictionHistorySerializer(predictions, many=True)
    
    return Response({
        "count": predictions.count(),
        "results": serializer.data
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([AllowAny])
def statistics(request):
    """Returns global statistics"""
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


@api_view(['POST'])
@permission_classes([IsAdminUser])
def retrain_model(request):
    """Retrains the model with new data ⚠️ Admin only"""
    return Response(
        {"message": "Retrain in progress... (feature to be implemented)"},
        status=status.HTTP_200_OK
    )


@api_view(['GET'])
@permission_classes([AllowAny])
def metrics(request):
    """Returns ML model performance metrics"""
    try:
        model = ModelInfo.objects.filter(is_active=True).first()
        
        if not model:
            return Response(
                {"error": "No active model found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        total_preds = Prediction.objects.count()
        
        metrics_data = {
            "accuracy": model.accuracy,
            "f1_score": model.f1_score if model.f1_score else 0.0,
            "recall": getattr(model, 'recall', 0.85),
            "precision": getattr(model, 'precision', 0.88),
            "total_predictions": total_preds
        }
        
        return Response(metrics_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        return Response(
            {"error": "Error retrieving metrics", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def graph1(request):
    """Processes a JPG/PNG image file"""
    return process_graph_image(request, "graph1")


@api_view(['POST'])
@permission_classes([AllowAny])
def graph2(request):
    """Processes a JPG/PNG image file"""
    return process_graph_image(request, "graph2")


def process_graph_image(request, graph_name):
    """Helper function to process graph images"""
    try:
        if 'file' not in request.FILES:
            return Response(
                {"error": "No file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES['file']
        
        if not image_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            return Response(
                {"error": "Unsupported file format. Use JPG, JPEG or PNG"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        import os
        from django.conf import settings
        from datetime import datetime
        
        graphs_dir = os.path.join(settings.MEDIA_ROOT, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{graph_name}_{timestamp}_{image_file.name}"
        filepath = os.path.join(graphs_dir, filename)
        
        with open(filepath, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        image_url = f"{settings.MEDIA_URL}graphs/{filename}"
        
        response_data = {
            "status": "success",
            "message": "Image processed successfully",
            "image_url": image_url,
            "metadata": {
                "filename": image_file.name,
                "size": image_file.size,
                "format": image_file.content_type,
                "saved_as": filename
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Error processing {graph_name} image: {str(e)}")
        return Response(
            {"error": "Error processing image", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def get_latest_graph1(request):
    """Returns the URL of the latest uploaded graph1"""
    return get_latest_graph(request, "graph1")


@api_view(['GET'])
@permission_classes([AllowAny])
def get_latest_graph2(request):
    """Returns the URL of the latest uploaded graph2"""
    return get_latest_graph(request, "graph2")


def get_latest_graph(request, graph_prefix):
    """Helper function to get latest graph"""
    import os
    from django.conf import settings
    
    graphs_dir = os.path.join(settings.MEDIA_ROOT, 'graphs')
    
    if not os.path.exists(graphs_dir):
        return Response({"error": "No graphs found"}, status=404)
    
    graph_files = [f for f in os.listdir(graphs_dir) if f.startswith(f'{graph_prefix}_')]
    
    if not graph_files:
        return Response({"error": f"No {graph_prefix} found"}, status=404)
    
    latest_graph = sorted(graph_files)[-1]
    
    return Response({
        "image_url": f"{settings.MEDIA_URL}graphs/{latest_graph}"
    })


def get_client_ip(request):
    """Gets the client's IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def update_model_prediction_count():
    """Increments the prediction counter for the active model"""
    model = ModelInfo.objects.filter(is_active=True).first()
    if model:
        model.total_predictions += 1
        model.save()