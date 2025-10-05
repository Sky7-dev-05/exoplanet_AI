"""
Main API Views - Your endpoints Nahine!
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

from ml_model.predict_exoplanet import predict_single, predict_batch

logger = logging.getLogger(__name__)


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
    
    Output:
    {
        "prediction": "Confirmed",
        "probability": 0.92,
        "confidence": "High",
        "message": "This planet is very likely confirmed"
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
    
    Input: CSV file with columns:
    - koi_score
    - koi_period (required)
    - koi_impact
    - koi_duration (required)
    - koi_depth
    - koi_prad (required)
    - koi_sma
    - koi_teq (optional)
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
        df = pd.read_csv(csv_file)
        
        required_cols = ['koi_period', 'koi_duration', 'koi_prad']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return Response(
                {"error": f"Missing columns: {', '.join(missing_cols)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
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
        
        results = predict_batch(df)
        
        summary = {"Confirmed": 0, "Candidate": 0, "False Positive": 0}
        
        for idx, result in enumerate(results):
            row = df.iloc[idx]
            
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
            
            pred_class = result['prediction']
            summary[pred_class] = summary.get(pred_class, 0) + 1
        
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
        logger.error(f"Error during batch processing: {str(e)}")
        return Response(
            {"error": "Error processing CSV", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def model_info(request):
    """
    Returns information about the current ML model

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
    predictions = Prediction.objects.all()[:50]
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
    """
    Retrains the model with new data
    ⚠️ Admin only
    """
    return Response(
        {"message": "Retrain in progress... (feature to be implemented)"},
        status=status.HTTP_200_OK
    )


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


@api_view(['POST', 'GET'])
@permission_classes([AllowAny])
def metrics(request):
    """
    Calculates and returns global metrics
    
    GET: Metrics based on current DB
    POST: Metrics based on JSON provided
    
    Input (POST - optional):
    {
        "predictions": [
            {"prediction": "Confirmed", "probability": 0.92},
            {"prediction": "Candidate", "probability": 0.65},
            ...
        ]
    }
    
    Output:
    {
        "total_predictions": 100,
        "confirmed": 60,
        "candidate": 25,
        "false_positive": 15,
        "average_probability": 0.78,
        "confidence_distribution": {
            "high": 70,
            "medium": 20,
            "low": 10
        }
    }
    """
    try:
        if request.method == 'POST' and request.data:
            predictions_data = request.data.get('predictions', [])
            
            if not predictions_data:
                return Response(
                    {"error": "The 'predictions' field is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            total = len(predictions_data)
            confirmed = sum(1 for p in predictions_data if 'Confirmed' in str(p.get('prediction', '')))
            candidate = sum(1 for p in predictions_data if 'Candidate' in str(p.get('prediction', '')))
            false_pos = sum(1 for p in predictions_data if 'False' in str(p.get('prediction', '')))
            
            probs = [p.get('probability', 0) for p in predictions_data if 'probability' in p]
            avg_prob = sum(probs) / len(probs) if probs else 0
            
            high_conf = sum(1 for p in probs if p > 0.8)
            medium_conf = sum(1 for p in probs if 0.5 < p <= 0.8)
            low_conf = sum(1 for p in probs if p <= 0.5)
            
        else:
            total = Prediction.objects.count()
            confirmed = Prediction.objects.filter(prediction__icontains="Confirmed").count()
            candidate = Prediction.objects.filter(prediction__icontains="Candidate").count()
            false_pos = Prediction.objects.filter(prediction__icontains="False").count()
            
            from django.db.models import Avg
            avg_prob = Prediction.objects.aggregate(Avg('probability'))['probability__avg'] or 0
            
            high_conf = Prediction.objects.filter(probability__gt=0.8).count()
            medium_conf = Prediction.objects.filter(probability__gt=0.5, probability__lte=0.8).count()
            low_conf = Prediction.objects.filter(probability__lte=0.5).count()
        
        metrics_data = {
            "total_predictions": total,
            "confirmed": confirmed,
            "candidate": candidate,
            "false_positive": false_pos,
            "confirmed_percentage": round((confirmed / total * 100) if total > 0 else 0, 2),
            "average_probability": round(avg_prob, 4),
            "confidence_distribution": {
                "high": high_conf,
                "medium": medium_conf,
                "low": low_conf
            }
        }
        
        return Response(metrics_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return Response(
            {"error": "Error calculating metrics", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def graph1(request):
    """
    Processes a JPG image file
    
    Input: JPG file
    Output: 
    {
        "status": "success",
        "image_url": "/media/graphs/graph1_processed.jpg",
        "metadata": {
            "filename": "graph1.jpg",
            "size": 12345,
            "format": "JPEG"
        }
    }
    """
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
        
        graphs_dir = os.path.join(settings.MEDIA_ROOT, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"graph1_{timestamp}_{image_file.name}"
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
        logger.error(f"Error processing graph1 image: {str(e)}")
        return Response(
            {"error": "Error processing image", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def graph2(request):
    """
    Processes a second JPG image file
    
    Input: JPG file
    Output: 
    {
        "status": "success",
        "image_url": "/media/graphs/graph2_processed.jpg",
        "metadata": {
            "filename": "graph2.jpg",
            "size": 12345,
            "format": "JPEG"
        }
    }
    """
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
        
        graphs_dir = os.path.join(settings.MEDIA_ROOT, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"graph2_{timestamp}_{image_file.name}"
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
        logger.error(f"Error processing graph2 image: {str(e)}")
        return Response(
            {"error": "Error processing image", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )