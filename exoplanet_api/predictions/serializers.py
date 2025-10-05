"""
Serializers for data validation and transformation
"""
from rest_framework import serializers
from .models import Prediction, ModelInfo


class PredictionInputSerializer(serializers.Serializer):
    """Validates input data for prediction (Kepler columns)"""
    koi_score = serializers.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        default=0.5,
        help_text="KOI score (0-1)"
    )
    koi_period = serializers.FloatField(
        required=True,
        min_value=0.001,  # ✅ CHANGÉ ICI
        help_text="Orbital period in days (REQUIRED, must be > 0)"
    )
    koi_impact = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Impact parameter"
    )
    koi_duration = serializers.FloatField(
        required=True,
        min_value=0.001,  # ✅ CHANGÉ ICI
        help_text="Transit duration in hours (REQUIRED, must be > 0)"
    )
    koi_depth = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Transit depth (ppm)"
    )
    koi_prad = serializers.FloatField(
        required=True,
        min_value=0.001,  # ✅ CHANGÉ ICI
        help_text="Planetary radius in Earth radii (REQUIRED, must be > 0)"
    )
    koi_sma = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Semi-major axis (AU)"
    )
    koi_teq = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0.0,
        default=300.0,
        help_text="Equilibrium temperature (K)"
    )
    koi_model_snr = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Signal-to-noise ratio"
    )


class PredictionOutputSerializer(serializers.Serializer):
    """Prediction response format"""
    prediction = serializers.CharField(help_text="Confirmed, Candidate, or False Positive")
    probability = serializers.FloatField(help_text="Probability between 0 and 1")
    confidence = serializers.CharField(help_text="High, Medium, or Low")
    message = serializers.CharField(help_text="Explanatory message")


class PredictionHistorySerializer(serializers.ModelSerializer):
    """Serializer for prediction history"""
    class Meta:
        model = Prediction
        fields = [
            'id',
            'koi_score',
            'koi_period',
            'koi_impact',
            'koi_duration',
            'koi_depth',
            'koi_prad',
            'koi_sma',
            'koi_teq',
            'koi_model_snr',
            'prediction',
            'probability',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ModelInfoSerializer(serializers.ModelSerializer):
    """Serializer for model information"""
    features_list = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelInfo
        fields = [
            'version',
            'accuracy',
            'f1_score',
            'trained_on',
            'features_list',
            'total_predictions',
            'is_active'
        ]
    
    def get_features_list(self, obj):
        """Converts features string to list"""
        return obj.features_used.split(',') if obj.features_used else []


class CSVUploadSerializer(serializers.Serializer):
    """Validates CSV file upload"""
    file = serializers.FileField(
        required=True,
        help_text="CSV file containing data to predict"
    )
    
    def validate_file(self, value):
        """Checks that it's a CSV file"""
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("File must be in CSV format")
        
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File is too large (max 10 MB)")
        
        return value


class BatchPredictionOutputSerializer(serializers.Serializer):
    """Response format for batch predictions (CSV)"""
    total_predictions = serializers.IntegerField()
    predictions = PredictionOutputSerializer(many=True)
    summary = serializers.DictField()