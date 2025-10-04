"""
Serializers pour la validation et transformation des données
"""
from rest_framework import serializers
from .models import Prediction, ModelInfo


class PredictionInputSerializer(serializers.Serializer):
    """
    Valide les données d'entrée pour la prédiction (colonnes Kepler)
    """
    koi_score = serializers.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        default=0.5,
        help_text="KOI score (0-1)"
    )
    koi_period = serializers.FloatField(
        required=True,
        min_value=0.0,
        help_text="Période orbitale en jours"
    )
    koi_impact = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Paramètre d'impact"
    )
    koi_duration = serializers.FloatField(
        required=True,
        min_value=0.0,
        help_text="Durée du transit en heures"
    )
    koi_depth = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Profondeur du transit (ppm)"
    )
    koi_prad = serializers.FloatField(
        required=True,
        min_value=0.0,
        help_text="Rayon planétaire (rayons terrestres)"
    )
    koi_sma = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Demi-grand axe (AU)"
    )
    koi_teq = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0.0,
        default=300.0,
        help_text="Température d'équilibre (K)"
    )
    koi_model_snr = serializers.FloatField(
        required=False,
        min_value=0.0,
        default=0.0,
        help_text="Signal-to-noise ratio"
    )


class PredictionOutputSerializer(serializers.Serializer):
    """
    Format de la réponse de prédiction
    """
    prediction = serializers.CharField(help_text="Confirmed, Candidate, ou False Positive")
    probability = serializers.FloatField(help_text="Probabilité entre 0 et 1")
    confidence = serializers.CharField(help_text="High, Medium, ou Low")
    message = serializers.CharField(help_text="Message explicatif")


class PredictionHistorySerializer(serializers.ModelSerializer):
    """
    Serializer pour l'historique des prédictions
    """
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
    """
    Serializer pour les informations du modèle
    """
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
        """Convertit le string features en liste"""
        return obj.features_used.split(',') if obj.features_used else []


class CSVUploadSerializer(serializers.Serializer):
    """
    Valide l'upload de fichier CSV
    """
    file = serializers.FileField(
        required=True,
        help_text="Fichier CSV contenant les données à prédire"
    )
    
    def validate_file(self, value):
        """Vérifie que c'est bien un CSV"""
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("Le fichier doit être au format CSV")
        
        # Vérifier la taille (10 MB max)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Le fichier est trop volumineux (max 10 MB)")
        
        return value


class BatchPredictionOutputSerializer(serializers.Serializer):
    """
    Format de réponse pour les prédictions en batch (CSV)
    """
    total_predictions = serializers.IntegerField()
    predictions = PredictionOutputSerializer(many=True)
    summary = serializers.DictField()