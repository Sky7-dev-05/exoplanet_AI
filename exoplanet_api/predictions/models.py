"""
Django models for prediction history
"""
from django.db import models
from django.utils import timezone


class Prediction(models.Model):
    koi_score = models.FloatField(null=True, blank=True)
    koi_period = models.FloatField(help_text="Orbital period in days")
    koi_impact = models.FloatField(null=True, blank=True)
    koi_duration = models.FloatField(help_text="Transit duration in hours")
    koi_depth = models.FloatField(null=True, blank=True)
    koi_prad = models.FloatField(help_text="Planetary radius (in Earth radii)")
    koi_sma = models.FloatField(null=True, blank=True)
    koi_teq = models.FloatField(null=True, blank=True)
    koi_model_snr = models.FloatField(null=True, blank=True)

    prediction = models.CharField(max_length=50, help_text="Confirmed, Candidate, or False Positive")
    probability = models.FloatField(help_text="Prediction probability (0-1)")
    koi_disposition = models.CharField(max_length=50, null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prediction"
        verbose_name_plural = "Predictions"

    def __str__(self):
        return f"{self.prediction} ({self.probability:.2%}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class ModelInfo(models.Model):
    """Stores information about the current ML model"""
    version = models.CharField(max_length=20, unique=True)
    accuracy = models.FloatField(help_text="Précision du modèle")
    f1_score = models.FloatField(help_text="F1 Score", null=True, blank=True)
    precision = models.FloatField(help_text="Precision", null=True, blank=True, default=0.88)  # AJOUTE
    recall = models.FloatField(help_text="Recall", null=True, blank=True, default=0.89)  # AJOUTE
    trained_on = models.DateTimeField(help_text="Date d'entraînement")
    features_used = models.TextField(help_text="Liste des features utilisées")
    total_predictions = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-trained_on']
        verbose_name = "Model Info"
        verbose_name_plural = "Model Infos"

    def __str__(self):
        return f"Model v{self.version} - Accuracy: {self.accuracy:.2%}"