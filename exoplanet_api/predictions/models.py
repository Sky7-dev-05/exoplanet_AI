"""
Modèles Django pour l'historique des prédictions
"""
from django.db import models
from django.utils import timezone


from django.db import models
from django.utils import timezone

class Prediction(models.Model):
    # Données d'entrée (features Powell)
    koi_score = models.FloatField(null=True, blank=True)
    koi_period = models.FloatField(help_text="Période orbitale en jours")
    koi_impact = models.FloatField(null=True, blank=True)
    koi_duration = models.FloatField(help_text="Durée du transit en heures")
    koi_depth = models.FloatField(null=True, blank=True)
    koi_prad = models.FloatField(help_text="Rayon planétaire (en rayons terrestres)")
    koi_sma = models.FloatField(null=True, blank=True)
    koi_teq = models.FloatField(null=True, blank=True)
    koi_model_snr = models.FloatField(null=True, blank=True)
    
    # Résultat de la prédiction
    prediction = models.CharField(max_length=50, help_text="Confirmed, Candidate, ou False Positive")
    probability = models.FloatField(help_text="Probabilité de la prédiction (0-1)")
    koi_disposition = models.CharField(max_length=50, null=True, blank=True)  # Classe réelle connue (optionnelle)
    
    # Métadonnées
    created_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prédiction"
        verbose_name_plural = "Prédictions"

    def __str__(self):
        return f"{self.prediction} ({self.probability:.2%}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class ModelInfo(models.Model):
    """
    Stocke les informations sur le modèle ML actuel
    """
    version = models.CharField(max_length=20, unique=True)
    accuracy = models.FloatField(help_text="Précision du modèle")
    f1_score = models.FloatField(help_text="F1 Score", null=True, blank=True)
    trained_on = models.DateTimeField(help_text="Date d'entraînement")
    features_used = models.TextField(help_text="Liste des features utilisées")
    total_predictions = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-trained_on']
        verbose_name = "Info Modèle"
        verbose_name_plural = "Infos Modèles"
    
    def __str__(self):
        return f"Model v{self.version} - Accuracy: {self.accuracy:.2%}"