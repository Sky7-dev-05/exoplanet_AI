"""
Configuration de l'interface admin Django
"""
from django.contrib import admin
from .models import Prediction, ModelInfo


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """
    Interface admin pour visualiser l'historique des prédictions
    """
    list_display = [
        'id',
        'prediction',
        'probability_percent',
        'koi_period',
        'koi_prad',
        'created_at',
        'ip_address'
    ]
    list_filter = ['prediction', 'created_at']
    search_fields = ['prediction', 'ip_address']
    ordering = ['-created_at']
    readonly_fields = ['created_at']
    
    def probability_percent(self, obj):
        """Affiche la probabilité en pourcentage"""
        return f"{obj.probability * 100:.1f}%"
    probability_percent.short_description = "Probabilité"
    
    fieldsets = (
        ('Données d\'entrée', {
            'fields': (
                'koi_score', 'koi_period', 'koi_impact', 'koi_duration', 
                'koi_depth', 'koi_prad', 'koi_sma', 'koi_teq', 'koi_model_snr'
            )
        }),
        ('Résultat', {
            'fields': ('prediction', 'probability', 'koi_disposition')
        }),
        ('Métadonnées', {
            'fields': ('created_at', 'ip_address'),
            'classes': ('collapse',)
        }),
    )
