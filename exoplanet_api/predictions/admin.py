"""
Django admin interface configuration
"""
from django.contrib import admin
from .models import Prediction, ModelInfo


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """Admin interface for viewing prediction history"""
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
        """Displays probability as percentage"""
        return f"{obj.probability * 100:.1f}%"
    probability_percent.short_description = "Probability"

    fieldsets = (
        ('Input Data', {
            'fields': (
                'koi_score', 'koi_period', 'koi_impact', 'koi_duration',
                'koi_depth', 'koi_prad', 'koi_sma', 'koi_teq', 'koi_model_snr'
            )
        }),
        ('Result', {
            'fields': ('prediction', 'probability', 'koi_disposition')
        }),
        ('Metadata', {
            'fields': ('created_at', 'ip_address'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    """Admin interface for model information"""
    list_display = [
        'version',
        'accuracy_percent',
        'f1_score',
        'trained_on',
        'total_predictions',
        'is_active'
    ]
    list_filter = ['is_active', 'trained_on']
    search_fields = ['version']
    ordering = ['-trained_on']
    readonly_fields = ['trained_on', 'total_predictions']

    def accuracy_percent(self, obj):
        """Displays accuracy as percentage"""
        return f"{obj.accuracy * 100:.1f}%"
    accuracy_percent.short_description = "Accuracy"

    fieldsets = (
        ('Model Information', {
            'fields': ('version', 'accuracy', 'f1_score', 'features_used')
        }),
        ('Status', {
            'fields': ('trained_on', 'total_predictions', 'is_active')
        }),
    )