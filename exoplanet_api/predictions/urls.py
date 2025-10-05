"""
API predictions routes
"""
from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('predict/', views.predict_exoplanet, name='predict'),
    path('predict-batch/', views.predict_batch_endpoint, name='predict-batch'),
    path('model-info/', views.model_info, name='model-info'),
    path('history/', views.prediction_history, name='history'),
    path('stats/', views.statistics, name='stats'),
    path('retrain/', views.retrain_model, name='retrain'),
    path('metrics/', views.metrics, name='metrics'),
    path('graph1/', views.graph1, name='graph1'),
    path('graph2/', views.graph2, name='graph2'),
    path('graph1/latest/', views.get_latest_graph1, name='get-latest-graph1'),
    path('graph2/latest/', views.get_latest_graph2, name='get-latest-graph2'),
    
]