"""
Main URLs for exoplanet_api project
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
    openapi.Info(
        title="Exoplanet Detection API",
        default_version='v1',
        description="""
        REST API for exoplanet detection using Machine Learning.
        Available endpoints:
        - POST /api/predict : Predict if a planet is confirmed
        - GET /api/model-info : Get ML model information
        - POST /api/retrain : Retrain the model (admin only)
        """,
        terms_of_service="https://www.nasa.gov/",
        contact=openapi.Contact(email="combarinahine@gmail.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('predictions.urls')),
    path('', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('api/docs/', schema_view.with_ui('swagger', cache_timeout=0), name='api-docs'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)