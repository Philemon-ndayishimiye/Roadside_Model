from django.urls import path
from .views import PredictFaultView, HealthCheckView

urlpatterns = [
    path('predict/', PredictFaultView.as_view(), name='predict'),
    path('health/', HealthCheckView.as_view(), name='health'),
]