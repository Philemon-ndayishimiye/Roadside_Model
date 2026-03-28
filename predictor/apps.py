from django.apps import AppConfig

class PredictorConfig(AppConfig):
    name = 'predictor'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        from predictor.model_loader import load_models
        load_models()