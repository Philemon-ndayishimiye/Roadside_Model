from django.apps import AppConfig
import os

class PredictorConfig(AppConfig):
    name = 'predictor'

    def ready(self):
        if os.environ.get('RUN_MAIN') != 'true':
            from predictor.model_loader import load_models
            load_models()