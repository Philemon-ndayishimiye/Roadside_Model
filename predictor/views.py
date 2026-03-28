import os
import re
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .model_loader import load_models

# Load all models once when Django starts
embed_model, rf_model, le_fault = load_models()


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s,\.\-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


class HealthCheckView(APIView):
    def get(self, request):
        return Response({
            'status' : '✅ API is running',
            'model'  : 'Random Forest',
            'classes': len(le_fault.classes_)
        })


class PredictFaultView(APIView):
    def post(self, request):
        symptom = request.data.get('symptom', '').strip()

        if not symptom:
            return Response(
                {'error': 'symptom field is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            cleaned   = clean_text(symptom)
            embedding = embed_model.encode([cleaned], convert_to_numpy=True)  # ← CHANGED
            pred      = rf_model.predict(embedding)[0]
            proba     = rf_model.predict_proba(embedding)[0]
            confidence  = float(proba.max() * 100)
            fault_label = le_fault.inverse_transform([pred])[0]

            top3_idx = np.argsort(proba)[::-1][:3]
            top3 = [
                {
                    'fault'     : le_fault.classes_[i],
                    'confidence': round(float(proba[i] * 100), 2)
                }
                for i in top3_idx
            ]

            return Response({
                'symptom'        : cleaned,
                'predicted_fault': fault_label,
                'confidence'     : round(confidence, 2),
                'top3'           : top3
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )