# model_loader.py
import os
import pickle
import tensorflow_hub as hub

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models'
)
USE_CACHE_DIR = os.path.join(MODELS_DIR, 'use_cache')

_embed_model = None
_rf_model    = None
_le_fault    = None

def load_models():
    global _embed_model, _rf_model, _le_fault

    if _embed_model is not None:
        return _embed_model, _rf_model, _le_fault

    os.makedirs(USE_CACHE_DIR, exist_ok=True)
    os.environ['TFHUB_CACHE_DIR'] = USE_CACHE_DIR

    print('⏳ Loading Universal Sentence Encoder...')
    _embed_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    print('✅ USE loaded!')

    print('⏳ Loading Random Forest...')
    with open(os.path.join(MODELS_DIR, 'RandomForest_fault_model.pkl'), 'rb') as f:
        _rf_model = pickle.load(f)
    print('✅ Random Forest loaded!')

    print('⏳ Loading Label Encoder...')
    with open(os.path.join(MODELS_DIR, 'le_fault.pkl'), 'rb') as f:
        _le_fault = pickle.load(f)
    print('✅ Label Encoder loaded!')

    print('🚀 All models ready!')
    return _embed_model, _rf_model, _le_fault