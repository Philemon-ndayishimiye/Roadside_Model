import os
import pickle
from sentence_transformers import SentenceTransformer

MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
)

_embed_model = None
_rf_model    = None
_le_fault    = None

def load_models():
    global _embed_model, _rf_model, _le_fault

    if _embed_model is not None:
        return _embed_model, _rf_model, _le_fault

    print('⏳ Loading sentence encoder...')
    _embed_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    print('✅ Encoder loaded!')

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