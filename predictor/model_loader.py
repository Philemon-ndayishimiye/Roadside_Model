import os
import pickle
import gdown
import tensorflow_hub as hub

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models'
)

USE_CACHE_DIR = os.path.join(MODELS_DIR, 'use_cache')

DRIVE_FILES = {
    'RandomForest_fault_model.pkl': '1FM_Ke2IUO1Q7jYxsFN4ZSw5Fqpjszb6v',
    'le_fault.pkl'                : '1sZy3pBbGbv8kpST5SY3luhGuVQQOsi0I',
    'fault_classes.json'          : '1b7uRZ5f6k_F2cCHj_A-oRUlQKxAgigqd',
}

# Module-level cache — loaded ONCE, reused forever
_embed_model = None
_rf_model    = None
_le_fault    = None


def download_if_missing():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(USE_CACHE_DIR, exist_ok=True)

    for filename, file_id in DRIVE_FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f'⏳ Downloading {filename} from Google Drive...')
            gdown.download(f'https://drive.google.com/uc?id={file_id}', filepath, quiet=False)
            print(f'✅ {filename} downloaded!')
        else:
            print(f'✅ {filename} already exists — skipping')


def load_models():
    """
    Called ONCE at Django startup via apps.py ready().
    Subsequent calls return the cached models immediately.
    """
    global _embed_model, _rf_model, _le_fault

    # Return cached models if already loaded
    if _embed_model is not None:
        return _embed_model, _rf_model, _le_fault

    download_if_missing()

    # Cache USE locally so it doesn't re-download on every deploy
    print('⏳ Loading Universal Sentence Encoder...')
    os.environ['TFHUB_CACHE_DIR'] = USE_CACHE_DIR
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