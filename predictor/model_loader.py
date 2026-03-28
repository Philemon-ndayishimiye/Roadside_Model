# model_loader.py
# Downloads ML model files from Google Drive on first startup
# This way we don't need to push large files to GitHub

import os
import pickle
import requests
import gdown          # pip install gdown
import tensorflow_hub as hub

# ── Folder where models will be saved ─────────────────────────
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models'
)

# ── Google Drive File IDs ──────────────────────────────────────
# Replace these with your actual Google Drive file IDs
DRIVE_FILES = {
    'RandomForest_fault_model.pkl': '1FM_Ke2IUO1Q7jYxsFN4ZSw5Fqpjszb6v',
    'le_fault.pkl'                : '1sZy3pBbGbv8kpST5SY3luhGuVQQOsi0I',
    'fault_classes.json'          : '1b7uRZ5f6k_F2cCHj_A-oRUlQKxAgigqd',
}





def download_if_missing():
    """
    Checks if model files exist locally.
    If not, downloads them from Google Drive.
    Called once when Django starts.
    """
    # Create models folder if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, file_id in DRIVE_FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)

        # Only download if file doesn't exist yet
        if not os.path.exists(filepath):
            print(f'⏳ Downloading {filename} from Google Drive...')
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)
            print(f'✅ {filename} downloaded!')
        else:
            print(f'✅ {filename} already exists — skipping download')


def load_models():
    """
    Downloads models if needed then loads them into memory.
    Returns: embed_model, rf_model, le_fault
    """
    # Download missing files first
    download_if_missing()

    # ── Load USE ──────────────────────────────────────────────
    print('⏳ Loading Universal Sentence Encoder...')
    embed_model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder/4'
    )
    print('✅ USE loaded!')

    # ── Load Random Forest ────────────────────────────────────
    print('⏳ Loading Random Forest...')
    rf_path = os.path.join(MODELS_DIR, 'RandomForest_fault_model.pkl')
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    print('✅ Random Forest loaded!')

    # ── Load Label Encoder ────────────────────────────────────
    print('⏳ Loading Label Encoder...')
    le_path = os.path.join(MODELS_DIR, 'le_fault.pkl')
    with open(le_path, 'rb') as f:
        le_fault = pickle.load(f)
    print('✅ Label Encoder loaded!')

    print('🚀 All models ready!')
    return embed_model, rf_model, le_fault