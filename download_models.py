# download_models.py
import os
os.environ['TFHUB_CACHE_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'use_cache'
)

from huggingface_hub import hf_hub_download
import tensorflow_hub as hub

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

REPO_ID = "philemonndayi/Roadaid_model"  # ← your HuggingFace repo

FILES = [
    'RandomForest_fault_model.pkl',
    'le_fault.pkl',
    'fault_classes.json',
]

# Download pkl files from HuggingFace
for filename in FILES:
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f'⏳ Downloading {filename}...')
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODELS_DIR)
        print(f'✅ {filename} done!')
    else:
        print(f'✅ {filename} already exists')

# Download and cache USE during build
print('⏳ Downloading Universal Sentence Encoder...')
hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
print('✅ USE cached!')