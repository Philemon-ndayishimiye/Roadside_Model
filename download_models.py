# download_models.py
import os
from huggingface_hub import hf_hub_download

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

REPO_ID = "philemonndayi/Roadaid_model"  # ← replace this

FILES = [
    'RandomForest_fault_model.pkl',
    'le_fault.pkl',
    'fault_classes.json',
]

for filename in FILES:
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f'⏳ Downloading {filename}...')
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR
        )
        print(f'✅ {filename} done!')
    else:
        print(f'✅ {filename} already exists')
