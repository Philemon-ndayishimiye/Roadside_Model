# download_models.py
import os
import gdown

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

DRIVE_FILES = {
    'RandomForest_fault_model.pkl': '1FM_Ke2IUO1Q7jYxsFN4ZSw5Fqpjszb6v',
    'le_fault.pkl'                : '1sZy3pBbGbv8kpST5SY3luhGuVQQOsi0I',
    'fault_classes.json'          : '1b7uRZ5f6k_F2cCHj_A-oRUlQKxAgigqd',
}

os.makedirs(MODELS_DIR, exist_ok=True)

for filename, file_id in DRIVE_FILES.items():
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f'⏳ Downloading {filename}...')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', filepath, quiet=False)
        print(f'✅ {filename} done!')
    else:
        print(f'✅ {filename} already exists')