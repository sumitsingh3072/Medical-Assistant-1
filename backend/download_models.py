import os
import sys
from huggingface_hub import snapshot_download

print("\n⬇️ Attempting to download models from Hugging Face...")

# Safely check for the secret token
token_path = '/run/secrets/HF_TOKEN'
token = open(token_path).read().strip() if os.path.exists(token_path) else None

try:
    snapshot_download(
        repo_id='sumitsingh3072/medivision-weights',
        local_dir='/app/model_assests',
        local_dir_use_symlinks=False,
        token=token
    )
    print("\n✅ DOWNLOAD SUCCESSFUL!")
except Exception as e:
    print(f"\n\n❌ DOWNLOAD ERROR: {e}\n\n")
    sys.exit(1)