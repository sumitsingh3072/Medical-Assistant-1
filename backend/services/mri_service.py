import threading
from pathlib import Path
from models.mri_model import load_mri_model, predict_mri

# Resolve backend root (one level up from services/)
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Cache
_cache_mri = {}
_lock = threading.Lock()


def _ensure_mri_model(mode: str = '3d', device: str = 'cpu') -> None:
    """Lazy-load only the requested MRI mode on first use."""
    if mode not in _cache_mri:
        with _lock:
            if mode not in _cache_mri:
                print(f"Lazy-loading MRI {mode} model...")
                _cache_mri[mode] = load_mri_model(mode, device)


def init_mri_models(device='cpu'):
    # Load only 3D by default; uncomment 2D if needed
    _ensure_mri_model('3d', device)


def process_mri(path: str, mode: str = '3d', device: str = 'cpu', top_k: int = 2):
    _ensure_mri_model(mode, device)
    return predict_mri(_cache_mri[mode], path, mode, device, top_k)


def is_supported_mri_file(filename: str, mode: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in (['.png', '.jpg', '.jpeg'] if mode == '2d' else ['.nii', '.nii.gz', '.dcm'])
