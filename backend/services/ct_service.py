import os
import threading
from pathlib import Path
from models.ct_model import load_ct_model, predict_ct

# Cache
_ct_models = {}
_lock = threading.Lock()


def _ensure_ct_model(mode: str, device: str = "cpu") -> None:
    """Lazy-load only the requested CT mode (2d or 3d) on first use."""
    if mode not in _ct_models:
        with _lock:
            if mode not in _ct_models:
                print(f"Lazy-loading CT {mode} model...")
                _ct_models[mode] = load_ct_model(mode=mode, device=device)


# Initialize — kept for backward compat but no longer called at startup
def init_ct_models(device: str = "cpu") -> None:
    """
    Load and cache both 2D and 3D CT models into memory.
    Call this once at application startup.
    """
    _ensure_ct_model('2d', device)
    _ensure_ct_model('3d', device)


# Process
def process_ct(image_path: str, mode: str = '2d', device: str = "cpu"):
    """
    Process a CT image or volume and return predictions.

    Args:
        image_path: Path to the CT image file (2D slice) or NIfTI/DICOM volume.
        mode: '2d' for slice classification, '3d' for volume analysis.
        device: Device to run inference on ('cpu' or 'cuda').

    Returns:
        For '2d': List of (class, probability) tuples sorted by probability.
        For '3d': Raw model output array (e.g., segmentation map or logits).
    """
    _ensure_ct_model(mode, device)

    model = _ct_models[mode]
    # Run prediction
    results = predict_ct(model, image_path, mode=mode, device=device)
    return results


# Validator
def is_supported_ct_file(fn, mode):
    ext = Path(fn).suffix.lower()
    return ext in (['.png','.jpg','.jpeg'] if mode=='2d' else ['.nii','.nii.gz','.dcm'])