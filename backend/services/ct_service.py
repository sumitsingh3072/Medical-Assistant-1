# backend/services/ct_service.py

import os
import torch
from models.ct_model import load_ct_model, predict_ct

# ---------------------------
# Initialize and cache CT models
# ---------------------------
_ct_models = {}

def init_ct_models(device: str = "cpu") -> None:
    """
    Load and cache both 2D and 3D CT models into memory.
    Call this once at application startup.
    """
    # Load 2D model
    _ct_models['2d'] = load_ct_model(mode='2d', device=device)
    # Load 3D model
    _ct_models['3d'] = load_ct_model(mode='3d', device=device)

# ---------------------------
# CT Processing Service
# ---------------------------
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
    if mode not in _ct_models:
        raise ValueError(f"Unsupported mode '{mode}'. Choose '2d' or '3d'.")

    model = _ct_models[mode]
    # Run prediction
    results = predict_ct(model, image_path, mode=mode, device=device)
    return results

# ---------------------------
# Optional: Utility to validate input
# ---------------------------
def is_supported_ct_file(filename: str, mode: str) -> bool:
    """
    Check if the file extension is supported for the given mode.
    """
    ext = os.path.splitext(filename)[1].lower()
    if mode == '2d':
        return ext in ['.png', '.jpg', '.jpeg']
    elif mode == '3d':
        return ext in ['.nii', '.nii.gz', '.dcm']
    else:
        return False
