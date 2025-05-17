# backend/services/mri_service.py

import os
import torch
from models.mri_model import load_mri_model, predict_mri

# ---------------------------
# Cache for MRI models
# ---------------------------
_mri_models = {}
DEFAULT_MRI_MODEL_2D = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\mri\2d\model.h5"
DEFAULT_MRI_MODEL_3D = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\mri\3d\resnet_200.pth"
# ---------------------------
# Initialize and load MRI models
# ---------------------------
def init_mri_models(device: str = "cpu") -> None:
    """
    Load and cache both 2D and 3D MRI models at application startup.
    """
    # Load 2D MRI classification model
    # _mri_models['2d'] = load_mri_model(mode='2d', device=device)
    # Load 3D MRI segmentation/classification model
    _mri_models['3d'] = load_mri_model(mode='3d',device=device)

# ---------------------------
# MRI Processing Service
# ---------------------------
def process_mri(input_path: str, mode: str = '2d', device: str = "cpu", top_k: int = 2):
    """
    Process MRI data and return predictions or segmentation output.

    Args:
        input_path: Path to MRI slice (2D) or volume (NIfTI/DICOM).
        mode: '2d' for slice classification, '3d' for volume analysis.
        device: 'cpu' or 'cuda'.
        top_k: Number of top predictions for classification.

    Returns:
        For '2d': List of (class, probability) tuples.
        For '3d': Raw output array (e.g., segmentation map or logits).
    """
    if mode not in _mri_models:
        raise ValueError(f"Unsupported mode '{mode}'. Choose '2d' or '3d'.")

    model = _mri_models[mode]
    results = predict_mri(model, input_path, mode=mode, device=device, top_k=top_k)
    return results

# ---------------------------
# Utility: Supported file check
# ---------------------------
def is_supported_mri_file(filename: str, mode: str) -> bool:
    """
    Check if the file extension is supported for the given mode.
    """
    ext = os.path.splitext(filename)[1].lower()
    if mode == '2d':
        return ext in ['.png', '.jpg', '.jpeg']
    elif mode == '3d':
        return ext in ['.nii', '.nii.gz', '.dcm']
    return False
