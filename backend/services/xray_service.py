# backend/services/xray_service.py

import os
import torch
from models.xray_model import load_chexnet_model, predict_xray

# Default path to the CheXNet weights
DEFAULT_WEIGHT_PATH = r"C:\ML_Projects\Medical-Assistant-1\model_assests\xray\xray.pth.tar"

# ---------------------------
# Cache for X-ray Model
# ---------------------------
_xray_model = None

# ---------------------------
# Initialize and load X-ray model
# ---------------------------
def init_xray_model(weight_path: str = DEFAULT_WEIGHT_PATH, device: str = "cpu") -> None:
    """
    Load and cache the CheXNet X-ray model at application startup.

    Args:
        weight_path: Path to CheXNet model weights file (.pth.tar)
        device: 'cpu' or 'cuda'
    """

    global _xray_model


    # Validate the weights path
    if not os.path.exists(weight_path):
        print(f"Trying to load X-ray weights from: {weight_path}")
        print(f"File exists? {os.path.exists(weight_path)}")
        raise FileNotFoundError(f"X-ray weights not found at {weight_path}")

    _xray_model = load_chexnet_model(weight_path=weight_path, device=device)

# Automatically initialize with default weights on import (optional)
try:
    init_xray_model()
except Exception:
    # Model will be initialized explicitly later if needed
    pass

# ---------------------------
# X-ray Processing Service
# ---------------------------
def process_xray(image_path: str, device: str = "cpu", top_k: int = 3) -> list:
    """
    Process an X-ray image and return top-k predicted conditions.

    Args:
        image_path: Path to the X-ray image file (.png, .jpg, .jpeg, .bmp)
        device: 'cpu' or 'cuda'
        top_k: Number of top predictions to return

    Returns:
        List of (class_label, probability) tuples sorted by probability.
    """
    if _xray_model is None:
        raise RuntimeError(
            "X-ray model has not been initialized. "
            "Ensure init_xray_model() is called with the correct weight path before processing."
        )

    # Validate file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
        raise ValueError(
            f"Unsupported X-ray file type '{ext}'. Supported types: .png, .jpg, .jpeg, .bmp"
        )

    # Run prediction
    results = predict_xray(_xray_model, image_path, top_k=top_k, device=device)
    return results
