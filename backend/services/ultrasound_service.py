# backend/services/ultrasound_service.py

import os
import torch
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(root)
from backend.models.ultrasound_model import (
    load_ultrasound_model,
    predict_ultrasound,
)

# -------------------------------------
# Default path to the USFM checkpoint
# -------------------------------------
DEFAULT_ULTRASOUND_CHECKPOINT = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\ultrasound\USFM_latest.pth"

# -------------------------------------
# Cache for Ultrasound Model
# -------------------------------------
_ultrasound_model = None

# -------------------------------------
# Initialize and load ultrasound model
# -------------------------------------
def init_ultrasound_model(
    checkpoint_path: str = DEFAULT_ULTRASOUND_CHECKPOINT,
    device: str = "cpu",
) -> None:
    """
    Load and cache the USFM-based Ultrasound model at application startup.

    Args:
        checkpoint_path: Path to the USFM foundation weights (.pth)
        device: 'cpu' or 'cuda'
    """
    global _ultrasound_model

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Ultrasound weights not found at {checkpoint_path}")

    # load_ultrasound_model handles instantiation, checkpoint loading, and .eval()
    _ultrasound_model = load_ultrasound_model(
        device=device,
        checkpoint_path=checkpoint_path
    )

# Automatically try to init on import (optional)
try:
    init_ultrasound_model()
except Exception:
    # Will be re-initialized explicitly later if needed
    _ultrasound_model = None

# -------------------------------------
# Ultrasound Processing Service
# -------------------------------------
def process_ultrasound(
    image_path: str,
    device: str = "cpu",
    top_k: int = 2,
) -> list[tuple[str, float]]:
    """
    Process an ultrasound image and return the top_k predicted conditions.

    Args:
        image_path: Path to the ultrasound image file (.png, .jpg, .jpeg, .bmp)
        device: 'cpu' or 'cuda'
        top_k: Number of top predictions to return

    Returns:
        List of (class_label, probability) tuples sorted by probability.
    """
    if _ultrasound_model is None:
        raise RuntimeError(
            "Ultrasound model has not been initialized. "
            "Call init_ultrasound_model() first."
        )

    # Validate file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
        raise ValueError(
            f"Unsupported ultrasound file type '{ext}'. "
            "Supported types: .png, .jpg, .jpeg, .bmp"
        )

    # Ensure model is on correct device
    _ultrasound_model.to(device)
    _ultrasound_model.eval()

    # Run prediction
    results = predict_ultrasound(
        model=_ultrasound_model,
        image_path=image_path,
        device=device,
        top_k=top_k
    )
    return results
