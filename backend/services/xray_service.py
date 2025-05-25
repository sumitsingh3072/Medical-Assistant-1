# backend/services/xray_service.py

import os
from pathlib import Path
import torch
from models.xray_model import load_chexnet_model, predict_xray

# Resolve project root and weight path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / 'model_assests' / 'xray' / 'xray.pth.tar'

_xray_model = None

def init_xray_model(weight_path: Path = DEFAULT_WEIGHT_PATH, device: str = 'cpu') -> None:
    """
    Load and cache the CheXNet X-ray model.
    """
    global _xray_model

    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"X-ray weights not found at {weight_path}")

    _xray_model = load_chexnet_model(str(weight_path), device=device)

# Try initializing on import
try:
    init_xray_model()
except Exception as e:
    print(f"Warning: could not load X-ray model: {e}")


def process_xray(image_path: str, device: str = 'cpu', top_k: int = 3) -> list:
    if _xray_model is None:
        raise RuntimeError("X-ray model not initialized. Call init_xray_model() first.")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
        raise ValueError(f"Unsupported file type: {ext}")

    return predict_xray(_xray_model, image_path, top_k=top_k, device=device)
