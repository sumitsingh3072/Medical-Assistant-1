from pathlib import Path
import torch
from models.ultrasound_model import load_ultrasound_model, predict_ultrasound

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ULTRASOUND_CHECKPOINT = PROJECT_ROOT / 'model_assests' / 'ultrasound' / 'USFM_latest.pth'

_ultrasound_model = None

def init_ultrasound_model(device: str = 'cpu', checkpoint_path: Path = DEFAULT_ULTRASOUND_CHECKPOINT) -> None:
    global _ultrasound_model
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ultrasound weights not found at {checkpoint_path}")
    _ultrasound_model = load_ultrasound_model(device=device, checkpoint_path=(checkpoint_path))

try:
    init_ultrasound_model()
except Exception as e:
    print(f"Warning: could not load ultrasound model: {e}")

def process_ultrasound(image_path: str, device: str = 'cpu', top_k: int = 2):
    if _ultrasound_model is None:
        raise RuntimeError("Ultrasound model not initialized.")
    ext = Path(image_path).suffix.lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
        raise ValueError(f"Unsupported file type: {ext}")
    _ultrasound_model.to(device).eval()
    return predict_ultrasound(_ultrasound_model, image_path, device, top_k)