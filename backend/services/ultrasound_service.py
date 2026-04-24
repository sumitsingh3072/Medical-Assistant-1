import threading
from pathlib import Path
from models.ultrasound_model import load_ultrasound_model, predict_ultrasound

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ULTRASOUND_CHECKPOINT = PROJECT_ROOT / 'model_assests' / 'ultrasound' / 'USFM_latest.pth'

_ultrasound_model = None
_lock = threading.Lock()


def init_ultrasound_model(device: str = 'cpu', checkpoint_path: Path = DEFAULT_ULTRASOUND_CHECKPOINT) -> None:
    global _ultrasound_model
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ultrasound weights not found at {checkpoint_path}")
    _ultrasound_model = load_ultrasound_model(device=device, checkpoint_path=(checkpoint_path))


def _ensure_model(device: str = 'cpu') -> None:
    """Lazy-load the model on first use (thread-safe)."""
    global _ultrasound_model
    if _ultrasound_model is None:
        with _lock:
            if _ultrasound_model is None:
                print("Lazy-loading ultrasound model...")
                init_ultrasound_model(device=device)


def process_ultrasound(image_path: str, device: str = 'cpu', top_k: int = 2):
    _ensure_model(device=device)

    ext = Path(image_path).suffix.lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
        raise ValueError(f"Unsupported file type: {ext}")
    _ultrasound_model.to(device).eval()
    return predict_ultrasound(_ultrasound_model, image_path, device, top_k)