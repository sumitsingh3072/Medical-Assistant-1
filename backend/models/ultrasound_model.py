import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Resolve project root and checkpoint path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ULTRASOUND_CHECKPOINT = PROJECT_ROOT / 'model_assests' / 'ultrasound' / 'USFM_latest.pth'
CLASS_NAMES = ["Normal", "Cyst", "Mass", "Fluid", "Other Anomaly"]
NUM_CLASSES = len(CLASS_NAMES)

class USFMUltrasoundClassifier(nn.Module):
    def __init__(self, checkpoint_path: Path = DEFAULT_ULTRASOUND_CHECKPOINT, device: str = 'cpu'):
        super().__init__()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Ultrasound weights not found at {checkpoint_path}")
        self.backbone = create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0, global_pool=''
        )
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        self.backbone.load_state_dict(ckpt, strict=False)
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, NUM_CLASSES),
            nn.Sigmoid()
        )
        self.to(device)
        self.eval()

    def forward(self, x):
        feats = self.backbone(x)
        cls = feats['cls'] if isinstance(feats, dict) else feats[:, 0]
        return self.head(cls)

ultrasound_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_ultrasound_model(device: str = 'cpu', checkpoint_path: Path = DEFAULT_ULTRASOUND_CHECKPOINT):
    return USFMUltrasoundClassifier(checkpoint_path=checkpoint_path, device=device)

def predict_ultrasound(model, image_path: str, device: str = 'cpu', top_k: int = 2):
    img = Image.open(image_path).convert('RGB')
    tensor = ultrasound_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad(): probs = model(tensor)[0].cpu().numpy()
    preds = [(CLASS_NAMES[i], float(probs[i])) for i in range(NUM_CLASSES)]
    return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]