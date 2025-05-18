# backend/models/ultrasound_model.py

import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms
from PIL import Image

# ---------------------------
# Configuration
# ---------------------------
USFM_CHECKPOINT = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\ultrasound\USFM_latest.pth"
CLASS_NAMES = ["Normal", "Cyst", "Mass", "Fluid", "Other Anomaly"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------
# Ultrasound Multi-Label ViT Classifier
# ---------------------------
class USFMUltrasoundClassifier(nn.Module):
    def __init__(self, checkpoint_path: str = USFM_CHECKPOINT, device: str = "cpu"):
        super().__init__()

        # 1) Create ViT-Base/16 with no head (num_classes=0 returns feature dict)
        self.backbone = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        # 2) Load USFM checkpoint (only matching keys; ignore extras)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.load_state_dict(ckpt, strict=False)

        # 3) Classification head: LayerNorm → Linear → Sigmoid
        embed_dim = self.backbone.embed_dim  # 768
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, NUM_CLASSES),
            nn.Sigmoid()
        )

        # 4) Move to device and set eval mode
        self.to(device)
        self.eval()

    def forward(self, x):
        # timm ViT returns a dict since num_classes=0
        feats = self.backbone(x)
        # extract CLS token embedding
        cls = feats["cls"] if isinstance(feats, dict) else feats[:, 0]
        return self.head(cls)

# ---------------------------
# Load Model Utility
# ---------------------------
def load_ultrasound_model(device: str = "cpu", checkpoint_path: str = USFM_CHECKPOINT) -> USFMUltrasoundClassifier:
    """
    Instantiates the USFMUltrasoundClassifier and returns it.
    """
    model = USFMUltrasoundClassifier(checkpoint_path=checkpoint_path, device=device)
    return model

# ---------------------------
# Image Preprocessing
# ---------------------------
ultrasound_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Inference Helper
# ---------------------------
def predict_ultrasound(
    model: USFMUltrasoundClassifier,
    image_path: str,
    device: str = "cpu",
    top_k: int = 2
):
    """
    Runs a forward pass on a single ultrasound image and returns
    the top_k class probabilities.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = ultrasound_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model(tensor)[0].cpu().numpy()

    # Build label/prob list and pick top_k
    all_preds = [(CLASS_NAMES[i], float(probs[i])) for i in range(NUM_CLASSES)]
    top_preds = sorted(all_preds, key=lambda x: x[1], reverse=True)[:top_k]
    return top_preds

# ---------------------------
# Example (commented out)
# ---------------------------
# if __name__ == "__main__":
#     m = load_ultrasound_model(device="cpu")
#     print(predict_ultrasound(m, "path/to/ultrasound.png", device="cpu", top_k=3))
