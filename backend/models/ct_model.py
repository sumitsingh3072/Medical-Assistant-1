import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from nibabel.loadsave import load as load_nifti
from pathlib import Path
import os

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CT_2D_WEIGHTS_PATH = BACKEND_ROOT / 'model_assests' / 'ct' / '2d' / 'ResNet50.pt'
CT_3D_WEIGHTS_PATH = BACKEND_ROOT / 'model_assests' / 'ct' / '3d' / 'resnet_200.pth'


class CTNet2D(nn.Module):
    def __init__(self, num_classes=2):
        super(CTNet2D, self).__init__()
        self.model = models.resnet50(weights=None)
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.model(x)

ct_transforms_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 3D CNN
def window_and_normalize(volume, hu_min=-150, hu_max=350):
    vol = np.clip(volume, hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)

def preprocess_ct_3d(vol):
    vol = window_and_normalize(vol)
    return np.resize(vol, (64,224,224))

class CTNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,32,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32,num_classes)
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0),-1))

ct_transforms_2d = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load
def load_ct_model(mode="2d", device="cpu"):
    if mode == "2d":
        model = CTNet2D()
        weights_path = CT_2D_WEIGHTS_PATH
    elif mode == "3d":
        model = CTNet3D()
        weights_path = CT_3D_WEIGHTS_PATH
    else:
        raise ValueError("Mode must be '2d' or '3d'.")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CT weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    raw_sd = checkpoint.get('state_dict', checkpoint)

    clean_sd = {}
    for k, v in raw_sd.items():
        nk = k.replace('module.', '')
        if mode == "2d":
            # remap 2D backbone keys under self.model
            if nk.startswith("backbone."):
                nk2 = nk.replace("backbone.", "model.")
            else:
                nk2 = "model." + nk
        else:
            nk2 = nk
        clean_sd[nk2] = v

    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()
    return model

# Predict
def predict_ct(model, image_path, mode="2d", device="cpu",
               thresh_low: float = 0.35,
               thresh_high: float = 0.65):
    """
    For 2D: returns list of (class, prob).
    For 3D: applies HU windowing, predicts, then classifies:
      prob_tumor > thresh_high    => 'Tumor'
      prob_tumor < thresh_low     => 'No Tumor'
      otherwise                   => 'Indeterminate'
    """
    if mode == "2d":
        image = Image.open(image_path).convert("RGB")
        input_tensor = ct_transforms_2d(image).unsqueeze(0).to(device)
    elif mode == "3d":
        nifti_img = load_nifti(image_path)
        volume = nifti_img.get_fdata()  # type: ignore
        volume = preprocess_ct_3d(volume)
        input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        raise ValueError("Mode must be '2d' or '3d'.")

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    classes = ["No Tumor", "Tumor"]

    if mode == "3d":
        prob_no, prob_tumor = float(probs[0]), float(probs[1])
        if prob_tumor >= thresh_high:
            label = "Tumor"
        elif prob_tumor <= thresh_low:
            label = "No Tumor"
        else:
            label = "Indeterminate"
        return [("No Tumor", prob_no),
        ("Tumor", prob_tumor),
        ("Label", label)
        ]
    else:
        # 2D: keep full distribution
        # When returning only top prediction
        return [(classes[np.argmax(probs)], float(np.max(probs)))]