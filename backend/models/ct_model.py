# backend/models/ct_model.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from nibabel.loadsave import load as load_nifti  # explicit import for NIfTI loading
import numpy as np
import os

# ---------------------------
# 2D Model: ResNet-based
# ---------------------------
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

# ---------------------------
# 3D Model: Placeholder for MedicalNet
# ---------------------------
class CTNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(CTNet3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------------------------
# Load Model Dispatcher
# ---------------------------
def load_ct_model(mode="2d", device="cpu"):
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_assets', 'ct'))
    if mode == "2d":
        model = CTNet2D()
        weights_path = os.path.join(assets_dir, "resnet_ct_2d.pth")
    elif mode == "3d":
        model = CTNet3D()
        weights_path = os.path.join(assets_dir, "resnet_ct_3d.pth")
    else:
        raise ValueError("Mode must be '2d' or '3d'.")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CT weights not found at {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()
    return model

# ---------------------------
# Prediction Dispatcher
# ---------------------------
def predict_ct(model, image_path, mode="2d", device="cpu"):
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
    return [(classes[i], float(probs[i])) for i in range(len(classes))]

# ---------------------------
# 3D Volume Preprocessing
# ---------------------------
def preprocess_ct_3d(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume = np.resize(volume, (64, 224, 224))
    return volume
