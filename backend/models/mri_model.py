# backend/models/mri_model.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import nibabel as nibabel
import numpy as np
from nibabel.loadsave import load as load_nifti


# ---------------------------
# Define MRI classification classes
# ---------------------------
MRI_CLASSES_2D = ["No Tumor", "Meningioma", "Glioma", "Pituitary Tumor"]
NUM_CLASSES_2D = len(MRI_CLASSES_2D)

# ---------------------------
# 2D MRI Classification Model
# ---------------------------
class MRINet2D(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES_2D):
        super(MRINet2D, self).__init__()
        # Use ResNet50 backbone
        self.model = models.resnet50(pretrained=False)
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.model(x)

# ---------------------------
# 3D MRI Segmentation/Class Model
# ---------------------------
# For 3D we treat it as a 4‐class volume classifier as well,
# returning a per‐class logit map.
NUM_CLASSES_3D = NUM_CLASSES_2D

class MRINet3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES_3D):
        super(MRINet3D, self).__init__()
        # Simple 3D CNN encoder / decoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, num_classes, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------------
# Preprocessing for 2D slices
# ---------------------------
mri_transforms_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Preprocessing for 3D volumes
# ---------------------------
def preprocess_mri_3d(nifti_path: str, target_shape=(64, 224, 224)) -> np.ndarray:
    """
    Load NIfTI file and preprocess to target_shape (D, H, W).
    Returns a normalized numpy array.
    """
    img = load_nifti(nifti_path)
    volume = img.get_fdata()  # type: ignore
    # ensure shape order (D, H, W)
    if volume.shape[0] not in target_shape:
        # assume volume is (H, W, D)
        volume = np.transpose(volume, (2, 0, 1))
    # Normalize to 0-1
    vol = (volume - volume.min()) / (volume.max() - volume.min())
    # Resize volume -- naive resize, consider better interpolation
    vol_resized = np.resize(vol, target_shape)
    return vol_resized

# ---------------------------
# Load Model Dispatcher
# ---------------------------
DEFAULT_MRI_MODEL_2D = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\mri\2d\model.h5"
DEFAULT_MRI_MODEL_3D = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\mri\3d\resnet_200.pth"
def load_mri_model(mode: str = "2d", device: str = "cpu") -> nn.Module:
    """
    Load MRI model in 2D or 3D mode. Provide weight_path accordingly.
    """
    if mode == "2d":
        model = MRINet2D(num_classes=NUM_CLASSES_2D)
        weight_path = DEFAULT_MRI_MODEL_2D
    elif mode == "3d":
        model = MRINet3D(in_channels=1, num_classes=NUM_CLASSES_3D)
        weight_path = DEFAULT_MRI_MODEL_3D
    else:
        raise ValueError("Mode must be '2d' or '3d'.")

    if weight_path is None or not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found at {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    # strip DataParallel prefix if present
    clean_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)

    model.to(device)
    model.eval()
    return model

# ---------------------------
# Prediction Dispatcher
# ---------------------------
def predict_mri(model: nn.Module,
                input_path: str,
                mode: str = "2d",
                device: str = "cpu",
                top_k: int = 2):
    """
    Predict on MRI data (2D slice or 3D volume).
    - For 2D: returns top_k class probabilities.
    - For 3D: averages over volume and returns top_k class probabilities.
    """
    if mode == "2d":
        image = Image.open(input_path).convert('RGB')
        tensor = mri_transforms_2d(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        preds = [(MRI_CLASSES_2D[i], float(probs[i])) for i in range(len(MRI_CLASSES_2D))]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]

    elif mode == "3d":
        volume = preprocess_mri_3d(input_path)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, D, H, W)
        with torch.no_grad():
            outputs = model(tensor)  # (1, num_classes, D, H, W)
            logits = outputs.mean(dim=[2, 3, 4])  # Global average over spatial dimensions
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (num_classes,)
        preds = [(MRI_CLASSES_2D[i], float(probs[i])) for i in range(len(MRI_CLASSES_2D))]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]

    else:
        raise ValueError("Mode must be '2d' or '3d'.")
