import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from nibabel.loadsave import load as load_nifti
from pathlib import Path

# Resolve backend root (one level up from models/)
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Paths under backend/model_assests/
WEIGHT_MRI_2D = BACKEND_ROOT / 'model_assests' / 'mri' / '2d' / 'model.h5'
WEIGHT_MRI_3D = BACKEND_ROOT / 'model_assests' / 'mri' / '3d' / 'resnet_200.pth'

MRI_CLASSES_2D = ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary Tumor']

# 2D preprocessing
mri_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 2D model definition
class MRINet2D(nn.Module):
    def __init__(self, num_classes=len(MRI_CLASSES_2D)):
        super().__init__()
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)

# 3D model definition
class MRINet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=len(MRI_CLASSES_2D)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.dec = nn.ConvTranspose3d(16, num_classes, 2, 2)

    def forward(self, x):
        return self.dec(self.enc(x))

# Model loader
def load_mri_model(mode='3d', device='cpu'):
    if mode == '2d':
        if not WEIGHT_MRI_2D.is_file():
            raise FileNotFoundError(f"MRI 2D weights not found at {WEIGHT_MRI_2D}")
        model = MRINet2D()
        ckpt = torch.load(str(WEIGHT_MRI_2D), map_location=device)
        sd = ckpt.get('state_dict', ckpt)

    elif mode == '3d':
        if not WEIGHT_MRI_3D.is_file():
            raise FileNotFoundError(f"MRI 3D weights not found at {WEIGHT_MRI_3D}")
        model = MRINet3D()
        ckpt = torch.load(str(WEIGHT_MRI_3D), map_location=device)
        sd = ckpt.get('state_dict', ckpt)

    else:
        raise ValueError("Mode must be '2d' or '3d'")

    # Clean up key names if loaded from DataParallel
    clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    return model.to(device).eval()

# Prediction helper
def predict_mri(model, path, mode='3d', device='cpu', top_k=2):
    if mode == '2d':
        img = Image.open(path).convert('RGB')
        inp = mri_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    else:
        volume = load_nifti(path).get_fdata() #type: ignore
        # normalize & resize
        vol = (volume - volume.min()) / (volume.max() - volume.min())
        vol_resized = np.resize(vol, (64, 224, 224))
        inp = torch.from_numpy(vol_resized).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(inp).mean(dim=[2, 3, 4])
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    preds = [(MRI_CLASSES_2D[i], float(probs[i])) for i in range(len(probs))]
    return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]
