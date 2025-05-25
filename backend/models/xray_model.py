#### xray_model.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# ensure we can import backend modules
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

class CheXNet(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNet, self).__init__()
        self.densenet121 = models.densenet121(weights=None)
        in_features = self.densenet121.classifier.in_features

        # Use a Sequential to match the weight keys (classifier.0.weight)
        self.densenet121.classifier = nn.Sequential( #type: ignore
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.densenet121(x)


def load_chexnet_model(weight_path: str, device: str = "cpu"):
    """
    Load the CheXNet model weights from given path.

    Args:
        weight_path: Absolute or relative path to .pth.tar file
        device: torch device string
    """
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weights not found at {weight_path}")

    model = CheXNet(num_classes=14)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

# Preprocessing transforms
xray_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]


def predict_xray(model, image_path: str, top_k: int = 3, device: str = "cpu"):
    """
    Predict top-k conditions for the given X-ray image.
    """
    image = Image.open(image_path).convert('RGB')
    input_tensor = xray_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    preds = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]
