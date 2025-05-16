import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath("backend"))
# ----------------------
# Define CheXNet Model
# ----------------------

class CheXNet(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNet, self).__init__()
        self.densenet121 = models.densenet121(weights=None)  # 'pretrained=False' is deprecated
        in_features = self.densenet121.classifier.in_features

        # Use a Sequential to match the weight keys (classifier.0.weight)
        self.densenet121.classifier = nn.Sequential( #type: ignore
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.densenet121(x)
# -------------------------------
# Load Pretrained CheXNet Weights
# -------------------------------
def load_chexnet_model(weight_path=r"model_assests\xray\xray.pth.tar", device="cpu"):
    model = CheXNet(num_classes=14)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

# ----------------------
# X-ray Image Transforms
# ----------------------
xray_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ------------------------
# ChestX-ray14 Class Labels
# ------------------------
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# -------------------------
# Predict Top-K Conditions
# -------------------------
def predict_xray(model, image_path, top_k=3, device="cpu"):
    image = Image.open(image_path).convert('RGB')
    input_tensor = xray_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    predictions = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]
