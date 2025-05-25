# inspect_mri_models.py

import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.models.mri_model import (
    MRINet2D, MRINet3D,
    NUM_CLASSES_2D, NUM_CLASSES_3D,
    MRI_CLASSES_2D
)

# Paths to your downloaded weights
WEIGHTS_2D = r"C:\ML_Projects\Medical-Assistant-1\backend\model_assests\mri\2d\model.h5"
WEIGHTS_3D = r"backend\model_assests\mri\3d\resnet_200.pth"

def explore_checkpoint(path):
    print(f"--- Exploring checkpoint: {path} ---")
    ckpt = torch.load(path, map_location="cpu")
    # If nested in 'state_dict', unwrap
    sd = ckpt.get("state_dict", ckpt)
    print(f"Top-level keys ({len(ckpt.keys())}): {list(ckpt.keys())[:10]}{'...' if len(ckpt.keys())>10 else ''}")
    print(f"Parameters in state_dict: {len(sd)}")
    # show first 5 param names + shapes
    for i, (k, v) in enumerate(sd.items()):
        print(f"  {i+1:2d}. {k:50s} {tuple(v.shape) if hasattr(v,'shape') else type(v)}")
        if i >= 4:
            break
    print()

    return sd

def inspect_model_2d(state_dict):
    print("--- Instantiating MRINet2D ---")
    model = MRINet2D(num_classes=NUM_CLASSES_2D)
    print(model)
    print()

    # Try loading weights
    print("Loading state_dict into MRINet2D (strict=False)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f" Missing keys ({len(missing)}): {missing}")
    print(f" Unexpected keys ({len(unexpected)}): {unexpected}")
    print()

def inspect_model_3d(state_dict):
    print("--- Instantiating MRINet3D ---")
    model = MRINet3D(in_channels=1, num_classes=NUM_CLASSES_3D)
    print(model)
    print()

    # Try loading weights
    print("Loading state_dict into MRINet3D (strict=False)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f" Missing keys ({len(missing)}): {missing}")
    print(f" Unexpected keys ({len(unexpected)}): {unexpected}")
    print()

def main():
    # check files exist
    if not os.path.exists(WEIGHTS_2D):
        print(f"2D weights not found at {WEIGHTS_2D}")
    else:
        sd2 = explore_checkpoint(WEIGHTS_2D)
        inspect_model_2d(sd2)

    if not os.path.exists(WEIGHTS_3D):
        print(f"3D weights not found at {WEIGHTS_3D}")
    else:
        sd3 = explore_checkpoint(WEIGHTS_3D)
        inspect_model_3d(sd3)

if __name__ == "__main__":
    main()
