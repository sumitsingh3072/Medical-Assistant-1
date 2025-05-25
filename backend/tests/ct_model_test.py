# test_ct_models.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.models.ct_model import load_ct_model, predict_ct

def test_ct_2d():
    # Path to a sample 2D CT slice image (PNG/JPG)
    sample_2d_path = r"backend\data\ct\2d\test_ct_2d.png"
    print("Testing 2D CT model...")
    model_2d = load_ct_model(mode="2d", device="cpu")
    results_2d = predict_ct(model_2d, sample_2d_path, mode="2d", device="cpu")
    print("2D Results:", results_2d)
    print()

def test_ct_3d():
    # Path to a sample 3D CT volume (NIfTI .nii or .nii.gz)
    sample_3d_path = r"backend\data\ct\3d\CT_Abdo.nii.gz"
    print("Testing 3D CT model...")
    model_3d = load_ct_model(mode="3d", device="cpu")
    results_3d = predict_ct(model_3d, sample_3d_path, mode="3d", device="cpu")
    print("3D Results:", results_3d)
    print()

if __name__ == "__main__":
    # Make sure these sample files exist before running
    sample_2d_path = r"backend\data\ct\2d\test_ct_2d.png"
    sample_3d_path = r"backend\data\ct\3d\CT_Abdo.nii.gz"

    if not os.path.exists(sample_2d_path):
        print(f"2D sample file not found: {sample_2d_path}")
    else:
        test_ct_2d()

    if not os.path.exists(sample_3d_path):
        print(f"3D sample file not found: {sample_3d_path}")
    else:
        test_ct_3d()
