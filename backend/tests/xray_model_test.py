# test_xray.py

import os
import sys
sys.path.append(os.path.abspath("backend"))
from services.xray_service import init_xray_model, process_xray

def test_xray(image_path: str, device: str = "cpu", top_k: int = 3):
    try:
        # Initialize model (optional if already initialized)
        init_xray_model(device=device)

        # Run prediction
        results = process_xray(image_path=image_path, device=device, top_k=top_k)

        # Print results
        print("\nX-Ray Prediction Results:")
        for label, prob in results:
            print(f"{label}: {prob:.4f}")

    except Exception as e:
        print(f"[ERROR] {e}")

# Example usage
if __name__ == "__main__":
    # Set your image path here
    image_path = r"backend\data\xray\test1.png"  # Replace with actual image
    test_xray(image_path=image_path, device="cpu", top_k=3)
