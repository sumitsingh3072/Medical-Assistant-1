# test_ultrasound.py

import os
import sys

# Ensure backend/ is on the import path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(root)

from services.ultrasound_service import init_ultrasound_model, process_ultrasound

def test_ultrasound():
    try:
        # --- Settings ---
        image_path = r"backend\data\ultrasound\c15.jpg"
        device = "cpu"
        top_k = 3  # number of predictions to return

        # Initialize the model (loads checkpoint, sets eval mode)
        init_ultrasound_model(device=device)

        # Run prediction
        results = process_ultrasound(
            image_path=image_path,
            device=device,
            top_k=top_k
        )

        # Print results
        print("\nUltrasound Prediction Results:")
        for label, prob in results:
            print(f"{label}: {prob:.4f}")

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_ultrasound()
