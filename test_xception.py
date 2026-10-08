import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

MODEL_DIR = Path(
    "backend/models/XceptionNet-Detector/Deepfake-Detection"
)

sys.path.insert(0, str(MODEL_DIR))

from network.models import model_selection
from dataset.transform import xception_default_data_transforms


IMAGE_PATH = "test.jpeg"
WEIGHTS_PATH = (
    MODEL_DIR
    / "weights"
    / "deepfake_c0_xception.pkl"
)


def main():

    print("=" * 60)
    print("TruthLens - Xception Test")
    print("=" * 60)

    # Device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    print(f"Device: {device}")

    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading Xception model...")

    model = model_selection(
        modelname="xception",
        num_out_classes=2,
        dropout=0.5
    )

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # Load image
    print(f"\nLoading image: {IMAGE_PATH}")

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not load {IMAGE_PATH}"
        )

    print(f"Image shape: {image.shape}")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Xception preprocessing
    preprocess = xception_default_data_transforms["test"]

    pil_image = Image.fromarray(image_rgb)

    tensor = preprocess(pil_image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # Inference
    print("\nRunning inference...")

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    probabilities = probabilities[0]

    real_probability = float(
        probabilities[0]
    )

    fake_probability = float(
        probabilities[1]
    )

    prediction = (
        "FAKE"
        if fake_probability > real_probability
        else "REAL"
    )

    # Result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"REAL probability : {real_probability:.4f}"
    )

    print(
        f"FAKE probability : {fake_probability:.4f}"
    )

    print(
        f"Prediction       : {prediction}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()