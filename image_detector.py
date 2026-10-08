"""Image deepfake detector using XceptionNet for TruthLens."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Locate repository and model directories
REPO_DIR = Path(__file__).resolve().parent
MODEL_DIR = REPO_DIR / "backend" / "models" / "XceptionNet-Detector" / "Deepfake-Detection"
WEIGHTS_PATH = MODEL_DIR / "weights" / "deepfake_c0_xception.pkl"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from dataset.transform import xception_default_data_transforms
from network.models import model_selection

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}


def _get_boundingbox(
    box: Tuple[int, int, int, int], width: int, height: int, scale: float = 1.3
) -> Tuple[int, int, int]:
    """Calculate enlarged quadratic bounding box matching detect_from_video.py.

    Args:
        box: (x, y, w, h) bounding box.
        width: Frame width.
        height: Frame height.
        scale: Scale factor to enlarge bounding box.

    Returns:
        (x1, y1, size_bb) in pixel coordinates.
    """
    x, y, w, h = box
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Top-left corner with boundary check
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    # Ensure box does not exceed frame dimensions
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


class FaceDetector:
    """Face detector supporting dlib, OpenCV Haar cascades, and skimage LBP cascades."""

    def __init__(self) -> None:
        self.detector_type = "none"
        self._dlib_detector = None
        self._cv2_cascade = None
        self._skimage_cascade = None

        # 1. Try dlib frontal face detector
        try:
            import dlib  # type: ignore

            self._dlib_detector = dlib.get_frontal_face_detector()
            self.detector_type = "dlib"
        except Exception:
            self._dlib_detector = None

        # 2. Try OpenCV Haar cascade
        if self._dlib_detector is None and hasattr(cv2, "CascadeClassifier"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self._cv2_cascade = cascade
                    self.detector_type = "opencv_cascade"
            except Exception:
                self._cv2_cascade = None

        # 3. Fallback to skimage LBP frontal face cascade (OpenCV format)
        if self._dlib_detector is None and self._cv2_cascade is None:
            try:
                from skimage.feature import Cascade
                from skimage import data as skimage_data

                self._skimage_cascade = Cascade(
                    skimage_data.lbp_frontal_face_cascade_filename()
                )
                self.detector_type = "skimage_cascade"
            except Exception:
                self._skimage_cascade = None

    def detect_faces(self, image_bgr: np.ndarray):
        """Detect faces in BGR image and return list of (x, y, w, h)."""
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if self._dlib_detector is not None:
            dlib_faces = self._dlib_detector(gray, 1)
            boxes = []
            for face in dlib_faces:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                boxes.append((x, y, w, h))
            return boxes

        if self._cv2_cascade is not None:
            faces = self._cv2_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        if self._skimage_cascade is not None:
            detected = self._skimage_cascade.detect_multi_scale(
                img=gray,
                scale_factor=1.2,
                step_ratio=1,
                min_size=(40, 40),
                max_size=(max(height, width), max(height, width)),
            )
            return [
                (int(d["c"]), int(d["r"]), int(d["width"]), int(d["height"]))
                for d in detected
            ]

        return []


class ImageDetector:
    """Reusable XceptionNet image deepfake detector."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = Path(weights_path) if weights_path else WEIGHTS_PATH

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Xception weights file not found at: {self.weights_path}"
            )

        # Initialize face detector
        self.face_detector = FaceDetector()

        # Initialize and load model
        try:
            self.model = model_selection(
                modelname="xception",
                num_out_classes=2,
                dropout=0.5,
            )
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Xception model on device '{self.device}': {e}"
            ) from e

        # Preprocessing transform from existing repository
        self.preprocess = xception_default_data_transforms["test"]

    def analyze(self, image_path: Union[str, Path]) -> Dict[str, Union[str, float, bool, None]]:
        """Run deepfake detection on an image.

        Args:
            image_path: Path to target image file.

        Returns:
            Dictionary containing modality, verdict, real/fake probabilities, confidence, face_detected.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. Supported formats: {supported}"
            )

        # Safely decode image and normalize color space (supports RGBA, Grayscale, etc.)
        try:
            with Image.open(path) as pil_raw:
                pil_raw.load()
                # Safely convert grayscale/RGBA to RGB
                pil_rgb = pil_raw.convert("RGB")
                image_rgb = np.array(pil_rgb)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to decode image '{image_path}': {e}") from e

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError(f"Image contains no data: {image_path}")

        height, width = image_bgr.shape[:2]

        # 1. Face detection
        faces = self.face_detector.detect_faces(image_bgr)

        # If no face is detected, do NOT force a prediction
        if not faces:
            return {
                "modality": "image",
                "verdict": "Inconclusive",
                "real_probability": None,
                "fake_probability": None,
                "confidence": 0.0,
                "face_detected": False,
            }

        # 2. Select largest / primary face by bounding box area (w * h)
        primary_face = max(faces, key=lambda b: b[2] * b[3])

        # 3. Crop face with quadratic enlargement
        x1, y1, size_bb = _get_boundingbox(primary_face, width, height, scale=1.3)
        if size_bb <= 0:
            return {
                "modality": "image",
                "verdict": "Inconclusive",
                "real_probability": None,
                "fake_probability": None,
                "confidence": 0.0,
                "face_detected": False,
            }

        cropped_bgr = image_bgr[y1 : y1 + size_bb, x1 : x1 + size_bb]
        if cropped_bgr.size == 0:
            return {
                "modality": "image",
                "verdict": "Inconclusive",
                "real_probability": None,
                "fake_probability": None,
                "confidence": 0.0,
                "face_detected": False,
            }

        # 4. Apply repository's exact Xception preprocessing
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
        tensor = self.preprocess(cropped_pil)
        tensor = tensor.unsqueeze(0).to(self.device)

        # 5. Xception inference & softmax
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        real_prob = round(float(probabilities[0].item()), 4)
        fake_prob = round(float(probabilities[1].item()), 4)

        # 6. Verdict determination
        if fake_prob >= 0.70:
            verdict = "Likely Manipulated"
            confidence = fake_prob
        elif real_prob >= 0.70:
            verdict = "Likely Authentic"
            confidence = real_prob
        else:
            verdict = "Inconclusive"
            confidence = round(max(real_prob, fake_prob), 4)

        return {
            "modality": "image",
            "verdict": verdict,
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "confidence": confidence,
            "face_detected": True,
        }


# Singleton instance to ensure model is loaded only once
_default_detector: Optional[ImageDetector] = None


def get_detector() -> ImageDetector:
    """Retrieve or initialize the singleton ImageDetector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ImageDetector()
    return _default_detector


def analyze_image(image_path: str) -> dict:
    """Analyze an image using the cached singleton ImageDetector.

    Args:
        image_path: Path to target image file.

    Returns:
        Dictionary containing modality, verdict, real/fake probabilities, confidence, face_detected.
    """
    detector = get_detector()
    return detector.analyze(image_path)


def main():
    """CLI entrypoint for image deepfake detector."""
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        default_sample = REPO_DIR / "test.jpeg"
        if default_sample.exists():
            test_path = str(default_sample)
        else:
            print("Usage: python image_detector.py <path/to/image>")
            sys.exit(1)

    try:
        detector = get_detector()
        result = detector.analyze(test_path)

        print(f"Image: {test_path}")
        print(f"Face detected: {result['face_detected']}")
        print(f"Real probability: {result['real_probability']}")
        print(f"Fake probability: {result['fake_probability']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Device: {detector.device}")
    except Exception as err:
        print(f"Error analyzing image: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
