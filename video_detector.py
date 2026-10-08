"""TruthLens Video Deepfake Detector.

Samples frames from a video stream and leverages image_detector.analyze_image
for face-based deepfake analysis.
"""

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np

# Ensure current directory is available on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from image_detector import analyze_image


def analyze_video(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Analyze a video file for deepfake manipulation.

    Args:
        video_path: Path to target video file.

    Returns:
        Dictionary containing modality, verdict, real_probability,
        fake_probability, confidence, and frames_analyzed.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        step = max(1, int(round(fps)))

        fake_probs = []
        frame_idx = 0
        sampled_count = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            while cap.isOpened() and sampled_count < 10:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % step == 0:
                    frame_path = tmp_path / f"frame_{sampled_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)

                    try:
                        res = analyze_image(str(frame_path))
                        if (
                            res.get("face_detected")
                            and res.get("fake_probability") is not None
                        ):
                            fake_probs.append(float(res["fake_probability"]))
                    finally:
                        if frame_path.exists():
                            frame_path.unlink()

                    sampled_count += 1

                frame_idx += 1
    finally:
        cap.release()

    # Zero valid face frames found
    if not fake_probs:
        return {
            "modality": "video",
            "verdict": "Inconclusive",
            "real_probability": None,
            "fake_probability": None,
            "confidence": 0.0,
            "frames_analyzed": 0,
            "frames_with_faces": 0,
        }

    # Aggregate probabilities across valid face frames
    avg_fake = round(float(sum(fake_probs) / len(fake_probs)), 4)
    avg_real = round(float(1.0 - avg_fake), 4)

    if avg_fake >= 0.70:
        verdict = "Likely Manipulated"
        confidence = avg_fake
    elif avg_real >= 0.70:
        verdict = "Likely Authentic"
        confidence = avg_real
    else:
        verdict = "Inconclusive"
        confidence = round(max(avg_real, avg_fake), 4)

    return {
        "modality": "video",
        "verdict": verdict,
        "real_probability": avg_real,
        "fake_probability": avg_fake,
        "confidence": confidence,
        "frames_analyzed": len(fake_probs),
        "frames_with_faces": len(fake_probs),
    }


def main():
    """CLI entrypoint for video deepfake detector."""
    if len(sys.argv) < 2:
        print("Usage: python video_detector.py <path/to/video>")
        sys.exit(1)

    video_path = sys.argv[1]
    try:
        result = analyze_video(video_path)

        print(f"Video: {video_path}")
        print(f"Frames analyzed: {result['frames_analyzed']}")
        print(f"Real probability: {result['real_probability']}")
        print(f"Fake probability: {result['fake_probability']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']}")
    except Exception as err:
        print(f"Error analyzing video: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
