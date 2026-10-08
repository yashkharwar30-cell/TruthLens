"""TruthLens Image Model Benchmark Script.

Evaluates image_detector.py on the test_dataset/ (real vs fake images),
calculates accuracy, false positives/negatives, and prints detailed reports.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_detector import analyze_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(folder: Path) -> List[Path]:
    """Discover all supported image files in a folder, sorted by filename."""
    if not folder.is_dir():
        return []
    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def format_val(val: Any) -> str:
    """Format float or None cleanly for display."""
    if val is None:
        return "None"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    # Locate dataset directory
    dataset_dir = SCRIPT_DIR / "test_dataset"
    if not dataset_dir.is_dir():
        dataset_dir = Path("test_dataset").resolve()

    real_dir = dataset_dir / "real"
    fake_dir = dataset_dir / "fake"

    real_images = find_images(real_dir)
    fake_images = find_images(fake_dir)

    if not real_images and not fake_images:
        print(f"Error: No images found in {dataset_dir} (checked real/ and fake/)")
        sys.exit(1)

    print("=" * 60)
    print("STARTING TRUTHLENS IMAGE MODEL BENCHMARK")
    print(f"Dataset location: {dataset_dir}")
    print(f"Discovered: {len(real_images)} real images, {len(fake_images)} fake images")
    print("=" * 60)

    records: List[Dict[str, Any]] = []

    # Process items: list of (expected_label, path)
    items_to_test = [("REAL", p) for p in real_images] + [("FAKE", p) for p in fake_images]

    real_correct = 0
    fake_correct = 0
    false_positives = 0
    false_negatives = 0
    no_face_count = 0

    for idx, (expected, img_path) in enumerate(items_to_test, 1):
        filename = img_path.name
        print(f"\n[{idx}/{len(items_to_test)}] Analyzing: {filename}")
        print(f"  Expected Label:    {expected}")

        try:
            res = analyze_image(str(img_path))
        except Exception as err:
            print(f"  Error analyzing image: {err}")
            res = {
                "modality": "image",
                "verdict": "Error",
                "real_probability": None,
                "fake_probability": None,
                "confidence": 0.0,
                "face_detected": False,
            }

        face_detected = bool(res.get("face_detected", False))
        fake_prob = res.get("fake_probability")
        real_prob = res.get("real_probability")
        confidence = res.get("confidence")
        verdict = res.get("verdict", "Unknown")

        print(f"  Face Detected:     {face_detected}")
        print(f"  Fake Probability:  {format_val(fake_prob)}")
        print(f"  Real Probability:  {format_val(real_prob)}")
        print(f"  Confidence:        {format_val(confidence)}")
        print(f"  Model Verdict:     {verdict}")

        # Classification rule:
        # fake_probability >= 0.70 => predicted FAKE
        # fake_probability < 0.70  => predicted REAL
        if not face_detected or fake_prob is None:
            pred = "NO_FACE"
            is_correct = False
            no_face_count += 1
        else:
            if fake_prob >= 0.70:
                pred = "FAKE"
            else:
                pred = "REAL"

            is_correct = (pred == expected)
            if expected == "REAL" and pred == "FAKE":
                false_positives += 1
            elif expected == "FAKE" and pred == "REAL":
                false_negatives += 1

        if is_correct:
            if expected == "REAL":
                real_correct += 1
            else:
                fake_correct += 1

        result_str = "CORRECT" if is_correct else "INCORRECT"
        print(f"  Classification:    {pred} -> {result_str}")

        records.append({
            "expected": expected,
            "filename": filename,
            "face_detected": face_detected,
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "confidence": confidence,
            "prediction": pred,
            "result": result_str,
        })

    total_real = len(real_images)
    total_fake = len(fake_images)
    total_images = len(items_to_test)
    overall_correct = real_correct + fake_correct

    real_acc = (real_correct / total_real * 100.0) if total_real > 0 else 0.0
    fake_acc = (fake_correct / total_fake * 100.0) if total_fake > 0 else 0.0
    overall_acc = (overall_correct / total_images * 100.0) if total_images > 0 else 0.0

    # Detailed Table
    print("\n" + "=" * 90)
    print("DETAILED RESULTS TABLE")
    print("=" * 90)
    header = f"{'Expected':<8} | {'File':<16} | {'Face':<5} | {'Fake Prob':<9} | {'Real Prob':<9} | {'Confidence':<10} | {'Prediction':<10} | {'Result':<9}"
    print(header)
    print("-" * len(header))

    for r in records:
        print(
            f"{r['expected']:<8} | "
            f"{r['filename']:<16} | "
            f"{str(r['face_detected']):<5} | "
            f"{format_val(r['fake_prob']):<9} | "
            f"{format_val(r['real_prob']):<9} | "
            f"{format_val(r['confidence']):<10} | "
            f"{r['prediction']:<10} | "
            f"{r['result']:<9}"
        )

    # Benchmark Summary
    print("\n" + "=" * 50)
    print("TRUTHLENS IMAGE MODEL BENCHMARK")
    print("=" * 50)
    print("\nREAL IMAGES")
    print(f"correct: {real_correct}/{total_real}")
    print(f"accuracy: {real_acc:.1f}%")

    print("\nFAKE IMAGES")
    print(f"correct: {fake_correct}/{total_fake}")
    print(f"accuracy: {fake_acc:.1f}%")

    print("\nOVERALL")
    print(f"correct: {overall_correct}/{total_images}")
    print(f"accuracy: {overall_acc:.1f}%")

    print(f"\nFalse Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"No Face Detected: {no_face_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
