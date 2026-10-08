"""TruthLens Multimodal Fusion Module.

Combines detection results from image, audio, and video modalities into an
aggregated authenticity verdict.
"""

from typing import Any, Dict, List, Optional


def fuse_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fuse detection results across multiple media modalities.

    Args:
        results: List of dictionaries from individual modality detectors.
                 Each dictionary may contain:
                 - 'modality': str (e.g., 'image', 'audio', 'video')
                 - 'fake_probability': float or None
                 - 'real_probability': float or None
                 - 'verdict': str
                 - 'confidence': float

    Returns:
        Dictionary containing aggregated verdict, confidence, combined
        probabilities, per-modality evidence, and list of modalities used.
    """
    valid_entries = []

    for r in results:
        if not isinstance(r, dict):
            continue
        fake_prob = r.get("fake_probability")
        # Accept only numeric fake_probability (ignore None, missing, non-numeric)
        if isinstance(fake_prob, (int, float)) and not isinstance(fake_prob, bool):
            modality = r.get("modality", "unknown")
            valid_entries.append((modality, float(fake_prob)))

    # If there are no usable results, return inconclusive defaults
    if not valid_entries:
        return {
            "verdict": "Inconclusive",
            "confidence": 0.0,
            "combined_fake_probability": None,
            "combined_real_probability": None,
            "evidence": [],
            "modalities_used": [],
        }

    # Aggregate fake probabilities
    fake_probs = [fp for _, fp in valid_entries]
    combined_fake = round(sum(fake_probs) / len(fake_probs), 4)
    combined_real = round(1.0 - combined_fake, 4)

    # Determine verdict based on 0.70 threshold
    if combined_fake >= 0.70:
        verdict = "Likely Manipulated"
    elif combined_real >= 0.70:
        verdict = "Likely Authentic"
    else:
        verdict = "Inconclusive"

    confidence = round(max(combined_fake, combined_real), 4)

    # Build evidence list and modalities_used list
    evidence_list: List[Dict[str, Any]] = []
    modalities_used: List[str] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        fake_prob = r.get("fake_probability")
        if isinstance(fake_prob, (int, float)) and not isinstance(fake_prob, bool):
            modality = r.get("modality", "unknown")
            if modality not in modalities_used:
                modalities_used.append(modality)

            detector_name = None
            if isinstance(r.get("evidence"), dict):
                detector_name = r["evidence"].get("detector")
            if not detector_name:
                detector_name = r.get("detector")
            if not detector_name:
                if modality == "image":
                    detector_name = "Xception Facial Manipulation Detector"
                elif modality == "audio":
                    detector_name = "Wav2Vec2 Deepfake Voice Detector"
                elif modality == "video":
                    detector_name = "Frame-level Facial Manipulation Detector"
                else:
                    detector_name = "Modality Detector"

            verdict_val = r.get("verdict")
            if not verdict_val:
                fp_float = float(fake_prob)
                if fp_float >= 0.70:
                    verdict_val = "Likely Manipulated"
                elif fp_float <= 0.30:
                    verdict_val = "Likely Authentic"
                else:
                    verdict_val = "Inconclusive"

            conf_val = r.get("confidence")
            if conf_val is None:
                conf_val = round(max(float(fake_prob), 1.0 - float(fake_prob)), 4)

            real_prob = r.get("real_probability")
            if real_prob is None:
                real_prob = round(1.0 - float(fake_prob), 4)

            evidence_list.append({
                "modality": modality,
                "verdict": verdict_val,
                "confidence": conf_val,
                "fake_probability": round(float(fake_prob), 4),
                "real_probability": real_prob,
                "detector": detector_name,
            })

    return {
        "verdict": verdict,
        "confidence": confidence,
        "combined_fake_probability": combined_fake,
        "combined_real_probability": combined_real,
        "evidence": evidence_list,
        "modalities_used": modalities_used,
    }


def run_tests() -> None:
    """Run standard test suite for fusion module."""
    import json

    print("=" * 60)
    print("TruthLens Fusion - Test Suite")
    print("=" * 60)

    # Test 1: Image only
    t1_input = [
        {
            "modality": "image",
            "fake_probability": 0.9899,
            "real_probability": 0.0101,
            "verdict": "Likely Manipulated",
            "confidence": 0.9899,
        }
    ]
    t1_res = fuse_results(t1_input)
    print("\n[Test 1] Image Only:")
    print(json.dumps(t1_res, indent=2))
    assert t1_res["verdict"] == "Likely Manipulated"
    assert t1_res["combined_fake_probability"] == 0.9899
    assert t1_res["combined_real_probability"] == 0.0101
    assert t1_res["modalities_used"] == ["image"]

    # Test 2: Audio only
    t2_input = [
        {
            "modality": "audio",
            "fake_probability": 0.1200,
            "real_probability": 0.8800,
            "verdict": "Likely Authentic",
            "confidence": 0.8800,
        }
    ]
    t2_res = fuse_results(t2_input)
    print("\n[Test 2] Audio Only:")
    print(json.dumps(t2_res, indent=2))
    assert t2_res["verdict"] == "Likely Authentic"
    assert t2_res["combined_fake_probability"] == 0.1200
    assert t2_res["combined_real_probability"] == 0.8800
    assert t2_res["modalities_used"] == ["audio"]

    # Test 3: Image + Audio
    t3_input = [
        {
            "modality": "image",
            "fake_probability": 0.8000,
            "real_probability": 0.2000,
        },
        {
            "modality": "audio",
            "fake_probability": 0.9000,
            "real_probability": 0.1000,
        },
    ]
    t3_res = fuse_results(t3_input)
    print("\n[Test 3] Image + Audio:")
    print(json.dumps(t3_res, indent=2))
    assert t3_res["verdict"] == "Likely Manipulated"
    assert t3_res["combined_fake_probability"] == 0.8500
    assert t3_res["combined_real_probability"] == 0.1500
    assert t3_res["confidence"] == 0.8500
    assert t3_res["modalities_used"] == ["image", "audio"]

    # Test 4: Image + Audio + Video
    t4_input = [
        {
            "modality": "image",
            "fake_probability": 0.9000,
            "real_probability": 0.1000,
        },
        {
            "modality": "audio",
            "fake_probability": 0.8500,
            "real_probability": 0.1500,
        },
        {
            "modality": "video",
            "fake_probability": 0.9500,
            "real_probability": 0.0500,
        },
    ]
    t4_res = fuse_results(t4_input)
    print("\n[Test 4] Image + Audio + Video:")
    print(json.dumps(t4_res, indent=2))
    assert t4_res["verdict"] == "Likely Manipulated"
    assert t4_res["combined_fake_probability"] == 0.9000
    assert t4_res["combined_real_probability"] == 0.1000
    assert t4_res["confidence"] == 0.9000
    assert t4_res["modalities_used"] == ["image", "audio", "video"]

    # Test 5: Conflicting Evidence
    t5_input = [
        {
            "modality": "image",
            "fake_probability": 0.9000,
            "real_probability": 0.1000,
        },
        {
            "modality": "audio",
            "fake_probability": 0.1000,
            "real_probability": 0.9000,
        },
    ]
    t5_res = fuse_results(t5_input)
    print("\n[Test 5] Conflicting Evidence:")
    print(json.dumps(t5_res, indent=2))
    assert t5_res["verdict"] == "Inconclusive"
    assert t5_res["combined_fake_probability"] == 0.5000
    assert t5_res["combined_real_probability"] == 0.5000
    assert t5_res["confidence"] == 0.5000
    assert t5_res["modalities_used"] == ["image", "audio"]

    # Test 6: Empty / None Input
    t6_input_empty = []
    t6_res_empty = fuse_results(t6_input_empty)
    print("\n[Test 6A] Empty Input:")
    print(json.dumps(t6_res_empty, indent=2))
    assert t6_res_empty["verdict"] == "Inconclusive"
    assert t6_res_empty["confidence"] == 0.0
    assert t6_res_empty["combined_fake_probability"] is None
    assert t6_res_empty["combined_real_probability"] is None
    assert t6_res_empty["evidence"] in ([], {})
    assert t6_res_empty["modalities_used"] == []

    t6_input_none = [
        {
            "modality": "image",
            "fake_probability": None,
            "real_probability": None,
            "face_detected": False,
        }
    ]
    t6_res_none = fuse_results(t6_input_none)
    print("\n[Test 6B] None/Unusable Input:")
    print(json.dumps(t6_res_none, indent=2))
    assert t6_res_none["verdict"] == "Inconclusive"
    assert t6_res_none["confidence"] == 0.0
    assert t6_res_none["combined_fake_probability"] is None
    assert t6_res_none["combined_real_probability"] is None
    assert t6_res_none["evidence"] in ([], {})
    assert t6_res_none["modalities_used"] == []

    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
