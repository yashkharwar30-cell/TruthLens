"""FastAPI Backend for TruthLens Multimodal Deepfake Detection Platform."""

import hashlib
import io
import mimetypes
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Resolve project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AUDIO_DIR = ROOT_DIR.parent / "deepfake-voice-detection-public"
if str(AUDIO_DIR) not in sys.path and AUDIO_DIR.is_dir():
    sys.path.insert(0, str(AUDIO_DIR))

# Import existing detector functions
from fusion import fuse_results
from image_detector import analyze_image
from video_detector import analyze_video

try:
    from audio_detector import analyze_audio
except ImportError:
    analyze_audio = None

# Initialize FastAPI application
app = FastAPI(
    title="TruthLens API",
    description="Multimodal deepfake detection API for Image, Audio, and Video.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload_to_temp(
    upload: UploadFile, default_ext: str = ".tmp"
) -> Tuple[Path, Dict[str, Any]]:
    """Save an UploadFile to a temporary file on disk, computing SHA-256 and metadata."""
    suffix = Path(upload.filename).suffix if upload.filename else default_ext
    if not suffix:
        suffix = default_ext

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    hasher = hashlib.sha256()
    file_size_bytes = 0

    try:
        with open(temp_path, "wb") as f:
            upload.file.seek(0)
            while True:
                chunk = upload.file.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
                file_size_bytes += len(chunk)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {e}"
        ) from e

    # Determine MIME type
    mime_type = upload.content_type
    if not mime_type or mime_type == "application/octet-stream":
        guessed_type, _ = mimetypes.guess_type(upload.filename or "")
        if not guessed_type and suffix:
            guessed_type, _ = mimetypes.guess_type(f"file{suffix}")
        if guessed_type:
            mime_type = guessed_type
        elif not mime_type:
            mime_type = "application/octet-stream"

    file_metadata = {
        "filename": upload.filename or "unknown",
        "sha256": hasher.hexdigest(),
        "file_size_bytes": file_size_bytes,
        "mime_type": mime_type,
    }

    return temp_path, file_metadata


def _extract_image_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely extract image metadata (width, height, format). Returns None for failed fields."""
    meta: Dict[str, Any] = {
        "width": None,
        "height": None,
        "format": None,
    }
    # 1. Try PIL (Pillow)
    try:
        from PIL import Image

        with Image.open(str(file_path)) as img:
            w, h = img.size
            fmt = img.format
            meta["width"] = int(w) if w is not None else None
            meta["height"] = int(h) if h is not None else None
            meta["format"] = str(fmt) if fmt is not None else None
            return meta
    except Exception:
        pass

    # 2. Fallback to OpenCV
    try:
        import cv2

        img_cv = cv2.imread(str(file_path))
        if img_cv is not None:
            h, w = img_cv.shape[:2]
            meta["width"] = int(w)
            meta["height"] = int(h)
            suffix = Path(file_path).suffix.lstrip(".").upper()
            meta["format"] = suffix if suffix else None
    except Exception:
        pass

    return meta


def _extract_audio_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely extract audio metadata (duration_seconds, sample_rate, channels). Returns None for failed fields."""
    meta: Dict[str, Any] = {
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
    }
    # 1. Try soundfile
    try:
        import soundfile as sf

        info = sf.info(str(file_path))
        if info.duration is not None:
            meta["duration_seconds"] = round(float(info.duration), 4)
        if info.samplerate is not None:
            meta["sample_rate"] = int(info.samplerate)
        if info.channels is not None:
            meta["channels"] = int(info.channels)
        return meta
    except Exception:
        pass

    # 2. Try standard library wave module (for WAV files)
    try:
        import wave

        with wave.open(str(file_path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            dur = round(frames / float(sr), 4) if sr > 0 else None
            meta["duration_seconds"] = dur
            meta["sample_rate"] = int(sr)
            meta["channels"] = int(ch)
            return meta
    except Exception:
        pass

    # 3. Try librosa
    try:
        import librosa

        dur = librosa.get_duration(path=str(file_path))
        sr = librosa.get_samplerate(str(file_path))
        if dur is not None:
            meta["duration_seconds"] = round(float(dur), 4)
        if sr is not None:
            meta["sample_rate"] = int(sr)
    except Exception:
        pass

    return meta


def _extract_video_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely extract video metadata (duration_seconds, width, height, fps, frame_count). Returns None for failed fields."""
    meta: Dict[str, Any] = {
        "duration_seconds": None,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
    }
    cap = None
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if w is not None and w > 0 and not np.isnan(w):
                meta["width"] = int(round(w))
            if h is not None and h > 0 and not np.isnan(h):
                meta["height"] = int(round(h))
            if fps_val is not None and fps_val > 0 and not np.isnan(fps_val):
                meta["fps"] = round(float(fps_val), 2)
            if frames is not None and frames > 0 and not np.isnan(frames):
                meta["frame_count"] = int(round(frames))

            if meta["fps"] and meta["frame_count"]:
                meta["duration_seconds"] = round(meta["frame_count"] / meta["fps"], 4)
    except Exception:
        pass
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    return meta


def _build_image_forensics(result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Build structured evidence and explanation for image analysis."""
    face_detected = bool(result.get("face_detected", False))
    fake_prob = result.get("fake_probability")
    real_prob = result.get("real_probability")
    confidence = result.get("confidence")
    verdict = result.get("verdict", "Inconclusive")

    if not face_detected:
        evidence = {
            "detector": "Xception Facial Manipulation Detector",
            "signal": "no detectable face",
            "message": "No suitable face was detected for facial manipulation analysis.",
            "face_detected": False,
        }
        explanation = "No suitable face was detected for facial manipulation analysis."
    else:
        evidence = {
            "detector": "Xception Facial Manipulation Detector",
            "signal": "facial manipulation",
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "face_detected": True,
        }
        if verdict == "Likely Manipulated":
            explanation = "The facial manipulation detector found a strong manipulation signal in the detected face."
        elif verdict == "Likely Authentic":
            explanation = "The facial manipulation detector found a strong authenticity signal for the detected face."
        else:
            explanation = "The detector evidence is not strong enough to produce a reliable verdict."

    return evidence, explanation


def _build_audio_forensics(result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Build structured evidence and explanation for audio analysis."""
    fake_prob = result.get("fake_probability")
    real_prob = result.get("real_probability")
    confidence = result.get("confidence")
    verdict = result.get("verdict", "Inconclusive")

    evidence = {
        "detector": "Wav2Vec2 Deepfake Voice Detector",
        "signal": "synthetic/manipulated voice probability",
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "confidence": confidence,
    }

    if verdict == "Likely Manipulated":
        explanation = "The voice detector produced a strong synthetic/manipulation probability."
    elif verdict == "Likely Authentic":
        explanation = "The voice detector produced a strong authentic voice probability."
    else:
        explanation = "The detector evidence is not strong enough to produce a reliable verdict."

    return evidence, explanation


def _build_video_forensics(result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Build structured evidence and explanation for video analysis."""
    fake_prob = result.get("fake_probability")
    real_prob = result.get("real_probability")
    confidence = result.get("confidence")
    verdict = result.get("verdict", "Inconclusive")
    frames_analyzed = result.get("frames_analyzed", 0)
    frames_with_faces = result.get("frames_with_faces", frames_analyzed)

    evidence = {
        "detector": "Frame-level Facial Manipulation Detector",
        "signal": "facial manipulation across sampled frames",
        "frames_analyzed": frames_analyzed,
        "frames_with_faces": frames_with_faces,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "confidence": confidence,
    }

    if frames_with_faces == 0:
        explanation = "No suitable faces were detected across the sampled video frames."
    elif verdict == "Likely Manipulated":
        explanation = "The frame-level facial manipulation detector found a strong manipulation signal across sampled frames."
    elif verdict == "Likely Authentic":
        explanation = "The frame-level facial manipulation detector found a strong authenticity signal across sampled frames."
    else:
        explanation = "The detector evidence is not strong enough to produce a reliable verdict."

    return evidence, explanation


def _build_overall_explanation(
    fusion_result: Dict[str, Any], individual_results: Dict[str, Any]
) -> str:
    """Build human-readable overall explanation for multimodal fusion."""
    modalities = list(individual_results.keys())
    if len(modalities) == 1:
        return f"Verdict is based on the available {modalities[0]} analysis."

    verdicts = [res.get("verdict") for res in individual_results.values()]
    has_manipulated = any(v == "Likely Manipulated" for v in verdicts)
    has_authentic = any(v == "Likely Authentic" for v in verdicts)

    if has_manipulated and has_authentic:
        return "The available modality signals conflict, so TruthLens reports an inconclusive result."

    if all(v == "Likely Manipulated" for v in verdicts):
        return "Multiple independent modality detectors produced strong manipulation signals."

    if all(v == "Likely Authentic" for v in verdicts):
        return "Multiple independent modality detectors produced strong authenticity signals."

    fusion_verdict = fusion_result.get("verdict")
    if fusion_verdict == "Likely Manipulated":
        return "Combined evidence from available modalities indicates a likely manipulation signal."
    elif fusion_verdict == "Likely Authentic":
        return "Combined evidence from available modalities indicates a likely authentic signal."
    else:
        return (
            "The available modality signals conflict, so TruthLens reports an inconclusive result."
            if (has_manipulated or has_authentic)
            else "The detector evidence is not strong enough to produce a reliable verdict."
        )


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "TruthLens",
    }


@app.post("/analyze/image")
def analyze_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Analyze an uploaded image file for deepfake manipulation."""
    upload = file or image
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    temp_path, file_meta = _save_upload_to_temp(upload, default_ext=".jpg")
    try:
        media_meta = _extract_image_metadata(temp_path)
        result = analyze_image(str(temp_path))
        evidence, explanation = _build_image_forensics(result)
        return {
            "filename": file_meta["filename"],
            "sha256": file_meta["sha256"],
            "file_size_bytes": file_meta["file_size_bytes"],
            "mime_type": file_meta["mime_type"],
            "width": media_meta.get("width"),
            "height": media_meta.get("height"),
            "format": media_meta.get("format"),
            "metadata": media_meta,
            **result,
            "evidence": evidence,
            "explanation": explanation,
        }
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing image: {err}"
        ) from err
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/analyze/audio")
def analyze_audio_endpoint(
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Analyze an uploaded audio file for deepfake voice manipulation."""
    if analyze_audio is None:
        raise HTTPException(
            status_code=500, detail="Audio detector module is not available."
        )

    upload = file or audio
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    temp_path, file_meta = _save_upload_to_temp(upload, default_ext=".wav")
    try:
        media_meta = _extract_audio_metadata(temp_path)
        result = analyze_audio(str(temp_path))
        evidence, explanation = _build_audio_forensics(result)
        return {
            "filename": file_meta["filename"],
            "sha256": file_meta["sha256"],
            "file_size_bytes": file_meta["file_size_bytes"],
            "mime_type": file_meta["mime_type"],
            "duration_seconds": media_meta.get("duration_seconds"),
            "sample_rate": media_meta.get("sample_rate"),
            "channels": media_meta.get("channels"),
            "metadata": media_meta,
            **result,
            "evidence": evidence,
            "explanation": explanation,
        }
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing audio: {err}"
        ) from err
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/analyze/video")
def analyze_video_endpoint(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Analyze an uploaded video file for deepfake manipulation."""
    upload = file or video
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    temp_path, file_meta = _save_upload_to_temp(upload, default_ext=".mp4")
    try:
        media_meta = _extract_video_metadata(temp_path)
        result = analyze_video(str(temp_path))
        evidence, explanation = _build_video_forensics(result)
        return {
            "filename": file_meta["filename"],
            "sha256": file_meta["sha256"],
            "file_size_bytes": file_meta["file_size_bytes"],
            "mime_type": file_meta["mime_type"],
            "duration_seconds": media_meta.get("duration_seconds"),
            "width": media_meta.get("width"),
            "height": media_meta.get("height"),
            "fps": media_meta.get("fps"),
            "frame_count": media_meta.get("frame_count"),
            "metadata": media_meta,
            **result,
            "evidence": evidence,
            "explanation": explanation,
        }
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing video: {err}"
        ) from err
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/analyze")
def analyze_multimodal(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Analyze multimodal media (image, audio, video) and produce fused verdict."""
    results_list: List[Dict[str, Any]] = []
    individual_results: Dict[str, Any] = {}
    temp_files: List[Path] = []

    try:
        # 1. Process image if provided
        if image and image.filename:
            img_path, img_meta = _save_upload_to_temp(image, default_ext=".jpg")
            temp_files.append(img_path)
            img_media_meta = _extract_image_metadata(img_path)
            img_res = analyze_image(str(img_path))
            img_evidence, img_explanation = _build_image_forensics(img_res)
            img_item = {
                "filename": img_meta["filename"],
                "sha256": img_meta["sha256"],
                "file_size_bytes": img_meta["file_size_bytes"],
                "mime_type": img_meta["mime_type"],
                "width": img_media_meta.get("width"),
                "height": img_media_meta.get("height"),
                "format": img_media_meta.get("format"),
                "metadata": img_media_meta,
                **img_res,
                "evidence": img_evidence,
                "explanation": img_explanation,
            }
            results_list.append(img_item)
            individual_results["image"] = img_item

        # 2. Process audio if provided
        if audio and audio.filename:
            if analyze_audio is None:
                raise HTTPException(
                    status_code=500, detail="Audio detector module is not available."
                )
            aud_path, aud_meta = _save_upload_to_temp(audio, default_ext=".wav")
            temp_files.append(aud_path)
            aud_media_meta = _extract_audio_metadata(aud_path)
            aud_res = analyze_audio(str(aud_path))
            aud_evidence, aud_explanation = _build_audio_forensics(aud_res)
            aud_item = {
                "filename": aud_meta["filename"],
                "sha256": aud_meta["sha256"],
                "file_size_bytes": aud_meta["file_size_bytes"],
                "mime_type": aud_meta["mime_type"],
                "duration_seconds": aud_media_meta.get("duration_seconds"),
                "sample_rate": aud_media_meta.get("sample_rate"),
                "channels": aud_media_meta.get("channels"),
                "metadata": aud_media_meta,
                **aud_res,
                "evidence": aud_evidence,
                "explanation": aud_explanation,
            }
            results_list.append(aud_item)
            individual_results["audio"] = aud_item

        # 3. Process video if provided
        if video and video.filename:
            vid_path, vid_meta = _save_upload_to_temp(video, default_ext=".mp4")
            temp_files.append(vid_path)
            vid_media_meta = _extract_video_metadata(vid_path)
            vid_res = analyze_video(str(vid_path))
            vid_evidence, vid_explanation = _build_video_forensics(vid_res)
            vid_item = {
                "filename": vid_meta["filename"],
                "sha256": vid_meta["sha256"],
                "file_size_bytes": vid_meta["file_size_bytes"],
                "mime_type": vid_meta["mime_type"],
                "duration_seconds": vid_media_meta.get("duration_seconds"),
                "width": vid_media_meta.get("width"),
                "height": vid_media_meta.get("height"),
                "fps": vid_media_meta.get("fps"),
                "frame_count": vid_media_meta.get("frame_count"),
                "metadata": vid_media_meta,
                **vid_res,
                "evidence": vid_evidence,
                "explanation": vid_explanation,
            }
            results_list.append(vid_item)
            individual_results["video"] = vid_item

        if not results_list:
            raise HTTPException(
                status_code=400,
                detail="No media files provided for analysis. Please upload at least one image, audio, or video file.",
            )

        # 4. Multimodal Fusion
        fusion_result = fuse_results(results_list)

        # Build structured fusion evidence list
        fusion_evidence = []
        for modality, res in individual_results.items():
            detector_name = (
                res.get("evidence", {}).get("detector")
                if isinstance(res.get("evidence"), dict)
                else None
            )
            if not detector_name:
                if modality == "image":
                    detector_name = "Xception Facial Manipulation Detector"
                elif modality == "audio":
                    detector_name = "Wav2Vec2 Deepfake Voice Detector"
                elif modality == "video":
                    detector_name = "Frame-level Facial Manipulation Detector"
                else:
                    detector_name = "Modality Detector"

            fusion_evidence.append({
                "modality": modality,
                "verdict": res.get("verdict"),
                "confidence": res.get("confidence"),
                "fake_probability": res.get("fake_probability"),
                "real_probability": res.get("real_probability"),
                "detector": detector_name,
            })

        fusion_result["evidence"] = fusion_evidence

        overall_explanation = _build_overall_explanation(
            fusion_result, individual_results
        )

        return {
            "status": "success",
            "results": individual_results,
            "fusion": fusion_result,
            "explanation": overall_explanation,
        }
    finally:
        for p in temp_files:
            if p.exists():
                p.unlink()


@app.post("/passport")
async def generate_passport(request: Request) -> StreamingResponse:
    """Generate an Authenticity Passport PDF from analysis results.

    Accepts the analysis result JSON (from /analyze, /analyze/image,
    /analyze/audio, or /analyze/video) and returns a professional PDF.
    Does NOT rerun any ML detectors.
    """
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON body: {exc}",
        ) from exc

    if not data or not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a non-empty JSON object containing analysis results.",
        )

    try:
        from backend.passport import generate_authenticity_passport_pdf
    except ImportError:
        try:
            from passport import generate_authenticity_passport_pdf
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Passport PDF generator module is not available.",
            )

    try:
        pdf_bytes = generate_authenticity_passport_pdf(data)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Authenticity Passport PDF: {exc}",
        ) from exc

    if not pdf_bytes:
        raise HTTPException(
            status_code=500,
            detail="Generated PDF is empty.",
        )

    # Derive a filename from analysis ID if available
    analysis_id = (
        data.get("analysis_id")
        or data.get("job_id")
        or data.get("id")
        or "truthlens"
    )
    pdf_filename = f"TruthLens_Passport_{analysis_id}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{pdf_filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)