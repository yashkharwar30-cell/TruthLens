"""Authenticity Passport PDF Generator for TruthLens."""

import io
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


def _fmt_prob(val: Any) -> str:
    """Format float probability to percentage string."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return f"{f * 100:.2f}%"
    except (ValueError, TypeError):
        return str(val)


def _fmt_size(num_bytes: Any) -> str:
    """Format bytes into readable string."""
    if num_bytes is None:
        return "N/A"
    try:
        nb = int(num_bytes)
        if nb < 1024:
            return f"{nb} bytes"
        elif nb < 1024 * 1024:
            return f"{nb:,} bytes ({nb / 1024:.1f} KB)"
        else:
            return f"{nb:,} bytes ({nb / (1024 * 1024):.2f} MB)"
    except (ValueError, TypeError):
        return str(num_bytes)


def generate_authenticity_passport_pdf(data: Dict[str, Any]) -> bytes:
    """Generate professional Authenticity Passport PDF from analysis data.

    Args:
        data: TruthLens analysis dictionary (either multimodal or single modality).

    Returns:
        Raw bytes of generated PDF.
    """
    # 1. Normalize data input structure
    if "results" in data and isinstance(data["results"], dict):
        individual_results = data["results"]
        fusion = data.get("fusion", {})
        overall_explanation = data.get("explanation", "")
    else:
        # Single modality payload passed directly
        modality = data.get("modality", "image")
        individual_results = {modality: data}
        verdict = data.get("verdict", "Inconclusive")
        confidence = data.get("confidence", 0.0)
        fake_prob = data.get("fake_probability")
        real_prob = data.get("real_probability")
        detector_name = (
            data.get("evidence", {}).get("detector")
            if isinstance(data.get("evidence"), dict)
            else "Modality Detector"
        )
        fusion = {
            "verdict": verdict,
            "confidence": confidence,
            "combined_fake_probability": fake_prob,
            "combined_real_probability": real_prob,
            "modalities_used": [modality],
            "evidence": [
                {
                    "modality": modality,
                    "verdict": verdict,
                    "confidence": confidence,
                    "fake_probability": fake_prob,
                    "real_probability": real_prob,
                    "detector": detector_name,
                }
            ],
        }
        overall_explanation = data.get(
            "explanation", f"Verdict is based on the available {modality} analysis."
        )

    # Generate or extract analysis ID and timestamp
    analysis_id = str(
        data.get("analysis_id")
        or data.get("job_id")
        or data.get("id")
        or f"TL-{uuid.uuid4().hex[:12].upper()}"
    )
    timestamp = str(
        data.get("timestamp")
        or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    verdict_str = str(fusion.get("verdict") or "Inconclusive")
    confidence_val = fusion.get("confidence")
    comb_fake = fusion.get("combined_fake_probability")
    comb_real = fusion.get("combined_real_probability")
    modalities_used = fusion.get("modalities_used") or list(individual_results.keys())

    # 2. Setup document and styles
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    c_dark = colors.HexColor("#0F172A")
    c_accent = colors.HexColor("#1E3A8A")
    c_muted = colors.HexColor("#475569")
    c_border = colors.HexColor("#CBD5E1")
    c_bg_light = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        "PassportTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_dark,
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "PassportSubtitle",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_accent,
        alignment=0,
    )
    sec_heading_style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark,
    )
    label_style = ParagraphStyle(
        "LabelCustom",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=c_muted,
    )
    mono_style = ParagraphStyle(
        "MonoCustom",
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=c_dark,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=c_muted,
    )

    story: List[Any] = []

    # -------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------
    header_table = Table(
        [
            [
                Paragraph("<b>TRUTHLENS</b>", title_style),
                Paragraph(
                    f"<font size=8 color='#64748B'><b>ANALYSIS ID:</b> {analysis_id}<br/><b>DATE:</b> {timestamp}</font>",
                    body_style,
                ),
            ],
            [
                Paragraph("AUTHENTICITY PASSPORT", subtitle_style),
                Paragraph("<font size=8 color='#2563EB'>Cryptographic Verification & Forensic Evidence</font>", body_style),
            ],
        ],
        colWidths=[320, 220],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceAfter=8))

    # -------------------------------------------------------------
    # SECTION 1: ANALYSIS
    # -------------------------------------------------------------
    story.append(Paragraph("<b>ANALYSIS</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    analysis_data = [
        [
            Paragraph("<b>Analysis ID:</b>", label_style),
            Paragraph(f"<font name='Courier'>{analysis_id}</font>", body_style),
            Paragraph("<b>Timestamp:</b>", label_style),
            Paragraph(timestamp, body_style),
        ],
        [
            Paragraph("<b>Modalities:</b>", label_style),
            Paragraph(", ".join(str(m).upper() for m in modalities_used), body_style),
            Paragraph("<b>System Version:</b>", label_style),
            Paragraph("TruthLens Forensic Core v1.0", body_style),
        ],
    ]
    t_analysis = Table(analysis_data, colWidths=[90, 180, 90, 180])
    t_analysis.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(t_analysis)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECTION 2: VERDICT
    # -------------------------------------------------------------
    story.append(Paragraph("<b>VERDICT</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    # Badge color based on verdict
    v_upper = verdict_str.upper()
    if "MANIPULATED" in v_upper:
        v_bg = colors.HexColor("#FEF2F2")
        v_text_color = colors.HexColor("#DC2626")
        v_border = colors.HexColor("#FCA5A5")
    elif "AUTHENTIC" in v_upper:
        v_bg = colors.HexColor("#F0FDF4")
        v_text_color = colors.HexColor("#16A34A")
        v_border = colors.HexColor("#86EFAC")
    else:
        v_bg = colors.HexColor("#FFFBEB")
        v_text_color = colors.HexColor("#D97706")
        v_border = colors.HexColor("#FDE68A")

    verdict_badge_style = ParagraphStyle(
        "VerdictBadge",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=v_text_color,
        alignment=1,
    )

    verdict_table_data = [
        [
            Paragraph(f"<b>OVERALL VERDICT: {verdict_str.upper()}</b>", verdict_badge_style),
            Paragraph(
                f"<b>Confidence:</b> {_fmt_prob(confidence_val)}<br/>"
                f"<b>Combined Fake Probability:</b> {_fmt_prob(comb_fake)}<br/>"
                f"<b>Combined Real Probability:</b> {_fmt_prob(comb_real)}<br/>"
                f"<b>Modalities Used:</b> {', '.join(str(m).title() for m in modalities_used)}",
                body_style,
            ),
        ]
    ]
    t_verdict = Table(verdict_table_data, colWidths=[240, 300])
    t_verdict.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), v_bg),
                ("BOX", (0, 0), (0, 0), 1, v_border),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_verdict)

    if overall_explanation:
        story.append(Spacer(1, 3))
        story.append(
            Paragraph(
                f"<b>Forensic Assessment:</b> {overall_explanation}",
                body_style,
            )
        )
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECTION 3: MEDIA (MEDIA FILES & METADATA)
    # -------------------------------------------------------------
    story.append(Paragraph("<b>MEDIA</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    media_rows = []
    for mod_name, item in individual_results.items():
        fname = item.get("filename", "unknown")
        mime = item.get("mime_type", "application/octet-stream")
        sz = _fmt_size(item.get("file_size_bytes"))
        sha = str(item.get("sha256") or "N/A")

        # Media metadata line
        meta_items = []
        if mod_name == "image":
            w = item.get("width")
            h = item.get("height")
            fmt = item.get("format")
            if w is not None and h is not None:
                meta_items.append(f"Dimensions: {w} × {h} px")
            if fmt:
                meta_items.append(f"Format: {fmt}")
        elif mod_name == "audio":
            dur = item.get("duration_seconds")
            sr = item.get("sample_rate")
            ch = item.get("channels")
            if dur is not None:
                meta_items.append(f"Duration: {dur:.2f} s")
            if sr is not None:
                meta_items.append(f"Sample Rate: {sr} Hz")
            if ch is not None:
                meta_items.append(f"Channels: {ch}")
        elif mod_name == "video":
            dur = item.get("duration_seconds")
            w = item.get("width")
            h = item.get("height")
            fps = item.get("fps")
            fc = item.get("frame_count")
            if dur is not None:
                meta_items.append(f"Duration: {dur:.2f} s")
            if w is not None and h is not None:
                meta_items.append(f"Dimensions: {w} × {h} px")
            if fps is not None:
                meta_items.append(f"FPS: {fps}")
            if fc is not None:
                meta_items.append(f"Frames: {fc}")

        meta_line = " | ".join(meta_items) if meta_items else "No metadata available"

        media_block = [
            Paragraph(f"<b>Modality:</b> {mod_name.upper()}", label_style),
            Paragraph(f"<b>Filename:</b> {fname}", body_style),
            Paragraph(f"<b>MIME Type:</b> {mime}", body_style),
            Paragraph(f"<b>File Size:</b> {sz}", body_style),
        ]
        media_rows.append(media_block)

        # SHA-256 row in monospace font
        sha_block = [
            Paragraph("<b>SHA-256:</b>", label_style),
            Paragraph(f"<font name='Courier' size=8>{sha}</font>", mono_style),
            Paragraph("<b>Media Metadata:</b>", label_style),
            Paragraph(f"<font size=8>{meta_line}</font>", body_style),
        ]
        media_rows.append(sha_block)

    t_media = Table(media_rows, colWidths=[90, 180, 110, 160])
    t_media.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(t_media)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECTION 4: FORENSIC EVIDENCE
    # -------------------------------------------------------------
    story.append(Paragraph("<b>FORENSIC EVIDENCE</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    evidence_table_data = [
        [
            Paragraph("<b>Modality</b>", label_style),
            Paragraph("<b>Detector & Signal</b>", label_style),
            Paragraph("<b>Verdict</b>", label_style),
            Paragraph("<b>Confidence</b>", label_style),
            Paragraph("<b>Fake / Real Prob.</b>", label_style),
        ]
    ]

    for mod_name, item in individual_results.items():
        ev = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
        det_name = ev.get("detector") or "Modality Detector"
        signal = ev.get("signal") or "analysis"
        v_val = item.get("verdict") or "Inconclusive"
        conf_val = _fmt_prob(item.get("confidence"))
        fake_p = _fmt_prob(item.get("fake_probability"))
        real_p = _fmt_prob(item.get("real_probability"))
        expl = item.get("explanation") or ""

        evidence_table_data.append(
            [
                Paragraph(f"<b>{mod_name.upper()}</b>", body_style),
                Paragraph(f"<b>{det_name}</b><br/><font size=8 color='#475569'>Signal: {signal}</font>", body_style),
                Paragraph(v_val, body_style),
                Paragraph(conf_val, body_style),
                Paragraph(f"Fake: {fake_p}<br/>Real: {real_p}", body_style),
            ]
        )
        if expl:
            evidence_table_data.append(
                [
                    "",
                    Paragraph(f"<font size=8 color='#334155'><i>Explanation: {expl}</i></font>", body_style),
                    "",
                    "",
                    "",
                ]
            )

    t_evidence = Table(evidence_table_data, colWidths=[65, 235, 90, 70, 80])
    t_evidence.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), c_bg_light),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(t_evidence)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECTION 5: FUSION (FUSION EVIDENCE)
    # -------------------------------------------------------------
    story.append(Paragraph("<b>FUSION</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    fusion_ev_list = fusion.get("evidence") or []
    fusion_table_data = [
        [
            Paragraph("<b>Modality / Source</b>", label_style),
            Paragraph("<b>Verdict</b>", label_style),
            Paragraph("<b>Contribution & Evidence</b>", label_style),
        ]
    ]

    for item in fusion_ev_list:
        if isinstance(item, dict):
            m_name = str(item.get("modality", "unknown")).upper()
            v_val = str(item.get("verdict", "Inconclusive"))
            fp = _fmt_prob(item.get("fake_probability"))
            conf = _fmt_prob(item.get("confidence"))
            fusion_table_data.append(
                [
                    Paragraph(f"<b>{m_name}</b>", body_style),
                    Paragraph(v_val, body_style),
                    Paragraph(f"{fp} fake probability (Confidence: {conf})", body_style),
                ]
            )

    # FINAL row
    fusion_table_data.append(
        [
            Paragraph("<b>FINAL FUSION</b>", ParagraphStyle("FinalLabel", parent=body_style, fontName="Helvetica-Bold", textColor=c_accent)),
            Paragraph(f"<b>{verdict_str}</b>", ParagraphStyle("FinalVerdict", parent=body_style, fontName="Helvetica-Bold", textColor=v_text_color)),
            Paragraph(f"<b>{_fmt_prob(confidence_val)} confidence</b> (Combined Fake: {_fmt_prob(comb_fake)})", body_style),
        ]
    )

    t_fusion = Table(fusion_table_data, colWidths=[120, 150, 270])
    t_fusion.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), c_bg_light),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(t_fusion)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECTION 6: VERIFICATION
    # -------------------------------------------------------------
    story.append(Paragraph("<b>VERIFICATION</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    verify_url = f"https://truthlens.local/verify/{analysis_id}"
    verification_data = [
        [
            Paragraph("<b>Verification URL (placeholder):</b>", label_style),
            Paragraph(f"<font name='Courier' color='#2563EB'>{verify_url}</font>", body_style),
        ],
        [
            Paragraph("<b>Verification Status:</b>", label_style),
            Paragraph("Verification URL (placeholder) - Local demonstration reference. No public cloud registry connection required.", disclaimer_style),
        ],
    ]
    t_verify = Table(verification_data, colWidths=[160, 380])
    t_verify.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(t_verify)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECTION 7: DISCLAIMER
    # -------------------------------------------------------------
    story.append(Paragraph("<b>DISCLAIMER</b>", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=5))

    disclaimer_text = (
        "TruthLens provides probabilistic forensic analysis and does not guarantee "
        "the authenticity or manipulation status of media."
    )
    t_disclaimer = Table(
        [[Paragraph(disclaimer_text, disclaimer_style)]],
        colWidths=[540],
    )
    t_disclaimer.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_disclaimer)

    # Build PDF in memory
    doc.build(story)
    return buf.getvalue()
