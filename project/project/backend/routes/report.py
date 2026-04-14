"""
Report generation routes.
POST /api/report/generate  - Generate & return a PDF medical report
"""

import io
import os
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
from PIL import Image as PILImage

from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR  = PROJECT_ROOT / "uploads"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ──────────────────────────────────────────────
#  Request / Response models
# ──────────────────────────────────────────────

class ReportRequest(BaseModel):
    record_id:      Optional[str] = None
    filename:       Optional[str] = "unknown"
    label:          Optional[str] = "unknown"
    confidence:     Optional[float] = 0.0
    has_lesion:     Optional[bool] = False
    mask_coverage:  Optional[float] = 0.0
    timestamp:      Optional[str] = None
    explanation:    Optional[str] = ""
    status:         Optional[str] = "success"
    # Relative or absolute URL paths stored in DB  e.g. /images/... or /outputs/...
    input_image:    Optional[str] = None
    mask_image:     Optional[str] = None
    overlay_image:  Optional[str] = None


# ──────────────────────────────────────────────
#  Colour palette
# ──────────────────────────────────────────────
NAVY       = colors.HexColor("#0d1b2a")
DARK_BLUE  = colors.HexColor("#1a2744")
MID_BLUE   = colors.HexColor("#1e3a5f")
ACCENT     = colors.HexColor("#3b82f6")
ACCENT2    = colors.HexColor("#60a5fa")
LIGHT_BLUE = colors.HexColor("#dbeafe")
WHITE      = colors.white
LIGHT_GREY = colors.HexColor("#f1f5f9")
MID_GREY   = colors.HexColor("#94a3b8")
DARK_GREY  = colors.HexColor("#334155")
TEXT_BODY  = colors.HexColor("#1e293b")

BENIGN_COLOR     = colors.HexColor("#d97706")
MALIGNANT_COLOR  = colors.HexColor("#dc2626")
NORMAL_COLOR     = colors.HexColor("#16a34a")
UNKNOWN_COLOR    = colors.HexColor("#6b7280")

LABEL_COLORS = {
    "benign":    BENIGN_COLOR,
    "malignant": MALIGNANT_COLOR,
    "normal":    NORMAL_COLOR,
}

LABEL_RISK = {
    "benign":    "LOW – MONITOR",
    "malignant": "HIGH – URGENT REVIEW",
    "normal":    "NONE – ROUTINE CARE",
    "unknown":   "UNDETERMINED",
}


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _resolve_path(url_path: Optional[str]) -> Optional[Path]:
    """
    Convert a stored URL like /images/foo.png or /outputs/bar.png
    to an absolute filesystem path.
    """
    if not url_path:
        return None
    p = url_path.lstrip("/")
    if p.startswith("images/"):
        return UPLOADS_DIR / p[len("images/"):]
    if p.startswith("outputs/"):
        return OUTPUTS_DIR / p[len("outputs/"):]
    return None


def _load_rl_image(url_path: Optional[str], width_mm: float, height_mm: float):
    """
    Return a ReportLab Image flowable or None if the file is missing.
    """
    path = _resolve_path(url_path)
    if not path or not path.exists():
        return None
    try:
        with PILImage.open(str(path)) as im:
            im.verify()           # quick integrity check
        img = RLImage(str(path), width=width_mm * mm, height=height_mm * mm)
        img.hAlign = "CENTER"
        return img
    except Exception as e:
        logger.warning(f"Could not load image {path}: {e}")
        return None


def _placeholder_image(label: str, width_mm: float, height_mm: float):
    """
    Return a simple grey placeholder drawing when an image is unavailable.
    """
    w, h = width_mm * mm, height_mm * mm
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=colors.HexColor("#e2e8f0"), strokeColor=MID_GREY, strokeWidth=1))
    d.add(String(w / 2, h / 2, f"{label}\nNot Available",
                 fontSize=8, fillColor=MID_GREY, textAnchor="middle"))
    return d


# ──────────────────────────────────────────────
#  PDF builder
# ──────────────────────────────────────────────

def _build_pdf(req: ReportRequest) -> bytes:
    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 18 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=12 * mm,
        bottomMargin=18 * mm,
        title="Medical Analysis Report",
        author="BreastAI Diagnostic System",
    )

    styles = getSampleStyleSheet()

    # ── custom styles ──────────────────────────
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle = S("sTitle",
               fontName="Helvetica-Bold", fontSize=17,
               textColor=WHITE, alignment=TA_CENTER, spaceAfter=2)
    sSubtitle = S("sSubtitle",
                  fontName="Helvetica", fontSize=9,
                  textColor=ACCENT2, alignment=TA_CENTER)
    sSectionHead = S("sSectionHead",
                     fontName="Helvetica-Bold", fontSize=10,
                     textColor=WHITE, spaceAfter=0, spaceBefore=0)
    sLabel = S("sLabel",
               fontName="Helvetica-Bold", fontSize=8,
               textColor=MID_GREY, spaceAfter=1)
    sValue = S("sValue",
               fontName="Helvetica-Bold", fontSize=11,
               textColor=TEXT_BODY, spaceAfter=2)
    sBody = S("sBody",
              fontName="Helvetica", fontSize=8.5,
              textColor=DARK_GREY, leading=13, alignment=TA_JUSTIFY)
    sDisclaimer = S("sDisclaimer",
                    fontName="Helvetica-Oblique", fontSize=7.5,
                    textColor=MID_GREY, alignment=TA_CENTER, leading=11)
    sImgLabel = S("sImgLabel",
                  fontName="Helvetica-Bold", fontSize=8,
                  textColor=DARK_GREY, alignment=TA_CENTER, spaceAfter=2)
    sRisk = S("sRisk",
              fontName="Helvetica-Bold", fontSize=9,
              textColor=WHITE, alignment=TA_CENTER)
    sFooter = S("sFooter",
                fontName="Helvetica", fontSize=7,
                textColor=MID_GREY, alignment=TA_CENTER)

    story = []
    usable_w = page_w - 2 * margin

    label_key  = (req.label or "unknown").lower()
    label_color = LABEL_COLORS.get(label_key, UNKNOWN_COLOR)
    label_text  = label_key.upper()
    risk_text   = LABEL_RISK.get(label_key, "UNDETERMINED")
    conf_pct    = f"{(req.confidence or 0) * 100:.1f}%"
    cov_pct     = f"{(req.mask_coverage or 0) * 100:.1f}%"

    # ── Timestamp ─────────────────────────────
    try:
        ts_obj = datetime.fromisoformat((req.timestamp or "").rstrip("Z"))
        ts_display = ts_obj.strftime("%d %B %Y  •  %H:%M UTC")
    except Exception:
        ts_display = req.timestamp or datetime.utcnow().strftime("%d %B %Y  •  %H:%M UTC")

    report_id = f"RPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # ══════════════════════════════════════════
    #  HEADER BANNER
    # ══════════════════════════════════════════
    header_data = [[
        Paragraph("🏥  BreastAI Diagnostic System", sTitle),
        Paragraph("Breast Ultrasound AI Analysis Report", sSubtitle),
    ]]
    header_tbl = Table(header_data, colWidths=[usable_w])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [NAVY]),
        ("BOX",           (0, 0), (-1, -1), 0, NAVY),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 4 * mm))

    # ══════════════════════════════════════════
    #  REPORT META
    # ══════════════════════════════════════════
    meta_rows = [
        [
            _cell_pair("Report ID",   report_id,  styles),
            _cell_pair("Analysis Date", ts_display, styles),
            _cell_pair("Status",  (req.status or "success").upper(), styles),
        ],
        [
            _cell_pair("Source File",  req.filename or "—", styles),
            _cell_pair("System",       "Attention U-Net + CNN", styles),
            _cell_pair("Dataset",      "BUSI (Breast Ultrasound Images)", styles),
        ],
    ]
    cw = usable_w / 3
    meta_tbl = Table(meta_rows, colWidths=[cw, cw, cw])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LIGHT_GREY, WHITE]),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════
    #  PREDICTION RESULT BANNER
    # ══════════════════════════════════════════
    result_inner = [
        [
            _verdict_cell(label_text, label_color, conf_pct, styles),
            _stats_cell("Area Coverage", cov_pct, styles),
            _stats_cell("Lesion Found",  "YES" if req.has_lesion else "NO", styles),
            _risk_cell(risk_text, label_color, styles),
        ]
    ]
    cw2 = [usable_w * 0.35, usable_w * 0.18, usable_w * 0.18, usable_w * 0.29]
    result_tbl = Table(result_inner, colWidths=cw2)
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), label_color),
        ("BACKGROUND",    (1, 0), (2, 0), DARK_BLUE),
        ("BACKGROUND",    (3, 0), (3, 0), MID_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("BOX",           (0, 0), (-1, -1), 1, DARK_BLUE),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════
    #  IMAGES SECTION
    # ══════════════════════════════════════════
    story.append(_section_header("ULTRASOUND IMAGES", usable_w))
    story.append(Spacer(1, 3 * mm))

    img_w_mm  = 54
    img_h_mm  = 50
    img_gap   = ((usable_w / mm) - 3 * img_w_mm) / 2  # horizontal gap between cells in mm

    def _img_cell(url, caption):
        rl_img = _load_rl_image(url, img_w_mm, img_h_mm)
        if rl_img is None:
            rl_img = _placeholder_image(caption, img_w_mm, img_h_mm)
        cap = Paragraph(caption, ParagraphStyle(
            "cap", fontName="Helvetica-Bold", fontSize=7.5,
            textColor=DARK_GREY, alignment=TA_CENTER, spaceBefore=3))
        return [rl_img, cap]

    img_row_data = [[
        _img_cell(req.input_image,   "Original Ultrasound"),
        _img_cell(req.mask_image,    "Segmentation Mask"),
        _img_cell(req.overlay_image, "Overlay / Highlighted Region"),
    ]]
    img_cw = usable_w / 3
    img_tbl = Table(img_row_data, colWidths=[img_cw, img_cw, img_cw])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BOX",           (0, 0), (-1, -1), 1,   colors.HexColor("#cbd5e1")),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════
    #  METRICS TABLE
    # ══════════════════════════════════════════
    story.append(_section_header("QUANTITATIVE METRICS", usable_w))
    story.append(Spacer(1, 2 * mm))

    metrics_data = [
        ["Metric", "Value", "Interpretation"],
        ["AI Classification", label_text, _interpret_label(label_key)],
        ["Confidence Score",  conf_pct, _interpret_confidence(req.confidence or 0)],
        ["Mask Coverage",     cov_pct,  _interpret_coverage(req.mask_coverage or 0)],
        ["Lesion Detected",   "Yes" if req.has_lesion else "No",
                              "Region of interest identified" if req.has_lesion else "No segmented region found"],
        ["Risk Level",        risk_text, "See clinical guidance below"],
    ]
    metrics_tbl = Table(metrics_data, colWidths=[usable_w * 0.28, usable_w * 0.20, usable_w * 0.52])
    metrics_tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
        # Data rows
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 1), (-1, -1), TEXT_BODY),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("ALIGN",         (1, 1), (1, -1), "CENTER"),
        ("FONTNAME",      (1, 1), (1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 1), (1, -1), label_color),
        # Borders
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("BOX",           (0, 0), (-1, -1), 1, DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(metrics_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════
    #  AI CLINICAL EXPLANATION
    # ══════════════════════════════════════════
    story.append(_section_header("AI CLINICAL EXPLANATION", usable_w))
    story.append(Spacer(1, 2 * mm))

    explanation_text = req.explanation or "No explanation available."
    exp_data = [[Paragraph(explanation_text, sBody)]]
    exp_tbl = Table(exp_data, colWidths=[usable_w])
    exp_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), WHITE),
        ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#bfdbfe")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(exp_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════
    #  DISCLAIMER
    # ══════════════════════════════════════════
    story.append(HRFlowable(width=usable_w, thickness=0.5, color=MID_GREY, spaceAfter=4))
    disclaimer = (
        "<b>⚠️  Medical Disclaimer:</b>  This report is generated by an AI-assisted diagnostic tool "
        "and is intended for informational and research purposes only. It does not constitute a medical "
        "diagnosis and must not replace the judgment of a qualified radiologist or clinician. "
        "All findings should be verified and interpreted by a certified medical professional before "
        "any clinical decision is made."
    )
    story.append(Paragraph(disclaimer, sDisclaimer))
    story.append(Spacer(1, 3 * mm))

    footer_text = (
        f"Report ID: {report_id}   |   Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}   |   "
        "BreastAI Diagnostic System — Attention U-Net + CNN   |   BUSI Dataset"
    )
    story.append(Paragraph(footer_text, sFooter))

    doc.build(story)
    return buf.getvalue()


# ──────────────────────────────────────────────
#  Mini cell builders
# ──────────────────────────────────────────────
def _cell_pair(label: str, value: str, styles):
    L = ParagraphStyle("l", fontName="Helvetica-Bold", fontSize=7,
                       textColor=MID_GREY, spaceAfter=1)
    V = ParagraphStyle("v", fontName="Helvetica", fontSize=8.5,
                       textColor=TEXT_BODY)
    return [Paragraph(label.upper(), L), Paragraph(str(value), V)]


def _verdict_cell(label_text, label_color, conf_pct, styles):
    sL = ParagraphStyle("vl", fontName="Helvetica-Bold", fontSize=7,
                        textColor=colors.HexColor("#fef3c7"), spaceAfter=1)
    sV = ParagraphStyle("vv", fontName="Helvetica-Bold", fontSize=18,
                        textColor=WHITE, spaceAfter=2)
    sC = ParagraphStyle("vc", fontName="Helvetica", fontSize=9,
                        textColor=colors.HexColor("#fef3c7"))
    return [
        Paragraph("CLASSIFICATION", sL),
        Paragraph(label_text, sV),
        Paragraph(f"Confidence: {conf_pct}", sC),
    ]


def _stats_cell(label: str, value: str, styles):
    sL = ParagraphStyle("sl", fontName="Helvetica-Bold", fontSize=7,
                        textColor=ACCENT2, spaceAfter=2)
    sV = ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=14,
                        textColor=WHITE)
    return [Paragraph(label.upper(), sL), Paragraph(value, sV)]


def _risk_cell(risk_text: str, label_color, styles):
    sL = ParagraphStyle("rl", fontName="Helvetica-Bold", fontSize=7,
                        textColor=ACCENT2, spaceAfter=2)
    sV = ParagraphStyle("rv", fontName="Helvetica-Bold", fontSize=10,
                        textColor=WHITE, leading=13)
    return [Paragraph("RISK LEVEL", sL), Paragraph(risk_text, sV)]


def _section_header(title: str, width):
    sHead = ParagraphStyle("sh", fontName="Helvetica-Bold", fontSize=9,
                           textColor=WHITE, spaceAfter=0)
    data = [[Paragraph(title, sHead)]]
    tbl = Table(data, colWidths=[width])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0, DARK_BLUE),
    ]))
    return tbl


def _interpret_label(label: str) -> str:
    return {
        "normal":    "Tissue appears normal; no malignancy indicators detected",
        "benign":    "Non-cancerous lesion; smooth margins, homogeneous texture",
        "malignant": "Malignant characteristics detected; irregular borders, possible shadowing",
    }.get(label, "Classification quality insufficient for interpretation")


def _interpret_confidence(conf: float) -> str:
    if conf >= 0.9: return "Very high confidence — reliable prediction"
    if conf >= 0.75: return "High confidence — prediction dependable"
    if conf >= 0.55: return "Moderate confidence — verify with specialist"
    return "Low confidence — manual radiologist review required"


def _interpret_coverage(cov: float) -> str:
    pct = cov * 100
    if pct < 1:  return "Negligible region — possibly normal"
    if pct < 5:  return "Small region of interest"
    if pct < 20: return "Moderate-sized region"
    return "Large region — thorough evaluation recommended"


# ══════════════════════════════════════════════
#  API endpoint
# ══════════════════════════════════════════════

@router.post("/report/generate")
async def generate_report(req: ReportRequest):
    """
    Accept analysis data (inline), generate a PDF, and stream it back.
    """
    if not req.label or req.label.lower() == "unknown":
        raise HTTPException(status_code=400, detail="Incomplete analysis data — cannot generate report")

    try:
        pdf_bytes = _build_pdf(req)
    except Exception as e:
        logger.exception(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"Medical_Report_{ts}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
