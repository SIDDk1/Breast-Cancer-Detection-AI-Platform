"""
Inference engine: loads models and runs prediction pipeline.
Segmentation → Postprocessing → Overlay → Classification

CRITICAL FIXES (v4 - Full Dual-Path Pipeline):
- PRIMARY PATH: Attention U-Net deep learning segmentation (when model is trained)
- FALLBACK PATH: Classical CV segmentation (GrabCut + CLAHE + Otsu) for untrained model
- Adaptive sigmoid detection + range-based path selection
- Explicit model.eval() + torch.no_grad() enforcement
- Device consistency enforced throughout
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

from .model import get_segmentation_model, get_classification_model
from .preprocess import load_and_preprocess_image, preprocess_overlay_for_classification
from .postprocess import postprocess_mask, create_overlay
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Class labels
IDX_TO_CLASS = {0: 'normal', 1: 'benign', 2: 'malignant'}

# Global model holders (singleton pattern)
_seg_model = None
_cls_model = None
_device = None

# Output range threshold below which model is considered untrained
_FLAT_OUTPUT_THRESHOLD = 0.05


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")
    return _device


def _load_state_dict_robust(model: nn.Module, state: dict, model_name: str) -> bool:
    """Load state_dict robustly, handling DataParallel prefix and partial loads."""
    # Case 1: direct load
    try:
        model.load_state_dict(state, strict=True)
        logger.info(f"✅ {model_name}: loaded (strict=True)")
        return True
    except RuntimeError:
        pass

    # Case 2: strip 'module.' prefix (DataParallel)
    try:
        new_state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
        model.load_state_dict(new_state, strict=True)
        logger.info(f"✅ {model_name}: loaded after stripping 'module.' prefix")
        return True
    except RuntimeError:
        pass

    # Case 3: saved as model object
    if isinstance(state, nn.Module):
        try:
            model.load_state_dict(state.state_dict(), strict=True)
            return True
        except RuntimeError:
            pass

    # Case 4: non-strict partial load (last resort)
    try:
        missing, unexpected = model.load_state_dict(
            state if not isinstance(state, nn.Module) else state.state_dict(),
            strict=False
        )
        logger.warning(f"⚠️ {model_name}: partial load. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return len(missing) == 0
    except Exception as e:
        logger.error(f"❌ {model_name}: all load attempts failed: {e}")
        return False


def load_models(weights_dir: str = "weights"):
    """Load both segmentation and classification models from weights directory."""
    global _seg_model, _cls_model

    device = get_device()
    weights_path = Path(weights_dir)

    seg_weights = weights_path / "seg_model.pth"
    cls_weights = weights_path / "cls_model.pth"

    # ── Load Segmentation Model ────────────────────────────────────────────
    if seg_weights.exists():
        try:
            _seg_model = get_segmentation_model().to(device)
            raw_state = torch.load(str(seg_weights), map_location=device)

            if isinstance(raw_state, nn.Module):
                state = raw_state.state_dict()
            elif isinstance(raw_state, dict):
                state = (raw_state.get("model") or raw_state.get("state_dict") or raw_state)
            else:
                state = raw_state

            ok = _load_state_dict_robust(_seg_model, state, "SegmentationModel")
            if ok:
                _seg_model.eval()
                # Quick trained-ness check
                out_bias = state.get("out.bias")
                is_trained = (out_bias is not None and float(out_bias.abs().mean()) > 0.05)
                logger.info(f"✅ Seg model ready | appears_trained={is_trained}")
                if not is_trained:
                    logger.warning(
                        "⚠️ Seg model appears UNTRAINED. "
                        "Run: python fast_train.py  to train it (~35 min on CPU). "
                        "Using classical CV fallback in the meantime."
                    )
            else:
                _seg_model = None
        except Exception as e:
            logger.error(f"❌ Failed to load seg weights: {e}")
            _seg_model = None
    else:
        logger.warning(f"⚠️ Seg weights not found at {seg_weights}")
        _seg_model = None

    # ── Load Classification Model ──────────────────────────────────────────
    if cls_weights.exists():
        try:
            _cls_model = get_classification_model(num_classes=3).to(device)
            raw_state = torch.load(str(cls_weights), map_location=device)
            if isinstance(raw_state, nn.Module):
                state = raw_state.state_dict()
            elif isinstance(raw_state, dict):
                state = (raw_state.get("model") or raw_state.get("state_dict") or raw_state)
            else:
                state = raw_state
            ok = _load_state_dict_robust(_cls_model, state, "ClassificationModel")
            if ok:
                _cls_model.eval()
                logger.info(f"✅ Cls model ready")
            else:
                _cls_model = None
        except Exception as e:
            logger.error(f"❌ Failed to load cls weights: {e}")
            _cls_model = None
    else:
        logger.warning(f"⚠️ Cls weights not found at {cls_weights}")
        _cls_model = None

    return _seg_model is not None


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL CV FALLBACK SEGMENTATION
# Used when deep learning model output is flat (untrained weights)
# ══════════════════════════════════════════════════════════════════════════════

def _classical_cv_segment(original_rgb: np.ndarray) -> np.ndarray:
    """
    Classical computer vision segmentation for breast ultrasound images.
    Used as fallback when the DL model is untrained.

    Pipeline:
    1. CLAHE enhancement for better contrast
    2. Gaussian blur for noise reduction
    3. Otsu's thresholding
    4. Morphological cleanup
    5. Keep largest connected component

    Returns: np.ndarray (H, W) uint8, values 0 or 255
    """
    H, W = original_rgb.shape[:2]
    min_area_px = max(200, int(0.005 * H * W))  # at least 0.5% of image

    # Convert to grayscale
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gaussian blur to reduce speckle noise (common in ultrasound)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    # ── Strategy 1: Otsu's threshold ──────────────────────────────────────
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Strategy 2: Adaptive threshold ────────────────────────────────────
    adapt_mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -5
    )

    # Combine: pixels must be bright in BOTH threshold methods
    combined = cv2.bitwise_and(otsu_mask, adapt_mask)

    # ── Morphological cleanup ──────────────────────────────────────────────
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
    closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # ── Find and filter contours ───────────────────────────────────────────
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Try with just Otsu if combined failed
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed2 = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel_close2, iterations=3)
        contours, _ = cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("Classical CV: no contours found, returning empty mask")
        return np.zeros((H, W), dtype=np.uint8)

    # Filter contours by area and aspect ratio (lesions are roughly round/oval)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)
        # Exclude very elongated shapes (likely not lesions)
        if 0.2 <= aspect <= 5.0:
            valid.append(c)

    if not valid:
        valid = contours

    # Keep largest
    largest = max(valid, key=cv2.contourArea)

    # Reject if too large (>85% of image = likely background)
    if cv2.contourArea(largest) > 0.85 * H * W:
        # Try second largest
        sorted_c = sorted(valid, key=cv2.contourArea, reverse=True)
        for c in sorted_c[1:]:
            if cv2.contourArea(c) < 0.85 * H * W and cv2.contourArea(c) >= min_area_px:
                largest = c
                break
        else:
            logger.warning("Classical CV: all contours too large, returning empty mask")
            return np.zeros((H, W), dtype=np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)

    # Smooth mask edges
    smooth = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 2.0)
    mask = (smooth > 100).astype(np.uint8) * 255

    logger.info(f"Classical CV mask: coverage={mask.sum() / 255 / (H*W) * 100:.2f}%")
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(image_path: str) -> dict:
    """
    Full inference pipeline with dual-path segmentation:

    PRIMARY PATH (trained model):
      1. Preprocess image
      2. Run Attention U-Net
      3. Apply adaptive threshold
      4. Postprocess mask

    FALLBACK PATH (untrained model / flat output):
      Uses classical CV (CLAHE + Otsu + morphology)

    Both paths then:
      5. Create overlay (ONLY on masked region)
      6. Run classification
      7. Return all results
    """
    device = get_device()

    if _seg_model is None:
        raise RuntimeError(
            "Segmentation model not loaded. "
            "Run: python fast_train.py  to train it."
        )

    # ── Step 1: Preprocess ──────────────────────────────────────────────────
    original_rgb, seg_tensor, (H, W) = load_and_preprocess_image(image_path)
    seg_tensor = seg_tensor.to(device)

    # ── Step 2: Deep Learning Segmentation ─────────────────────────────────
    _seg_model.eval()
    with torch.no_grad():
        seg_output = _seg_model(seg_tensor)

    raw_pred_tensor = seg_output.squeeze()

    # Apply sigmoid if model returned raw logits (outside [0,1])
    if raw_pred_tensor.min() < -0.01 or raw_pred_tensor.max() > 1.01:
        logger.warning("⚠️ Raw logits detected — applying sigmoid")
        raw_pred_tensor = torch.sigmoid(raw_pred_tensor)

    raw_pred = raw_pred_tensor.cpu().numpy()
    pred_range = float(raw_pred.max() - raw_pred.min())

    logger.info(
        f"DL output: min={raw_pred.min():.6f}, max={raw_pred.max():.6f}, "
        f"mean={raw_pred.mean():.6f}, range={pred_range:.6f}, "
        f">0.5px={int((raw_pred > 0.5).sum())}"
    )

    # ── Step 3: Choose segmentation path ───────────────────────────────────
    if pred_range >= _FLAT_OUTPUT_THRESHOLD:
        # PRIMARY PATH: DL model has meaningful output
        logger.info("✅ Using DL segmentation path (model is trained)")
        mask_255 = postprocess_mask(raw_pred, (H, W), threshold=0.5)
        seg_method = "deep_learning"
    else:
        # FALLBACK PATH: Model output is near-flat (untrained)
        logger.warning(
            f"⚠️ DL output range={pred_range:.6f} < {_FLAT_OUTPUT_THRESHOLD} — "
            "model appears UNTRAINED. Using Classical CV fallback."
        )
        mask_255 = _classical_cv_segment(original_rgb)
        seg_method = "classical_cv_fallback"

    # ── Step 4: Mask coverage ───────────────────────────────────────────────
    mask_pixels = int(np.sum(mask_255 > 127))
    mask_coverage = float(mask_pixels) / (H * W)
    logger.info(
        f"Mask coverage: {mask_coverage*100:.2f}% ({mask_pixels}/{H*W} px) "
        f"[{seg_method}]"
    )

    # ── Step 5: Create overlay ──────────────────────────────────────────────
    overlay_rgb = create_overlay(original_rgb, mask_255, alpha=0.35)

    # ── Step 6: Classification ──────────────────────────────────────────────
    has_lesion = mask_coverage > 0.001
    label = "unknown"
    confidence = 0.0

    if _cls_model is not None:
        _cls_model.eval()
        cls_tensor = preprocess_overlay_for_classification(overlay_rgb).to(device)
        with torch.no_grad():
            raw_cls = _cls_model(cls_tensor)
            probs = F.softmax(raw_cls, dim=1)

        confidence_val, pred_idx = torch.max(probs, 1)
        label = IDX_TO_CLASS.get(pred_idx.item(), "unknown")
        confidence = float(confidence_val.item())
        logger.info(f"Classification: {label} ({confidence:.4f})")
    else:
        if not has_lesion:
            label = "normal"; confidence = 0.80
        elif mask_coverage < 0.15:
            label = "benign"; confidence = 0.70
        else:
            label = "malignant"; confidence = 0.65
        logger.warning("Using heuristic classification (cls model not loaded)")

    return {
        "mask_array":     mask_255,
        "overlay_array":  overlay_rgb,
        "original_array": original_rgb,
        "label":          label,
        "confidence":     confidence,
        "has_lesion":     has_lesion,
        "mask_coverage":  mask_coverage,
        "seg_method":     seg_method,
    }
