"""
Helper utilities for image I/O, base64 encoding, etc.
"""

import os
import cv2
import base64
import numpy as np
from pathlib import Path
from datetime import datetime


def save_image(img_array: np.ndarray, directory: str, filename: str, is_rgb: bool = True) -> str:
    """
    Save a numpy image array to disk.
    Returns the saved file path.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    if is_rgb:
        # Convert RGB → BGR for cv2
        save_arr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        save_arr = img_array

    cv2.imwrite(path, save_arr)
    return path


def ndarray_to_base64(img_array: np.ndarray, is_rgb: bool = True) -> str:
    """
    Convert numpy image to base64-encoded PNG string.
    """
    if is_rgb and len(img_array.shape) == 3:
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_array

    _, buffer = cv2.imencode('.png', bgr)
    return base64.b64encode(buffer).decode('utf-8')


def generate_filename(prefix: str, ext: str = "png") -> str:
    """Generate a timestamped filename."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.{ext}"


def build_explanation(label: str, confidence: float, has_lesion: bool, mask_coverage: float) -> str:
    """
    Generate a human-readable explanation for the prediction.
    """
    coverage_pct = mask_coverage * 100

    if not has_lesion:
        return (
            "No significant lesion detected in this ultrasound image. "
            "The segmentation model found no meaningful region of interest. "
            "This could indicate a normal tissue or a low-contrast lesion. "
            "Clinical evaluation is recommended."
        )

    if label == "normal":
        return (
            f"The AI analysis indicates NORMAL tissue with {confidence*100:.1f}% confidence. "
            f"A small region ({coverage_pct:.1f}% of image) was highlighted, "
            "but it does not show characteristics typical of malignant or benign lesions. "
            "No immediate clinical concern detected."
        )
    elif label == "benign":
        return (
            f"The AI analysis indicates a BENIGN lesion with {confidence*100:.1f}% confidence. "
            f"The segmented region covers {coverage_pct:.1f}% of the image. "
            "Benign lesions are typically non-cancerous, with smooth borders and homogeneous texture. "
            "Follow-up imaging may be recommended based on clinical judgment."
        )
    elif label == "malignant":
        return (
            f"⚠️ The AI analysis indicates a MALIGNANT lesion with {confidence*100:.1f}% confidence. "
            f"The segmented region covers {coverage_pct:.1f}% of the image. "
            "Malignant lesions may show irregular borders, heterogeneous texture, or posterior shadowing. "
            "URGENT clinical evaluation and biopsy are strongly recommended."
        )
    else:
        return "Analysis complete. Please consult a radiologist for clinical interpretation."
