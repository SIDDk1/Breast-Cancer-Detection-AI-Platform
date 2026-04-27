"""
Postprocessing utilities: clean and refine raw model output masks,
generate overlay images, and keep only the largest connected component.

CRITICAL FIXES (v3 - Complete Rewrite):
- ADAPTIVE THRESHOLDING: Uses Otsu's method on prediction map to handle flat outputs
- Relative threshold based on output range (not fixed 0.5 when range is tiny)
- Minimum contour area set as fraction of total image area
- Overlay ONLY applies color where mask==1, never to entire image
- Proper alpha blending formula exactly as specified
- Robust morphological cleanup with size-adaptive kernels
"""

import cv2
import numpy as np


def postprocess_mask(
    raw_pred: np.ndarray,
    original_hw: tuple,
    threshold: float = 0.5,
    min_area_ratio: float = 0.001,   # At least 0.1% of image must be masked
) -> np.ndarray:
    """
    Convert raw model sigmoid output → clean binary mask.

    ADAPTIVE STRATEGY:
    1. Clip to [0, 1]
    2. Analyze output range to detect near-flat predictions
    3. If model output range < 0.05 (nearly flat = weak/untrained model):
       → Use Otsu's thresholding on the raw prediction map
       → Use relative threshold (percentile-based)
    4. Otherwise use standard fixed threshold (0.5) with fallback ladder
    5. Resize to original image dimensions
    6. Morphological opening (noise removal) + closing (fill holes)
    7. Keep only the largest contour (if area >= min_area_ratio of image)

    Returns: np.ndarray uint8 (H, W), values are 0 or 255
    """
    H, W = original_hw
    total_pixels = H * W
    min_area_px = max(100, int(min_area_ratio * total_pixels))

    # Step 1: Ensure values are in [0, 1]
    pred = np.clip(raw_pred.squeeze().astype(np.float32), 0.0, 1.0)

    # Step 2: Analyze output range
    pred_min = pred.min()
    pred_max = pred.max()
    pred_range = pred_max - pred_min
    pred_mean = pred.mean()

    # Step 3: Choose thresholding strategy based on model output quality
    binary = None
    used_threshold = threshold

    if pred_range < 0.05:
        # FLAT OUTPUT CASE: Model output is nearly constant (untrained / weak model)
        # Use OTSU thresholding on the raw prediction map rescaled to 8-bit
        # This finds the natural split in the histogram if any signal exists

        # Rescale to 8-bit for Otsu
        if pred_range > 1e-6:
            pred_8bit = ((pred - pred_min) / pred_range * 255).astype(np.uint8)
        else:
            # Truly flat output — try a relative approach using percentile
            # Take top N% pixels as "foreground candidates"
            threshold_percentile = np.percentile(pred, 90)
            if threshold_percentile > pred_mean:
                candidate = (pred >= threshold_percentile).astype(np.uint8)
                if candidate.sum() >= min_area_px:
                    binary = candidate
            # If still nothing, return empty mask - can't segment
            if binary is None:
                return np.zeros((H, W), dtype=np.uint8)

        if binary is None:
            # Apply Otsu
            otsu_threshold, otsu_binary = cv2.threshold(
                pred_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if otsu_binary.sum() > min_area_px * 255:
                binary = (otsu_binary > 0).astype(np.uint8)
                used_threshold = float(otsu_threshold) / 255.0 * pred_range + pred_min

            if binary is None or binary.sum() < min_area_px:
                # Otsu failed — try top percentiles
                for pct in [85, 75, 65]:
                    thr_val = np.percentile(pred, pct)
                    candidate = (pred >= thr_val).astype(np.uint8)
                    if candidate.sum() >= min_area_px:
                        binary = candidate
                        used_threshold = thr_val
                        break

    else:
        # NORMAL OUTPUT CASE: Model has meaningful variation
        # Try standard threshold ladder
        thresholds_to_try = [threshold, 0.4, 0.35, 0.3, 0.25, 0.2]
        for thr in thresholds_to_try:
            candidate = (pred > thr).astype(np.uint8)
            if candidate.sum() >= min_area_px:
                binary = candidate
                used_threshold = thr
                break

        # Last resort: Otsu on scaled prediction
        if binary is None:
            pred_8bit = (pred * 255).astype(np.uint8)
            otsu_threshold, otsu_binary = cv2.threshold(
                pred_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if otsu_binary.sum() > min_area_px * 255:
                binary = (otsu_binary > 0).astype(np.uint8)

    if binary is None:
        # Truly no detectable region — return all zeros
        return np.zeros((H, W), dtype=np.uint8)

    # Step 4: Resize to original image size
    binary_resized = cv2.resize(binary.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # Step 5: Morphological cleanup
    # Size kernels adaptively based on image size
    open_k = max(3, min(7, H // 80))
    close_k = max(5, min(15, H // 40))

    # Ensure odd kernel sizes
    if open_k % 2 == 0: open_k += 1
    if close_k % 2 == 0: close_k += 1

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    opened = cv2.morphologyEx(binary_resized, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
    closed = cv2.morphologyEx(opened,         cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 6: Keep only the largest contour
    binary_255   = (closed * 255).astype(np.uint8)
    contours, _  = cv2.findContours(binary_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Morphology removed everything — try without morphology
        binary_255_raw = (binary_resized * 255).astype(np.uint8)
        contours, _    = cv2.findContours(binary_255_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((H, W), dtype=np.uint8)
        binary_255 = binary_255_raw

    # Filter: keep only contour whose area >= min_area_px
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]

    if not valid_contours:
        # Relax: keep the largest regardless of size threshold
        largest  = max(contours, key=cv2.contourArea)
        valid_contours = [largest]

    # Take largest valid contour
    largest = max(valid_contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)

    clean_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)

    # Optional: smooth mask edges with Gaussian blur + re-threshold
    smooth = cv2.GaussianBlur(clean_mask.astype(np.float32), (5, 5), 1.0)
    clean_mask = (smooth > 127).astype(np.uint8) * 255

    return clean_mask


def create_overlay(
    original_rgb: np.ndarray,
    mask_255: np.ndarray,
    alpha: float = 0.4,
    color: tuple = (220, 50, 50),   # Red in RGB
) -> np.ndarray:
    """
    Alpha-blend: highlight EXACTLY the segmented region with a colored tint.
    Pixels OUTSIDE the mask are left UNCHANGED.

    Formula (ONLY where mask == 255):
        output = (1 - alpha) * original + alpha * color

    Parameters:
        original_rgb: (H, W, 3) uint8 RGB image
        mask_255:     (H, W)    uint8 mask (0 or 255)
        alpha:        blend strength for color (0=original only, 1=solid color)
        color:        RGB tuple for highlight color

    Returns: (H, W, 3) uint8 RGB overlay
    """
    # Safety: ensure mask matches original size
    H, W = original_rgb.shape[:2]
    if mask_255.shape != (H, W):
        mask_255 = cv2.resize(mask_255, (W, H), interpolation=cv2.INTER_NEAREST)

    # Work in float32 for precision
    overlay = original_rgb.copy().astype(np.float32)

    # Boolean mask — only True where mask is foreground
    mask_bool = mask_255 > 127

    # CRITICAL: Only blend where mask is active — NEVER touch rest of image
    if mask_bool.any():
        color_arr = np.array(color, dtype=np.float32)  # shape (3,)

        # Alpha blend: (1-alpha)*original + alpha*color — ONLY on mask region
        overlay[mask_bool] = (
            (1.0 - alpha) * overlay[mask_bool] +
            alpha * color_arr
        )

    # Clip and convert back to uint8
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw contour border for clean edge highlighting
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Yellow contour border (visible on all backgrounds)
        cv2.drawContours(overlay_uint8, contours, -1, (255, 215, 0), 2)

    return overlay_uint8
