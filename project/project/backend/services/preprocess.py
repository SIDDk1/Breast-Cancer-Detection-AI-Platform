"""
Preprocessing utilities: prepare raw image for model inference.
Matches the exact pipeline used during training.

CRITICAL FIXES (v2):
- Normalization mean/std expanded to 3-channel tuples to match albumentations
  requirements when processing RGB images (NOT 1-channel tuples).
- Consistent IMAGE_SIZE = 256 matching training pipeline.
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 256

# ── Segmentation normalization ──────────────────────────────────────────────
# FIXED: was (0.0,)/(1.0,) — 1-element tuples. Albumentations processes
# each channel independently and requires one value PER CHANNEL when the
# image is RGB (3-ch). Using a single element causes channel-0-only application
# for some albumentations versions, leaving ch-1 and ch-2 un-normalized and
# producing wildly different tensor values → causes full-white mask.
# Training normalises with mean=0, std=1 for all channels (i.e. /255 implicit
# in ToTensorV2 is NOT done — ToTensorV2 divides by 255 before Normalize runs).
# But A.Normalize DOES divide by std=1 after subtracting mean=0, so values are
# in [0,1] range which matches training.
SEG_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
])

# ── Classification normalization ─────────────────────────────────────────────
# ImageNet stats (correct 3-tuple format, unchanged)
CLS_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_and_preprocess_image(image_path: str):
    """
    Load image from disk, return:
    - original_rgb: np.ndarray (H, W, 3) uint8 in RGB  — stored for overlay
    - seg_tensor:   torch.Tensor (1, 3, 256, 256)       — fed to seg model
    - original_hw:  tuple (H, W)                        — for mask resize
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR → RGB (training also used cv2 + cvtColor)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    original_rgb = image_rgb.copy()

    # Apply segmentation transform
    aug = SEG_TRANSFORM(image=image_rgb)
    # ToTensorV2 produces (C, H, W) float32 tensor
    seg_tensor = aug["image"].float().unsqueeze(0)  # (1, 3, 256, 256)

    return original_rgb, seg_tensor, (H, W)


def preprocess_overlay_for_classification(overlay_rgb: np.ndarray) -> torch.Tensor:
    """
    Prepare an overlay image (RGB np.ndarray) for classification model inference.
    Returns: torch.Tensor (1, 3, 256, 256)
    """
    aug = CLS_TRANSFORM(image=overlay_rgb)
    return aug["image"].float().unsqueeze(0)
