"""
Offline training script for Attention U-Net segmentation and CNN classification.
Run this script ONCE to train and save weights.

Usage:
    python -m backend.services.train

DO NOT run during API serving. Inference only uses saved weights.
"""

import os
import sys
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.services.model import get_segmentation_model, get_classification_model
from backend.services.dataset import (
    BreastSegDataset, ClassificationDataset, CLASS_TO_IDX,
    get_seg_train_transform, get_seg_val_transform,
    get_cls_train_transform, get_cls_val_transform,
    prepare_flat_dataset
)
from backend.utils.logger import get_logger
from backend.services.postprocess import postprocess_mask, create_overlay

logger = get_logger("train")

# ─────────────── CONFIG ───────────────
SEG_EPOCHS = 50     # Enough epochs for convergence; early stopping will cut this short
CLS_EPOCHS = 30     # Enough for classification head to converge
BATCH_SIZE = 8      # Increase from 4 → 8 for better gradient estimates
LR = 3e-4           # Slightly higher LR, scheduler will reduce as needed
PATIENCE = 10       # Early stopping patience
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT.parent / "use-this-one" / "Dataset_BUSI_with_GT"
FLAT_IMAGES = ROOT / "temp" / "flat_images"
FLAT_MASKS = ROOT / "temp" / "flat_masks"
OVERLAY_DIR = ROOT / "temp" / "overlay_dataset"
WEIGHTS_DIR = ROOT / "weights"


# ─────────────── LOSS ───────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice(pred, target)


# ─────────────── METRICS ───────────────

def pixel_accuracy(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    return (pred_bin == target).float().mean().item()


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return ((2 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)).item()


# ─────────────── SEGMENTATION TRAINING ───────────────

def train_segmentation():
    logger.info("=" * 60)
    logger.info("PHASE 1: Segmentation Training (Attention U-Net)")
    logger.info(f"Device: {DEVICE} | Epochs: {SEG_EPOCHS} | Batch: {BATCH_SIZE}")
    logger.info("=" * 60)

    # Prepare flat dataset
    if not FLAT_IMAGES.exists() or len(list(FLAT_IMAGES.glob("*.png"))) == 0:
        logger.info("Preparing flat dataset from BUSI...")
        FLAT_IMAGES.mkdir(parents=True, exist_ok=True)
        FLAT_MASKS.mkdir(parents=True, exist_ok=True)
        count, skipped = prepare_flat_dataset(str(DATASET_PATH), str(FLAT_IMAGES), str(FLAT_MASKS))
        logger.info(f"Dataset prepared: {count} samples, {skipped} skipped (no mask)")
    else:
        count = len(list(FLAT_IMAGES.glob("*.png")))
        logger.info(f"Using existing flat dataset: {count} images")

    # Create datasets with proper train/val split
    all_dataset = BreastSegDataset(str(FLAT_IMAGES), str(FLAT_MASKS), transform=None)
    n_total = len(all_dataset)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    # Get random split indices
    indices = list(range(n_total))
    import random; random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = BreastSegDataset(str(FLAT_IMAGES), str(FLAT_MASKS), transform=get_seg_train_transform())
    val_ds_obj = BreastSegDataset(str(FLAT_IMAGES), str(FLAT_MASKS), transform=get_seg_val_transform())

    from torch.utils.data import Subset
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds_obj, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    logger.info(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    # Model
    model = get_segmentation_model().to(DEVICE)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(SEG_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_dice = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{SEG_EPOCHS} [Train]", leave=False):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                train_acc += pixel_accuracy(preds, masks)
                train_dice += dice_coefficient(preds, masks)

        n_batches = len(train_loader)
        train_loss /= n_batches
        train_acc /= n_batches
        train_dice /= n_batches

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE).float()
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_acc += pixel_accuracy(preds, masks)
                val_dice += dice_coefficient(preds, masks)

        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_acc /= n_val_batches
        val_dice /= n_val_batches

        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch+1:02d}/{SEG_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {val_dice:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  ✅ New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  ⏹ Early stopping triggered at epoch {epoch+1}")
                break

    # Save best model
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    seg_path = WEIGHTS_DIR / "seg_model.pth"
    torch.save(best_state, str(seg_path))
    logger.info(f"✅ Segmentation model saved: {seg_path}")
    return model, best_state


# ─────────────── OVERLAY GENERATION ───────────────

def generate_overlay_dataset(seg_model):
    logger.info("=" * 60)
    logger.info("PHASE 2: Generating Overlay Images for Classification")
    logger.info("=" * 60)

    seg_model.eval()
    seg_model = seg_model.to(DEVICE)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # FIXED: Use 3-channel tuples to match inference preprocessing.
    # Single-element tuples (0.0,) caused channel-0-only normalization in some
    # albumentations versions, creating a train/inference mismatch for 3-ch RGB images.
    seg_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    total = 0
    for category in ['normal', 'benign', 'malignant']:
        cat_path = DATASET_PATH / category
        out_path = OVERLAY_DIR / category
        out_path.mkdir(parents=True, exist_ok=True)

        files = [f for f in sorted(os.listdir(str(cat_path)))
                 if f.endswith('.png') and '_mask' not in f]
        logger.info(f"  {category}: {len(files)} images")

        for filename in tqdm(files, desc=f"Overlay {category}", leave=False):
            img_path = cat_path / filename
            save_path = out_path / filename

            if save_path.exists():
                total += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]

            aug = seg_transform(image=img_rgb)
            tensor = aug["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = seg_model(tensor).squeeze().cpu().numpy()

            mask = postprocess_mask(pred, (H, W), threshold=0.35)
            overlay = create_overlay(img_rgb, mask, alpha=0.3)

            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            total += 1

    logger.info(f"✅ Generated {total} overlay images")


# ─────────────── CLASSIFICATION TRAINING ───────────────

def train_classification():
    logger.info("=" * 60)
    logger.info("PHASE 3: Classification Training (CNN)")
    logger.info(f"Device: {DEVICE} | Epochs: {CLS_EPOCHS}")
    logger.info("=" * 60)

    train_ds = ClassificationDataset(str(OVERLAY_DIR), CLASS_TO_IDX, transform=get_cls_train_transform())
    val_ds = ClassificationDataset(str(OVERLAY_DIR), CLASS_TO_IDX, transform=get_cls_val_transform())
    n_total = len(train_ds)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val

    train_sub, _ = random_split(train_ds, [n_train, n_val])
    _, val_sub = random_split(val_ds, [n_train, n_val])

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info(f"Train: {len(train_sub)} | Val: {len(val_sub)}")

    model = get_classification_model(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(CLS_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for imgs, labels in tqdm(train_loader, desc=f"Cls Epoch {epoch+1}/{CLS_EPOCHS} [Train]", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / len(train_sub)
        val_acc = val_correct / len(val_sub)

        scheduler.step(val_loss)
        logger.info(
            f"Epoch {epoch+1:02d}/{CLS_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  ✅ New best: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    cls_path = WEIGHTS_DIR / "cls_model.pth"
    torch.save(best_state, str(cls_path))
    logger.info(f"✅ Classification model saved: {cls_path}")


# ─────────────── MAIN ───────────────

if __name__ == "__main__":
    logger.info(f"🚀 Training on device: {DEVICE}")
    logger.info(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    # Phase 1: Train segmentation
    seg_model, best_seg_state = train_segmentation()

    # Load best state for overlay generation
    from backend.services.model import get_segmentation_model
    seg_model_for_overlay = get_segmentation_model().to(DEVICE)
    seg_model_for_overlay.load_state_dict(best_seg_state)

    # Phase 2: Generate overlays
    generate_overlay_dataset(seg_model_for_overlay)

    # Phase 3: Train classification
    train_classification()

    logger.info("=" * 60)
    logger.info("✅ ALL TRAINING COMPLETE")
    logger.info(f"Weights saved in: {WEIGHTS_DIR}")
    logger.info("You can now start the API server.")
    logger.info("=" * 60)
