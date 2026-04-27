"""
Fast CPU-optimized training script for the Attention U-Net segmentation model.
Designed to complete in ~30-45 minutes on CPU (no GPU required).

Optimizations vs full training:
  - IMAGE_SIZE = 128  (vs 256 → 4× faster conv operations)
  - SEG_EPOCHS = 8    (with early stopping after PATIENCE=4)
  - BATCH_SIZE = 16   (larger batch, better gradient estimates)
  - CLS_EPOCHS = 10   (classification is lighter)
  - NO tqdm progress bars (faster I/O)

Usage:
    python fast_train.py

After training completes, restart the backend server.
"""

import os
import sys
import copy
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.services.model import get_segmentation_model, get_classification_model
from backend.services.dataset import (
    BreastSegDataset, ClassificationDataset, CLASS_TO_IDX,
    get_seg_train_transform, get_seg_val_transform,
    get_cls_train_transform, get_cls_val_transform,
    prepare_flat_dataset, IMAGE_SIZE as FULL_SIZE
)
from backend.services.postprocess import postprocess_mask, create_overlay
from backend.utils.logger import get_logger

logger = get_logger("fast_train")

# ── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128   # Half-resolution for 4× speedup
SEG_EPOCHS   = 8    # Early stopping will cut this down
CLS_EPOCHS   = 10
BATCH_SIZE   = 16
LR           = 5e-4
PATIENCE     = 4    # Aggressive early stopping
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT         = Path(__file__).resolve().parent
DATASET_PATH = ROOT.parent / "use-this-one" / "Dataset_BUSI_with_GT"
FLAT_IMAGES  = ROOT / "temp" / "flat_images"
FLAT_MASKS   = ROOT / "temp" / "flat_masks"
OVERLAY_DIR  = ROOT / "temp" / "overlay_dataset_fast"
WEIGHTS_DIR  = ROOT / "weights"

print(f"🚀 Fast Training | Device: {DEVICE} | Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"   Dataset: {DATASET_PATH}")


# ── LOSS ────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        p = pred.view(-1)
        t = target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, w=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.w = w

    def forward(self, pred, target):
        return self.w * self.bce(pred, target) + (1 - self.w) * self.dice(pred, target)


def dice_coeff(pred, target, thr=0.5):
    p = (pred > thr).float()
    inter = (p * target).sum()
    return ((2 * inter + 1e-6) / (p.sum() + target.sum() + 1e-6)).item()


# ── FAST TRANSFORMS (128×128) ────────────────────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2

def fast_seg_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

def fast_seg_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

def fast_cls_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def fast_cls_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── PREPARE DATASET ──────────────────────────────────────────────────────────
def prepare():
    if FLAT_IMAGES.exists() and len(list(FLAT_IMAGES.glob("*.png"))) > 0:
        n = len(list(FLAT_IMAGES.glob("*.png")))
        print(f"✅ Using existing flat dataset: {n} images")
        return n
    print("📦 Preparing flat dataset from BUSI...")
    FLAT_IMAGES.mkdir(parents=True, exist_ok=True)
    FLAT_MASKS.mkdir(parents=True, exist_ok=True)
    count, skipped = prepare_flat_dataset(str(DATASET_PATH), str(FLAT_IMAGES), str(FLAT_MASKS))
    print(f"   Prepared {count} samples ({skipped} skipped)")
    return count


# ── TRAIN SEGMENTATION ────────────────────────────────────────────────────────
def train_seg():
    print("\n" + "="*60)
    print("PHASE 1: Segmentation Training (Attention U-Net @ 128×128)")
    print("="*60)

    n = prepare()

    # Full dataset with splits
    all_ds = BreastSegDataset(str(FLAT_IMAGES), str(FLAT_MASKS), transform=None)
    n_total = len(all_ds)
    n_val   = max(10, int(0.15 * n_total))
    n_train = n_total - n_val

    import random
    idxs = list(range(n_total))
    random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    val_idxs   = idxs[n_train:]

    train_ds = BreastSegDataset(str(FLAT_IMAGES), str(FLAT_MASKS), transform=fast_seg_train_transform())
    val_ds   = BreastSegDataset(str(FLAT_IMAGES), str(FLAT_MASKS), transform=fast_seg_val_transform())

    train_loader = DataLoader(Subset(train_ds, train_idxs), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(Subset(val_ds,   val_idxs),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"   Train: {n_train} | Val: {n_val} | Batches/epoch: {len(train_loader)}")

    model     = get_segmentation_model().to(DEVICE)
    criterion = BCEDiceLoss(w=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss  = float('inf')
    best_val_dice  = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(SEG_EPOCHS):
        # ── Train ──
        model.train()
        t_loss = t_dice = 0.0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            with torch.no_grad():
                t_dice += dice_coeff(preds, masks)
            # Print batch progress
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"   Ep{epoch+1}/{SEG_EPOCHS} [{i+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}", flush=True)

        t_loss /= len(train_loader)
        t_dice /= len(train_loader)

        # ── Validate ──
        model.eval()
        v_loss = v_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(DEVICE)
                masks = masks.to(DEVICE).float()
                preds = model(imgs)
                v_loss += criterion(preds, masks).item()
                v_dice += dice_coeff(preds, masks)

        v_loss /= len(val_loader)
        v_dice /= len(val_loader)
        scheduler.step(v_loss)

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{SEG_EPOCHS}")
        print(f"  Train → loss={t_loss:.4f} | dice={t_dice:.4f}")
        print(f"  Val   → loss={v_loss:.4f} | dice={v_dice:.4f}")
        print(f"{'='*50}\n")

        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_val_dice  = v_dice
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
            print(f"   ✅ New best! val_loss={best_val_loss:.4f} | val_dice={best_val_dice:.4f}")
            # Save checkpoint immediately
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, str(WEIGHTS_DIR / "seg_model.pth"))
            print(f"   💾 Checkpoint saved: weights/seg_model.pth")
        else:
            patience_count += 1
            print(f"   No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"   ⏹ Early stopping at epoch {epoch+1}")
                break

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(WEIGHTS_DIR / "seg_model.pth"))
    print(f"\n✅ Segmentation model saved! Best val dice={best_val_dice:.4f}")
    return model, best_state


# ── GENERATE OVERLAYS ─────────────────────────────────────────────────────────
def generate_overlays(seg_model, best_state):
    print("\n" + "="*60)
    print("PHASE 2: Generating Overlay Dataset for Classification")
    print("="*60)

    seg_model.eval()
    seg_model = seg_model.to(DEVICE)

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
        print(f"   {category}: {len(files)} images")

        for filename in files:
            img_path  = cat_path / filename
            save_path = out_path / filename
            if save_path.exists():
                total += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]

            aug    = seg_transform(image=img_rgb)
            tensor = aug["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = seg_model(tensor).squeeze().cpu().numpy()

            pred_range = float(pred.max() - pred.min())
            mask = postprocess_mask(pred, (H, W), threshold=0.5)
            overlay = create_overlay(img_rgb, mask, alpha=0.3)
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            total += 1

    print(f"✅ Generated {total} overlay images")


# ── TRAIN CLASSIFICATION ──────────────────────────────────────────────────────
def train_cls():
    print("\n" + "="*60)
    print("PHASE 3: Classification Training (CNN)")
    print("="*60)

    train_ds = ClassificationDataset(str(OVERLAY_DIR), CLASS_TO_IDX, transform=fast_cls_train_transform())
    val_ds   = ClassificationDataset(str(OVERLAY_DIR), CLASS_TO_IDX, transform=fast_cls_val_transform())

    if len(train_ds) == 0:
        print("⚠️ No overlay images found — skipping classification training")
        return

    n_total = len(train_ds)
    n_val   = max(5, int(0.15 * n_total))
    n_train = n_total - n_val

    train_loader = DataLoader(Subset(train_ds, list(range(n_train))), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(Subset(val_ds,   list(range(n_train, n_total))), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"   Train: {n_train} | Val: {n_val}")

    model     = get_classification_model(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state    = None
    patience_count = 0

    for epoch in range(CLS_EPOCHS):
        model.train()
        t_loss = t_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += (out.argmax(1) == labels).sum().item()

        model.eval()
        v_loss = v_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out    = model(imgs)
                v_loss += criterion(out, labels).item()
                v_correct += (out.argmax(1) == labels).sum().item()

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        t_acc   = t_correct / n_train
        v_acc   = v_correct / n_val
        scheduler.step(v_loss)

        print(f"Epoch {epoch+1}/{CLS_EPOCHS}: train_loss={t_loss:.4f} acc={t_acc:.3f} | "
              f"val_loss={v_loss:.4f} acc={v_acc:.3f}")

        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
            print(f"  ✅ New best cls model saved")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(WEIGHTS_DIR / "cls_model.pth"))
    print(f"\n✅ Classification model saved!")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    print("\n" + "🔥"*30)
    print("FAST CPU TRAINING — Attention U-Net + CNN Classifier")
    print("🔥"*30 + "\n")

    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    # Phase 1: Segmentation
    seg_model, best_seg_state = train_seg()

    # Phase 2: Overlay generation
    generate_overlays(seg_model, best_seg_state)

    # Phase 3: Classification
    train_cls()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ ALL TRAINING COMPLETE in {elapsed/60:.1f} minutes")
    print(f"   Weights saved in: {WEIGHTS_DIR}")
    print("   NOW RESTART THE BACKEND SERVER to load new weights.")
    print(f"{'='*60}\n")
