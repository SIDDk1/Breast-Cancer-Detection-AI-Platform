"""
Dataset definitions for breast cancer segmentation and classification.
Uses local dataset structure from Dataset_BUSI_with_GT.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Class mappings
CLASS_TO_IDX = {'normal': 0, 'benign': 1, 'malignant': 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMAGE_SIZE = 256


def get_seg_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        # FIXED: 3-channel tuples to match RGB input (was single-element tuples)
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


def get_seg_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # FIXED: 3-channel tuples to match RGB input (was single-element tuples)
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


def get_cls_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_cls_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class BreastSegDataset(Dataset):
    """
    Segmentation dataset. Collects image/mask pairs from flat_images/flat_masks dirs.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Only include images that have corresponding masks
        all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.images = [f for f in all_images if os.path.exists(os.path.join(mask_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = mask.astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


class ClassificationDataset(Dataset):
    """
    Classification dataset built from overlay images by class folder.
    """
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_name, label in class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
            for img_name in sorted(os.listdir(class_path)):
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.long)


def prepare_flat_dataset(root_path, flat_image_dir, flat_mask_dir):
    """
    Flattens BUSI dataset from class subfolders into flat image/mask dirs.
    Uses only the primary mask file (no _mask_1, _mask_2).
    """
    import shutil
    os.makedirs(flat_image_dir, exist_ok=True)
    os.makedirs(flat_mask_dir, exist_ok=True)

    count = 0
    skipped = 0
    for category in sorted(os.listdir(root_path)):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue
        for filename in sorted(os.listdir(category_path)):
            if filename.endswith('.png') and '_mask' not in filename:
                img_src = os.path.join(category_path, filename)
                img_dst = os.path.join(flat_image_dir, f"{category}_{filename}")

                # Use only the primary mask
                mask_src = os.path.join(category_path, filename.replace('.png', '_mask.png'))
                mask_dst = os.path.join(flat_mask_dir, f"{category}_{filename}")

                if os.path.exists(mask_src):
                    if not os.path.exists(img_dst):
                        shutil.copy(img_src, img_dst)
                    if not os.path.exists(mask_dst):
                        shutil.copy(mask_src, mask_dst)
                    count += 1
                else:
                    skipped += 1

    return count, skipped
