"""Check dataset paths and flat dirs."""
from pathlib import Path
import os

flat_images = Path(r'e:/final_year/project/temp/flat_images')
flat_masks  = Path(r'e:/final_year/project/temp/flat_masks')
overlay_dir = Path(r'e:/final_year/project/temp/overlay_dataset')

count_fi = len(list(flat_images.glob('*.png'))) if flat_images.exists() else 0
count_fm = len(list(flat_masks.glob('*.png')))  if flat_masks.exists() else 0
print(f'flat_images: {flat_images.exists()}, {count_fi} files')
print(f'flat_masks:  {flat_masks.exists()}, {count_fm} files')
print(f'overlay_dir: {overlay_dir.exists()}')

if overlay_dir.exists():
    for cat in ['normal','benign','malignant']:
        d = overlay_dir / cat
        n = len(list(d.glob('*.png'))) if d.exists() else 0
        print(f'  overlay/{cat}: {n} files')

dataset_path = Path(r'e:/final_year/use-this-one/Dataset_BUSI_with_GT')
for cat in ['normal','benign','malignant']:
    d = dataset_path / cat
    if d.exists():
        imgs  = [f for f in os.listdir(str(d)) if f.endswith('.png') and '_mask' not in f]
        masks = [f for f in os.listdir(str(d)) if f.endswith('_mask.png')]
        print(f'BUSI/{cat}: {len(imgs)} images, {len(masks)} masks')
    else:
        print(f'BUSI/{cat}: NOT FOUND')
