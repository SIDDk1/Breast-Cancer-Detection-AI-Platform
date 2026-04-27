"""
Diagnostic script: analyze model weights and inference output.
Run from project root: python diagnose_model.py
"""
import sys
sys.path.insert(0, r'e:/final_year/project')
import torch
import numpy as np
from pathlib import Path

weights_dir = Path(r'e:/final_year/project/weights')

# ── Analyze seg model weights ──────────────────────────────────────────────────
print("=== SEG MODEL WEIGHT ANALYSIS ===")
raw_state = torch.load(str(weights_dir / "seg_model.pth"), map_location="cpu")
print(f"Type: {type(raw_state)}")
print(f"Total keys: {len(raw_state)}")
print(f"File size: {(weights_dir / 'seg_model.pth').stat().st_size / 1e6:.2f} MB")

all_means = []
all_stds = []
for k, v in raw_state.items():
    if "weight" in k and len(v.shape) >= 2:
        m = v.abs().mean().item()
        s = v.std().item()
        all_means.append(m)
        all_stds.append(s)

print(f"Weight tensors checked: {len(all_means)}")
print(f"Avg abs mean: {np.mean(all_means):.6f}")
print(f"Avg std: {np.mean(all_stds):.6f}")

# Check output bias — if near 0, this is untrained
out_bias = raw_state.get("out.bias")
out_weight = raw_state.get("out.weight")
if out_bias is not None:
    print(f"Output bias: {out_bias} (near 0 = untrained/dummy)")
if out_weight is not None:
    print(f"Output weight: shape={out_weight.shape}, mean={out_weight.mean():.6f}")

# ── Run inference on a test image ─────────────────────────────────────────────
print("\n=== INFERENCE TEST ===")
from backend.services.model import get_segmentation_model
from backend.services.preprocess import load_and_preprocess_image

seg_model = get_segmentation_model()
seg_model.load_state_dict(raw_state, strict=True)
seg_model.eval()
print("Model loaded successfully")

uploads_dir = Path(r"e:/final_year/project/uploads")
all_imgs = list(uploads_dir.glob("*.png"))[:3]
print(f"Test images: {[p.name for p in all_imgs]}")

for img_path in all_imgs:
    original_rgb, seg_tensor, (H, W) = load_and_preprocess_image(str(img_path))
    print(f"\nImage: {img_path.name} (H={H}, W={W})")
    print(f"  Input tensor: min={seg_tensor.min():.4f}, max={seg_tensor.max():.4f}, mean={seg_tensor.mean():.4f}")
    
    with torch.no_grad():
        out = seg_model(seg_tensor)
    
    raw = out.squeeze().numpy()
    print(f"  RAW OUTPUT: min={raw.min():.6f}, max={raw.max():.6f}, mean={raw.mean():.6f}")
    print(f"  Pixels > 0.5: {(raw > 0.5).sum()}")
    print(f"  Pixels > 0.4: {(raw > 0.4).sum()}")
    print(f"  Pixels > 0.3: {(raw > 0.3).sum()}")
    print(f"  Pixels > 0.1: {(raw > 0.1).sum()}")
    print(f"  Range of output: [{raw.min():.6f}, {raw.max():.6f}]  -- range={(raw.max()-raw.min()):.6f}")

print("\n=== DIAGNOSIS COMPLETE ===")
print("If output range is very small (< 0.01), model weights are UNTRAINED/DUMMY.")
print("Solution: the model needs real trained weights OR we need an adaptive threshold.")
