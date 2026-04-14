"""
Quick test of the classical CV fallback and postprocess fixes.
Run: python test_cv_fallback.py
"""
import sys
sys.path.insert(0, r'e:/final_year/project')
import cv2
import numpy as np
from pathlib import Path
from backend.services.inference import run_inference, load_models
from backend.utils.logger import get_logger

logger = get_logger("test")

print("Loading models...")
ok = load_models(r"e:/final_year/project/weights")
print(f"Models loaded: {ok}")

uploads_dir = Path(r"e:/final_year/project/uploads")
test_imgs = list(uploads_dir.glob("*.png"))[:3]
print(f"\nTesting on {len(test_imgs)} images...\n")

for img_path in test_imgs:
    print(f"{'='*50}")
    print(f"Image: {img_path.name}")
    try:
        result = run_inference(str(img_path))
        mask   = result["mask_array"]
        overlay = result["overlay_array"]
        orig    = result["original_array"]

        mask_px = int((mask > 127).sum())
        H, W    = mask.shape
        coverage = mask_px / (H * W) * 100

        print(f"  Method:   {result.get('seg_method', 'unknown')}")
        print(f"  Coverage: {coverage:.2f}% ({mask_px} pixels)")
        print(f"  Mask empty?: {'YES ❌' if mask_px == 0 else 'NO ✅'}")
        print(f"  Overlay all-red check:")
        # Check if overlay is all-red (bug) vs selective
        if mask_px > 0:
            mask_bool = mask > 127
            non_mask_overlay = overlay[~mask_bool]
            orig_non_mask    = orig[~mask_bool]
            if len(non_mask_overlay) > 0:
                diff = np.abs(non_mask_overlay.astype(float) - orig_non_mask.astype(float)).mean()
                print(f"    Non-mask region diff from original: {diff:.4f} (should be ~0)")
                print(f"    {'✅ CORRECT: non-mask unchanged' if diff < 5.0 else '❌ WRONG: non-mask modified'}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")

        # Save outputs for visual inspection
        out_dir = Path(r"e:/final_year/project/test_outputs")
        out_dir.mkdir(exist_ok=True)
        stem = img_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Saved to: test_outputs/{stem}_mask.png")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
    print()

print("✅ Test complete. Check test_outputs/ folder for visual results.")
