import torch
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.services.model import get_segmentation_model, get_classification_model

print("Generating initialized weights for fast testing...")
weights_dir = Path(__file__).resolve().parent.parent.parent / "weights"
weights_dir.mkdir(exist_ok=True)

seg_model = get_segmentation_model()
torch.save(seg_model.state_dict(), str(weights_dir / "seg_model.pth"))

cls_model = get_classification_model(num_classes=3)
torch.save(cls_model.state_dict(), str(weights_dir / "cls_model.pth"))
print(f"Weights saved in {weights_dir}. You can now start the API.")
