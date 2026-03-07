"""Download MobileNetV2 pretrained weights from torchvision and save as .npz.
Requires: pip install torch torchvision
Run once, then use train_petnet_transfer.py (which doesn't need torch).
"""
import numpy as np

try:
    import torch
    import torchvision.models as models
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch torchvision")
    print("This script is only needed once to download weights.")
    exit(1)

print("Downloading MobileNetV2 pretrained on ImageNet...")
try:
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
except (ImportError, TypeError):
    model = models.mobilenet_v2(pretrained=True)

model.eval()
sd = model.state_dict()

# Save as .npz with original PyTorch key names
arrays = {}
for k, v in sd.items():
    if 'num_batches_tracked' in k:
        continue  # skip batch count trackers
    arrays[k] = v.cpu().numpy()

out_path = "mobilenet_v2_weights.npz"
np.savez(out_path, **arrays)
print(f"Saved {len(arrays)} tensors to {out_path}")
print(f"File size: {__import__('os').path.getsize(out_path) / 1024 / 1024:.1f} MB")
