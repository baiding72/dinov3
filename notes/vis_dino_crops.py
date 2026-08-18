"""Visualize DINOv3 global vs local crops and their loss pairing on a nuImages frame.

Uses the same crop recipe as dinov3_vitl16_lvd1689m_distilled.yaml:
  global: 2 crops, scale [0.32, 1.0], size 256
  local : 8 crops, scale [0.05, 0.32], size 112
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

sys.path.insert(0, "/Users/baiding/Desktop/dinov3")

GLOBAL_SCALE = (0.32, 1.0)
LOCAL_SCALE = (0.05, 0.32)
G_SIZE = 256
L_SIZE = 112
N_LOCAL = 8

IMG_PATH = (
    "/Users/baiding/nuimages/samples/CAM_FRONT/"
    "n006-2018-09-17-12-15-45-0400__CAM_FRONT__1537201092012482.jpg"
)
OUT_PATH = "/Users/baiding/Desktop/dinov3/notes/dino_crops_vis.png"

torch.manual_seed(0)
np.random.seed(0)


def sample_crop(img, scale, size):
    """RandomResizedCrop source box in original coords + the resized crop."""
    i, j, h, w = T.RandomResizedCrop.get_params(img, scale=scale, ratio=(3 / 4, 4 / 3))
    crop = F.resized_crop(img, i, j, h, w, (size, size), F.InterpolationMode.BICUBIC)
    return crop, (j, i, w, h)  # box = (x, y, w, h)


pil = Image.open(IMG_PATH).convert("RGB")
W, H = pil.size

global_crops, global_boxes = [], []
for _ in range(2):
    c, box = sample_crop(pil, GLOBAL_SCALE, G_SIZE)
    global_crops.append(c)
    global_boxes.append(box)

local_crops, local_boxes = [], []
for _ in range(N_LOCAL):
    c, box = sample_crop(pil, LOCAL_SCALE, L_SIZE)
    local_crops.append(c)
    local_boxes.append(box)

print("image size:", pil.size)
print("global boxes:", global_boxes)
print("local boxes:", local_boxes)

fig = plt.figure(figsize=(16, 15))
gs = fig.add_gridspec(4, 4)

# Row 1: original + crop source regions
ax = fig.add_subplot(gs[0, :])
ax.imshow(pil)
for k, (x, y, w, h) in enumerate(global_boxes):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="blue", lw=4, label="global" if k == 0 else None))
for k, (x, y, w, h) in enumerate(local_boxes):
    ax.add_patch(
        Rectangle((x, y), w, h, fill=False, ec="red", lw=2, ls="--", label="local" if k == 0 else None)
    )
ax.set_title(
    f"original {W}x{H}: 2 global (blue, 32%-100% area) + {N_LOCAL} local (red, 5%-32% area)",
    fontsize=13,
)
ax.legend(loc="upper right")
ax.axis("off")

# Row 2: global crops
ax = fig.add_subplot(gs[1, 0:2])
ax.imshow(global_crops[0])
ax.set_title(f"global crop 1\n{G_SIZE}x{G_SIZE} -> {G_SIZE//16}x{G_SIZE//16} patches")
ax.axis("off")
ax = fig.add_subplot(gs[1, 2])
ax.imshow(global_crops[1])
ax.set_title(f"global crop 2\n{G_SIZE}x{G_SIZE} -> {G_SIZE//16}x{G_SIZE//16} patches")
ax.axis("off")
ax = fig.add_subplot(gs[1, 3])
ax.text(0.5, 0.5, "teacher sees ONLY these 2 global crops\n(student also sees 8 local)", ha="center",
        va="center", fontsize=12, wrap=True)
ax.axis("off")

# Row 3-4: local crops
for k, crop in enumerate(local_crops):
    ax = fig.add_subplot(gs[2 + k // 4, k % 4])
    ax.imshow(crop)
    ax.set_title(f"local {k+1}\n{L_SIZE}x{L_SIZE} -> {L_SIZE//16}x{L_SIZE//16} patches", fontsize=9)
    ax.axis("off")

fig.suptitle("DINOv3 crops on nuImages CAM_FRONT (seed=0)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PATH, dpi=110)
print("saved", OUT_PATH)
