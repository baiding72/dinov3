"""Visualize metric 2 (object region classification) end-to-end on one nuImages frame.

Pipeline shown: original 1600x900 -> short-edge 910x512 -> center crop 512x512
-> 32x32 patch grid -> per-bbox patch selection -> mean pool [1024] -> Linear(1024->C).
Same image as debug_0 for cross-referencing with the drivable pipeline.
"""

import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

ROOT = "/Users/baiding/nuimages"
META = os.path.join(ROOT, "v1.0-mini")
OUT = "/Users/baiding/Desktop/dinov3/notes/object_probe_vis.png"
IMG_NAME = "n006-2018-09-17-12-15-45-0400__CAM_FRONT__1537201092012482.jpg"


def load(fn):
    with open(os.path.join(META, fn)) as f:
        return json.load(f)


sample_data = load("sample_data.json")
object_ann = load("object_ann.json")
category = {c["token"]: c["name"] for c in load("category.json")}
calibrated = load("calibrated_sensor.json")
sensors = {s["token"]: s["channel"] for s in load("sensor.json")}
ch_by_cs = {c["token"]: sensors[c["sensor_token"]] for c in calibrated}

sd = next(s for s in sample_data if s["filename"].endswith(IMG_NAME))
objs = [a for a in object_ann if a["sample_data_token"] == sd["token"]]


def bbox_xyxy(a):
    b = a["bbox"]
    if isinstance(b, str):
        b = b[1:-1].split(",")
    return [float(v) for v in b]


def transform_bbox(x1, y1, x2, y2, scale, left, top):
    return x1 * scale - left, y1 * scale - top, x2 * scale - left, y2 * scale - top


def patch_span(x1, y1, x2, y2):
    """Patch grid coords covered by a bbox in 512x512 space."""
    px1 = max(0, int(x1 // 16))
    py1 = max(0, int(y1 // 16))
    px2 = min(32, int(math.ceil(x2 / 16)))
    py2 = min(32, int(math.ceil(y2 / 16)))
    return px1, py1, px2, py2


img = np.asarray(Image.open(os.path.join(ROOT, sd["filename"])).convert("RGB"))
H, W = img.shape[:2]
scale = 512 / min(H, W)
rw, rh = round(W * scale), round(H * scale)
resized = np.asarray(Image.fromarray(img).resize((rw, rh), Image.BICUBIC))
left = (rw - 512) // 2
top = (rh - 512) // 2
cropped = resized[top : top + 512, left : left + 512]

boxes = [bbox_xyxy(a) for a in objs]
names = [category.get(a["category_token"], "?") for a in objs]
tboxes = [transform_bbox(*b, scale, left, top) for b in boxes]


def clip_to_crop(tb):
    x1, y1, x2, y2 = tb
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(512.0, x2), min(512.0, y2)
    return (cx1, cy1, cx2, cy2) if cx2 > cx1 and cy2 > cy1 else None


clipped = [clip_to_crop(tb) for tb in tboxes]
spans = [patch_span(*cb) if cb is not None else None for cb in clipped]
areas = [None if cb is None else (cb[2] - cb[0]) * (cb[3] - cb[1]) for cb in clipped]
order = sorted(range(len(objs)), key=lambda i: -(areas[i] if areas[i] is not None else 0))

print(f"keyframe: {IMG_NAME} | n_objects={len(objs)}")
for i in order:
    if spans[i] is None:
        print(f"  {names[i]:32s} bbox_orig={[round(v) for v in boxes[i]]} -> OUTSIDE crop, dropped")
    else:
        print(f"  {names[i]:32s} bbox_orig={[round(v) for v in boxes[i]]} "
              f"patches={spans[i]} n_patches={(spans[i][2]-spans[i][0])*(spans[i][3]-spans[i][1])}")

cmap = plt.get_cmap("tab10")
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3)

# 1. original + bboxes
ax = fig.add_subplot(gs[0, 0])
ax.imshow(img)
for i in order:
    x1, y1, x2, y2 = boxes[i]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=cmap(i % 10), lw=1.5))
    ax.text(x1, max(0, y1 - 8), names[i].split(".")[-1], fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc=cmap(i % 10), alpha=0.8))
ax.set_title(f"1) original {W}x{H} + object bboxes ({len(objs)} objects)")
ax.axis("off")

# 2. resized
ax = fig.add_subplot(gs[0, 1])
ax.imshow(resized)
for i in order:
    x1, y1, x2, y2 = tboxes[i]
    if clipped[i] is None:
        continue
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=cmap(i % 10), lw=1.5))
ax.set_title(f"2) short-edge resize -> {rw}x{rh} (bbox x{scale:.4f}; crop-outside dropped)")
ax.axis("off")

# 3. crop
ax = fig.add_subplot(gs[0, 2])
ax.imshow(cropped)
for i in order:
    x1, y1, x2, y2 = tboxes[i]
    if clipped[i] is None:
        continue
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=cmap(i % 10), lw=1.5))
    ax.text(x1, max(0, y1 - 6), f"#{i}", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc=cmap(i % 10), alpha=0.8))
ax.set_title(f"3) center crop 512x512 (offset_x={left})")
ax.axis("off")

# 4. crop + 32x32 grid + selected patches
ax = fig.add_subplot(gs[1, 0])
ax.imshow(cropped)
for i in range(1, 32):
    ax.axvline(i * 16, color="gray", lw=0.3, alpha=0.5)
    ax.axhline(i * 16, color="gray", lw=0.3, alpha=0.5)
for i in order:
    if spans[i] is None:
        continue
    px1, py1, px2, py2 = spans[i]
    for gy in range(py1, py2):
        for gx in range(px1, px2):
            ax.add_patch(Rectangle((gx * 16, gy * 16), 16, 16, fill=True,
                                   fc=cmap(i % 10), alpha=0.35, ec=cmap(i % 10), lw=0.5))
ax.set_title("4) 32x32 grid: patches inside each bbox (colored by object)")
ax.axis("off")

# 5. pooling + head schematic (representative object = largest)
ax = fig.add_subplot(gs[1, 1])
rep = order[0]
rep = next(i for i in order if spans[i] is not None)
px1, py1, px2, py2 = spans[rep]
n_patch = (px2 - px1) * (py2 - py1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
# patch grid
gx0, gy0, cell = 0.3, 6.5, 0.55
for r in range(min(3, py2 - py1)):
    for c in range(min(3, px2 - px1)):
        ax.add_patch(Rectangle((gx0 + c * cell, gy0 - r * cell), cell, cell,
                               fc=cmap(rep % 10), alpha=0.5, ec="k", lw=0.6))
ax.text(gx0, gy0 + 0.8, f"object #{rep}: {n_patch} patches x 1024d", fontsize=9)
ax.add_patch(FancyArrowPatch((gx0 + 1.9, gy0 - 0.8), (3.2, 5.2),
                             arrowstyle="-|>", mutation_scale=14, color="k"))
ax.text(2.0, 5.6, "mean pool", fontsize=9)
# pooled vector
ax.add_patch(Rectangle((3.2, 4.1), 2.6, 1.3, fc="lightblue", ec="k"))
ax.text(4.5, 4.75, "[1024]", ha="center", fontsize=10)
ax.add_patch(FancyArrowPatch((5.8, 4.75), (6.9, 4.75), arrowstyle="-|>", mutation_scale=14, color="k"))
ax.text(5.9, 5.15, "Linear(1024->C)", fontsize=8)
# logits
for k in range(8):
    h = 0.7
    ax.add_patch(Rectangle((7.0, 3.6 + k * 0.85), 1.6, h, fc="lightgray", ec="k"))
ax.text(7.8, 3.45, "logits [C]", ha="center", fontsize=9)
ax.text(7.0, 8.8, "schematic, not real values", fontsize=8, style="italic", color="gray")
ax.set_title("5) bbox patches -> mean pool -> Linear head")

# 6. per-object stats
ax = fig.add_subplot(gs[1, 2])
lines = [f"{'#':>2} {'category':<28} {'patches':>8} {'bbox(patch)':>14}"]
for i in order:
    if spans[i] is None:
        lines.append(f"{i:>2} {names[i]:<28} {'dropped':>8}")
        continue
    px1, py1, px2, py2 = spans[i]
    n = (px2 - px1) * (py2 - py1)
    lines.append(f"{i:>2} {names[i]:<28} {n:>8} [{px1},{py1}]-[{px2},{py2}]".rstrip())
ax.text(0.02, 0.97, "\n".join(lines), va="top", fontfamily="monospace", fontsize=8)
ax.set_title("6) per-object patch spans")
ax.axis("off")

fig.suptitle(f"Metric 2 pipeline: object region classification ({IMG_NAME})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=110)
print("saved", OUT)
