"""Visualize metric 3 (CLS multi-label classification) end-to-end on one nuImages frame.

Pipeline: original 1600x900 -> center crop 512x512 -> backbone CLS [1024]
-> Linear(1024->C)+sigmoid -> [C] probs; label = multi-hot of coarse classes present
in the keyframe (from object_ann). Metric: per-class AP -> mAP.
Same image as debug_0 for cross-referencing.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

ROOT = "/Users/baiding/nuimages"
META = os.path.join(ROOT, "v1.0-mini")
OUT = "/Users/baiding/Desktop/dinov3/notes/cls_probe_vis.png"
IMG_NAME = "n006-2018-09-17-12-15-45-0400__CAM_FRONT__1537201092012482.jpg"
COARSE_ORDER = ["vehicle", "human", "movable_object", "static_object", "animal"]


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
coarse_of = {a["token"]: category.get(a["category_token"], "?").split(".")[0] for a in objs}
present = sorted({c for c in coarse_of.values()})
multi_hot = [1.0 if c in present else 0.0 for c in COARSE_ORDER]


def bbox_xyxy(a):
    b = a["bbox"]
    if isinstance(b, str):
        b = b[1:-1].split(",")
    return [float(v) for v in b]


img = np.asarray(Image.open(os.path.join(ROOT, sd["filename"])).convert("RGB"))
H, W = img.shape[:2]
scale = 512 / min(H, W)
rw, rh = round(W * scale), round(H * scale)
resized = np.asarray(Image.fromarray(img).resize((rw, rh), Image.BICUBIC))
left = (rw - 512) // 2
top = (rh - 512) // 2
cropped = resized[top : top + 512, left : left + 512]

cmap = plt.get_cmap("tab10")
class_color = {c: cmap(i % 10) for i, c in enumerate(COARSE_ORDER)}

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3)

# 1. original + bboxes colored by coarse class
ax = fig.add_subplot(gs[0, 0])
ax.imshow(img)
handles = []
for a in objs:
    c = coarse_of[a["token"]]
    x1, y1, x2, y2 = bbox_xyxy(a)
    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=class_color[c], lw=1.5))
for c in COARSE_ORDER:
    if c in present:
        handles.append(plt.Line2D([0], [0], color=class_color[c], lw=3, label=c))
ax.legend(handles=handles, loc="upper right", fontsize=9)
ax.set_title(f"1) original {W}x{H}: objects colored by coarse class\nclasses present: {present}")
ax.axis("off")

# 2. crop + 32x32 grid + CLS concept
ax = fig.add_subplot(gs[0, 1])
ax.imshow(cropped)
for i in range(1, 32):
    ax.axvline(i * 16, color="gray", lw=0.3, alpha=0.4)
    ax.axhline(i * 16, color="gray", lw=0.3, alpha=0.4)
ax.set_title("2) center crop 512x512 -> 32x32 patches\n(metric 3 uses only the CLS token, not patches)")
ax.axis("off")

# 3. multi-hot label
ax = fig.add_subplot(gs[0, 2])
ax.set_xlim(0, 6)
ax.set_ylim(-0.5, len(COARSE_ORDER) + 0.5)
ax.axis("off")
for k, c in enumerate(COARSE_ORDER):
    val = multi_hot[k]
    y = len(COARSE_ORDER) - 1 - k
    ax.add_patch(Rectangle((0.4, y - 0.35), 1.4, 0.7, fc="lightgreen" if val else "lightgray", ec="k"))
    ax.text(2.1, y, f"{c} = {int(val)}", fontsize=11, va="center")
ax.text(0.4, len(COARSE_ORDER) + 0.2,
        "3) label: multi-hot of classes present\n(aggregate object_ann of this keyframe)",
        fontsize=9)
ax.set_title("multi-hot label [5]")

# 4. backbone -> CLS -> head -> probs (schematic)
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis("off")
ax.add_patch(Rectangle((0.3, 3.4), 2.4, 2.4, fc="lightyellow", ec="k"))
ax.text(1.5, 4.6, "512x512", ha="center", fontsize=10)
ax.add_patch(FancyArrowPatch((2.7, 4.6), (4.0, 4.6), arrowstyle="-|>", mutation_scale=16, color="k"))
ax.text(3.0, 5.0, "ViT-L/16", fontsize=9)
ax.add_patch(Rectangle((4.0, 3.9), 2.0, 1.4, fc="lightblue", ec="k"))
ax.text(5.0, 4.6, "CLS\n[1024]", ha="center", fontsize=9)
ax.add_patch(FancyArrowPatch((6.0, 4.6), (7.3, 4.6), arrowstyle="-|>", mutation_scale=16, color="k"))
ax.text(6.1, 5.0, "Linear(1024->5)\n+ sigmoid", fontsize=8)
for k in range(5):
    ax.add_patch(Rectangle((7.3, 2.6 + k * 0.9), 1.4, 0.7, fc="lightgray", ec="k"))
ax.text(8.8, 4.0, "probs [5]", fontsize=10, va="center")
ax.text(0.3, 9.0, "4) inference: whole image -> one CLS token -> 5 sigmoid probs", fontsize=10)
ax.text(0.3, 1.2, "loss = BCE(probs, multi-hot label)", fontsize=10)
ax.text(0.3, 0.5, "schematic, not real values", fontsize=8, style="italic", color="gray")
ax.set_title("CLS global head")

# 5. per-class AP / mAP explanation
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
lines = [
    "5) per-class AP -> mAP",
    "",
    "for class c:",
    "  score = sigmoid prob of class c",
    "  label = 1 if class present in frame",
    "  rank val frames by score",
    "  AP_c = area under PR curve",
    "",
    "mAP = mean(AP_c) over 5 classes",
    "",
    "why AP: multi-label, so accuracy is",
    "meaningless; AP measures ranking",
    "quality per class, robust to",
    "class imbalance.",
]
ax.text(0.02, 0.97, "\n".join(lines), va="top", fontfamily="monospace", fontsize=9)

# 6. per-class presence stats for this frame
ax = fig.add_subplot(gs[1, 2])
from collections import Counter
counts = Counter(coarse_of.values())
lines = [f"{'class':<16} {'present':>7} {'objects':>8}"]
for c in COARSE_ORDER:
    lines.append(f"{c:<16} {int(c in present):>7} {counts.get(c, 0):>8}")
ax.text(0.02, 0.97, "\n".join(lines), va="top", fontfamily="monospace", fontsize=10)
ax.set_title("6) this keyframe's class presence")
ax.axis("off")

fig.suptitle(f"Metric 3 pipeline: CLS multi-label classification ({IMG_NAME})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=110)
print("saved", OUT)
print("classes present:", present)
print("multi-hot:", dict(zip(COARSE_ORDER, multi_hot)))
