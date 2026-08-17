import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "/Users/baiding/nuimages"
META = os.path.join(ROOT, "v1.0-mini")
OUT = "/Users/baiding/Desktop/dinov3/notes/probe_debug_mini"
os.makedirs(OUT, exist_ok=True)


def load(fn):
    with open(os.path.join(META, fn)) as f:
        return json.load(f)


sample_data = load("sample_data.json")
surface_ann = load("surface_ann.json")
object_ann = load("object_ann.json")
category = {c["token"]: c["name"] for c in load("category.json")}
calibrated = load("calibrated_sensor.json")
sensors = {s["token"]: s["channel"] for s in load("sensor.json")}
ch_by_cs = {c["token"]: sensors[c["sensor_token"]] for c in calibrated}

sd_by_token = {s["token"]: s for s in sample_data}
oa_by_sd = {}
for a in object_ann:
    oa_by_sd.setdefault(a["sample_data_token"], []).append(a)
sa_by_sd = {}
for a in surface_ann:
    sa_by_sd.setdefault(a["sample_data_token"], []).append(a)

# CAM_FRONT keyframes (exact channel via calibrated_sensor -> sensor)
front_keys = [
    s for s in sample_data
    if s["is_key_frame"] and ch_by_cs.get(s["calibrated_sensor_token"]) == "CAM_FRONT"
]
front_keys.sort(key=lambda s: s["filename"])

# ---- decode helper: reuse devkit mask_decode ----
sys.path.insert(0, "/Users/baiding/Desktop/dinov3/tools/nuscenes-devkit/python-sdk")
from nuimages.utils.utils import mask_decode  # noqa: E402


def drivable_mask(sd_token, h=900, w=1600):
    """OR-merge all flat.driveable_surface records of one keyframe."""
    masks = []
    for a in sa_by_sd.get(sd_token, []):
        if category.get(a["category_token"]) == "flat.driveable_surface":
            masks.append(mask_decode(a["mask"]).astype(bool))
    if not masks:
        return None
    out = np.zeros((h, w), dtype=bool)
    for m in masks:
        out |= m
    return out


def ego_mask(sd_token, h=900, w=1600):
    for a in sa_by_sd.get(sd_token, []):
        if category.get(a["category_token"]) == "vehicle.ego":
            return mask_decode(a["mask"]).astype(bool)
    return None


def bbox_xyxy(a):
    b = a["bbox"]
    if isinstance(b, str):
        b = b[1:-1].split(",")
    return [float(v) for v in b]


def short_resize_center_crop(img_np, target=512):
    """short-edge resize (bicubic) + center crop, returns (cropped, resized, offset_x, scale)."""
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    h, w = img_np.shape[:2]
    scale = target / min(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    pil = Image.fromarray(img_np)
    resized = np.asarray(pil.resize((new_w, new_h), Image.BICUBIC))
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    cropped = resized[top : top + target, left : left + target]
    return cropped, resized, left, top, scale, (new_w, new_h)


def mask_to_grid(mask_crop, target=32):
    """area-average downsample to target grid, then threshold >0.5."""
    h, w = mask_crop.shape[:2]
    pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
    small = np.asarray(pil.resize((target, target), Image.BOX), dtype=np.float32) / 255.0
    grid = small > 0.5
    return small, grid


def overlay(img, mask, color=(0, 200, 0), alpha=0.45):
    out = img.copy()
    for c, val in enumerate(color):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + val * alpha, out[..., c])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_one(sd, idx):
    path = os.path.join(ROOT, sd["filename"])
    img_u8 = np.asarray(Image.open(path).convert("RGB"))
    img = img_u8.astype(np.float32)
    driv = drivable_mask(sd["token"])
    ego = ego_mask(sd["token"])
    objs = oa_by_sd.get(sd["token"], [])

    cropped, resized, left, top, scale, (rw, rh) = short_resize_center_crop(img)
    # mask through same geometry (resize bilinear -> crop), grid via area average
    mask_rs = np.asarray(
        Image.fromarray((driv * 255).astype(np.uint8)).resize((rw, rh), Image.BILINEAR)
    ) > 127
    mask_crop = mask_rs[top : top + 512, left : left + 512]
    small_frac, grid = mask_to_grid(mask_crop)
    grid_up = np.asarray(
        Image.fromarray((grid * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST), dtype=bool
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(f"{os.path.basename(sd['filename'])}\nscale={scale:.4f} resized={rw}x{rh} crop_offset_x={left} drivable_frac_full={driv.mean():.3f} crop_frac={mask_crop.mean():.3f} grid_frac(>0.5)={grid.mean():.3f}", fontsize=10)

    # 1. original + drivable + ego + bboxes
    ax = axes[0][0]
    show = overlay(img_u8, driv, (0, 180, 0))
    if ego is not None:
        show = overlay(show, ego, (255, 80, 80), alpha=0.5)
    ax.imshow(show)
    for a in objs[:12]:
        x1, y1, x2, y2 = bbox_xyxy(a)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=1))
    ax.set_title(f"original 1600x900 (green=drivable, red=ego, yellow=obj)")
    ax.axis("off")

    # 2. resized 910x512 + mask
    ax = axes[0][1]
    ax.imshow(overlay(resized, mask_rs))
    ax.set_title(f"short-edge resize -> {rw}x{rh}")
    ax.axis("off")

    # 3. crop 512x512 + mask
    ax = axes[0][2]
    ax.imshow(overlay(cropped, mask_crop))
    ax.set_title("center crop 512x512 + mask")
    ax.axis("off")

    # 4. grid overlay (nearest upscale) + grid lines
    ax = axes[1][0]
    ax.imshow(overlay(cropped, grid_up, (255, 180, 0), alpha=0.4))
    for i in range(1, 32):
        ax.axvline(i * 16, color="gray", lw=0.3, alpha=0.6)
        ax.axhline(i * 16, color="gray", lw=0.3, alpha=0.6)
    ax.set_title("32x32 grid (area-avg > 0.5)")
    ax.axis("off")

    # 5. raw area-average heatmap
    ax = axes[1][1]
    im = ax.imshow(small_frac, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("raw area-avg fraction [32x32]")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 6. bbox -> patch selection on crop
    ax = axes[1][2]
    ax.imshow(cropped)
    for a in objs[:12]:
        x1, y1, x2, y2 = bbox_xyxy(a)
        x1, y1 = x1 * scale - left, y1 * scale - top
        x2, y2 = x2 * scale - left, y2 * scale - top
        px1, py1 = max(0, int(x1 // 16)), max(0, int(y1 // 16))
        px2, py2 = min(32, int(np.ceil(x2 / 16))), min(32, int(np.ceil(y2 / 16)))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=1))
        for gy in range(py1, py2):
            for gx in range(px1, px2):
                ax.add_patch(
                    plt.Rectangle((gx * 16, gy * 16), 16, 16, fill=False, ec="cyan", lw=0.6, alpha=0.8)
                )
    ax.set_title("bbox -> patch grid selection (yellow=box, cyan=patches)")
    ax.axis("off")

    fig.tight_layout()
    out = os.path.join(OUT, f"debug_{idx}_{os.path.basename(sd['filename']).replace('.jpg', '')}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out} | full_frac={driv.mean():.3f} crop_frac={mask_crop.mean():.3f} grid_frac={grid.mean():.3f} n_obj={len(objs)}")


# global sanity over all CAM_FRONT keyframes: grid positive fraction distribution
fracs = []
for sd in front_keys:
    driv = drivable_mask(sd["token"])
    if driv is None or driv.mean() == 0:
        fracs.append(0.0)
        continue
    _, resized, left, top, scale, (rw, rh) = short_resize_center_crop(np.zeros((900, 1600, 3), dtype=np.uint8))
    del resized
    mask_rs = np.asarray(
        Image.fromarray((driv * 255).astype(np.uint8)).resize((rw, rh), Image.BILINEAR)
    ) > 127
    mask_crop = mask_rs[top : top + 512, left : left + 512]
    _, grid = mask_to_grid(mask_crop)
    fracs.append(grid.mean())
print("mini CAM_FRONT keyframes:", len(front_keys), "| grid positive fraction min/mean/max =",
      min(fracs), np.mean(fracs), max(fracs))

for i, sd in enumerate(front_keys[:2]):
    draw_one(sd, i)
