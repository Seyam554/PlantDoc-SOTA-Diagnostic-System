r"""
Leaf-ROI cropping by vegetation segmentation - FPGA-cheap, dependency-light.

The "in the wild" difficulty in PlantDoc is mostly framing/background: Grad-CAM
studies show classifiers keying on pot/hand/soil, not the lesion. Cropping to the
leaf removes that. We use the Excess-Green index (ExG = 2G - R - B), a classic
vegetation index, plus a hue gate so yellow/brown diseased tissue is still kept.

Pipeline (single pass over the image):
  1. ExG = 2*G - R - B     (per pixel: 2 adds + 1 shift)
  2. mask = (ExG > t_exg) OR (green-ish/olive/brown hue AND mid value)
  3. optional light morphological open/close via max/min pooling (torch, 3x3)
  4. bbox = 2nd..98th percentile of mask coordinates  (robust to speckle)
  5. pad by `pad`, square-pad the short side, clamp to image
  If the mask covers < `min_frac` of the image, or the bbox spans ~the whole
  frame, fall back to a mild centre crop (`center_fallback` of each side).

FPGA note: steps 1-2 are a few LUTs/DSPs per pixel; step 3 is 3x3 line buffers;
step 4 is running min/max of the row/col index where mask==1 (4 registers + 4
comparators). One raster pass, no framebuffer needed for the mask itself.

CLI:
  python fpga/leaf_roi.py --in "PlantDoc-Dataset/test/Tomato leaf" --out /tmp/roi_preview --limit 24 --grid
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _morph(mask_bool, k=3, rounds=1):
    """3x3 close then open. Uses torch pooling if available, else returns as-is."""
    if not _HAS_TORCH:
        return mask_bool
    m = torch.from_numpy(mask_bool.astype(np.float32))[None, None]
    pad = k // 2
    for _ in range(rounds):
        m = F.max_pool2d(m, k, 1, pad)      # dilate
        m = -F.max_pool2d(-m, k, 1, pad)    # erode  -> close
    for _ in range(rounds):
        m = -F.max_pool2d(-m, k, 1, pad)    # erode
        m = F.max_pool2d(m, k, 1, pad)      # dilate -> open
    return (m[0, 0].numpy() > 0.5)


def leaf_bbox(img_rgb, t_exg=8, min_frac=0.03, morph=True):
    """img_rgb: HxWx3 uint8. Returns (x0, y0, x1, y1) or None (=use full image)."""
    a = img_rgb.astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * G - R - B
    veg = exg > t_exg

    # hue gate for chlorotic / necrotic (yellow-brown) tissue still attached to leaf
    mx = a.max(-1); mn = a.min(-1)
    val = mx
    sat = np.where(mx > 0, (mx - mn) * 255 // np.maximum(mx, 1), 0)
    warm = (R >= G) & (G >= B) & (sat > 40) & (val > 40) & (val < 245)
    mask = veg | (warm & (exg > -60))

    if morph and _HAS_TORCH:
        mask = _morph(mask, 3, 1)
    if mask.mean() < min_frac:
        return None

    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    x0, x1 = np.percentile(xs, 2), np.percentile(xs, 98)
    y0, y1 = np.percentile(ys, 2), np.percentile(ys, 98)
    return int(x0), int(y0), int(x1), int(y1)


def crop_to_leaf(pil_img, pad=0.08, square=True, min_frac=0.03, center_fallback=0.85):
    """PIL.Image -> cropped PIL.Image (RGB).

    If a vegetation ROI is found, crop to it (padded, optionally squared).
    Otherwise (or if the ROI spans the whole frame) fall back to a mild centre
    crop (`center_fallback` of each side) which still removes frame clutter.
    center_fallback >= 1.0 disables the fallback (return the full image)."""
    im = pil_img.convert("RGB")
    w, h = im.size

    def _centre():
        if center_fallback >= 1.0:
            return im
        cw, ch = int(w * center_fallback), int(h * center_fallback)
        ox, oy = (w - cw) // 2, (h - ch) // 2
        return im.crop((ox, oy, ox + cw, oy + ch))

    bb = leaf_bbox(np.asarray(im), min_frac=min_frac)
    if bb is None or (bb[2] - bb[0]) < 8 or (bb[3] - bb[1]) < 8:
        return _centre()
    x0, y0, x1, y1 = bb
    bw, bh = x1 - x0, y1 - y0
    if bw * bh > 0.92 * w * h:                     # "leaf everywhere" -> no localisation
        return _centre()
    px, py = int(bw * pad), int(bh * pad)
    x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
    if square:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return im
    return im.crop((x0, y0, x1, y1))


def _iter_images(path):
    if os.path.isdir(path):
        for r, _, ns in os.walk(path):
            for n in ns:
                if os.path.splitext(n)[1].lower() in IMG_EXT:
                    yield os.path.join(r, n)
    else:
        yield from glob.glob(path, recursive=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="file / glob / directory")
    ap.add_argument("--out", default=None, help="dir to write cropped previews")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad", type=float, default=0.08)
    ap.add_argument("--grid", action="store_true", help="save a before/after grid png")
    args = ap.parse_args()

    files = sorted(_iter_images(args.inp))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("no images"); return
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    pairs, hit, area_ratio = [], 0, []
    for f in files:
        im = Image.open(f)
        c = crop_to_leaf(im, pad=args.pad)
        o = im.convert("RGB")
        changed = c.size != o.size
        hit += int(changed)
        area_ratio.append((c.size[0] * c.size[1]) / (o.size[0] * o.size[1]))
        pairs.append((o, c, os.path.basename(f), changed))
        if args.out:
            c.save(os.path.join(args.out, os.path.basename(f)))
    print(f"{len(files)} images | cropped {hit} ({100*hit/len(files):.0f}%) | "
          f"mean kept area {100*np.mean(area_ratio):.0f}%")

    if args.grid:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = min(len(pairs), 12)
            fig, ax = plt.subplots(n, 2, figsize=(5, 2.4 * n))
            if n == 1:
                ax = ax[None]
            for i in range(n):
                orig, crp, name, changed = pairs[i]
                ax[i, 0].imshow(orig); ax[i, 0].set_title(name, fontsize=7); ax[i, 0].axis("off")
                ax[i, 1].imshow(crp); ax[i, 1].set_title("ROI" + ("" if changed else " (full)"), fontsize=7); ax[i, 1].axis("off")
            fig.tight_layout()
            out = (args.out or ".") + "/roi_grid.png"
            fig.savefig(out, dpi=130); plt.close(fig)
            print(f"grid -> {out}")
        except Exception as e:
            print(f"[warn] grid skipped: {e}")


if __name__ == "__main__":
    main()
