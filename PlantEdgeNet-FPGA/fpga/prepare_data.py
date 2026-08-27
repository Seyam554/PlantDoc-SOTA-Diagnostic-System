r"""
Build the training corpus for the >=80% target (see research/ACCURACY_TO_80.md).

Produces  <dst>/{train,test}/<class>/*.jpg  where:
  * train = leaf-ROI-cropped PlantDoc-train
            (+ mapped PlantVillage, if --plantvillage given)
            (+ web images, if --web given)
  * test  = leaf-ROI-cropped PlantDoc-test  ONLY   (honest evaluation)

All images are cropped with fpga/leaf_roi.crop_to_leaf and saved at --size
(long side), so downstream training can resize to 64/96/112 freely.

PlantVillage class folder names are mapped into PlantDoc's label space with
PV_TO_PLANTDOC below; PV classes with no PlantDoc counterpart are skipped, and
PlantDoc classes with no PV source simply get no extra data.

Examples:
  python fpga/prepare_data.py --src PlantDoc-Dataset --dst PlantDoc-Cropped
  python fpga/prepare_data.py --src PlantDoc-Dataset --dst PlantDoc-Cropped-Plus \
         --plantvillage datasets/PlantVillage --web datasets/web_scraped --size 224
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.leaf_roi import crop_to_leaf

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# PlantVillage folder name (lowercased, non-alnum -> space, collapsed) -> PlantDoc class
PV_TO_PLANTDOC = {
    "apple apple scab": "Apple Scab Leaf",
    "apple black rot": "Apple leaf",
    "apple cedar apple rust": "Apple rust leaf",
    "apple healthy": "Apple leaf",
    "blueberry healthy": "Blueberry leaf",
    "cherry including sour powdery mildew": "Cherry leaf",
    "cherry including sour healthy": "Cherry leaf",
    "corn maize cercospora leaf spot gray leaf spot": "Corn Gray leaf spot",
    "corn maize common rust": "Corn rust leaf",
    "corn maize northern leaf blight": "Corn leaf blight",
    "corn maize healthy": "Corn rust leaf",
    "grape black rot": "grape leaf black rot",
    "grape esca black measles": "grape leaf",
    "grape leaf blight isariopsis leaf spot": "grape leaf",
    "grape healthy": "grape leaf",
    "peach bacterial spot": "Peach leaf",
    "peach healthy": "Peach leaf",
    "pepper bell bacterial spot": "Bell_pepper leaf spot",
    "pepper bell healthy": "Bell_pepper leaf",
    "potato early blight": "Potato leaf early blight",
    "potato late blight": "Potato leaf late blight",
    "potato healthy": "Potato leaf early blight",
    "raspberry healthy": "Raspberry leaf",
    "soybean healthy": "Soyabean leaf",
    "squash powdery mildew": "Squash Powdery mildew leaf",
    "strawberry leaf scorch": "Strawberry leaf",
    "strawberry healthy": "Strawberry leaf",
    "tomato bacterial spot": "Tomato leaf bacterial spot",
    "tomato early blight": "Tomato Early blight leaf",
    "tomato late blight": "Tomato leaf late blight",
    "tomato leaf mold": "Tomato mold leaf",
    "tomato septoria leaf spot": "Tomato Septoria leaf spot",
    "tomato spider mites two spotted spider mite": "Tomato two spotted spider mites leaf",
    "tomato target spot": "Tomato leaf",
    "tomato tomato yellow leaf curl virus": "Tomato leaf yellow virus",
    "tomato tomato mosaic virus": "Tomato leaf mosaic virus",
    "tomato healthy": "Tomato leaf",
}


def _norm(name):
    s = "".join(c.lower() if c.isalnum() else " " for c in name)
    return " ".join(s.split())


def _save_cropped(args_tuple):
    src, dst, size, crop = args_tuple
    if os.path.exists(dst):
        return 0
    try:
        im = Image.open(src)
        if crop:
            im = crop_to_leaf(im)
        im.thumbnail((size, size), Image.BICUBIC)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.convert("RGB").save(dst, quality=92)
        return 1
    except Exception as e:
        print(f"[skip] {src}: {e}")
        return 0


def collect_plantdoc(src_split):
    out = []
    for cls in sorted(os.listdir(src_split)):
        d = os.path.join(src_split, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                out.append((cls, os.path.join(d, fn), f"pd_{fn}"))
    return out


def collect_plantvillage(pv_root, classes):
    out, unmapped = [], set()
    for folder in sorted(os.listdir(pv_root)):
        d = os.path.join(pv_root, folder)
        if not os.path.isdir(d):
            continue
        target = PV_TO_PLANTDOC.get(_norm(folder))
        if target is None or target not in classes:
            unmapped.add(folder)
            continue
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                out.append((target, os.path.join(d, fn), f"pv_{folder[:12]}_{fn}"))
    if unmapped:
        print(f"[pv] {len(unmapped)} PV folders not mapped: {sorted(list(unmapped))[:6]}...")
    return out


def collect_web(web_root, classes):
    out = []
    for cls in sorted(os.listdir(web_root)):
        d = os.path.join(web_root, cls)
        if not os.path.isdir(d) or cls not in classes:
            if os.path.isdir(d):
                print(f"[web] folder '{cls}' not a PlantDoc class - skipped")
            continue
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                out.append((cls, os.path.join(d, fn), f"web_{fn}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="PlantDoc-Dataset")
    ap.add_argument("--dst", default="PlantDoc-Cropped")
    ap.add_argument("--plantvillage", default=None, help="PlantVillage root (ImageFolder of ~38 classes)")
    ap.add_argument("--web", default=None, help="web images root (folders named exactly like PlantDoc classes)")
    ap.add_argument("--size", type=int, default=224, help="max side after crop")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-crop", action="store_true", help="copy/resize without leaf-ROI crop")
    args = ap.parse_args()
    crop = not args.no_crop

    classes = sorted(d for d in os.listdir(os.path.join(args.src, "train"))
                     if os.path.isdir(os.path.join(args.src, "train", d)))
    print(f"{len(classes)} PlantDoc classes")

    jobs = []
    for cls, sp, name in collect_plantdoc(os.path.join(args.src, "test")):
        jobs.append((sp, os.path.join(args.dst, "test", cls, name), args.size, crop))
    n_test = len(jobs)

    train_items = collect_plantdoc(os.path.join(args.src, "train"))
    n_pd = len(train_items)
    n_pv = n_web = 0
    if args.plantvillage:
        pv = collect_plantvillage(args.plantvillage, set(classes))
        train_items += pv; n_pv = len(pv)
    if args.web:
        wb = collect_web(args.web, set(classes))
        train_items += wb; n_web = len(wb)
    for cls, sp, name in train_items:
        jobs.append((sp, os.path.join(args.dst, "train", cls, name), args.size, crop))

    print(f"train: PlantDoc={n_pd}  PlantVillage={n_pv}  web={n_web}  total={n_pd+n_pv+n_web}")
    print(f"test : PlantDoc={n_test}")
    print(f"writing -> {args.dst}/  (crop={'HSV/ExG leaf-ROI' if crop else 'off'}, size={args.size})")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_save_cropped, jobs)):
            done += r
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"done: wrote {done} new images into {args.dst}/")
    print(f"\nnext:\n  python fpga/train_fpga.py --data-dir {args.dst} --img-size 96 --width 1.5 --se "
          f"--epochs 200 --iters-per-epoch 120 --mixup --adabn-batches 50 --qat \\\n"
          f"      --teachers checkpoints_sota/dinov2_vits14_best.pth <convnext.pth> <effnet.pth>")


if __name__ == "__main__":
    main()
