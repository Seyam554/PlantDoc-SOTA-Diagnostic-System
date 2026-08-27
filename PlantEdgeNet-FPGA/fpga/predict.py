r"""
Run a trained PlantEdgeNet checkpoint on individual images.

Works with any checkpoint produced by fpga/train_fpga.py:
  * plantedgenet_w<W>_fp32.pth       (float model)
  * plantedgenet_w<W>_int8_ptq.pth   (BN-folded + fake-quant INT8 model)
  * plantedgenet_w<W>_int8_qat.pth   (Brevitas QAT INT8 model)

Examples:
  .\.venv-1\Scripts\python.exe fpga\predict.py --ckpt fpga\checkpoints_fpga\plantedgenet_w1.5_int8_ptq.pth ^
      --images "PlantDoc-Cropped\test\Tomato leaf" "PlantDoc-Cropped\test\Apple rust leaf" --topk 5

  # whole test folder -> accuracy (true label = parent folder name) + grid png
  .\.venv-1\Scripts\python.exe fpga\predict.py --ckpt fpga\checkpoints_fpga\plantedgenet_w1.5_int8_ptq.pth ^
      --images "PlantDoc-Cropped\test" --score --tta --grid results_fpga\preds_grid.png
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.model_tiny import build_model
from fpga.train_fpga import MEAN, STD, IMG_EXT, _fuse_convbnact, _wrap_quant

try:
    from PIL import Image
except Exception:
    print("Pillow required: pip install pillow")
    raise


def gather_images(patterns):
    files = []
    for p in patterns:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    if os.path.splitext(n)[1].lower() in IMG_EXT:
                        files.append(os.path.join(root, n))
        elif any(ch in p for ch in "*?["):
            files.extend(glob.glob(p, recursive=True))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[warn] no match: {p}")
    return sorted(dict.fromkeys(files))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    classes = ck["classes"]
    a = ck.get("args", {})
    width = ck.get("width_mult", a.get("width", 1.75))
    img_size = ck.get("img_size", a.get("img_size", 64))
    se = ck.get("se", a.get("se", False))
    sd = ck["state_dict"]
    if not se:
        se = any(".se.fc1." in k for k in sd)
    is_int8 = any(("w_scale" in k) or ("act_scale" in k) or (".base." in k) for k in sd)

    model = build_model(width_mult=width, num_classes=len(classes), se=se, assert_budget=False)
    if is_int8:
        model = _fuse_convbnact(model)
        _wrap_quant(model)
        missing, _ = model.load_state_dict(sd, strict=False)
    else:
        rd = next(iter(model.state_dict().values())).dtype
        model.load_state_dict({k: (v if v.is_floating_point() else v.to(rd)) for k, v in sd.items()}, strict=True)
        missing = []
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    model.eval().to(device)
    if is_int8:
        kind = "INT8-QAT" if "qat" in str(ck.get("quant", {})).lower() else "INT8-PTQ"
    else:
        kind = "FP32"
    metrics = ck.get("metrics") or ck.get("int8_metrics") or ck.get("fp32_metrics") or {}
    return model, classes, img_size, kind, metrics


def preprocess(path, size):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="files / globs / directories")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--score", action="store_true", help="true label = parent folder name; report accuracy")
    ap.add_argument("--tta", action="store_true", help="average with horizontal flip")
    ap.add_argument("--grid", default=None, help="save a PNG grid of images + predictions")
    ap.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, size, kind, train_metrics = load_model(args.ckpt, device)
    cls_idx = {c.lower(): i for i, c in enumerate(classes)}
    files = gather_images(args.images)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("no images found."); return
    print(f"model: {kind}  |  {len(classes)} classes  |  input {size}x{size}  |  "
          f"train test_acc {train_metrics.get('accuracy', float('nan'))*100:.2f}%")
    print(f"scoring {len(files)} image(s) on {device}\n")

    rows = []
    correct = total = 0
    for i in range(0, len(files), args.batch):
        chunk = files[i:i + args.batch]
        xb = torch.stack([preprocess(f, size) for f in chunk]).to(device)
        logits = model(xb)
        if args.tta:
            logits = logits + model(torch.flip(xb, dims=[-1]))
        probs = F.softmax(logits, dim=1).cpu()
        for f, p in zip(chunk, probs):
            tk = torch.topk(p, min(args.topk, len(classes)))
            pred = classes[tk.indices[0]]
            true = os.path.basename(os.path.dirname(f))
            hit = ""
            if args.score and true.lower() in cls_idx:
                total += 1
                ok = (tk.indices[0].item() == cls_idx[true.lower()])
                correct += int(ok)
                hit = "  OK" if ok else f"  x (true: {true})"
            top_str = ", ".join(f"{classes[idx]} {p[idx]*100:.1f}%" for idx in tk.indices)
            print(f"{os.path.relpath(f):70s} -> {pred:28s} [{top_str}]{hit}")
            rows.append((f, pred, p))

    if args.score and total:
        print(f"\naccuracy: {correct}/{total} = {100*correct/total:.2f}%")

    if args.grid:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = min(len(rows), 25)
            cols = 5
            r = int(np.ceil(n / cols))
            fig, axes = plt.subplots(r, cols, figsize=(cols * 2.6, r * 2.8))
            for ax in np.array(axes).ravel():
                ax.axis("off")
            for k, (f, pred, p) in enumerate(rows[:n]):
                ax = np.array(axes).ravel()[k]
                with Image.open(f) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{pred}\n{p.max().item()*100:.0f}%", fontsize=7)
            fig.tight_layout()
            os.makedirs(os.path.dirname(args.grid) or ".", exist_ok=True)
            fig.savefig(args.grid, dpi=150)
            plt.close(fig)
            print(f"grid -> {args.grid}")
        except Exception as e:
            print(f"[warn] grid skipped: {e}")


if __name__ == "__main__":
    main()
