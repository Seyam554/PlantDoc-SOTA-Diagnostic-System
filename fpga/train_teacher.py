r"""
Fine-tune a timm backbone into a KD teacher for PlantDoc.

Produces a checkpoint in the format fpga/kd.load_teachers expects:
    {"state_dict", "model_name", "args": {"img_size": ...}, "classes", "metrics"}

Then pass it to train_fpga.py --teachers <this.pth>.

Needs: timm (pip install timm). GPU-cached + GPU-augmented like train_fpga.py.

Examples:
  python fpga/train_teacher.py --arch convnext_tiny.fb_in22k_ft_in1k --data-dir PlantDoc-Cropped --img-size 224 --epochs 40 --mixup
  python fpga/train_teacher.py --arch efficientnet_b3 --data-dir PlantDoc-Cropped --img-size 288 --epochs 40 --mixup
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.train_fpga import cache_split, GpuAug, evaluate, save_confusion_png, Bar, mixup_cutmix

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="timm model name, e.g. convnext_tiny.fb_in22k_ft_in1k / efficientnet_b3")
    ap.add_argument("--data-dir", default="PlantDoc-Cropped")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--iters-per-epoch", type=int, default=120)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-dir", default="checkpoints_sota")
    ap.add_argument("--out-dir", default="results_sota")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tr_x, tr_y, classes = cache_split(os.path.join(args.data_dir, "train"), args.img_size, args.workers, "train")
    te_x, te_y, _ = cache_split(os.path.join(args.data_dir, "test"), args.img_size, args.workers, "test")
    K, N = len(classes), tr_x.shape[0]

    model = timm.create_model(args.arch, pretrained=True, num_classes=K, drop_path_rate=args.drop_path)
    model = model.to(device, memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"teacher {args.arch}  params={n_params/1e6:.1f}M  img={args.img_size}  classes={K}  "
          f"train={N}  test={te_x.shape[0]}")

    aug_tr = GpuAug(args.img_size, train=True, strength="light").to(device)
    aug_ev = GpuAug(args.img_size, train=False).to(device)

    head_keys = ("head", "fc", "classifier")
    head, body = [], []
    for n, p in model.named_parameters():
        (head if any(k in n for k in head_keys) else body).append(p)
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * 10}], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        actx = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        actx = lambda: torch.cuda.amp.autocast(enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    clean = args.arch.replace("/", "_").replace(".", "_")
    best = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        bar = Bar(args.iters_per_epoch, prefix=f"teacher ep {ep:03d}/{args.epochs}")
        run = 0.0
        for _ in range(args.iters_per_epoch):
            idx = torch.randint(0, N, (args.batch_size,))
            x = aug_tr(tr_x[idx].to(device), chunks=8)
            y = tr_y[idx].to(device)
            soft = None
            if args.mixup:
                x, soft = mixup_cutmix(x, y, K)
            opt.zero_grad(set_to_none=True)
            with actx():
                out = model(x)
                loss = torch.sum(-soft * F.log_softmax(out, 1), 1).mean() if soft is not None else ce(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item(); bar.update(1, loss=f"{run/max(bar.n,1):.3f}")
        bar.close(); sched.step()
        m, cm = evaluate(model, te_x, te_y, aug_ev, device, K, batch=64, tta=True, prefix=f"  eval {ep:03d}")
        print(f"  ep {ep:03d}  loss {run/args.iters_per_epoch:.3f}  test_acc {m['accuracy']*100:.2f}%  "
              f"[{(time.time()-t0)/60:.1f} min]")
        if m["accuracy"] > best:
            best = m["accuracy"]
            torch.save({"state_dict": model.state_dict(), "model_name": args.arch,
                        "args": {"img_size": args.img_size}, "classes": classes, "metrics": m},
                       os.path.join(args.save_dir, f"{clean}_best.pth"))
            save_confusion_png(cm, classes, f"{args.arch}  {m['accuracy']*100:.2f}%",
                               os.path.join(args.out_dir, f"{clean}_confusion_matrix.png"))
            print(f"    -> saved {args.save_dir}/{clean}_best.pth ({best*100:.2f}%)")

    print(f"\nteacher done: best {best*100:.2f}%  ->  {args.save_dir}/{clean}_best.pth")
    print(f"use:  --teachers checkpoints_sota/dinov2_vits14_best.pth {args.save_dir}/{clean}_best.pth")


if __name__ == "__main__":
    main()
