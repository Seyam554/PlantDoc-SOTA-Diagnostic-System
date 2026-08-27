r"""
Standalone knowledge-distillation / pretraining for PlantEdgeNet.

Most users should run fpga/train_fpga.py (it does multi-teacher KD, SE,
adaptive-BN and QAT in one shot). This script is the lighter tool for:

  --stage pretrain : train the student on a big auxiliary set (PlantVillage as
                     an ImageFolder) with no teacher -> produces an --init
                     checkpoint for train_fpga.py.
  --stage distill  : multi-teacher KD fine-tune on a target set.

Self-contained (torch/torchvision/numpy only); shares KD math with
fpga/train_fpga.py via fpga/kd.py.

Examples:
  python fpga/distill.py --stage pretrain --data datasets/PlantVillage --width 1.5 --se --img-size 96 --epochs 40
  python fpga/distill.py --stage distill  --data PlantDoc-Cropped --width 1.5 --se --img-size 96 \
         --teachers checkpoints_sota/dinov2_vits14_best.pth checkpoints_sota/convnext_tiny_fb_in22k_ft_in1k_best.pth
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.model_tiny import build_model, count_params
from fpga.kd import load_teachers, teacher_soft, kd_kl, dist_loss

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def loaders(data, img_size, bs, workers):
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2, 0.03),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    te_tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = datasets.ImageFolder(os.path.join(data, "train"), tr_tf)
    te_dir = os.path.join(data, "test")
    te = datasets.ImageFolder(te_dir, te_tf) if os.path.isdir(te_dir) else None
    pin = torch.cuda.is_available()
    tl = DataLoader(tr, bs, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
    el = DataLoader(te, bs, shuffle=False, num_workers=workers, pin_memory=pin) if te else None
    return tl, el, tr.classes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    k, cm = None, None
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1) + model(torch.flip(x, [-1])).softmax(1)
        pred = p.argmax(1).cpu()
        if k is None:
            k = p.shape[1]; cm = np.zeros((k, k), np.int64)
        for a, b in zip(y.numpy(), pred.numpy()):
            cm[a, b] += 1
        correct += (pred == y).sum().item(); total += y.numel()
    f1s = []
    for c in range(k):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    return {"accuracy": correct / max(total, 1), "macro_f1": float(np.mean(f1s))}


def rand_mix(x, y, k, p=0.5, alpha=1.0):
    y1 = F.one_hot(y, k).float()
    if torch.rand(1).item() > p:
        return x, y1
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    if torch.rand(1).item() < 0.5:
        x = lam * x + (1 - lam) * x[idx]
    else:
        _, _, h, w = x.shape
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
        y0, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x0, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y0:y2, x0:x2] = x[idx, :, y0:y2, x0:x2]
        lam = 1 - (y2 - y0) * (x2 - x0) / (h * w)
    return x, lam * y1 + (1 - lam) * y1[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "distill"], required=True)
    ap.add_argument("--data", default="PlantDoc-Cropped")
    ap.add_argument("--width", type=float, default=1.5)
    ap.add_argument("--se", action="store_true")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teachers", nargs="+", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--kd-alpha", type=float, default=0.3)
    ap.add_argument("--kd-T", type=float, default=4.0)
    ap.add_argument("--kd-beta", type=float, default=1.0)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--save-dir", default="fpga/checkpoints_fpga")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl, el, classes = loaders(args.data, args.img_size, args.batch_size, args.workers)
    K = len(classes)
    print(f"{args.stage}: {len(tl.dataset)} train / {len(el.dataset) if el else 0} test / {K} classes")

    student = build_model(width_mult=args.width, num_classes=K, se=args.se, assert_budget=True).to(device)
    print(f"student params: {count_params(student):,}")
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location="cpu")
        print(student.load_state_dict(sd.get("state_dict", sd), strict=False))

    teachers = load_teachers(args.teachers, K, device) if args.stage == "distill" else []
    if args.stage == "distill" and not teachers:
        print("[warn] no teachers loaded - distill stage will be plain CE training")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        actx = lambda: torch.amp.autocast("cuda", enabled=device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        actx = lambda: torch.cuda.amp.autocast(enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = {k: v.detach().clone().float() for k, v in student.state_dict().items()}

    best, tag, gstep = 0.0, f"plantedgenet_w{args.width}_{args.stage}", 0
    for epoch in range(1, args.epochs + 1):
        student.train(); t0 = time.time(); run = 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            soft = None
            if args.mix:
                x, soft = rand_mix(x, y, K)
            opt.zero_grad(set_to_none=True)
            with actx():
                sl = student(x)
                hard = torch.sum(-soft * F.log_softmax(sl, 1), 1).mean() if soft is not None else ce(sl, y)
                loss = hard
                if teachers:
                    ts = teacher_soft(teachers, x, args.kd_T)
                    loss = args.kd_alpha * hard + (1 - args.kd_alpha) * kd_kl(sl, ts, args.kd_T)
                    if args.kd_beta > 0:
                        loss = loss + args.kd_beta * dist_loss(sl, ts, T=args.kd_T)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0); gstep += 1
            d = min(0.999, (gstep + 1) / (gstep + 10))
            with torch.no_grad():
                for k, v in student.state_dict().items():
                    if ema[k].is_floating_point():
                        ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                    else:
                        ema[k].copy_(v)
        sched.step()
        acc = f1 = float("nan")
        if el:
            m = evaluate(student, el, device); acc, f1 = m["accuracy"], m["macro_f1"]
        print(f"epoch {epoch:03d}/{args.epochs} ({time.time()-t0:.1f}s) loss {run/len(tl.dataset):.4f} "
              f"test_acc {acc*100:.2f}% f1 {f1*100:.2f}%")
        ckpt = {"state_dict": student.state_dict(), "ema": ema, "classes": classes,
                "width_mult": args.width, "se": args.se, "img_size": args.img_size,
                "metrics": {"accuracy": acc}, "args": vars(args), "epoch": epoch}
        torch.save(ckpt, os.path.join(args.save_dir, f"{tag}_last.pth"))
        if not el or acc > best:
            best = acc if el else best
            torch.save(ckpt, os.path.join(args.save_dir, f"{tag}.pth"))

    print(f"done. best {best*100:.2f}%  ->  {args.save_dir}/{tag}.pth"
          + ("" if args.stage != "pretrain" else f"\nuse as: train_fpga.py --init {args.save_dir}/{tag}.pth"))


if __name__ == "__main__":
    main()
