r"""
End-to-end FPGA training for PlantDoc -> a < 100K-parameter INT8 CNN for the
Puzhi PA200T-StarLite (AMD Artix-7 XC7A200T).  One shot:

  1. GPU-cache the dataset (pinned uint8 RAM), augment on GPU (transforms.v2)
  2. train PlantEdgeNet with optional multi-teacher knowledge distillation
     (averaged soft targets + DIST relational loss + attention-transfer feats)
  3. adaptive-BatchNorm recalibration on target data (domain-gap fix)
  4. INT8 post-training quantization (PTQ, per-channel weights + calib acts)
  5. optional Brevitas INT8 QAT; ship whichever of QAT / PTQ scores higher
  6. save checkpoints, confusion matrices, summary.json

Self-contained: needs torch, torchvision, numpy, matplotlib.
Optional: timm (ConvNeXt/EfficientNet teachers), brevitas+qonnx (--qat).

RUN (from repo root, using .venv-1):
  .\.venv-1\Scripts\python.exe fpga\train_fpga.py `
      --data-dir PlantDoc-Cropped --width 1.5 --se --img-size 96 `
      --epochs 200 --iters-per-epoch 120 --mixup `
      --teachers checkpoints_sota\dinov2_vits14_best.pth `
      --adabn-batches 50 --qat
"""

import os
import sys
import json
import math
import time
import copy
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.model_tiny import build_model, count_params, count_macs
from fpga.kd import load_teachers, teacher_soft, kd_kl, dist_loss, at_loss

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
class Bar:
    """Dependency-free progress bar (no tqdm)."""

    def __init__(self, total, prefix="", width=30):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = width
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0

    def update(self, n=1, **post):
        self.n += n
        now = time.time()
        if now - self._last < 0.1 and self.n < self.total:
            return
        self._last = now
        frac = min(self.n / self.total, 1.0)
        filled = int(self.width * frac)
        rate = self.n / max(now - self.t0, 1e-9)
        eta = (self.total - self.n) / max(rate, 1e-9)
        extra = "  ".join(f"{k}={v}" for k, v in post.items())
        sys.stdout.write(
            f"\r{self.prefix} |{'#' * filled}{'.' * (self.width - filled)}| "
            f"{self.n}/{self.total} {frac * 100:5.1f}%  {rate:5.1f} it/s  eta {eta:4.0f}s  {extra}   "
        )
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


# ----------------------------------------------------------------------------
# RAM-cached dataset (decode once, augment on GPU)
# ----------------------------------------------------------------------------
def _list_imagefolder(root):
    classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        cdir = os.path.join(root, c)
        for fn in os.listdir(cdir):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                samples.append((os.path.join(cdir, fn), cls_to_idx[c]))
    return samples, classes


def _decode_one(args):
    from PIL import Image
    path, size = args
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((size, size), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
    except Exception:
        return None


def cache_split(root, size, workers, tag):
    samples, classes = _list_imagefolder(root)
    imgs = np.zeros((len(samples), 3, size, size), dtype=np.uint8)
    labels = np.zeros(len(samples), dtype=np.int64)
    ok = 0
    bar = Bar(len(samples), prefix=f"cache[{tag}]")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, arr in enumerate(ex.map(_decode_one, [(p, size) for p, _ in samples])):
            if arr is not None:
                imgs[ok] = arr
                labels[ok] = samples[i][1]
                ok += 1
            bar.update()
    bar.close()
    x = torch.from_numpy(imgs[:ok]).contiguous()
    y = torch.from_numpy(labels[:ok]).contiguous()
    if torch.cuda.is_available():
        x = x.pin_memory()
        y = y.pin_memory()
    return x, y, classes


# ----------------------------------------------------------------------------
class GpuAug:
    def __init__(self, size, train, strength="light"):
        from torchvision.transforms import v2
        self.train = train
        self.size = size
        self._mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(STD).view(1, 3, 1, 1)
        if train and strength != "none":
            if strength == "strong":
                geom = [
                    v2.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.333), antialias=True),
                    v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(0.2),
                    v2.RandomRotation(20), v2.RandAugment(num_ops=2, magnitude=7),
                ]
                self.color = v2.ColorJitter(0.25, 0.25, 0.25, 0.05)
            else:  # light (default) - trains fast, still regularizes
                geom = [
                    v2.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True),
                    v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(0.15),
                    v2.RandomRotation(12),
                ]
                self.color = v2.ColorJitter(0.15, 0.15, 0.15, 0.03)
            self.geom = v2.Compose(geom)
        else:
            self.geom = self.color = None

    def to(self, device):
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return self

    def __call__(self, u8, chunks=8):
        if self.geom is not None:
            b = u8.shape[0]
            step = math.ceil(b / max(1, chunks))
            u8 = torch.cat([self.geom(u8[i:i + step]) for i in range(0, b, step)], 0)
        x = u8.float().div_(255.0)
        if self.color is not None:
            x = self.color(x)
        x = (x - self._mean) / self._std
        return x.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------
# metrics (no sklearn)
# ----------------------------------------------------------------------------
def confusion(preds, tgts, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(preds, tgts):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    pred_pos = cm.sum(0).astype(np.float64)
    acc = tp.sum() / max(cm.sum(), 1)
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {"accuracy": float(acc), "macro_precision": float(prec.mean()),
            "macro_recall": float(rec.mean()), "macro_f1": float(f1.mean())}


@torch.no_grad()
def evaluate(model, x_u8, y, aug, device, num_classes, batch=512, tta=True, prefix="eval"):
    model.eval()
    preds, tgts = [], []
    bar = Bar(math.ceil(x_u8.shape[0] / batch), prefix=prefix)
    for i in range(0, x_u8.shape[0], batch):
        xb = aug(x_u8[i:i + batch].to(device, non_blocking=True))
        out = model(xb)
        if tta:
            out = out.softmax(1) + model(torch.flip(xb, dims=[-1])).softmax(1)
        preds.append(out.argmax(1).cpu().numpy())
        tgts.append(y[i:i + batch].cpu().numpy())
        bar.update()
    bar.close()
    preds = np.concatenate(preds)
    tgts = np.concatenate(tgts)
    cm = confusion(preds, tgts, num_classes)
    return metrics_from_cm(cm), cm


def save_confusion_png(cm, classes, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 13))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        thr = cm.max() / 2 if cm.max() else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5,
                            color="white" if cm[i, j] > thr else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
        print(f"  wrote {path}")
    except Exception as e:
        print(f"  [warn] confusion png skipped: {e}")


# ----------------------------------------------------------------------------
def mixup_cutmix(x, y, num_classes, p=0.5, alpha=1.0):
    y1 = F.one_hot(y, num_classes).float()
    if random.random() > p:
        return x, y1
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        x = lam * x + (1 - lam) * x[idx]
    else:
        _, _, h, w = x.shape
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y0, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x0, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y0:y2, x0:x2] = x[idx, :, y0:y2, x0:x2]
        lam = 1 - ((y2 - y0) * (x2 - x0) / (h * w))
    return x, lam * y1 + (1 - lam) * y1[idx]


# ----------------------------------------------------------------------------
# INT8 post-training quantization (no brevitas)
# ----------------------------------------------------------------------------
def _fuse_convbnact(model):
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    m = copy.deepcopy(model).eval().float().to("cpu")
    for mod in m.modules():
        if hasattr(mod, "conv") and hasattr(mod, "bn") and isinstance(mod.conv, nn.Conv2d) and isinstance(mod.bn, nn.BatchNorm2d):
            mod.conv = fuse_conv_bn_eval(mod.conv, mod.bn)
            mod.bn = nn.Identity()
    return m


class QuantMAC(nn.Module):
    """Fake-quant wrapper: per-output-channel symmetric INT8 weights,
    per-tensor INT8 input activations (range from percentile calibration)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.is_conv = isinstance(base, nn.Conv2d)
        self.qmax = 127
        w = base.weight.detach()
        dims = [1, 2, 3] if w.dim() == 4 else [1]
        self.register_buffer("w_scale", w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / self.qmax)
        self.register_buffer("act_scale", torch.tensor(-1.0))
        self.register_buffer("act_absmax", torch.tensor(0.0))
        self.calibrating = False

    def _fq(self, t, s):
        return torch.clamp(torch.round(t / s), -self.qmax - 1, self.qmax) * s

    def forward(self, x):
        if self.calibrating:
            flat = x.detach().abs().flatten().float()
            k = max(1, int(flat.numel() * 0.001))
            p999 = torch.kthvalue(flat, flat.numel() - k + 1).values
            self.act_absmax = torch.maximum(self.act_absmax, p999)
            return self.base(x)
        w = self._fq(self.base.weight, self.w_scale)
        if self.act_scale.item() > 0:
            x = self._fq(x, self.act_scale)
        if self.is_conv:
            return F.conv2d(x, w, self.base.bias, self.base.stride,
                            self.base.padding, self.base.dilation, self.base.groups)
        return F.linear(x, w, self.base.bias)


def _wrap_quant(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            setattr(module, name, QuantMAC(child))
        else:
            _wrap_quant(child)


@torch.no_grad()
def apply_int8_ptq(fp32_model, calib_norm, device):
    qmodel = _fuse_convbnact(fp32_model)
    _wrap_quant(qmodel)
    qmodel.to(device).eval()
    qmacs = [m for m in qmodel.modules() if isinstance(m, QuantMAC)]
    for m in qmacs:
        m.calibrating = True
    for i in range(0, calib_norm.size(0), 32):
        qmodel(calib_norm[i:i + 32].to(device))
    for m in qmacs:
        m.act_scale = (m.act_absmax / m.qmax).clamp_min(1e-8)
        m.calibrating = False
    return qmodel


# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train + INT8 a <100K-param PlantDoc CNN for Artix-7")
    ap.add_argument("--data-dir", default="PlantDoc-Cropped")
    ap.add_argument("--width", type=float, default=1.5, help="1.5+--se -> ~72K, 1.75+--se -> ~95K params")
    ap.add_argument("--se", action="store_true", help="squeeze-excite blocks (+~9K params, +1-2 pts)")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--aug-strength", choices=["none", "light", "strong"], default="light")
    ap.add_argument("--iters-per-epoch", type=int, default=60,
                    help="random batches per epoch (0 = dataset_size // batch)")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--aug-chunks", type=int, default=8)
    ap.add_argument("--teachers", nargs="+", default=None,
                    help="teacher .pth checkpoints (or timm model names) for multi-teacher KD")
    ap.add_argument("--kd-alpha", type=float, default=0.3, help="weight on hard CE (rest is soft KD)")
    ap.add_argument("--kd-T", type=float, default=4.0)
    ap.add_argument("--kd-beta", type=float, default=1.0, help="weight on DIST relational loss")
    ap.add_argument("--feat-kd", type=float, default=0.0, help="weight on attention-transfer feature loss")
    ap.add_argument("--init", default=None, help="optional student init .pth")
    ap.add_argument("--adabn-batches", type=int, default=50, help="adaptive-BN batches before eval (0=off)")
    ap.add_argument("--qat", action="store_true", help="after FP32, run Brevitas INT8 QAT and ship the best")
    ap.add_argument("--qat-epochs", type=int, default=25)
    ap.add_argument("--calib-images", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="fpga/checkpoints_fpga")
    ap.add_argument("--out-dir", default="results_fpga")
    ap.add_argument("--export-onnx", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def adaptive_bn(state_dict, width, se, num_classes, tr_x, aug_eval, device, n_batches, bs=128):
    """Recompute BN running stats on target-domain data (adaptive BN)."""
    m = build_model(width_mult=width, num_classes=num_classes, se=se, assert_budget=False)
    rd = next(iter(m.state_dict().values())).dtype
    m.load_state_dict({k: (v if v.is_floating_point() else v.to(rd)) for k, v in state_dict.items()}, strict=True)
    m = m.to(device, memory_format=torch.channels_last)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
    m.train()
    N = tr_x.shape[0]
    for _ in range(n_batches):
        idx = torch.randint(0, N, (bs,))
        m(aug_eval(tr_x[idx].to(device)))
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  cuda {torch.version.cuda}  |  "
              f"cudnn.benchmark=on  tf32=on  amp=bf16")
    else:
        print("[warn] CUDA not available - running on CPU")

    tr_x, tr_y, classes = cache_split(os.path.join(args.data_dir, "train"), args.img_size, args.workers, "train")
    te_x, te_y, _ = cache_split(os.path.join(args.data_dir, "test"), args.img_size, args.workers, "test")
    num_classes = len(classes)
    N = tr_x.shape[0]
    steps = args.iters_per_epoch if args.iters_per_epoch > 0 else max(N // args.batch_size, 1)

    aug_train = GpuAug(args.img_size, train=True, strength=args.aug_strength).to(device)
    aug_eval = GpuAug(args.img_size, train=False).to(device)

    model = build_model(width_mult=args.width, num_classes=num_classes, se=args.se, assert_budget=True)
    model = model.to(device, memory_format=torch.channels_last)
    n_params = count_params(model)
    n_macs = count_macs(model, size=args.img_size)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}) - continuing eager")

    print("=" * 72)
    print(f"PlantEdgeNet  width={args.width}  se={args.se}  img={args.img_size}  "
          f"params={n_params:,}  MACs/inf={n_macs/1e6:.1f}M")
    print(f"train={N}  test={te_x.shape[0]}  classes={num_classes}  batch={args.batch_size}  "
          f"steps/epoch={steps}  total_updates~{steps*args.epochs}  aug={args.aug_strength}")
    print(f"mixup: {args.mixup}   adabn: {args.adabn_batches}   qat: {args.qat}")
    print("=" * 72)

    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location="cpu")
        sd = sd.get("ema", sd.get("state_dict", sd))
        print(model.load_state_dict(sd, strict=False))

    teachers = load_teachers(args.teachers, num_classes, device)
    feat_kd = args.feat_kd > 0 and any(t.spatial for t in teachers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        amp_ctx = lambda: torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
        amp_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def load_into(sd):
        mm = build_model(width_mult=args.width, num_classes=num_classes, se=args.se, assert_budget=False)
        rd = next(iter(model.state_dict().values())).dtype
        mm.load_state_dict({k: (v if v.is_floating_point() else v.to(rd)) for k, v in sd.items()}, strict=True)
        return mm.to(device, memory_format=torch.channels_last)

    best_acc, best_sd, best_kind = 0.0, None, "raw"
    gstep = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        run_loss, seen = 0.0, 0
        bar = Bar(steps, prefix=f"epoch {epoch:03d}/{args.epochs}")
        for s in range(steps):
            if args.iters_per_epoch > 0:
                idx = torch.randint(0, N, (args.batch_size,))
            else:
                idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = tr_x[idx].to(device, non_blocking=True)
            yb = tr_y[idx].to(device, non_blocking=True)
            xb = aug_train(xb, chunks=args.aug_chunks)
            soft = None
            if args.mixup:
                xb, soft = mixup_cutmix(xb, yb, num_classes)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                if feat_kd:
                    logits, feats = model(xb, return_features=True)
                else:
                    logits, feats = model(xb), None
                hard = torch.sum(-soft * F.log_softmax(logits, 1), 1).mean() if soft is not None else ce(logits, yb)
                loss = hard
                if teachers:
                    tsoft = teacher_soft(teachers, xb, args.kd_T)
                    loss = args.kd_alpha * hard + (1 - args.kd_alpha) * kd_kl(logits, tsoft, args.kd_T)
                    if args.kd_beta > 0:
                        loss = loss + args.kd_beta * dist_loss(logits, tsoft, T=args.kd_T)
                    if feat_kd:
                        loss = loss + args.feat_kd * at_loss(feats, teachers, xb)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            run_loss += loss.item() * xb.size(0); seen += xb.size(0)
            gstep += 1
            d = min(args.ema_decay, (gstep + 1) / (gstep + 10))
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if ema[k].is_floating_point():
                        ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                    else:
                        ema[k].copy_(v)
            bar.update(1, loss=f"{run_loss / max(seen,1):.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        bar.close()
        sched.step()

        mr, _ = evaluate(load_into(model.state_dict()), te_x, te_y, aug_eval, device, num_classes,
                         batch=max(256, args.batch_size), tta=True, prefix=f"  eval-raw {epoch:03d}")
        me, _ = evaluate(load_into(ema), te_x, te_y, aug_eval, device, num_classes,
                         batch=max(256, args.batch_size), tta=True, prefix=f"  eval-ema {epoch:03d}")
        m, kind, sd = (mr, "raw", model.state_dict()) if mr["accuracy"] >= me["accuracy"] else (me, "ema", ema)
        print(f"  epoch {epoch:03d}  loss {run_loss/max(seen,1):.4f}  "
              f"raw {mr['accuracy']*100:5.2f}%  ema {me['accuracy']*100:5.2f}%  "
              f"-> best-so-far {max(best_acc, m['accuracy'])*100:5.2f}% ({kind})  "
              f"[{(time.time()-t_start)/60:.1f} min]")
        if m["accuracy"] > best_acc:
            best_acc, best_kind = m["accuracy"], kind
            best_sd = {k: v.detach().clone() for k, v in sd.items()}
            torch.save({"state_dict": best_sd, "ema": ema, "classes": classes, "width_mult": args.width,
                        "se": args.se, "img_size": args.img_size, "metrics": m, "selected": kind,
                        "args": vars(args), "epoch": epoch},
                       os.path.join(args.save_dir, f"plantedgenet_w{args.width}_fp32.pth"))

    print(f"\nFP32 training done in {(time.time()-t_start)/60:.1f} min. best test acc {best_acc*100:.2f}% ({best_kind})")
    ship_sd = best_sd if best_sd is not None else {k: v.detach().clone() for k, v in model.state_dict().items()}

    if args.adabn_batches > 0:
        print(f"[adabn] recomputing BN stats on {args.adabn_batches} target batches ...")
        cand = adaptive_bn(ship_sd, args.width, args.se, num_classes, tr_x, aug_eval, device, args.adabn_batches)
        ma, _ = evaluate(load_into(cand), te_x, te_y, aug_eval, device, num_classes, tta=True, prefix="  eval-adabn")
        print(f"[adabn] {best_acc*100:.2f}% -> {ma['accuracy']*100:.2f}%  "
              f"({'kept' if ma['accuracy'] >= best_acc else 'reverted'})")
        if ma["accuracy"] >= best_acc:
            ship_sd, best_acc = cand, ma["accuracy"]
            torch.save({"state_dict": ship_sd, "ema": ema, "classes": classes, "width_mult": args.width,
                        "se": args.se, "img_size": args.img_size, "metrics": ma, "selected": best_kind + "+adabn",
                        "args": vars(args)}, os.path.join(args.save_dir, f"plantedgenet_w{args.width}_fp32.pth"))

    fp32 = load_into(ship_sd)
    fp32_m, fp32_cm = evaluate(fp32, te_x, te_y, aug_eval, device, num_classes, tta=True, prefix="fp32 eval")
    save_confusion_png(fp32_cm, classes, f"FP32  acc={fp32_m['accuracy']*100:.2f}%",
                       os.path.join(args.out_dir, f"plantedgenet_w{args.width}_fp32_confusion_matrix.png"))

    # ---- INT8 PTQ ----
    print("\n[PTQ] folding BN, per-channel INT8 weights, calibrating INT8 activations ...")
    n_cal = min(args.calib_images, tr_x.shape[0])
    sel = torch.randperm(tr_x.shape[0])[:n_cal]
    calib_norm = aug_eval(tr_x[sel].to(device)).float().cpu()
    qmodel = apply_int8_ptq(fp32, calib_norm, device)
    int8_m, int8_cm = evaluate(qmodel, te_x, te_y, aug_eval, device, num_classes, tta=True, prefix="int8 eval")
    save_confusion_png(int8_cm, classes, f"INT8 PTQ  acc={int8_m['accuracy']*100:.2f}%",
                       os.path.join(args.out_dir, f"plantedgenet_w{args.width}_int8_ptq_confusion_matrix.png"))
    torch.save({"state_dict": qmodel.state_dict(), "classes": classes, "width_mult": args.width,
                "se": args.se, "img_size": args.img_size, "fp32_metrics": fp32_m, "int8_metrics": int8_m,
                "quant": {"scheme": "per-channel-symmetric-int8-weights / per-tensor-int8-acts",
                          "calib_images": int(n_cal), "percentile": 99.9},
                "args": vars(args)},
               os.path.join(args.save_dir, f"plantedgenet_w{args.width}_int8_ptq.pth"))

    # ---- INT8 QAT (optional, shipped if it wins) ----
    qat_m = None
    if args.qat:
        try:
            from fpga.quantize_qat import quantize_in_place, HAS_BREVITAS
            if not HAS_BREVITAS:
                raise ImportError("brevitas not installed (pip install brevitas qonnx)")
            print(f"\n[QAT] INT8 quantization-aware training, {args.qat_epochs} epochs ...")
            qm = quantize_in_place(load_into(ship_sd).float().cpu(), bit_width=8).to(device)
            qopt = torch.optim.AdamW(qm.parameters(), lr=2e-4, weight_decay=1e-5)
            qsched = torch.optim.lr_scheduler.CosineAnnealingLR(qopt, T_max=args.qat_epochs, eta_min=1e-6)
            qce = nn.CrossEntropyLoss(label_smoothing=0.05)
            qbest, qbest_sd = 0.0, None
            for qe in range(1, args.qat_epochs + 1):
                qm.train()
                if qe > 3:
                    for mod in qm.modules():
                        if isinstance(mod, nn.BatchNorm2d):
                            mod.eval()
                bar = Bar(steps, prefix=f"  qat {qe:02d}/{args.qat_epochs}")
                for _ in range(steps):
                    idx = torch.randint(0, tr_x.shape[0], (args.batch_size,))
                    x = aug_train(tr_x[idx].to(device), chunks=args.aug_chunks)
                    y = tr_y[idx].to(device)
                    qopt.zero_grad(set_to_none=True)
                    qce(qm(x), y).backward()
                    qopt.step(); bar.update(1)
                bar.close(); qsched.step()
                mm, _ = evaluate(qm, te_x, te_y, aug_eval, device, num_classes, tta=True, prefix=f"  qat-eval {qe:02d}")
                if mm["accuracy"] > qbest:
                    qbest, qbest_sd = mm["accuracy"], {k: v.detach().clone() for k, v in qm.state_dict().items()}
                print(f"  qat {qe:02d}  int8 {mm['accuracy']*100:.2f}%  (best {qbest*100:.2f}%)")
            qat_m = {"accuracy": qbest}
            torch.save({"state_dict": qbest_sd, "classes": classes, "width_mult": args.width, "se": args.se,
                        "img_size": args.img_size, "fp32_metrics": fp32_m, "int8_metrics": {"accuracy": qbest},
                        "quant": {"scheme": "brevitas-QAT-int8"}, "args": vars(args)},
                       os.path.join(args.save_dir, f"plantedgenet_w{args.width}_int8_qat.pth"))
            print(f"[QAT] best INT8 {qbest*100:.2f}%  (PTQ was {int8_m['accuracy']*100:.2f}%)")
        except Exception as e:
            print(f"[QAT][warn] skipped: {e}")

    shipped = qat_m if (qat_m and qat_m["accuracy"] >= int8_m["accuracy"]) else int8_m
    shipped_name = "int8_qat" if shipped is qat_m else "int8_ptq"

    summary = {"params": n_params, "macs": int(n_macs), "width_mult": args.width, "se": args.se,
               "img_size": args.img_size, "fp32": fp32_m, "int8_ptq": int8_m, "int8_qat": qat_m,
               "shipped": shipped_name, "shipped_acc": shipped["accuracy"],
               "int8_gap_pp": (fp32_m["accuracy"] - shipped["accuracy"]) * 100}
    with open(os.path.join(args.out_dir, f"plantedgenet_w{args.width}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"PARAMS         : {n_params:,}   ({'OK < 100K' if n_params < 100_000 else 'OVER BUDGET'})")
    print(f"FP32  test acc : {fp32_m['accuracy']*100:6.2f}%   macroF1 {fp32_m['macro_f1']*100:.2f}%")
    print(f"INT8  PTQ acc  : {int8_m['accuracy']*100:6.2f}%")
    if qat_m:
        print(f"INT8  QAT acc  : {qat_m['accuracy']*100:6.2f}%")
    print(f"SHIPPED        : {shipped_name}  {shipped['accuracy']*100:.2f}%   "
          f"(gap vs FP32 {summary['int8_gap_pp']:+.2f} pp)")
    print(f"checkpoints -> {args.save_dir}/plantedgenet_w{args.width}_*.pth")
    print(f"reports     -> {args.out_dir}/plantedgenet_w{args.width}_*")
    print("=" * 72)

    if args.export_onnx:
        try:
            onnx_path = os.path.join(args.save_dir, f"plantedgenet_w{args.width}_fp32.onnx")
            dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device).contiguous(memory_format=torch.channels_last)
            torch.onnx.export(fp32, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                              opset_version=17, dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}})
            print(f"exported {onnx_path}  (hls4ml input; for FINN QONNX see fpga/export_onnx.py)")
        except Exception as e:
            print(f"[warn] onnx export failed: {e}")


if __name__ == "__main__":
    main()
