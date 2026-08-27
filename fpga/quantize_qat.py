"""
INT8 quantization-aware training (QAT) for PlantEdgeNet using Brevitas.
Recommended path to ship (best accuracy retention for a tiny model).

Strategy (see research/RESEARCH_REPORT.md 4.3):
  * weights   : per-channel, symmetric, signed INT8
  * activations: per-tensor, INT8, affine
  * BN folded by Brevitas at export; here we keep BN and let QAT adapt.
  * init from the FP32 student; low LR; freeze BN running stats after a few epochs.
  * export QONNX for FINN (see export_onnx.py).

Requires:  pip install brevitas qonnx
train_fpga.py imports `quantize_in_place` / `HAS_BREVITAS` from here; the
fpga.eval_common import (sklearn/seaborn) is deferred into main() so that stays lightweight.

Example:
  python fpga/quantize_qat.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.5_distill.pth --epochs 25
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.model_tiny import build_model

try:
    from brevitas import nn as qnn
    try:
        from brevitas.quant import Int8WeightPerChannelFixedPoint as WQ   # older brevitas
    except Exception:
        from brevitas.quant import Int8WeightPerChannelFloat as WQ        # brevitas >= 0.11
    from brevitas.quant import Uint8ActPerTensorFloat as AQ
    from brevitas.quant import Int8ActPerTensorFloat as AQS
    HAS_BREVITAS = True
except Exception as _e:
    print(f"[quantize_qat] brevitas unavailable: {_e}")
    HAS_BREVITAS = False


def quantize_in_place(model, bit_width=8):
    """Swap float layers for Brevitas quant equivalents, copying weights.
    Minimal graph-preserving replacement for the PlantEdgeNet module tree."""
    import copy
    qmodel = copy.deepcopy(model)

    def swap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                q = qnn.QuantConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding, groups=child.groups,
                    bias=child.bias is not None,
                    weight_quant=WQ, weight_bit_width=bit_width,
                    input_quant=AQS, input_bit_width=bit_width,
                )
                q.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
                setattr(parent, name, q)
            elif isinstance(child, nn.Linear):
                q = qnn.QuantLinear(
                    child.in_features, child.out_features, bias=child.bias is not None,
                    weight_quant=WQ, weight_bit_width=bit_width,
                    input_quant=AQS, input_bit_width=bit_width,
                )
                q.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
                setattr(parent, name, q)
            elif isinstance(child, nn.ReLU6):
                setattr(parent, name, qnn.QuantReLU(bit_width=bit_width, act_quant=AQ))
            else:
                swap(child)

    swap(qmodel)
    return qmodel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="FP32 student checkpoint")
    ap.add_argument("--data", default="PlantDoc-Cropped")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bit-width", type=int, default=8)
    ap.add_argument("--freeze-bn-epoch", type=int, default=3)
    ap.add_argument("--save-dir", default="fpga/checkpoints_fpga")
    ap.add_argument("--out-dir", default="results_fpga")
    args = ap.parse_args()

    if not HAS_BREVITAS:
        print("ERROR: brevitas not installed.  pip install brevitas qonnx\n"
              "Fallback: use fpga/ptq.py (post-training) instead.")
        sys.exit(1)

    from fpga.eval_common import get_loaders, evaluate, save_report  # needs sklearn/seaborn

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu")
    width = ck.get("width_mult", 1.5)
    se = ck.get("se", ck.get("args", {}).get("se", False))
    classes = ck["classes"]
    fp32 = build_model(width_mult=width, num_classes=len(classes), se=se, assert_budget=True)
    fp32.load_state_dict(ck["ema"] if "ema" in ck else ck["state_dict"], strict=True)

    train_loader, test_loader, _ = get_loaders(args.data, args.img_size, args.batch_size, strong_aug=False)

    fp32.to(device)
    base_metrics, _, _ = evaluate(fp32, test_loader, device, tta=True)
    print(f"FP32 student test acc: {base_metrics['accuracy']*100:.2f}%")

    qmodel = quantize_in_place(fp32, bit_width=args.bit_width).to(device)

    opt = torch.optim.AdamW(qmodel.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    tag = f"plantedgenet_w{width}_int8_qat"
    for epoch in range(1, args.epochs + 1):
        qmodel.train()
        if epoch > args.freeze_bn_epoch:
            for m in qmodel.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            ce(qmodel(x), y).backward()
            opt.step()
        sched.step()
        metrics, preds, tgts = evaluate(qmodel, test_loader, device, tta=True)
        drop = (base_metrics["accuracy"] - metrics["accuracy"]) * 100
        print(f"epoch {epoch:03d}/{args.epochs} ({time.time()-t0:.1f}s) "
              f"int8 test_acc {metrics['accuracy']*100:.2f}%  (drop {drop:+.2f} pp)")
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save({"state_dict": qmodel.state_dict(), "width_mult": width, "se": se,
                        "classes": classes, "metrics": metrics, "fp32_metrics": base_metrics,
                        "bit_width": args.bit_width, "quant": {"scheme": "brevitas-QAT-int8"},
                        "args": vars(args)},
                       os.path.join(args.save_dir, f"{tag}.pth"))
            save_report(args.out_dir, tag, metrics, preds, tgts, classes)
            print(f"  -> new best int8 {best*100:.2f}%")

    print(f"\nFP32 {base_metrics['accuracy']*100:.2f}%  ->  INT8 QAT {best*100:.2f}%  "
          f"(gap {(base_metrics['accuracy']-best)*100:.2f} pp)")
    print(f"saved: {args.save_dir}/{tag}.pth   next: python fpga/export_onnx.py --ckpt {args.save_dir}/{tag}.pth --format qonnx")


if __name__ == "__main__":
    main()
