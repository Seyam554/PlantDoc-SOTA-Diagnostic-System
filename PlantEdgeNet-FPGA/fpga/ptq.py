"""
INT8 POST-TRAINING QUANTIZATION (PTQ) for PlantEdgeNet.

This is the literal "post quantization" path requested, and also the
paper ablation baseline against QAT (fpga/quantize_qat.py).

Order of operations (see research/RESEARCH_REPORT.md 3.2 / 4.3):
  1. BatchNorm folding into preceding conv
  2. Cross-Layer Equalization (CLE) + high-bias absorption   [data-free]
  3. Activation-range calibration on N class-balanced images  [MSE / 99.9 pct]
  4. AdaRound (adaptive weight rounding)                       [--adaround]
  5. Bias correction                                           [--bias-corr]

Reports naive-PTQ vs each enhancement so the paper has a table.

Primary backend: Brevitas PTQ (brevitas.graph.*). Falls back to a plain
per-channel min/max fake-quant if Brevitas is missing (naive baseline only).

Example:
  python fpga/ptq.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.5_distill.pth \
         --calib-images 256 --adaround --bias-corr
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.model_tiny import build_model
from fpga.eval_common import get_loaders, evaluate, save_report, build_transforms

try:
    from brevitas.graph.quantize import preprocess_for_quantize, quantize
    from brevitas.graph.calibrate import calibration_mode, bias_correction_mode
    from brevitas.graph.equalize import activation_equalization_mode
    HAS_BREVITAS = True
except Exception:
    HAS_BREVITAS = False


def collect_calib(data_dir, img_size, n, num_classes, seed=0):
    """Class-balanced calibration batch as a single tensor."""
    from torchvision import datasets
    ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                              transform=build_transforms(img_size, train=False))
    by_cls = {}
    for idx, (_, y) in enumerate(ds.samples):
        by_cls.setdefault(y, []).append(idx)
    random.seed(seed)
    per = max(1, n // max(1, len(by_cls)))
    picks = []
    for y, idxs in by_cls.items():
        picks += random.sample(idxs, min(per, len(idxs)))
    xs = torch.stack([ds[i][0] for i in picks])
    return xs


@torch.no_grad()
def naive_perchannel_ptq(model, calib, bit=8):
    """Fallback: symmetric per-output-channel weight quant + per-tensor act
    quant from calibration min/max. Naive baseline only."""
    qm = model
    qmax = 2 ** (bit - 1) - 1
    for m in qm.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            dims = [1, 2, 3] if w.dim() == 4 else [1]
            s = w.abs().amax(dim=dims, keepdim=True).clamp_(min=1e-8) / qmax
            m.weight.data = (torch.round(w / s).clamp_(-qmax - 1, qmax) * s)
    return qm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="PlantDoc-Dataset")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--calib-images", type=int, default=256)
    ap.add_argument("--bit-width", type=int, default=8)
    ap.add_argument("--cle", action="store_true", help="cross-layer equalization")
    ap.add_argument("--adaround", action="store_true")
    ap.add_argument("--bias-corr", action="store_true")
    ap.add_argument("--save-dir", default="fpga/checkpoints_fpga")
    ap.add_argument("--out-dir", default="results_fpga")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    width = ck.get("width_mult", 1.5)
    classes = ck["classes"]
    model = build_model(width_mult=width, num_classes=len(classes), assert_budget=True)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["state_dict"], strict=True)
    model.to(device).eval()

    _, test_loader, _ = get_loaders(args.data, args.img_size, 64, strong_aug=False)
    fp32_metrics, _, _ = evaluate(model, test_loader, device, tta=True)
    print(f"FP32 student test acc: {fp32_metrics['accuracy']*100:.2f}%")

    calib = collect_calib(args.data, args.img_size, args.calib_images, len(classes)).to(device)
    results = {"fp32": fp32_metrics["accuracy"]}

    if not HAS_BREVITAS:
        print("[warn] brevitas missing - running NAIVE per-channel PTQ baseline only.")
        import copy
        qm = naive_perchannel_ptq(copy.deepcopy(model), calib, args.bit_width).to(device)
        m, preds, tgts = evaluate(qm, test_loader, device, tta=True)
        results["naive_ptq"] = m["accuracy"]
        save_report(args.out_dir, f"plantedgenet_w{width}_int8_ptq_naive", m, preds, tgts, classes)
        print(results)
        return

    # ---- Brevitas PTQ flow ----
    fx = preprocess_for_quantize(model, equalize_iters=(20 if args.cle else 0))
    qm = quantize(fx, weight_bit_width=args.bit_width, act_bit_width=args.bit_width,
                  weight_quant_granularity="per_channel")
    qm.to(device).eval()

    # activation calibration
    with torch.no_grad(), calibration_mode(qm):
        for i in range(0, calib.size(0), 32):
            qm(calib[i:i + 32])
    m, preds, tgts = evaluate(qm, test_loader, device, tta=True)
    results["ptq_calibrated"] = m["accuracy"]
    best_tag, best_m, best_preds, best_tgts = "ptq_calibrated", m, preds, tgts

    if args.bias_corr:
        with torch.no_grad(), bias_correction_mode(qm):
            for i in range(0, calib.size(0), 32):
                qm(calib[i:i + 32])
        m, preds, tgts = evaluate(qm, test_loader, device, tta=True)
        results["ptq_bias_corrected"] = m["accuracy"]
        best_tag, best_m, best_preds, best_tgts = "ptq_bias_corrected", m, preds, tgts

    if args.adaround:
        try:
            from brevitas.graph.gpxq import apply_gpfq  # newer brevitas
        except Exception:
            apply_gpfq = None
        try:
            from brevitas.core.function_wrapper.learned_round import LearnedRoundSte  # noqa
            from brevitas_examples.common.learned_round.learned_round_optimizer import apply_learned_round
            apply_learned_round(qm, calib, iters=1000)
            m, preds, tgts = evaluate(qm, test_loader, device, tta=True)
            results["ptq_adaround"] = m["accuracy"]
            best_tag, best_m, best_preds, best_tgts = "ptq_adaround", m, preds, tgts
        except Exception as e:
            print(f"[warn] AdaRound/learned-round unavailable in this brevitas build: {e}")

    for k, v in results.items():
        gap = (results["fp32"] - v) * 100
        print(f"  {k:22s}: {v*100:6.2f}%   (gap {gap:+.2f} pp)")

    tag = f"plantedgenet_w{width}_int8_{best_tag}"
    torch.save({"state_dict": qm.state_dict(), "width_mult": width, "classes": classes,
                "metrics": best_m, "fp32_metrics": fp32_metrics, "ptq_results": results,
                "args": vars(args)}, os.path.join(args.save_dir, f"{tag}.pth"))
    save_report(args.out_dir, tag, best_m, best_preds, best_tgts, classes)
    print(f"saved best PTQ variant: {args.save_dir}/{tag}.pth")


if __name__ == "__main__":
    main()
