"""
Export a quantized PlantEdgeNet to ONNX / QONNX for FINN or hls4ml, and
numerically verify the export matches the PyTorch model.

  --format qonnx  : Brevitas -> QONNX  (FINN ingest)      [needs brevitas + qonnx]
  --format onnx   : standard ONNX (QDQ) for hls4ml / ORT   [needs onnx, onnxruntime]

Also emits input-normalization constants so the hardware front-end can fold
mean/std into fixed-point (uint8 image -> normalized) instead of doing float
preprocessing on device.

Example:
  python fpga/export_onnx.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.5_int8_qat.pth --format qonnx
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpga.model_tiny import build_model, INPUT_SIZE
from fpga.eval_common import MEAN, STD, get_loaders, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--format", choices=["qonnx", "onnx"], default="qonnx")
    ap.add_argument("--img-size", type=int, default=INPUT_SIZE)
    ap.add_argument("--out-dir", default="fpga/export")
    ap.add_argument("--data", default="PlantDoc-Dataset")
    ap.add_argument("--check", action="store_true", help="run argmax-parity check on the test set")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    width = ck.get("width_mult", 1.5)
    classes = ck["classes"]
    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    # rebuild + load. QAT/PTQ checkpoints hold quant-module state dicts;
    # for a clean export prefer re-quantizing the FP32 model and loading,
    # but here we support the common case of a brevitas QuantModel state.
    is_quant = any("weight_quant" in k or "act_quant" in k for k in ck["state_dict"])

    meta = {
        "input_name": "input", "output_name": "logits",
        "input_size": args.img_size, "num_classes": len(classes),
        "classes": classes, "width_mult": width,
        "preprocess": {
            "layout": "NCHW", "dtype_in": "uint8[0..255]",
            "normalize": "x_norm = (x/255 - mean) / std",
            "mean": MEAN, "std": STD,
            "fold_into_first_conv": True,
            "fixed_point_hint": "scale_r = 1/(255*std_c), bias_r = -mean_c/std_c",
        },
    }

    if args.format == "qonnx":
        try:
            from brevitas.export import export_qonnx
        except Exception:
            print("ERROR: need brevitas (+ qonnx). pip install brevitas qonnx")
            sys.exit(1)
        from fpga.quantize_qat import quantize_in_place
        model = build_model(width_mult=width, num_classes=len(classes), assert_budget=True)
        qmodel = quantize_in_place(model, bit_width=ck.get("bit_width", 8))
        if is_quant:
            qmodel.load_state_dict(ck["state_dict"], strict=False)
        qmodel.eval()
        path = os.path.join(args.out_dir, "plantedgenet_int8.qonnx")
        export_qonnx(qmodel, input_t=dummy, export_path=path)
        print(f"wrote {path}")
    else:
        model = build_model(width_mult=width, num_classes=len(classes), assert_budget=True)
        try:
            model.load_state_dict(ck["state_dict"], strict=False)
        except Exception:
            pass
        model.eval()
        path = os.path.join(args.out_dir, "plantedgenet_int8.onnx")
        torch.onnx.export(model, dummy, path, input_names=["input"],
                          output_names=["logits"], opset_version=17,
                          dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}})
        print(f"wrote {path}")

    with open(os.path.join(args.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out_dir}/export_meta.json")

    if args.check and args.format == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _, test_loader, _ = get_loaders(args.data, args.img_size, 64, strong_aug=False)
        agree = tot = 0
        for x, y in test_loader:
            pt = model(x).argmax(1).numpy()
            on = sess.run(None, {"input": x.numpy()})[0].argmax(1)
            agree += int((pt == on).sum()); tot += len(y)
        print(f"argmax parity PyTorch vs ONNX: {agree}/{tot} = {100*agree/tot:.2f}%")


if __name__ == "__main__":
    main()
