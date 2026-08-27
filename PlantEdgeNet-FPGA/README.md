# PlantEdgeNet-FPGA

A **self-contained sub-project**: a **< 100,000-parameter INT8 CNN** for PlantDoc plant-disease
classification, distilled from the DINOv2 / ConvNeXt / EfficientNet models in the parent repo and
targeted at the **Puzhi PA200T-StarLite** board (**AMD Artix-7 XC7A200T**, part `xc7a200tfbg484-2`)
via FINN / hls4ml / Vivado.

Everything needed lives in this folder: model, data prep, knowledge-distillation, training,
quantization (PTQ + Brevitas QAT), ONNX/QONNX export, inference, FPGA build configs, and the full
research write-up.

```
PlantEdgeNet-FPGA/
├── README.md                 ← you are here (operational runbook)
├── references.md             ← the papers this design is built on (with links)
├── models.py                 ← vendored: baseline backbone defs (teacher loading)
├── models_sota.py            ← vendored: DINOv2 / ConvNeXt / Swin defs (teacher loading)
├── research/
│   ├── RESEARCH_REPORT.md     ← hardware analysis, literature, architecture + quant recipe, risks
│   ├── ACCURACY_TO_80.md      ← deep research: every lever to reach ≥80%, ranked, with projections
│   └── IMPLEMENTATION_PLAN.md ← 10 milestones (M0–M9) + acceptance criteria
└── fpga/
    ├── model_tiny.py          ← PlantEdgeNet: depthwise-separable CNN + optional Squeeze-Excite
    ├── leaf_roi.py            ← HSV / Excess-Green leaf-ROI crop (FPGA-cheap) + preview grid
    ├── prepare_data.py        ← build PlantDoc-Cropped[-Plus]: crop + map PlantVillage + fold web data
    ├── kd.py                  ← multi-teacher soft targets + DIST relational loss + attention transfer
    ├── train_teacher.py       ← fine-tune a timm backbone (ConvNeXt/EfficientNet) into a KD teacher
    ├── train_fpga.py          ← ONE-SHOT: GPU-cached train + KD + adaptive-BN + INT8 PTQ + INT8 QAT
    ├── distill.py             ← standalone PlantVillage-pretrain / multi-teacher-distill
    ├── quantize_qat.py        ← Brevitas INT8 QAT (per-channel symmetric weights) + QONNX path
    ├── ptq.py                 ← INT8 PTQ ablation (CLE + bias-correction + AdaRound)
    ├── export_onnx.py         ← QONNX (FINN) / ONNX (hls4ml) export + PyTorch↔ONNX parity check
    ├── predict.py             ← run any checkpoint (FP32 / INT8-PTQ / INT8-QAT) on images
    ├── finn_build.py          ← FINN dataflow build config for xc7a200tfbg484-2 (run in FINN Docker)
    ├── hls4ml_build.py        ← hls4ml build config for xc7a200tfbg484-2 (needs Vitis HLS)
    ├── eval_common.py         ← shared loaders/metrics for the standalone ptq.py / quantize_qat.py mains
    └── requirements-fpga.txt
```

---

## Why a new model (not "quantize DINOv2")

Post-training quantization changes **bits per weight** (FP32 → INT8 = 4× smaller memory), **not the
number of weights**. DINOv2 ViT-S/14 has ~22 M params — 220× over a 100 K budget — and a ViT
(LayerNorm / GELU / softmax attention) maps poorly to a small FP-less FPGA. So the SOTA models become
**teachers**, and the deployed model is a new tiny CNN, **`PlantEdgeNet`** (`fpga/model_tiny.py`):

| config | params | MACs/inf @96px |
|---|---|---|
| `--width 1.0` | ~34 K | ~5.8 M |
| `--width 1.5 --se` (**default**) | **~72 K** | ~12 M |
| `--width 1.75 --se` | ~95 K | ~16 M |

## Target hardware — XC7A200T-2FBG484I

| Resource | Value | Consequence |
|---|---|---|
| Logic / DSP / BRAM | 215 K cells · 740 DSP48E1 · ~13 Mbit (~1.63 MB) | a ~72 K-param INT8 model (~72 KB weights) fits **entirely on-chip**; no DDR for weights |
| CPU | **none** (plain Artix-7, not Zynq) | **no Vitis-AI DPU**; drive with MicroBlaze + AXI-DMA, or JTAG-to-AXI for bench tests |
| Clock target | 100–150 MHz | INT8 DSP packing ≈ 1 480 MAC/clk → ~148 GMAC/s vs ~12–16 MMAC/inference |

## The ≥80% target — honest assessment

EfficientNet-B3 (12 M params) scores 73 % on the PlantDoc test split; lightweight SOTA ~82 % needs
~2 M params + augmentation. ≥80 % at < 100 K params is very ambitious. Levers, all within budget:

| Lever | Expected gain | Where |
|---|---|---|
| **Leaf-ROI crop** (HSV / Excess-Green, FPGA-cheap) | **+10–20 pts** | `fpga/leaf_roi.py`, `fpga/prepare_data.py` |
| **External training data** (PlantVillage mapped + web); test stays PlantDoc | +3–8 pts (only *with* crop) | `fpga/prepare_data.py --plantvillage` |
| **Multi-teacher KD** (DINOv2 + ConvNeXt + EfficientNet; soft + DIST + attention transfer) | +2–5 pts | `fpga/kd.py`, `fpga/train_teacher.py` |
| **96 px input + Squeeze-Excite + EMA + adaptive-BN** | +4–9 pts | `fpga/train_fpga.py` |
| **QAT instead of PTQ** | +1–3 pts | `fpga/quantize_qat.py` (`--qat`) |

**Projected:** FP32 ≈ 76–82 %, INT8 (QAT) ≈ 75–81 %. Fallback: relax to ~300 K params (still
< 512 KB INT8, still fits the XC7A200T) — see `research/ACCURACY_TO_80.md` §F.

---

## Runbook

Uses the parent repo's `.venv-1` (Python 3.14, torch 2.13+cu126). **Run every command from the parent
repo root** (`ICCIT_Paper/` or your clone) — the scripts self-locate this folder, and the datasets
(`PlantDoc-Dataset/`, `PlantDoc-Cropped/`, `checkpoints_sota/`) live at the repo root. Paths below
assume that. (To run from inside `PlantEdgeNet-FPGA/`, prefix data paths with `../`.)

### 0 — deps
```powershell
.\.venv-1\Scripts\python.exe -m pip install timm brevitas qonnx onnx onnxruntime
```

### 1 — build the cropped corpus
```powershell
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\prepare_data.py --src PlantDoc-Dataset --dst PlantDoc-Cropped --size 160
# + external TRAIN data (optional):  ... --dst PlantDoc-Cropped-Plus --plantvillage "<PlantVillage root>" --web <web_root>
# baseline without crop:             ... --dst PlantDoc-Raw --no-crop
```
Preview: `python PlantEdgeNet-FPGA\fpga\leaf_roi.py --in "PlantDoc-Dataset\test\Tomato leaf" --grid --out roi_preview`

### 2 — (optional) build KD teachers
A **bare timm name** as `--teachers` is wrong (random head → noise). Fine-tune real teachers first:
```powershell
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\train_teacher.py --arch convnext_tiny.fb_in22k_ft_in1k --data-dir PlantDoc-Cropped --img-size 224 --epochs 40 --mixup
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\train_teacher.py --arch efficientnet_b3               --data-dir PlantDoc-Cropped --img-size 288 --epochs 40 --mixup
```

### 3 — train + INT8 (one shot)
```powershell
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\train_fpga.py `
  --data-dir PlantDoc-Cropped --width 1.5 --se --img-size 96 `
  --epochs 200 --batch-size 128 --iters-per-epoch 120 --mixup `
  --teachers checkpoints_sota\dinov2_vits14_best.pth checkpoints_sota\convnext_tiny_fb_in22k_ft_in1k_best.pth checkpoints_sota\efficientnet_b3_best.pth `
  --kd-beta 1.0 --feat-kd 0.5 --adabn-batches 50 --qat --calib-images 256

# minimal (DINOv2 teacher only):
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\train_fpga.py --data-dir PlantDoc-Cropped --width 1.5 --se --img-size 96 `
  --epochs 200 --iters-per-epoch 120 --mixup --teachers checkpoints_sota\dinov2_vits14_best.pth --adabn-batches 50 --qat
```
Pipeline: RAM-cache dataset → GPU aug (`transforms.v2`, channels_last + cudnn.benchmark + TF32 + bf16)
→ train with multi-teacher KD (soft + `--kd-beta` DIST + `--feat-kd` attention transfer) + EMA
→ adaptive-BatchNorm recalibration (kept only if it helps) → INT8 PTQ → INT8 QAT → **ship the higher**.

Outputs: `PlantEdgeNet-FPGA/fpga/checkpoints_fpga/plantedgenet_w1.5_{fp32,int8_ptq,int8_qat}.pth`,
`results_fpga/plantedgenet_w1.5_summary.json` (`params`, `macs`, `fp32`, `int8_ptq`, `int8_qat`,
`shipped`, `shipped_acc`, `int8_gap_pp`), confusion-matrix PNGs.

### 4 — test images
```powershell
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\predict.py `
  --ckpt PlantEdgeNet-FPGA\fpga\checkpoints_fpga\plantedgenet_w1.5_int8_qat.pth `
  --images "PlantDoc-Cropped\test" --score --tta --grid results_fpga\preds_grid.png
```

### 5 — export for the FPGA toolchain
```powershell
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\export_onnx.py --ckpt <int8_qat.pth> --format qonnx        # FINN
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\export_onnx.py --ckpt <int8_ptq.pth> --format onnx --check  # hls4ml + parity
```

### 6 — FPGA build (Vivado)
* **FINN** (in the FINN Docker image): `python fpga/finn_build.py` → stitched IP + OOC synth; open the
  Vivado project (`xc7a200tfbg484-2`), add MicroBlaze + AXI-DMA (or JTAG-to-AXI), implement, close
  timing ≥ 100 MHz, generate the bitstream.
* **hls4ml** (Vitis HLS on PATH): `python fpga/hls4ml_build.py` — cross-check / fallback.
* **On-board eval:** push the 236 PlantDoc test images through the bitstream; the confusion matrix must
  match the software INT8 model within rounding.

### Acceptance criteria
| # | Criterion |
|---|---|
| A1 | params **< 100,000** (asserted in `model_tiny.py`) |
| A2 | FP32 student PlantDoc test top-1 **≥ 62 %** (stretch ≥ 80 %) |
| A3 | shipped INT8 within **≤ 1.5 %** of FP32 |
| A4 | ONNX / QONNX argmax agreement **= 100 %** on the test set |
| A5 | fits `xc7a200tfbg484-2`: DSP ≤ 700, BRAM ≤ 360, LUT ≤ 130 k, timing met ≥ 100 MHz |
| A6 | on-board predictions match software INT8 within rounding |

---

## Dependencies

`torch, torchvision, numpy, matplotlib` cover `model_tiny / leaf_roi / prepare_data / kd /
train_teacher / train_fpga / distill / predict`. Optional: `timm` (ConvNeXt/EfficientNet teachers),
`brevitas` + `qonnx` (`--qat`, FINN export), `onnx` + `onnxruntime` (hls4ml path, parity check),
`scikit-learn` + `seaborn` (only the standalone `ptq.py` / `quantize_qat.py` mains). Missing optional
deps degrade gracefully (fewer teachers / PTQ ships instead of QAT).
