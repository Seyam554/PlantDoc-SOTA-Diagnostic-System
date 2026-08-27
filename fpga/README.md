# `fpga/` — sub-100K-param INT8 PlantDoc classifier for Artix-7 XC7A200T

Full rationale: [`../research/RESEARCH_REPORT.md`](../research/RESEARCH_REPORT.md)
Step-by-step milestones: [`../research/IMPLEMENTATION_PLAN.md`](../research/IMPLEMENTATION_PLAN.md)

## TL;DR

The project's "SOTA" model (`DINOv2 ViT-S/14`, ~22M params, 66.9% PlantDoc test) **cannot**
be quantized down to <100K parameters — quantization shrinks *bits/weight*, not *weight count*,
and a ViT is hostile to a small FP-less FPGA. Instead:

```
DINOv2 / ConvNeXt teacher ──KD──▶ PlantEdgeNet student (~11K–95K params, depthwise-separable CNN)
                                        │
                                        ├─ INT8 QAT  (quantize_qat.py, Brevitas)   ← ship this
                                        └─ INT8 PTQ  (ptq.py, AdaRound+bias-corr)   ← "post quantization" + ablation
                                        │
                                        ▼
                              QONNX / ONNX  (export_onnx.py)
                                        │
                          FINN  ──or──  hls4ml   →   Vivado (xc7a200tfbg484-2)  →  bitstream
```

## Environment

The checked-in `.venv` is broken (built on another machine — `pyvenv.cfg` points at a
missing `C:\Users\tazwa\...` Python). Recreate:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r fpga\requirements-fpga.txt
```

## Run order

### One-shot: `train_fpga.py` (train + INT8 PTQ in a single run, GPU-first, progress bars)

```powershell
.\.venv-1\Scripts\python.exe fpga\train_fpga.py `
  --data-dir PlantDoc-Dataset --width 1.75 --img-size 64 `
  --epochs 120 --batch-size 256 --lr 3e-3 --mixup --aug-chunks 8 `
  --teacher checkpoints_sota\dinov2_vits14_best.pth --calib-images 256
```

- caches the whole dataset into a pinned uint8 RAM tensor once (~8 s), augments on the GPU
  (`torchvision.transforms.v2`), uses channels_last + cudnn.benchmark + TF32 + bf16 AMP.
- `--width 1.75` = **88,812 params**. The model is tiny, so a laptop GPU won't sit at 100%;
  push `--batch-size 512 --aug-chunks 16` for higher utilization.
- outputs: `fpga/checkpoints_fpga/plantedgenet_w1.75_{fp32,int8_ptq}.pth`,
  `results_fpga/plantedgenet_w1.75_summary.json` + FP32/INT8 confusion PNGs.
- first epoch is slow (cudnn autotune + v2 warmup); steady state is ~30 it/s on an RTX 3050.

### Multi-stage flow (finer control)

| Step | Command | Output |
|---|---|---|
| 0. sanity | `python fpga/model_tiny.py --width 1.75` | prints params (~89K) / MACs, asserts <100K |
| 1. teacher *(opt.)* | train ConvNeXt via existing `train_sota.py`, or reuse `checkpoints_sota/dinov2_vits14_best.pth` | teacher `.pth` |
| 2a. pre-train *(opt., big win)* | `python fpga/distill.py --stage pretrain --data datasets/PlantVillage --width 1.5 --epochs 60` | `checkpoints_fpga/plantedgenet_w1.5_pretrain.pth` |
| 2b. distill | `python fpga/distill.py --stage distill --data PlantDoc-Dataset --width 1.5 --teacher checkpoints_sota/dinov2_vits14_best.pth --init fpga/checkpoints_fpga/plantedgenet_w1.5_pretrain.pth --mix` | `..._distill.pth` + metrics/CM in `results_fpga/` |
| 3. INT8 QAT | `python fpga/quantize_qat.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.5_distill.pth --epochs 25` | `..._int8_qat.pth` (+ FP32-vs-INT8 gap printed) |
| 4. INT8 PTQ ablation | `python fpga/ptq.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.5_distill.pth --calib-images 256 --cle --adaround --bias-corr` | table: naive / calibrated / bias-corr / adaround |
| 5. export | `python fpga/export_onnx.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.5_int8_qat.pth --format qonnx` | `fpga/export/plantedgenet_int8.qonnx` + `export_meta.json` |
| 5b. parity check | `python fpga/export_onnx.py --ckpt ..._distill.pth --format onnx --check` | argmax agreement PyTorch vs ONNX Runtime |
| 6. FINN | *(inside FINN Docker)* `python fpga/finn_build.py` | `fpga/finn_out/` stitched IP + OOC synth report |
| 7. hls4ml *(parallel/fallback)* | *(with Vitis HLS on PATH)* `python fpga/hls4ml_build.py` | `fpga/hls4ml_prj/` + exported IP |
| 8. Vivado | open the generated project, add MicroBlaze + AXI-DMA (or JTAG-to-AXI), implement, close timing, gen bitstream | `.bit` + utilization/timing/power |
| 9. on-board eval | push the 236 PlantDoc test images through the bitstream, rebuild confusion matrix, compare to SW INT8 | `results_fpga/hw_*.json` / `hw_confusion_matrix.png` |

## Targets (acceptance)

- params **< 100,000** (asserted in `model_tiny.py`)
- FP32 student PlantDoc test top-1 **≥ 62%** (stretch ≥ 70%)
- shipped INT8 within **≤ 1.5%** of FP32 student
- ONNX/QONNX argmax agreement **= 100%** on the test set
- fits `xc7a200tfbg484-2`: **DSP ≤ 700, BRAM ≤ 360, LUT ≤ 130k**, timing met at ≥ 100 MHz
- on-board predictions match SW INT8 within rounding

## Device notes (PA200T-StarLite)

- `xc7a200tfbg484-2`: 215,360 logic cells, 740 DSP48E1, ~13 Mbit BRAM, 1 GB DDR3, GbE, HDMI, camera header.
- **No hardened CPU** (plain Artix-7, not Zynq) ⇒ **no Vitis-AI DPU**. Drive the accelerator with a
  MicroBlaze soft-core + AXI-DMA, or JTAG-to-AXI for bench validation.
- A ~70K-param INT8 model = ~70 KB weights ⇒ fits entirely in on-chip BRAM; DDR3 only needed for the frame buffer.
- INT8 packing → ~2 MACs/DSP → ~1,480 MAC/clk; at 100 MHz that is ~148 GMAC/s vs ~20 MMAC/inference.

## What's a template vs runnable

- Runnable now (given torch + brevitas): `model_tiny.py`, `eval_common.py`, `distill.py`, `quantize_qat.py`, `ptq.py`, `export_onnx.py`.
- Toolchain templates (need FINN Docker / Vitis HLS, and per-layer folding tuning): `finn_build.py`, `hls4ml_build.py`.
