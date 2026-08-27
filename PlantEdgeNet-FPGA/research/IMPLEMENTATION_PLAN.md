# Implementation Plan — PlantEdgeNet: <100K-param INT8 PlantDoc classifier on Artix-7 XC7A200T (Vivado)

Companion to `research/RESEARCH_REPORT.md`. Starter code lives in `fpga/`.

## Pipeline overview

```
[FP32 teacher]  DINOv2 / ConvNeXt  (existing checkpoints_sota/, or train stronger)
      │  knowledge distillation
      ▼
[FP32 student]  PlantEdgeNet  (fpga/model_tiny.py)  ~11K-95K params (w=1.5 -> ~70K recommended)
      │  (a) INT8 QAT  (fpga/quantize_qat.py, Brevitas)      ← ship this
      │  (b) INT8 PTQ  (fpga/ptq.py, AdaRound+bias-corr)     ← literal "post quantization" + ablation
      ▼
[QONNX / ONNX INT8]  (fpga/export_onnx.py)
      │  FINN compiler (Docker)   OR   hls4ml   OR   hand HLS
      ▼
[Vivado 2023.x/2024.x project]  xc7a200tfbg484-2
      ▼
[bitstream]  MicroBlaze + AXI-DMA + camera/UART   →   on-board 236-image eval
```

## Milestones

### M0 — Environment (0.5 day)
- `python -m venv` already present (`.venv`). `pip install -r fpga/requirements-fpga.txt`.
- Install Vivado 2023.2 or 2024.1 (WebPACK covers xc7a200t). Optional: Docker + FINN (`git clone https://github.com/Xilinx/finn`).
- Sanity: `python fpga/model_tiny.py --width 1.0 --summary` prints param/MAC count (< 100K assert).

### M1 — Stronger teacher (1–2 days, optional but recommended)
- Train ConvNeXt-Base (`timm`) or fine-tune DINOv2 with the existing `train_sota.py` machinery, target ≥ 73% PlantDoc test.
- If skipping: use `checkpoints_sota/dinov2_vits14_best.pth` (66.9%) as teacher directly.

### M2 — FP32 student with KD (2–4 days)
1. (Optional, big win) Pre-train `PlantEdgeNet` on PlantVillage: `python fpga/distill.py --stage pretrain --data <plantvillage>`.
2. KD fine-tune on PlantDoc: `python fpga/distill.py --stage distill --teacher checkpoints_sota/dinov2_vits14_best.pth --width 1.0`.
3. Target: FP32 student ≥ 62% test (stretch 70%). Save `fpga/checkpoints_fpga/plantedgenet_w1.0_fp32.pth`.
4. Gate: if < 55%, raise `--width` to 1.25, add CutMix/MixUp, more epochs, verify PlantVillage pre-train loaded.

### M3 — INT8 QAT (1–2 days)
- `python fpga/quantize_qat.py --ckpt fpga/checkpoints_fpga/plantedgenet_w1.0_fp32.pth --epochs 25`.
- Per-channel symmetric INT8 weights, per-tensor INT8 acts, BN folded, LSQ if available.
- Gate: QAT INT8 test accuracy within 1.5% of FP32 student. Save `..._int8_qat.pth`.

### M4 — INT8 PTQ ablation (0.5 day)
- `python fpga/ptq.py --ckpt ..._fp32.pth --calib-images 256 --adaround`.
- Record naive-PTQ vs CLE+bias-corr vs +AdaRound. Table goes in the paper.

### M5 — Export + numerical check (0.5 day)
- `python fpga/export_onnx.py --ckpt ..._int8_qat.pth --format qonnx` → `fpga/export/plantedgenet_int8.onnx`.
- Verify: PyTorch-INT8 logits vs onnxruntime/QONNX logits match (max abs diff, argmax agreement = 100% on test set).

### M6 — FINN build (3–7 days)
- FINN dataflow build script (`fpga/finn_build.py`, template provided): set `folding_config` so DSP ≤ 600, BRAM ≤ 300, `synth_clk_period_ns = 10` (100 MHz), `fpga_part = "xc7a200tfbg484-2"`.
- Outputs stitched IP + Vivado project. Run synth/impl in Vivado, close timing.
- Deliverable: `utilization + timing + latency + throughput` report.

### M7 — hls4ml cross-check (2–3 days, parallel)
- `fpga/hls4ml_build.py` template: `io_stream`, `Resource` strategy, `ap_fixed<8,4>` / INT8, `Part = xc7a200tfbg484-2`.
- Second resource/latency data point + fallback path.

### M8 — Board integration + on-hardware eval (3–5 days)
- Vivado block design: camera/HDMI-in or Ethernet frame source → preprocess (crop+resize to 64×64 uint8) → accelerator IP → MicroBlaze reads class → UART/Ethernet out.
- Bench bring-up via JTAG-to-AXI: push the 236 test images, capture predictions, rebuild confusion matrix, compare to SW INT8 (must match within rounding).
- Deliverable: `results_fpga/hw_confusion_matrix.png`, `results_fpga/hw_benchmark_metrics.json`, power estimate.

### M9 — Paper artifacts (2 days)
- Tables: param/MAC/accuracy across `width_mult ∈ {0.5,1.0,1.25}`; FP32 vs QAT vs PTQ variants; FINN vs hls4ml resources; fps/latency/power; per-class F1 vs DINOv2 teacher.
- Figures: architecture, accuracy-vs-precision curve, accuracy-vs-params Pareto, HW confusion matrix.

## Acceptance criteria

| # | Criterion |
|---|---|
| A1 | Student parameter count **< 100,000** (asserted in `model_tiny.py`) |
| A2 | FP32 student PlantDoc test top-1 **≥ 62%** (stretch ≥ 70%) |
| A3 | INT8 (shipped variant) within **≤ 1.5%** of FP32 student |
| A4 | ONNX/QONNX argmax agreement with PyTorch INT8 = **100%** on test set |
| A5 | Fits `xc7a200tfbg484-2`: **DSP ≤ 700, BRAM ≤ 360, LUT ≤ 130k**, timing met at ≥ 100 MHz |
| A6 | On-board predictions match software INT8 within rounding; HW accuracy reported |
| A7 | All claims backed by a committed metrics JSON + confusion matrix |

## File map (starter code in `fpga/`)

| File | Purpose | Status |
|---|---|---|
| `model_tiny.py` | `PlantEdgeNet` definition, param/MAC counter, `--summary` | provided, runnable |
| `distill.py` | PlantVillage pre-train + KD fine-tune on PlantDoc | provided, runnable (needs data paths) |
| `quantize_qat.py` | Brevitas INT8 QAT + eval | provided; needs `brevitas` installed |
| `ptq.py` | INT8 PTQ: calibration + CLE + bias-corr + AdaRound hook | provided; Brevitas PTQ |
| `export_onnx.py` | QONNX/ONNX export + numerical parity check | provided |
| `finn_build.py` | FINN dataflow build config for xc7a200t | template |
| `hls4ml_build.py` | hls4ml build config for xc7a200t | template |
| `eval_common.py` | shared dataloader (reuses repo `dataset.py` transforms at 64px) + metrics | provided |
| `requirements-fpga.txt` | extra deps | provided |
| `README.md` | run order, Vivado notes, no-PS integration notes | provided |
