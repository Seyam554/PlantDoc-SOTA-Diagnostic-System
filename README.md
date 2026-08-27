# 🌿 PlantDoc-SOTA: Diagnostic System + Sub-100K INT8 FPGA Classifier

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20%7C%20CUDA%2012.x-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg?style=flat&logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![DINOv2 ViT](https://img.shields.io/badge/DINOv2-Vision%20Transformer-0080FF.svg?style=flat&logo=meta&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![FPGA](https://img.shields.io/badge/Target-AMD%20Artix--7%20XC7A200T-FFC627.svg?style=flat&logo=amd&logoColor=black)](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/artix-7.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Two connected efforts on the benchmark **PlantDoc dataset** (Singh et al., CoDS-COMAD 2020) —
in-the-wild plant disease detection & fine-grained classification:

1. **Part A — the 3-stage diagnostic system** (YOLOv11 leaf detector → DINOv2 ViT classifier → TTA aggregation).
2. **Part B — [`PlantEdgeNet-FPGA/`](PlantEdgeNet-FPGA/): a < 100,000-parameter INT8 CNN** distilled from
   the Part-A models, targeting the **Puzhi PA200T-StarLite (AMD Artix-7 XC7A200T)** via FINN / hls4ml /
   Vivado. Goal: keep accuracy as high as possible (target ≥ 80 % on the PlantDoc test split) at a size
   that fits on-chip BRAM. **This is a self-contained sub-folder** — model, data prep, KD, training,
   quantization, export, inference, FPGA build configs, and the full research write-up all live there.

> **For AI agents:** for Part A read this file. For Part B read
> [`PlantEdgeNet-FPGA/README.md`](PlantEdgeNet-FPGA/README.md) (the canonical runbook), then
> `PlantEdgeNet-FPGA/research/{RESEARCH_REPORT,ACCURACY_TO_80,IMPLEMENTATION_PLAN}.md` and
> `PlantEdgeNet-FPGA/references.md`. Those contain the complete design rationale, literature basis,
> accuracy levers, milestone plan, and every command.

---

## 📌 1. Part A — 3-Stage Diagnostic System

```
 Raw in-the-wild photo ──► Stage 1: YOLOv11 leaf localization ──► Stage 2: DINOv2 ViT-S/14 ──► Stage 3: TTA
 (complex background)       (rejects soil / hands / glare)          (518px patch-token classifier)   aggregation + overlays
```

| Stage | Model | Key settings |
|---|---|---|
| 1. Leaf detection | Ultralytics **YOLOv11n** (COCO → PlantDoc) | 640 px, 20 epochs |
| 2. Disease classification | **DINOv2 ViT-S/14** + 2-layer head | 518 px, differential LR (1e-5 backbone / 5e-4 head), label smoothing 0.1 |
| 3. Aggregation | Test-Time Augmentation | original + h-flip vote, colour-coded boxes |

### Benchmark results

**Stage 1 — Leaf detection (mAP @ 0.50 IoU)**

| Model | Weights | mAP@0.50 | Precision | Recall |
|---|---|---|---|---|
| **YOLOv11 (this repo)** | YOLOv11n (COCO + PlantDoc) | **59.40 %** | 41.90 % | 64.20 % |
| Faster R-CNN (InceptionResNet-V2) | Singh et al. 2020 (COCO) | 38.90 % | — | — |
| MobileNet-SSD | Singh et al. 2020 (COCO) | 32.80 % | — | — |

**Stage 2 — Disease classification (PlantDoc test split, 236 images, 28 classes)**

| Model | Backbone | Top-1 | Weighted-P | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| **DINOv2 ViT-S/14** | Meta LVD-142M SSL | **66.95 %** (77.97 % val) | 73.73 % | 0.6399 | 0.6600 |
| ResNet-50 | ImageNet-1k | 61.44 % | 64.35 % | 0.5980 | 0.6151 |
| VGG-16 | ImageNet-1k | 57.20 % | 61.58 % | 0.5459 | 0.5715 |
| Paper baseline (Singh et al. 2020) | ImageNet-1k | 44.52 % | — | 0.4400 | — |

---

## 🎯 2. Part B — the FPGA sub-project ([`PlantEdgeNet-FPGA/`](PlantEdgeNet-FPGA/))

Self-contained. File paths in this section are relative to `PlantEdgeNet-FPGA/` (e.g.
`fpga/model_tiny.py` = `PlantEdgeNet-FPGA/fpga/model_tiny.py`).

### 2.1 The problem and the honest constraints

* **Post-training quantization does NOT reduce parameter count.** FP32 → INT8 is 4× smaller memory,
  *same* number of weights. DINOv2 ViT-S/14 has ~22 M params — 220× over a 100 K budget — and a ViT
  (LayerNorm / GELU / softmax attention) maps poorly to a small FP-less FPGA. So the SOTA model becomes
  a **teacher**, not the deployed model.
* **The deployed model is a new tiny CNN: `PlantEdgeNet`** (`fpga/model_tiny.py`) — a MobileNet-style
  depthwise-separable network with optional Squeeze-Excite, 64 or 96 px input, ReLU6, GAP + one FC head.

  | config | params | MACs/inf (96 px) |
  |---|---|---|
  | `--width 1.0` | ~34 K | ~5.8 M |
  | `--width 1.5 --se` (**default / recommended**) | **~72 K** | ~12 M |
  | `--width 1.75 --se` | ~95 K | ~16 M |

* **≥ 80 % on the honest PlantDoc test split with < 100 K params is very ambitious** (EfficientNet-B3 at
  12 M params scores 73 %; lightweight SOTA ~82 % needs ~2 M params + augmentation). The levers that get
  close, all kept within the < 100 K budget:

  | Lever | Expected gain | Where |
  |---|---|---|
  | **Leaf-ROI crop** (HSV / Excess-Green segmentation, FPGA-cheap) | **+10–20 pts** | `fpga/leaf_roi.py`, `fpga/prepare_data.py` |
  | **External training data** (PlantVillage mapped + web), test stays PlantDoc | +3–8 pts (only *with* crop) | `fpga/prepare_data.py --plantvillage` |
  | **Multi-teacher KD** (DINOv2 + ConvNeXt + EfficientNet, soft targets + DIST + attention-transfer) | +2–5 pts | `fpga/kd.py`, `fpga/train_teacher.py`, `--teachers` |
  | **96 px input + Squeeze-Excite + EMA + adaptive-BN** | +4–9 pts combined | `fpga/train_fpga.py` |
  | **QAT instead of PTQ for the INT8 step** | +1–3 pts (PTQ hurts depthwise nets) | `fpga/quantize_qat.py`, `--qat` |

  **Projected:** FP32 ≈ 76–82 %, INT8 (QAT) ≈ 75–81 %. Fallback if it lands 77–79 %: relax to ~300 K
  params (still < 512 KB INT8, still fits the XC7A200T) — see `research/ACCURACY_TO_80.md` §F.

### 2.2 Target hardware — Puzhi PA200T-StarLite

| Item | Value | Consequence |
|---|---|---|
| FPGA | **XC7A200T-2FBG484I** (Artix-7) | 28 nm, **no hardened CPU** (not a Zynq), **no Vitis-AI DPU** |
| Logic / DSP / BRAM | 215 K cells · 740 DSP48E1 · ~13 Mbit (~1.63 MB) BRAM | a ~72 K-param INT8 model (~72 KB weights) fits **entirely on-chip** — no DDR for weights |
| External RAM / I/O | 1 GB DDR3 · HDMI · GbE · camera header · dual 40-pin | drive the accelerator with a MicroBlaze soft-core + AXI-DMA, or JTAG-to-AXI for bench tests |
| Clock target | 100–150 MHz | INT8 DSP packing ≈ 1 480 MAC/clk → ~148 GMAC/s vs ~12–16 MMAC/inference |

### 2.3 Deployment flow (all terminate in Vivado)

```
PlantEdgeNet (Brevitas QAT) ──► QONNX ──► FINN compiler ──► stitched IP + Vivado project ──► synth/impl ──► .bit
                              └► ONNX  ──► hls4ml         ──► HLS IP ─────────────────────────┘  (cross-check / fallback)
```
`fpga/finn_build.py` and `fpga/hls4ml_build.py` are configured for `xc7a200tfbg484-2`.
`fpga/export_onnx.py` produces QONNX (FINN) or ONNX (hls4ml) + a numerical-parity check.

---

## 📚 3. Research documents (read these for the full picture)

| File | Contents |
|---|---|
| `PlantEdgeNet-FPGA/README.md` | The command-level runbook for the FPGA sub-project. |
| `PlantEdgeNet-FPGA/research/RESEARCH_REPORT.md` | Hardware analysis, literature synthesis (KD, tiny CNNs, INT8 best practice, FINN vs hls4ml), recommended architecture + quantization recipe, risk table, ~35 sources. |
| `PlantEdgeNet-FPGA/research/ACCURACY_TO_80.md` | Deep dive on reaching ≥ 80 %: the honest ceiling, every lever ranked by expected gain, the selected path, projected numbers, fallback. |
| `PlantEdgeNet-FPGA/research/IMPLEMENTATION_PLAN.md` | 10 milestones (M0–M9) with acceptance criteria and a file map. |
| `PlantEdgeNet-FPGA/references.md` | All papers the FPGA design is built on, grouped by topic, with links. |
| `Papers/` | Part-A research PDFs (PlantDoc, DINOv2, ConvNeXt, Swin, PlantXViT). |

---

## 💻 4. Environment

**GPU box (training):** Python 3.11–3.14, CUDA 12.x GPU. This project was developed with an RTX 3050
Laptop GPU using a venv named `.venv-1`.

```bash
git clone https://github.com/Seyam554/PlantDoc-SOTA-Diagnostic-System.git
cd PlantDoc-SOTA-Diagnostic-System

python -m venv .venv-1
# Windows:  .\.venv-1\Scripts\Activate.ps1     Linux/Mac:  source .venv-1/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt                                # Part A
pip install -r PlantEdgeNet-FPGA/fpga/requirements-fpga.txt    # Part B extras: timm, brevitas, qonnx, onnx, onnxruntime
```

The Part-B scripts (`train_fpga.py`, `predict.py`, `prepare_data.py`, `leaf_roi.py`, `train_teacher.py`, `kd.py`) need
only **torch, torchvision, numpy, matplotlib**. `timm` is needed for ConvNeXt/EfficientNet teachers;
`brevitas` + `qonnx` for `--qat` and the FINN export. Missing optional deps degrade gracefully
(fewer teachers / PTQ ships instead of QAT).

> **Note:** do **not** run this project from inside a cloud-synced folder (OneDrive/Dropbox). Sync can
> silently roll back or partially restore files mid-work. Clone to a local path.

---

## 📂 5. Repository layout

```
Part A (3-stage diagnostic system)
├── dataset.py / dataset_sota.py        # loaders (224 px base / 518 px RandAugment)
├── models.py / models_sota.py          # VGG16·ResNet50·MobileNetV2 / DINOv2·ConvNeXt·Swin
├── train.py / train_sota.py            # baseline / differential-LR SOTA trainers
├── evaluate.py / evaluate_sota.py      # evaluation (+ TTA)
├── train_stage1_yolo.py               # YOLOv11 leaf detector
├── voc_to_yolo.py / extract_*.py       # dataset prep
├── pipeline_end_to_end.py             # 3-stage integrated diagnosis
├── run_random_tests.py                # batch diagnosis, timestamped output/
├── results/ · results_sota/ · output/  # metrics, confusion matrices, annotated images

Part B  ──  PlantEdgeNet-FPGA/  (self-contained sub-100K INT8 classifier for Artix-7)
├── README.md               # canonical runbook for the FPGA sub-project
├── references.md            # papers the design is built on (grouped, with links)
├── models.py / models_sota.py   # vendored backbone defs (for loading teacher checkpoints)
├── research/
│   ├── RESEARCH_REPORT.md        # full design + literature basis
│   ├── ACCURACY_TO_80.md         # deep research: how to push the model to ≥80%
│   └── IMPLEMENTATION_PLAN.md    # 10-milestone plan + acceptance criteria
└── fpga/
    ├── model_tiny.py       # PlantEdgeNet: depthwise-separable CNN + optional Squeeze-Excite, <100K params
    ├── leaf_roi.py         # HSV / Excess-Green leaf-ROI crop (FPGA-cheap) + preview grid
    ├── prepare_data.py     # build PlantDoc-Cropped[-Plus]/{train,test}: crop + map PlantVillage + fold web data
    ├── kd.py               # multi-teacher soft targets + DIST relational loss + attention-transfer feature loss
    ├── train_teacher.py    # fine-tune a timm backbone (ConvNeXt/EfficientNet) into a KD teacher
    ├── train_fpga.py       # ONE-SHOT: GPU-cached train + KD + adaptive-BN + INT8 PTQ + INT8 QAT + summary
    ├── distill.py          # standalone PlantVillage-pretrain / multi-teacher-distill (finer control)
    ├── quantize_qat.py     # Brevitas INT8 QAT (per-channel sym weights) + QONNX export path
    ├── ptq.py              # INT8 PTQ ablation: CLE + bias-correction + AdaRound
    ├── export_onnx.py      # QONNX (FINN) / ONNX (hls4ml) export + PyTorch↔ONNX parity check
    ├── predict.py          # run any checkpoint (FP32 / INT8-PTQ / INT8-QAT) on images, report accuracy + grid
    ├── finn_build.py       # FINN dataflow build config for xc7a200tfbg484-2  (run inside FINN Docker)
    ├── hls4ml_build.py     # hls4ml build config for xc7a200tfbg484-2       (needs Vitis HLS)
    ├── eval_common.py      # shared loaders/metrics for the standalone ptq.py / quantize_qat.py mains
    └── requirements-fpga.txt
```

---

## 🚀 6. Reproduce Part A (diagnostic system)

```bash
# datasets
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git            && python extract_dataset.py
git clone https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git && python extract_obj_detection.py && python voc_to_yolo.py

# train
python train_stage1_yolo.py --epochs 20 --batch 16 --imgsz 640
python train_sota.py --model dinov2_vits14 --img-size 518 --epochs 25 --batch-size 16 --lr-head 5e-4 --lr-backbone 1e-5
python evaluate_sota.py --checkpoint checkpoints_sota/dinov2_vits14_best.pth --use-tta

# run
python pipeline_end_to_end.py --image "path/to/photo.jpg"
python run_random_tests.py --num-images 10
```

---

## 🧪 7. Reproduce Part B (sub-100K INT8 FPGA model)

The full runbook lives in **[`PlantEdgeNet-FPGA/README.md`](PlantEdgeNet-FPGA/README.md)**. At a glance
(run every command from this repo root, using `.venv-1`):

| Step | Script | What |
|---|---|---|
| 0 | `pip install timm brevitas qonnx onnx onnxruntime` | optional deps (teachers, QAT, export) |
| 1 | `PlantEdgeNet-FPGA/fpga/prepare_data.py` | leaf-ROI crop → `PlantDoc-Cropped/` (`--plantvillage` / `--web` fold extra **train** data; test stays PlantDoc) |
| 2 | `PlantEdgeNet-FPGA/fpga/train_teacher.py` | *(optional)* fine-tune ConvNeXt / EfficientNet teachers |
| 3 | `PlantEdgeNet-FPGA/fpga/train_fpga.py` | one shot: GPU-cached train + multi-teacher KD (soft + DIST + attention transfer) + EMA + adaptive-BN + INT8 **PTQ** + INT8 **QAT** → ships the better |
| 4 | `PlantEdgeNet-FPGA/fpga/predict.py` | run FP32 / INT8-PTQ / INT8-QAT checkpoints on images, report accuracy + grid |
| 5 | `PlantEdgeNet-FPGA/fpga/export_onnx.py` | QONNX (FINN) / ONNX (hls4ml) + PyTorch↔ONNX parity check |
| 6 | `PlantEdgeNet-FPGA/fpga/{finn_build,hls4ml_build}.py` | Vivado build for `xc7a200tfbg484-2`; then on-board eval on the 236 test images |

Recommended Step 3 (DINOv2 teacher only, no timm needed):
```powershell
.\.venv-1\Scripts\python.exe PlantEdgeNet-FPGA\fpga\train_fpga.py --data-dir PlantDoc-Cropped `
  --width 1.5 --se --img-size 96 --epochs 200 --iters-per-epoch 120 --mixup `
  --teachers checkpoints_sota\dinov2_vits14_best.pth --adabn-batches 50 --qat
```
Outputs: `PlantEdgeNet-FPGA/fpga/checkpoints_fpga/plantedgenet_w1.5_{fp32,int8_ptq,int8_qat}.pth`,
`results_fpga/plantedgenet_w1.5_summary.json` (`params`, `macs`, `fp32`, `int8_ptq`, `int8_qat`,
`shipped`, `shipped_acc`, `int8_gap_pp`), confusion-matrix PNGs.

**Acceptance criteria** (full list in `PlantEdgeNet-FPGA/research/IMPLEMENTATION_PLAN.md`):
params < 100 000 · FP32 test ≥ 62 % (stretch ≥ 80 %) · shipped INT8 within ≤ 1.5 % of FP32 ·
ONNX/QONNX argmax agreement = 100 % · fits `xc7a200tfbg484-2` (DSP ≤ 700, BRAM ≤ 360, LUT ≤ 130 k,
timing met ≥ 100 MHz) · on-board predictions match software INT8.

---

## 📊 8. Dataset notes

* **PlantDoc classification:** 28 classes, 2 336 train / 236 test, in-the-wild. Noisy, class-imbalanced —
  the biggest accuracy lever is **data + leaf-ROI cropping**, not the last bits of quantization.
* **PlantDoc-Object-Detection:** Pascal-VOC boxes; `voc_to_yolo.py` converts to YOLO format.
* All large artifacts (datasets, `*.pth`, venvs, `PlantDoc-Cropped*/`, `checkpoints_fpga/`,
  `results_fpga/`) are git-ignored — regenerate them with the runbooks above.

---

## 📜 9. License & acknowledgements

MIT License. Thanks to the authors of **PlantDoc** (Singh et al., IIT Gandhinagar), **Ultralytics YOLO**,
**Meta AI DINOv2**, **Xilinx/AMD FINN & Brevitas**, **Fast ML hls4ml**, and **timm**.
