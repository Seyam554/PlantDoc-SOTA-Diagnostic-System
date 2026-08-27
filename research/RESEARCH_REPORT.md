# Deep-Research Report — Sub-100K-Parameter INT8 Plant-Disease Classifier for the Puzhi PA200T-StarLite (Artix-7 XC7A200T)

Date: 2026-08-27
Scope: how to take the project's "SOTA" PlantDoc model and produce a model that (a) has **< 100,000 parameters**, (b) runs in **INT8** on the **AMD Artix-7 XC7A200T** through the **Vivado** toolchain, and (c) loses as little accuracy as possible.

---

## 0. Executive summary / the honest headline

1. **Post-training quantization (PTQ) cannot meet the < 100K-parameter requirement.** Quantization changes *bits per weight* (FP32 → INT8 = 4× smaller memory), **not the number of weights**. The current SOTA model, `DINOv2 ViT-S/14`, has **≈ 22.1 M parameters**. INT8 PTQ makes it ≈ 22 MB → ≈ 5.5 MB; it is still ~221× over budget and still a Vision Transformer.
2. **Vision Transformers are a bad fit for Artix-7.** LayerNorm, GELU, softmax attention, and dynamic reshapes map poorly to a small FP-less FPGA fabric. Every practical FPGA leaf-disease classifier in the literature is a **small quantized CNN**.
3. **The workable path is model replacement, not model compression of the ViT:** design a **new depthwise-separable CNN student of 15K–95K parameters**, train it with **knowledge distillation (KD)** using the existing SOTA model (and/or a stronger ConvNeXt) as the **teacher**, then quantize the student to INT8 with **quantization-aware training (QAT, primary)** and **PTQ with AdaRound + bias-correction (fallback / "post-quantization" as literally requested)**.
4. **On the accuracy target:** "perfect / lossless" accuracy is not attainable on PlantDoc at this size. Realistic, defensible targets:
   - FP32 student with KD + PlantVillage pre-training + strong augmentation: **≈ 62–72 % top-1** on the 236-image PlantDoc test split (teacher `DINOv2` is currently 66.9 %; a better teacher raises this ceiling).
   - INT8 QAT student vs its own FP32: **within 0.5–1.5 %**.
   - INT8 PTQ (AdaRound) student vs its own FP32: **within 1–3 %**.
5. **Resource feasibility on XC7A200T is comfortable.** A ~60K-param INT8 model = ~60 KB of weights, which fits entirely in on-chip BRAM (~13 Mbit ≈ 1.6 MB). DDR3 is not needed for weights. 740 DSP48E1 slices with INT8 packing ≈ 1,480 MAC/cycle → at 100 MHz ≈ 148 GMAC/s, versus ~10–30 MMAC per inference → **> 1,000 fps is theoretically available**; realistic pipelined designs land 100–1,000 fps at < 1 W.

---

## 1. Target hardware — Puzhi PA200T-StarLite

| Item | Value | Consequence for the design |
|---|---|---|
| FPGA | AMD/Xilinx **XC7A200T-2FBG484I**, Artix-7 family | 28 nm, no hardened CPU (not a Zynq), no PCIe hard block, **no Vitis-AI DPU support** |
| Logic | 215,360 logic cells / 33,650 slices / 269,200 FF / 134,600 LUT | Enough for a FINN/hls4ml dataflow accelerator for a small CNN |
| DSP | **740 × DSP48E1** | INT8×INT8 with the Xilinx "2 MACs/DSP" packing trick → ~1,480 INT8 MAC/clk |
| BRAM | ~365 × 36 Kb ≈ **13.1 Mbit ≈ 1.63 MB** | Holds all weights + activations of a < 100K-param INT8 net on-chip |
| External RAM | 1 GB DDR3 | Only needed for input image buffering / frame store, not weights |
| I/O | HDMI out, Gigabit Ethernet, SD, USB-UART, camera header, dual 40-pin | Feed images via camera header or Ethernet; return class over UART/Ethernet |
| Clock / power | 5 V / 1 A board | Target 100–150 MHz accelerator clock, aim < 1 W dynamic |

**Host/control implication:** with no ARM PS you drive the accelerator with a **MicroBlaze soft-core + AXI-DMA**, or a pure-RTL front end reading the camera and streaming to the IP, with results over UART/Ethernet. This matters for the integration step, not the model.

---

## 2. Where the project stands today (baseline)

| Model | Params | PlantDoc test top-1 | Source |
|---|---|---|---|
| VGG16 (head-tuned) | 138 M | 57.2 % | `results/vgg16_benchmark_metrics.json` |
| ResNet50 (head-tuned) | 25.6 M | 61.4 % | `results/resnet50_benchmark_metrics.json` |
| **DINOv2 ViT-S/14 + head + TTA** | **≈ 22.1 M** | **66.9 %** | `results_sota/dinov2_vits14_benchmark_metrics.json` |
| External reference — EfficientNet-B3 on PlantDoc (2025) | 12 M | 73.3 % | Benchmarking In-the-Wild Multimodal Plant Disease Recognition |
| External reference — MobileNetV2 on *augmented* PlantDoc | 3.5 M | ~89.9 %* | Comparative Analysis of Lightweight DL Models |

\* Numbers on "augmented PlantDoc" or PlantDoc+web are **not comparable** to this project's 236-image held-out split; treat them as optimistic ceilings, not targets.

**Dataset:** 28 classes, 2,336 train / 236 test images, in-the-wild. Small + noisy + class-imbalanced → the single biggest accuracy lever is **data** (PlantVillage pre-training, aggressive augmentation, and KD), not the last 2 bits of quantization.

---

## 3. Literature synthesis

### 3.1 Making the model small enough (< 100K params)

- **Depthwise-separable convolutions** (MobileNet-style) are the standard primitive. A DW 3×3 + PW 1×1 pair costs `C·9 + C·C'` vs `C·C'·9` for standard conv — an 8–9× parameter cut at similar accuracy for small nets. *(CNN Accelerator on FPGA Using Depthwise Separable Convolution; DeepDive; MobileNet-friendly DL for Plant Disease Detection.)*
- **Knowledge distillation is the key accuracy recovery mechanism at this scale.** Reported results: transferring an ensemble/teacher into ShuffleNetV2 kept 98.5 % accuracy with **163× fewer parameters and 43.6× speed-up**; ALNet reaches **0.17 M params / 677 KB**; student models down to **~295 KB** via KD. Cross-architecture KD (ViT teacher → CNN student) is an established technique and directly applicable here (teacher = DINOv2 / ConvNeXt, student = tiny CNN). *(Knowledge Distillation Facilitates the Lightweight… Plant Diseases Detection Model; Cross-Architecture Knowledge Distillation, ACCV 2022; Lightweight Plant Disease Detection with Relationship-Based KD, 2025.)*
- **Hardware-aware NAS** (e.g. NASH) can search a sub-100K CNN under explicit LUT/DSP/BRAM constraints and co-optimize quantization; optional stretch goal, not required for a first result. *(NASH: NAS for Hardware-Optimized ML Models.)*
- **What to keep dense:** classes = 28, so a GAP → single FC(→28) head costs `Clast·28` params (≈ 2–5K). Avoid large FC layers entirely.

### 3.2 INT8 quantization that preserves accuracy

Consensus best practices from the two quantization whitepapers (Nagel et al. 2021; Krishnamoorthi 2018) and the NVIDIA integer-quantization study:

| Technique | Effect | Use here |
|---|---|---|
| **BatchNorm folding** into the preceding conv before quant | removes a source of range mismatch | always, first step |
| **Per-channel (per-output-channel) weight scales**, **per-tensor activation scales** | recovers most of the PTQ gap for conv nets | always |
| **Symmetric signed INT8 weights** (zero-point = 0), affine (asymmetric) INT8 activations, or symmetric activations after ReLU | symmetric weights are what FPGA MAC datapaths and FINN/hls4ml expect | always |
| **Cross-Layer Equalization (CLE)** + **absorbing high biases** | data-free range balancing across conv layers | PTQ path |
| **Bias correction** | cancels the mean quantization error introduced into activations | PTQ path |
| **AdaRound** (adaptive rounding, layer-wise) | closes most of the remaining PTQ gap; often < 1 % from FP32 for CNNs | PTQ path — this is the "post-quantization" method to use |
| **QAT with fake-quant / LSQ (learned step size)** | best accuracy; simulates INT8 in the forward pass, learns around it | primary path |
| **Calibration set** of 128–512 representative training images (class-balanced) | sets activation ranges (percentile/MSE, not plain min-max) | both paths; use MSE or 99.9-percentile |
| **Accumulator width check** | INT8×INT8 over a K-element dot product needs ≤ 32-bit accumulators; verify no overflow / add clipping-aware training | relevant for FINN; see *QNNs for Low-Precision Accumulation with Guaranteed Overflow Avoidance* |
| **Keep input pre-processing simple** | fixed-point normalize (multiply-shift), no per-image float mean/std on device | design the student to take uint8 [0,255] input and fold normalization into the first layer |

**PTQ vs QAT, quantified:** across the literature QAT "almost always" beats PTQ and can recover the large majority of the drop (e.g. ~96 % of the degradation recovered in reported LLM/vision cases); PTQ+AdaRound is usually "good enough" (≤ 1–3 %) for 8-bit CNNs but the gap **widens for very small models**, which is exactly this regime. → **Recommendation: ship QAT; keep PTQ+AdaRound as the fast baseline and ablation.**

### 3.3 Getting it onto Artix-7 through Vivado

Three viable flows, all terminating in Vivado synthesis/implementation:

| Flow | How it reaches Vivado | Strengths | Weaknesses | Verdict |
|---|---|---|---|---|
| **Brevitas → QONNX → FINN** | FINN emits per-layer HLS + a stitched IP / Vivado project; you run synth+impl in Vivado | purpose-built for QNNs, arbitrary bit-widths (INT8…binary), true dataflow (high fps), streaming weights in BRAM, MVAU/VVAU support depthwise | steeper setup (Docker), accelerator is model-specific, needs a driver (MicroBlaze/AXI) | **Primary.** Best match for "quantized CNN on a small AMD FPGA via Vivado." |
| **QKeras/Brevitas → hls4ml → Vivado HLS IP** | hls4ml generates an HLS project; export IP → Vivado block design | simplest, smallest BRAM for tiny models, `io_stream` for CNNs, well documented (MLPerf Tiny) | higher latency than FINN for the same net; less flexible for deep nets | **Strong alternative / cross-check.** Good for the first bring-up and for a ≤ 60K-param model. |
| **Hand-written Vivado HLS** tiled conv + INT8 MAC array, weights in BRAM | you write the HLS, synialize IP in Vitis HLS, integrate in Vivado | maximal control of DSP/BRAM, no external toolchain | most engineering effort, easiest to get wrong (accumulator, padding, requant) | **Only if** FINN/hls4ml resource results are unsatisfactory. |

Reference data points: a comparison on the same class of hardware ran an **hls4ml CNN of 58,115 params at 8–12-bit** successfully (fewer BRAMs, higher latency) against a much larger binary FINN net (more BRAM, 18× lower latency) — i.e. both tools work; FINN trades BRAM for speed. The Arty A7-100T (same Artix-7 family, half this device) is a documented hls4ml/FINN target. *(hls4ml vs FINN comparison; Fast CNNs on FPGAs with hls4ml; Benchmarking QNNs on FPGAs with FINN; Open-source FPGA-ML codesign for MLPerf Tiny.)*

**Why not Vitis-AI / DPU:** the DPUCZ/DPUCV accelerators require Zynq-7000/UltraScale+/Versal. The XC7A200T is plain Artix-7 with no PS — **DPU is not an option**, confirmed as a likely dead-end during research. This is why FINN/hls4ml (which emit generic RTL/IP) are the route.

---

## 4. Recommended method (what to actually build)

### 4.1 Architecture — `PlantEdgeNet` student, ≈ 15K–95K params (configurable `width_mult`)

Input **64×64×3 uint8** (normalization folded into first conv). All convs INT8-quantizable, ReLU6 activations, BN folded at export.

```
Stem      : Conv3x3  s2  3  -> 16w      + BN + ReLU6     -> 32x32
Block 1   : DWConv3x3 s1 16w + PWConv1x1 16w->24w        -> 32x32
Block 2   : DWConv3x3 s2 24w + PWConv1x1 24w->32w        -> 16x16
Block 3   : DWConv3x3 s1 32w + PWConv1x1 32w->48w        -> 16x16
Block 4   : DWConv3x3 s2 48w + PWConv1x1 48w->64w        ->  8x8
Block 5   : DWConv3x3 s1 64w + PWConv1x1 64w->96w        ->  8x8
Block 6   : DWConv3x3 s2 96w + PWConv1x1 96w->128w       ->  4x4
Head      : GlobalAvgPool -> Dropout -> FC 128w -> 28
```

Parameter count (channels rounded to multiples of 8; analytically verified — see `fpga/model_tiny.py`):
- `w = 0.5` → ≈ **11K params** (fast bring-up)
- `w = 1.0` → ≈ **32K params** (safe default)
- `w = 1.5` → ≈ **70K params** (**recommended** — uses the budget for accuracy)
- `w = 1.75` → ≈ **95K params** (accuracy-max, still < 100K; asserted at build time)

MACs at `w=1.0`, 64×64 input ≈ **8–12 M** per inference; `w=1.5` ≈ **18–24 M**.
Note: the < 100K budget is loose here — spend it. The accuracy bottleneck is data + KD, not capacity.

Design choices that matter for the FPGA:
- **64×64 input** (not 224): keeps the largest activation tensor at 32×32×16 INT8 = 16 KB, trivtarget for BRAM; 224 would blow BRAM and force DDR streaming.
- **ReLU6**, not GELU/SiLU: piecewise-linear, free in fixed point.
- **No residual adds across different scales**; keep skip connections INT8-aligned or omit — simpler requant.
- **GAP instead of flatten+FC**: kills the parameter blow-up and is a single accumulate on hardware.
- **Stride-2 convs instead of maxpool** where possible (fewer distinct hardware ops), or keep maxpool (also cheap) — either is fine.

### 4.2 Training pipeline (accuracy recovery order)

1. **Teacher.** Use the best available: current `checkpoints_sota/dinov2_vits14_best.pth` (66.9 %) as a baseline teacher; **strongly recommended** to first train a better teacher (ConvNeXt-Base or fine-tuned DINOv2, expected 72–78 %) since the student ceiling ≈ teacher − a few %.
2. **Student pre-training on PlantVillage** (54k images, 38 classes, lab conditions) → then **map/transfer to the 28 PlantDoc classes**. This is the single largest realistic accuracy gain for an in-the-wild small model.
3. **Knowledge distillation fine-tune on PlantDoc:**
   - Loss = `α · CE(student, hard_label) + (1-α) · KL(softmax(student/T), softmax(teacher/T)) · T²`, with `T ≈ 4`, `α ≈ 0.3`.
   - Optional **feature distillation**: project a mid student feature map to teacher feature dim with a 1×1 conv and add an MSE term (helps small students).
   - Heavy augmentation: RandAugment + random resized crop + horizontal/vertical flip + slight color jitter + MixUp/CutMix (CutMix helps in-the-wild leaf data).
   - Class-balanced sampler or class-weighted loss (PlantDoc is imbalanced).
   - Cosine LR, label smoothing 0.1, EMA weights, ~100–200 epochs.
4. **Record FP32 student accuracy** (this is the number INT8 is measured against).

### 4.3 Quantization — two paths, run both, ship the better

**Path A — INT8 QAT (primary, recommended to ship):**
- Framework: **Brevitas** (native QONNX/FINN export). Alternative: PyTorch `torch.ao.quantization` FX QAT if you go hls4ml/manual.
- Weights: per-channel symmetric signed INT8. Activations: per-tensor INT8 (unsigned after ReLU, or signed) affine.
- Fold BN, then insert fake-quant. Initialize from the FP32 student. LR = 1/10–1/100 of final training LR. 15–30 epochs. Freeze BN running stats after ~3 epochs.
- Prefer **LSQ** (learned step size) quantizers if available — best small-model results.
- Keep **input** quant at INT8 (uint8 image → scale = 1/255 folded into first conv weight scale).
- Export **QONNX**; validate numerically against the Brevitas model.

**Path B — INT8 PTQ (the literal "post-quantization" request; ship if it's within tolerance, always keep as ablation):**
- Calibration set: 256 class-balanced training images, MSE or 99.9-percentile range estimation.
- Apply, in order: **BN folding → Cross-Layer Equalization → high-bias absorption → AdaRound (weights) → bias correction**.
- Tools: Brevitas PTQ (`brevitas.graph.quantize` + `brevitas.graph.calibrate` + AdaRound), or AIMET, or ONNX Runtime static quant (per-channel) for a quick number.
- Per-channel weights are non-negotiable here; per-tensor weights will cost several %.

**Expected outcome ordering:** `FP32 student ≥ QAT INT8 ≳ PTQ+AdaRound INT8 > naive PTQ INT8`. If QAT INT8 is within ~1 % of FP32 and PTQ is within ~2–3 %, either is defensible for the paper; QAT is the safer claim.

### 4.4 FPGA deployment (Vivado)

1. **QONNX/ONNX → FINN** (Docker). Set `folding` (PE/SIMD) so DSP ≤ ~600 and BRAM ≤ ~300 with margin; target `fclk = 100–150 MHz`.
2. FINN emits a **stitched IP / Vivado project**; run **synthesis + implementation + timing closure in Vivado 2023.x/2024.x** for `xc7a200tfbg484-2`.
3. Build a block design: `camera/AXI-Stream source → (resize/crop to 64×64, uint8) → FINN IP → AXI-Stream sink → MicroBlaze → UART/Ethernet`. Or host-driven via JTAG-to-AXI for first tests.
4. **On-board validation:** run the same 236-image PlantDoc test set through the bitstream, compare per-class confusion matrix to the software INT8 model — they must match within rounding.
5. Report: LUT/FF/DSP/BRAM utilization, `fmax`, latency (cycles + µs), throughput (fps), estimated power (Vivado report), and accuracy (FP32 vs SW-INT8 vs HW-INT8).

**hls4ml cross-check (parallel, low cost):** convert the same student with hls4ml (`io_stream`, `Resource` strategy, `ap_fixed<8,·>` / INT8), export IP, get a second utilization/latency point. Useful comparison content for an ICCIT paper and a fallback if FINN timing/resources disappoint.

---

## 5. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Sub-100K student underfits PlantDoc badly (< 50 %) | medium | PlantVillage pre-train + KD + CutMix; allow `width_mult` up to 1.25 (92K); revisit 96×96 input if BRAM allows |
| INT8 drop larger than expected (tiny model, per-tensor acts) | medium | per-channel weights; QAT with LSQ; keep first/last layer calibration extra-careful; 99.9-percentile activation ranges |
| FINN timing closure fails at 100 MHz on Artix-7 | low–medium | reduce PE/SIMD folding, lower `fclk` to 75–100 MHz, or switch that layer to hls4ml IP |
| No PS on XC7A200T complicates data movement | certain (design fact) | plan MicroBlaze + AXI-DMA from day 1; JTAG-to-AXI for bench validation |
| Teacher (DINOv2, 66.9 %) too weak to lift student | medium | train ConvNeXt-Base / fine-tune DINOv2 to ≥ 73 % first; multi-teacher KD |
| "Perfect accuracy" expectation | certain | set explicit targets in §0.4; frame the paper as accuracy/efficiency trade-off, report the Pareto point |

---

## 6. Papers downloaded to `Papers/` (open access)

| File | Why it's here |
|---|---|
| `White_Paper_on_Neural_Network_Quantization_Nagel.pdf` | canonical PTQ+QAT best-practice reference (CLE, bias-corr, AdaRound, per-channel) |
| `Quantizing_Deep_ConvNets_Efficient_Inference_Whitepaper.pdf` | Krishnamoorthi 2018 — per-channel vs per-tensor, QAT recipe for CNNs |
| `Integer_Quantization_Principles_Empirical_Eval_NVIDIA.pdf` | INT8 principles + empirical accuracy study, calibration methods |
| `Benchmarking_QNNs_on_FPGAs_with_FINN.pdf` | FINN accuracy/throughput/resource trade-offs 2–8 bit |
| `Fast_CNNs_on_FPGAs_with_hls4ml.pdf` | hls4ml CNN flow, `io_stream`, quantized conv on FPGA |
| `FPGA-ML_codesign_MLPerf_Tiny.pdf` | hls4ml + FINN co-design on tiny image models, Artix-class targets |
| `QNNs_Low_Precision_Accumulation_Overflow_Avoidance.pdf` | accumulator bit-width guarantees for INT8 dataflow |
| `NASH_NAS_for_Hardware_Optimized_ML.pdf` | hardware-aware NAS + quantization co-search (stretch goal) |
| `CNN_Accelerator_FPGA_Depthwise_Separable_Conv.pdf` | depthwise-separable conv accelerator design on FPGA |
| `DeepDive_AlgoArch_CoDesign_Separable_ConvNets.pdf` | algorithm/architecture co-design for separable convnets on FPGA |
| `Benchmarking_In_the_Wild_Multimodal_Plant_Disease.pdf` | current PlantDoc SOTA numbers + versatile baseline |
| `Mobile_Friendly_DL_Plant_Disease_Detection.pdf` | MobileNet-class models for plant disease, KD context |
| `Comparative_Lightweight_DL_Memory_Constrained.pdf` | ShuffleNetV2 / MobileNetV2/V3 comparison on PlantDoc |
| `Lightweight_Explainable_CNN_Plant_Disease_Diagnosis.pdf` | compact CNN design for plant disease, param/FLOP budgets |
| `Lightweight_Transfer_Learning_Leaf_Diseases_MultiPlant.pdf` | transfer-learning small architecture across multiple plants |
| `Efficient_DL_Infrastructures_Embedded_Survey.pdf` | survey: quantization + embedded/FPGA inference landscape |
| `ConvNeXt_V2_Co-designing_and_Scaling_ConvNets.pdf` | candidate stronger teacher architecture |
| `Swin_Transformer_Hierarchical_Vision_Transformer.pdf` | candidate teacher / prior project reference |

Blocked (not open access via direct fetch, cite from abstract only): MDPI *FPGA-Based Low-Power CNN Accelerator Integrating DIST for Rice Leaf Disease* (10.3390/electronics14091704); *Knowledge Distillation Facilitates the Lightweight… Plant Diseases Detection Model* (Plant Phenomics, 10.34133/plantphenomics.0062); MDPI *Integer Intelligence: A Reproducible Path from Training to FPGA* (10.3390/electronics15051117 — has an Artix-7 CMOD A7 INT8 datapath worth reading if you can access it).

---

## 7. Sources (web)

- Knowledge Distillation Facilitates the Lightweight and Efficient Plant Diseases Detection Model — https://spj.science.org/doi/10.34133/plantphenomics.0062 / https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10308957/
- Lightweight Plant Disease Detection With Adaptive Multi-Scale Model and Relationship-Based Knowledge Distillation — https://onlinelibrary.wiley.com/doi/10.1111/exsy.70059
- A lightweight and explainable CNN model for empowering plant disease diagnosis — https://www.nature.com/articles/s41598-025-94083-1
- Plant pest and disease lightweight identification model by fusing tensor features and knowledge distillation — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11617168/
- FPGA-accelerated CNN for real-time plant disease identification — https://www.sciencedirect.com/science/article/abs/pii/S0168169923001035
- A Lightweight Quantized CNN Model for Plant Disease Recognition — https://link.springer.com/article/10.1007/s13369-023-08280-z
- FPGA-Based Low-Power High-Performance CNN Accelerator Integrating DIST for Rice Leaf Disease Classification — https://doi.org/10.3390/electronics14091704
- Brevitas Export — FINN documentation — https://finn.readthedocs.io/en/latest/brevitas_export.html
- FINN Quickstart — https://xilinx.github.io/finn/quickstart.html
- Benchmarking Quantized Neural Networks on FPGAs with FINN — https://arxiv.org/abs/2102.01341
- Integer Intelligence: A Reproducible Path from Training to FPGA — https://doi.org/10.3390/electronics15051117
- Fast convolutional neural networks on FPGAs with hls4ml — https://arxiv.org/abs/2101.05108
- Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark — https://arxiv.org/abs/2206.11791
- NASH: Neural Architecture Search for Hardware-Optimized ML Models — https://arxiv.org/abs/2403.01845
- Quantized Neural Networks for Low-Precision Accumulation with Guaranteed Overflow Avoidance — https://arxiv.org/abs/2301.13376
- A White Paper on Neural Network Quantization (Nagel et al.) — https://arxiv.org/abs/2106.08295
- Quantizing deep convolutional networks for efficient inference: A whitepaper (Krishnamoorthi) — https://arxiv.org/abs/1806.08342
- Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation (NVIDIA) — https://arxiv.org/abs/2004.09602
- QAT vs PTQ (overview) — https://medium.com/better-ml/quantization-aware-training-qat-vs-post-training-quantization-ptq-cd3244f43d9a
- Achieving FP32 Accuracy for INT8 Inference Using QAT with TensorRT — https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/
- QAT for LLMs with PyTorch — https://pytorch.org/blog/quantization-aware-training/
- A CNN Accelerator on FPGA Using Depthwise Separable Convolution — https://arxiv.org/abs/1809.01536
- Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline — https://arxiv.org/abs/2408.03120
- Comparative Analysis of Lightweight Deep Learning Models for Memory-Constrained Devices — https://arxiv.org/abs/2505.03303
- Mobile-Friendly Deep Learning for Plant Disease Detection — https://arxiv.org/abs/2508.10817
- RTR_Lite_MobileNetV2: A lightweight and efficient model for plant disease detection — https://www.sciencedirect.com/science/article/pii/S2214662825000271
- Evaluation of lightweight and efficient deep learning models for plant disease classification — https://link.springer.com/article/10.1007/s43926-026-00310-0
- Puzhi PA200T-StarLite product page — https://www.en.puzhi.com/Product/AMD-FPGA-Development-Board/Artix-7/PA200T-StarLite
- Puzhi PA200T-StarLite detail — https://www.en.puzhi.com/detail/429.html
- XC7A200T specifications — https://pcbsync.com/artix-7-200t/
- hls4ml vs FINN common interface / comparison — https://www.researchgate.net/figure/HLS4ML-and-FINN-common-interface_fig2_375705484
- A Low Memory Requirement MobileNets Accelerator Based on FPGA — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9854863/
