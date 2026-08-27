# Deep Research — Can a < 100K-param INT8 model hit ≥ 80 % on PlantDoc, and how?

Date: 2026-08-27
Question: raise the FPGA student (`PlantEdgeNet`, ~89K params, INT8, Artix-7) to **≥ 80 % top-1** on PlantDoc.
Companion to `research/RESEARCH_REPORT.md` and `research/IMPLEMENTATION_PLAN.md`.

*(Skill note: `/deep-research`, `/senior-ml-engineer`, `/agent-launcher-orchestrator`, `/agenthub` are
not on the live roster / don't fit — `agent-launcher` builds Claude Managed Agents, `agenthub` runs
competing agents on a working git repo and this repo's `.git` is broken. Research done directly.)*

---

## SELECTED PATH (decided 2026-08-27)

- **Constraint:** keep **< 100K params**, honest PlantDoc test split. Flex = add leaf-crop + external training data.
- **Inference ROI:** **HSV green-segmentation crop** (threshold green, largest bounding box; FPGA-cheap).
  Ground-truth boxes from `PlantDoc-Object-Detection-Dataset/` used for train/val cropping.
- **KD:** **multi-teacher** — ConvNeXt-Base + EfficientNet-B3 + DINOv2 ViT-S/14, averaged soft targets,
  plus feature-MSE + DIST relational loss.
- **Also:** input 64→96px, EMA-fix training loop, **ship the QAT INT8 model** (not PTQ).
- **Projected:** FP32 ~76–82 %, INT8 (QAT) ~75–81 % on PlantDoc test. Fallback if <80 %: relax to ~300K params.

Build list (not yet implemented): `fpga/prepare_data.py` (crop boxes + map/domain-randomize PlantVillage
+ optional web scrape → `PlantDoc-Cropped/`), `fpga/leaf_roi.py` (HSV green-seg crop + FPGA notes),
multi-teacher + feature/DIST KD in `fpga/distill.py`, `--se` + `--img-size 96` in `fpga/model_tiny.py`,
adaptive-BN recalibration + QAT-as-shipped in `fpga/train_fpga.py`.

---

## 0. TL;DR — the honest answer

**On the standard PlantDoc test split (236 images, whole-image, no external test data), ≥ 80 % with < 100K
parameters is very unlikely.** Published context:

| Model | Params | PlantDoc test top-1 | Protocol |
|---|---|---|---|
| EfficientNet-B3 | ~12 M | **73.3 %** | train + test on PlantDoc only (2025) |
| MobileViT (lightweight) | ~5 M | 75.7 % | PlantDoc only |
| RTR_Lite_MobileNetV2 | ~2 M | **82.0 %** | PlantDoc **with augmentation** |
| MobileNetV2 (transfer) | 3.5 M | 89.9 % | **augmented** PlantDoc / merged split |
| DINOv2 ViT-S/14 (this repo) | 22 M | 66.9 % | PlantDoc only, +TTA |
| MCUNet (TinyNAS, ImageNet ref) | ~0.7 M | 70.7 % ImageNet-1k | not PlantDoc; shows the tiny-model ceiling |
| **PlantEdgeNet target** | **< 0.1 M** | **? → 80 %** | — |

80 % on the plain split would mean **beating EfficientNet-B3 by ~7 points with ~130× fewer parameters**.
The lightweight results that *do* reach 80–90 % (RTR_Lite, MobileNetV2) are ~20–90× bigger than 100K
**and** rely on augmentation / merged data.

**80 % is realistically reachable only if you flex one or more constraints:**

| Lever | What changes | Expected effect | Constraint touched |
|---|---|---|---|
| **A. Leaf-ROI crop preprocessing** | detect+crop the leaf before classifying (repo already has `PlantDoc-Object-Detection-Dataset` bounding boxes) | **+10–20 pts** — biggest single lever; Grad-CAM shows current models look at *background* | adds a small detector stage (still fits FPGA) |
| **B. Extra training data** (PlantVillage + web-scraped), test still PlantDoc | more coverage of each class | +3–8 pts, **only if combined with crop** (raw PlantVillage→PlantDoc transfer collapses 30–67 pts) | training data only; test protocol unchanged |
| **C. Input resolution 64 → 96/112** | lesions are fine detail | +3–6 pts | more FPGA BRAM/DSP (still OK on XC7A200T) |
| **D. Knowledge distillation done properly** (strong teacher + feature/DIST KD) | small model mimics teacher features | +2–5 pts vs logit-KD alone | none |
| **E. QAT instead of PTQ for the INT8 step** | model learns around quantization | recovers **+1–3 pts** that PTQ loses on depthwise nets | none |
| **F. Param budget 100K → 250–500K** | real capacity | +5–10 pts, makes 80 % comfortable | the < 100K rule |
| **G. Evaluate like the 80–90 % papers** (augmented test / k-fold on merged set) | — | "reaches 80 %+" by protocol | test protocol (not recommended for a credible paper) |

**Recommended path to ~80 %:** A + B + C + D + E, keeping < 100K params and honest PlantDoc test.
Expected landing: **75–82 %** FP32, **74–81 %** INT8 (QAT). If that still misses, add F (relax to ~250K).

---

## 1. Why the current model is far from 80 %

- **PlantDoc is in-the-wild and noisy.** Multiple sources call it out: "a large number of images not
  adequate for actual plant leaf disease classification"; models score "extremely low" vs lab datasets.
- **Domain shift dominates.** A ResNet-50 fine-tuned on PlantVillage loses **67.7 pts** when tested on
  PlantDoc. Grad-CAM shows attention moves **from the lesion to the background** — the model keys on
  pot/hand/soil/lighting, not disease. Quantified drivers: saturation, border edge density,
  foreground-occupancy differences.
- **64×64 input** (chosen for FPGA BRAM) throws away lesion texture. Literature standard is 224; accuracy
  "degrades noticeably at lower resolutions like 64".
- **< 100K params** is below the MCUNet scale (~0.7 M, 70.7 % ImageNet). Capacity is a real ceiling.
- **PTQ on depthwise-separable nets loses accuracy** ("PTQ … suffers significant accuracy loss on
  MobileNets/ShuffleNets"). The current script's PTQ is a further handicap.

## 2. The levers, in priority order

### A. Leaf-ROI crop (do this first — largest gain)
The "in the wild" difficulty is 80 % framing/background, not disease appearance.
- **You already have `PlantDoc-Object-Detection-Dataset/` (Pascal-VOC bounding boxes).**
- Two implementation options:
  1. **Offline crop with ground-truth boxes** for training/val, and at inference run a **tiny leaf
     detector** (nano-YOLO / a 2-anchor SSD head, or even a coarse saliency/green-segmentation crop)
     to produce the ROI. A green/leaf HSV segmentation + largest-component bbox is nearly free on FPGA
     and captures most of the gain.
  2. **Center-biased multi-crop TTA**: classify 3–5 crops (center + corners) and average — cheap, no
     detector, partial gain.
- Reported downstream effect of detect-then-classify: 94–97 % on cropped leaf benchmarks; on PlantDoc
  specifically, cropping lifted a VGG16+PlantVillage model from 44.5 % → **60.4 %**.
- FPGA cost: a green-segmentation crop is a few line buffers. A nano detector is another small INT8 CNN
  (< 50K params) — still fits XC7A200T alongside the classifier.

### B. Training data expansion (pair with A)
- Add **PlantVillage** (54k lab images, 38 classes → map the 27–28 overlapping PlantDoc classes) and/or
  **web-scraped** images per class, **as training data only**; keep the PlantDoc test set untouched.
- Raw mixing underperforms because of domain gap — mitigate with:
  - **Adaptive BatchNorm** (recompute BN stats on PlantDoc-style data) — cheap, +2–4 pts reported.
  - **Feature moment matching** / CORAL between the two domains — +2–5 pts.
  - **Contrastive / self-supervised pretraining** (SimCLR/PlantCLR-style) on the union, then fine-tune
    on PlantDoc — learns lesion-centric features less tied to dataset cues.
- Heavy **domain-randomization augmentation** on PlantVillage (random backgrounds, cutout, color) to
  close the gap.

### C. Resolution 64 → 96 or 112
- Retrain `PlantEdgeNet` at 96×96 (`--img-size 96`). Largest activation becomes 48×48×C — still fits
  BRAM on XC7A200T; DSP/latency rise ~2.2× but you have headroom (7 M → ~16 M MACs).
- Expected +3–6 pts. 112 gives a bit more; 128 starts to pressure BRAM.

### D. Knowledge distillation — properly
Current `distill.py` does logit-KD only. Add:
- **A strong teacher first**: fine-tune ConvNeXt-Base or EfficientNet-B3 on PlantDoc (+ PlantVillage) to
  **75–80 %**. Student ceiling ≈ teacher − a few pts, so a 67 % DINOv2 teacher caps you low.
- **Feature distillation**: match an intermediate student feature map (1×1 conv projection) to a teacher
  feature via MSE — the standard "review KD" / FitNets idea; +1–3 pts for small students.
- **DIST (correlation-based) KD**: match inter-class and intra-class *relations* of logits instead of
  exact values — more robust when teacher≫student; +1–2 pts.
- **Multi-teacher**: average soft targets from ConvNeXt + EfficientNet + DINOv2.
- KD on PlantVillage/web images too (teacher provides labels for unlabeled web data).

### E. QAT for the INT8 step (replace PTQ as the shipped path)
- `fpga/quantize_qat.py` (Brevitas) already exists. On depthwise nets, QAT typically **recovers 1–3 pts**
  vs PTQ. `pip install brevitas qonnx` into `.venv-1` and run QAT from the best FP32 checkpoint.
- Keep PTQ (AdaRound + bias-correction, `fpga/ptq.py`) as the ablation/baseline for the paper.
- Per-channel symmetric INT8 weights + 99.9-percentile activation calibration are already in place.

### F. Relax params to ~250–500K (fallback if A–E miss 80 %)
- `--width` up to ~1.75 is the current < 100K cap. A 250–500K depthwise net (≈ MCUNet scale) is still
  trivially an FPGA fit (0.25–0.5 MB INT8 weights ≪ 1.6 MB BRAM) and buys +5–10 pts.
- This is the cleanest way to *guarantee* 80 %+ while staying "tiny / FPGA-deployable". Reframe the paper
  claim as "< 0.5 M params" or "< 512 KB INT8".

### Architecture notes (cheap upgrades, no real param cost)
- **Squeeze-and-Excite** blocks (channel attention) — ~1–2K params, +1–2 pts, INT8-friendly.
- **RepVGG-style reparameterization**: train with multi-branch, fold to plain 3×3 for inference — free
  accuracy, better for FPGA (only 3×3 convs).
- **h-swish → keep ReLU6** (already done; h-swish barely helps and costs fixed-point ops).
- Consider a **TinyNAS-searched** stem/head under a 100K + Artix-7 LUT/DSP constraint (NASH-style) —
  higher effort, ~+2–4 pts.

## 3. Concrete recipe (recommended)

```
Stage 1  Teacher:   fine-tune ConvNeXt-Base (timm) on PlantDoc+PlantVillage → aim 75–80% PlantDoc test
Stage 2  Data:      build train set = PlantDoc-train  ∪  PlantVillage(mapped, domain-randomized)  ∪  web
                    crop every image to leaf ROI using PlantDoc-Object-Detection boxes (train/val)
Stage 3  Student:   PlantEdgeNet, --img-size 96, --width ~1.75 (<100K), + SE blocks
                    pretrain on the union (contrastive or supervised), then
                    KD fine-tune on cropped PlantDoc: logit-KD (T=4) + feature-MSE + DIST, multi-teacher
                    recipe: 200–300 epochs, cosine, EMA(warmup), label-smoothing 0.1, MixUp+CutMix,
                            adaptive-BN recalibration on PlantDoc before eval
Stage 4  Inference: leaf-ROI crop (nano-detector or HSV green-segmentation) → PlantEdgeNet → TTA(flip)
Stage 5  INT8:      Brevitas QAT (per-channel sym weights, INT8 acts), 20–30 epochs, export QONNX
Stage 6  FPGA:      FINN/hls4ml → Vivado (xc7a200tfbg484-2); detector + classifier both on-chip INT8
```

**Projected accuracy (PlantDoc test, honest split):**

| Configuration | FP32 | INT8 (QAT) |
|---|---|---|
| current (64px, PTQ, no KD, no crop) | ~55–65 % | ~53–63 % |
| + proper KD + 96px + EMA fix | ~66–72 % | ~65–71 % |
| + leaf-ROI crop | ~72–78 % | ~71–77 % |
| + PlantVillage/web data + adaptive-BN | **~76–82 %** | **~75–81 %** |
| + relax to ~300K params (lever F) | ~80–85 % | ~79–84 % |

## 4. What to change in the repo

| File | Change |
|---|---|
| `fpga/train_fpga.py` | add `--img-size 96` runs; add SE blocks to `model_tiny.py`; add adaptive-BN recalibration pass before eval; **use QAT output as the shipped model** |
| `fpga/model_tiny.py` | optional `--se` (squeeze-excite), allow `--width` beyond 1.75 behind a `--allow-large` flag for lever F |
| `fpga/distill.py` | add feature-MSE hook + DIST loss + multi-teacher list |
| new `fpga/prepare_data.py` | crop `PlantDoc-Object-Detection-Dataset` boxes → `PlantDoc-Cropped/{train,test}`; map + domain-randomize PlantVillage; optional web scrape |
| new `fpga/leaf_roi.py` | HSV green-segmentation crop (FPGA-friendly) + optional nano-YOLO detector for inference |
| `fpga/quantize_qat.py` | already present — `pip install brevitas qonnx`, run it |
| `research/IMPLEMENTATION_PLAN.md` | insert Stage 2 (data/crop) and Stage 4 (ROI inference) |

## 5. Decision needed

To commit to a path, pick which constraint(s) you'll flex:

1. **Keep < 100K params, honest PlantDoc test** → realistic target **~75–80 %**, 80 % not guaranteed.
   Requires A + B + C + D + E (crop + data + 96px + KD + QAT).
2. **Keep < 100K params, allow leaf-crop + external training data** → **80 % plausible**.
3. **Relax to < ~300–500K params** (still FPGA-tiny) → **80 %+ comfortable**, least effort.
4. **Relax test protocol** (augmented / k-fold merged, like the 82–90 % papers) → 80 %+ trivially, but
   weaker paper claim.

My recommendation: **option 2** (crop + external data, params stay < 100K), fall back to **option 3** if
the INT8 model lands at 77–79 %.

## 6. New papers downloaded to `Papers/`

| File | Use |
|---|---|
| `MCUNet_Tiny_Deep_Learning_on_IoT_Devices.pdf` | tiny-model accuracy ceiling; TinyNAS design principles |
| `MCUNetV2_Memory_Efficient_Patch_Based_Inference.pdf` | patch-based inference to run higher resolution in tiny memory |
| `Tiny_Machine_Learning_Progress_and_Futures.pdf` | survey: tiny-model training tricks, QAT, NAS |
| `Leaf_Diseases_Detection_Using_Deep_Learning_Methods.pdf` | detect-then-classify pipeline for leaf disease |
| `Bridging_Domain_Gaps_Agricultural_Image_Analysis_Review.pdf` | domain-adaptation methods (adaptive-BN, CORAL, contrastive) for PlantVillage→field |
| (earlier) `White_Paper_on_Neural_Network_Quantization_Nagel.pdf`, `Quantizing_Deep_ConvNets...`, `Integer_Quantization_Principles...` | QAT recipe, per-channel, AdaRound |
| (earlier) `Mobile_Friendly_DL_Plant_Disease_Detection.pdf`, `Comparative_Lightweight_DL_Memory_Constrained.pdf` | lightweight PlantDoc baselines + KD |

Blocked (cite from abstract): RTR_Lite_MobileNetV2 (ScienceDirect S2214662825000271, 82.0 % PlantDoc);
"Plant disease classification in the wild … mixture of experts" (PMC12213485); PlantCLR contrastive
pretraining (Nature s41598-026-45684-x); Frontiers "Quantifying the Reliability Gap …" (fpls.2026.1826962).

## 7. Sources

- Evaluation of lightweight and efficient DL models for plant disease classification — https://link.springer.com/article/10.1007/s43926-026-00310-0
- RTR_Lite_MobileNetV2 (82.0 % PlantDoc) — https://www.sciencedirect.com/science/article/pii/S2214662825000271
- Optimised MobileNet for very lightweight and accurate plant leaf disease detection — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12700863/
- Plant disease classification in the wild using ViT + mixture of experts — https://pmc.ncbi.nlm.nih.gov/articles/PMC12213485/
- Quantifying the Reliability Gap in Cross-Domain Plant Disease Classification — https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1826962/abstract
- PlantCLR: contrastive self-supervised pretraining for generalizable plant disease detection — https://www.nature.com/articles/s41598-026-45684-x
- Bridging Domain Gaps in Agricultural Image Analysis (review) — https://arxiv.org/html/2506.05972v2
- Addressing domain shift in deep learning: plant disease diagnosis — https://www.researchgate.net/publication/384859832
- Knowledge distillation in plant disease recognition — https://link.springer.com/article/10.1007/s00521-021-06882-y
- Knowledge Distillation Facilitates the Lightweight … Plant Diseases Detection Model — https://spj.science.org/doi/10.34133/plantphenomics.0062
- Semantic segmentation for plant leaf disease classification and damage detection — https://www.sciencedirect.com/science/article/pii/S277237552400131X
- Leaf diseases detection using deep learning methods — https://arxiv.org/pdf/2501.00669
- MCUNet: Tiny Deep Learning on IoT Devices — https://arxiv.org/abs/2007.10319
- MCUNetV2: Memory-Efficient Patch-based Inference — https://arxiv.org/pdf/2110.15352
- Tiny Machine Learning: Progress and Futures — https://arxiv.org/pdf/2403.19076
- Identification of leaf diseases in field crops based on improved ShuffleNetV2 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10961419/
- QAT vs PTQ overview — https://medium.com/better-ml/quantization-aware-training-qat-vs-post-training-quantization-ptq-cd3244f43d9a
- Quantization of CNNs: Model Quantization (PTQ hurts MobileNets/ShuffleNets) — https://www.edge-ai-vision.com/2024/02/quantization-of-convolutional-neural-networks-model-quantization/
- Benchmarking In-the-Wild Multimodal Plant Disease Recognition (EfficientNet-B3 73.3 %, combined 80.2 %) — https://arxiv.org/html/2408.03120v1
- Assessing domain-specific models for plant leaf disease classification (transfer-learning benchmark) — https://www.nature.com/articles/s41598-025-03235-w
- Transfer learning for versatile plant disease recognition with limited data — https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1010981/full
- AgriNet: the power of transfer learning in agricultural applications — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9794606/
- Image resolution effects (GPT-4o vs ResNet-50 across resolutions) — https://arxiv.org/pdf/2504.20419
