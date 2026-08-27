# References — PlantEdgeNet-FPGA

The papers this design is built on. Full per-claim citations are in
`research/RESEARCH_REPORT.md` §6–7 and `research/ACCURACY_TO_80.md` §6–7.
(PDFs are not committed to keep the repo small — download from the links below.)

## Quantization (INT8 PTQ / QAT best practice)

- **A White Paper on Neural Network Quantization** — Nagel et al., 2021 — https://arxiv.org/abs/2106.08295
  (CLE, bias correction, AdaRound, per-channel weights — the canonical PTQ/QAT reference)
- **Quantizing deep convolutional networks for efficient inference: A whitepaper** — Krishnamoorthi, 2018 — https://arxiv.org/abs/1806.08342
- **Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation** — NVIDIA, 2020 — https://arxiv.org/abs/2004.09602
- **Quantized Neural Networks for Low-Precision Accumulation with Guaranteed Overflow Avoidance** — 2023 — https://arxiv.org/abs/2301.13376
- Model Quantization overview (PTQ hurts MobileNets/ShuffleNets) — https://www.edge-ai-vision.com/2024/02/quantization-of-convolutional-neural-networks-model-quantization/

## FPGA deployment of quantized CNNs

- **Benchmarking Quantized Neural Networks on FPGAs with FINN** — 2021 — https://arxiv.org/abs/2102.01341
- **Fast convolutional neural networks on FPGAs with hls4ml** — 2021 — https://arxiv.org/abs/2101.05108
- **Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark** — 2022 — https://arxiv.org/abs/2206.11791
- **A CNN Accelerator on FPGA Using Depthwise Separable Convolution** — 2018 — https://arxiv.org/abs/1809.01536
- **DeepDive: Algorithm/Architecture Co-Design for Deep Separable CNNs** — 2020 — https://arxiv.org/abs/2007.09490
- **NASH: Neural Architecture Search for Hardware-Optimized ML Models** — 2024 — https://arxiv.org/abs/2403.01845
- Brevitas → QONNX → FINN flow — https://finn.readthedocs.io/en/latest/brevitas_export.html · https://xilinx.github.io/finn/quickstart.html
- Integer Intelligence: A Reproducible Path from Training to FPGA (Artix-7 CMOD A7 INT8 datapath) — https://doi.org/10.3390/electronics15051117

## Tiny models / knowledge distillation

- **MCUNet: Tiny Deep Learning on IoT Devices** — 2020 — https://arxiv.org/abs/2007.10319
- **MCUNetV2: Memory-Efficient Patch-based Inference** — 2021 — https://arxiv.org/abs/2110.15352
- **Tiny Machine Learning: Progress and Futures** — 2024 — https://arxiv.org/abs/2403.19076
- **Cross-Architecture Knowledge Distillation** — ACCV 2022 — https://openaccess.thecvf.com/content/ACCV2022/papers/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.pdf
- **Knowledge Distillation Facilitates the Lightweight and Efficient Plant Diseases Detection Model** — Plant Phenomics 2023 — https://spj.science.org/doi/10.34133/plantphenomics.0062
- Knowledge distillation in plant disease recognition — https://link.springer.com/article/10.1007/s00521-021-06882-y
- Efficient Deep Learning Infrastructures for Embedded Computing Systems (survey) — https://arxiv.org/abs/2411.01431

## PlantDoc / in-the-wild plant disease

- **PlantDoc: A Dataset for Visual Plant Disease Detection** — Singh et al., CoDS-COMAD 2020 — https://arxiv.org/abs/1911.10317
- **Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline** — 2024 (EfficientNet-B3 73.3 %, combined 80.2 %) — https://arxiv.org/abs/2408.03120
- RTR_Lite_MobileNetV2 (82.0 % PlantDoc, ~2 M params + aug) — https://www.sciencedirect.com/science/article/pii/S2214662825000271
- Comparative Analysis of Lightweight DL Models for Memory-Constrained Devices — https://arxiv.org/abs/2505.03303
- Mobile-Friendly Deep Learning for Plant Disease Detection — https://arxiv.org/abs/2508.10817
- Bridging Domain Gaps in Agricultural Image Analysis (adaptive-BN, CORAL, contrastive) — https://arxiv.org/html/2506.05972v2
- Leaf diseases detection using deep learning methods (detect-then-classify) — https://arxiv.org/abs/2501.00669
- A lightweight and explainable CNN model for empowering plant disease diagnosis — https://www.nature.com/articles/s41598-025-94083-1

## Teacher architectures

- **DINOv2: Learning Robust Visual Features without Supervision** — Oquab et al., 2023 — https://arxiv.org/abs/2304.07193
- **ConvNeXt V2: Co-designing and Scaling ConvNets** — Woo et al., CVPR 2023 — https://arxiv.org/abs/2301.00808
- **Swin Transformer: Hierarchical Vision Transformer** — Liu et al., ICCV 2021 — https://arxiv.org/abs/2103.14030

## Target board

- Puzhi PA200T-StarLite — https://www.en.puzhi.com/Product/AMD-FPGA-Development-Board/Artix-7/PA200T-StarLite · https://www.en.puzhi.com/detail/429.html
- Xilinx XC7A200T specifications — https://pcbsync.com/artix-7-200t/
