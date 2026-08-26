# 🌿 PlantDoc-SOTA: 3-Stage Plant Disease Diagnostic & Detection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.13 (CUDA 13.0)](https://img.shields.io/badge/PyTorch-2.13%20%7C%20CUDA%2013.0-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg?style=flat&logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![DINOv2 ViT](https://img.shields.io/badge/DINOv2-Vision%20Transformer-0080FF.svg?style=flat&logo=meta&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end computer vision system designed for **in-the-wild plant disease detection and fine-grained classification** on the benchmark **PlantDoc Dataset** (Singh et al., CoDS-COMAD 2020), addressing real-world farm conditions with complex backgrounds, multi-leaf scenes, and variable lighting.

---

## 📌 1. System Architecture

The diagnostic framework implements a **3-stage synergistic pipeline**:

```
 ┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
 │ Raw In-the-Wild Photo  │ ──►  │ Stage 1: YOLOv11 Leaf  │ ──►  │ Stage 2: DINOv2 ViT    │ ──► Final Diagnosis &
 │ (Complex Background)   │      │ Localization & Detector│      │ Fine-Grained Classifier│     Annotated Visuals
 └────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

1. **Stage 1: Leaf Localization & Noise Rejection (YOLOv11)**
   * Localizes individual leaves across multi-leaf images.
   * Filters out background noise (soil, sunlight glare, hands, equipment).
2. **Stage 2: Fine-Grained Disease Classification (DINOv2 ViT-S/14)**
   * High-resolution ($518 \times 518$) patch token analysis using Meta's self-supervised LVD-142M foundation features.
   * Layer-wise differential learning rates ($10^{-5}$ backbone, $5\times 10^{-4}$ head) with label smoothing ($\epsilon=0.1$).
3. **Stage 3: Diagnostic Aggregation & Test-Time Augmentation (TTA)**
   * Multi-view voting (original + horizontal flip) and color-coded bounding box overlays.

---

## 📊 2. Experimental Benchmark Results

### A. Stage 1: Leaf Object Detection (mAP @ 0.50 IoU)

| Model Architecture | Published Source / Pretrained Weights | mAP @ 0.50 IoU | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv11 (Our Stage 1)** | **Ultralytics YOLOv11n (COCO + PlantDoc)** | **59.40%** | **41.90%** | **64.20%** |
| **Faster R-CNN (InceptionResNet-V2)** | *Singh et al. 2020 (COCO)* | *38.90%* | — | — |
| **Faster R-CNN (InceptionResNet-V2)** | *Singh et al. 2020 (iNaturalist)* | *36.10%* | — | — |
| **MobileNet SSD** | *Singh et al. 2020 (COCO)* | *32.80%* | — | — |

*Our YOLOv11 detector outperforms the paper's best detector by **+20.5% absolute mAP**.*

---

### B. Stage 2: Disease Classification on PlantDoc Test Split

| Model Architecture | Pretrained Backbone | Top-1 Test Accuracy | Weighted Precision | Macro F1-Score | Weighted F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DINOv2 (ViT-S/14)** | **Meta LVD-142M Self-Supervised** | **66.95%** *(77.97% Val)* | **73.73%** | **0.6399** | **0.6600** |
| **ResNet-50** | ImageNet-1k | **61.44%** | 64.35% | 0.5980 | 0.6151 |
| **VGG-16 (Ours)** | ImageNet-1k | **57.20%** | 61.58% | 0.5459 | 0.5715 |
| *Paper Published Baseline (Singh et al. 2020)* | ImageNet-1k | *44.52%* | — | *0.4400* | — |

---

## 📚 3. Research Papers

The `Papers/` folder contains key research papers grounding this work:

1. **`PlantDoc_Dataset_Visual_Plant_Disease_Detection.pdf`** (*Singh et al., CoDS-COMAD 2020*)
2. **`DINOv2_Learning_Robust_Visual_Features_Without_Supervision.pdf`** (*Oquab et al., Meta AI 2023*)
3. **`Swin_Transformer_Hierarchical_Vision_Transformer.pdf`** (*Liu et al., Microsoft Research, ICCV 2021*)
4. **`ConvNeXt_V2_Co-designing_and_Scaling_ConvNets.pdf`** (*Woo et al., Meta AI, CVPR 2023*)
5. **`PlantXViT_Explainable_Vision_Transformer_for_Plant_Disease.pdf`** (*arXiv:2207.07919*)

---

## 💻 4. Installation & Quickstart

### Prerequisites
* Python 3.10+
* NVIDIA GPU (tested on RTX 5070 Ti with CUDA 13.0)

### Setup Environment
```bash
# Clone this repository
git clone https://github.com/Seyam554/PlantDoc-SOTA-Diagnostic-System.git
cd PlantDoc-SOTA-Diagnostic-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 5. Usage Guide

### A. Download & Extract Dataset
```bash
python download_papers.py
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
python extract_dataset.py

git clone https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git
python extract_obj_detection.py
python voc_to_yolo.py
```

### B. Train Models
```bash
# 1. Train Stage 1 YOLOv11 Leaf Detector
python train_stage1_yolo.py --epochs 20 --batch 16 --imgsz 640

# 2. Train Stage 2 DINOv2 Vision Transformer
python train_sota.py --model dinov2_vits14 --img-size 518 --epochs 25 --batch-size 16 --lr-head 0.0005 --lr-backbone 0.00001
```

### C. Run End-to-End Inference
```bash
# Diagnose a single raw image
python pipeline_end_to_end.py --image "path/to/crop_photo.jpg"

# Run automated batch diagnosis on random test images (creates a new timestamped folder for each run)
python run_random_tests.py --num-images 10
```

---

## 📂 6. Repository Layout

```
├── dataset.py                # Base dataset loader & augmentations
├── dataset_sota.py           # High-resolution 518x518 RandAugment loader
├── models.py                 # Baseline models (VGG16, ResNet50, MobileNetV2)
├── models_sota.py            # SOTA architectures (DINOv2, ConvNeXt, Swin)
├── train.py                  # Baseline training loop (SGD / AdamW)
├── train_sota.py             # SOTA differential LR & label smoothing trainer
├── train_stage1_yolo.py      # Stage 1 YOLOv11 detector trainer
├── evaluate.py               # Baseline evaluation script
├── evaluate_sota.py          # SOTA evaluation script with TTA
├── voc_to_yolo.py            # Pascal VOC to YOLO converter
├── pipeline_end_to_end.py    # 3-Stage integrated diagnostic pipeline
├── run_random_tests.py       # Random test evaluator with timestamped output folders
├── download_papers.py        # Automated paper downloader
├── requirements.txt          # Python dependencies
├── Papers/                   # 5 full research papers in PDF format
├── results/                  # Confusion matrix plots and metrics (VGG16, ResNet50)
├── results_sota/             # Confusion matrix plots and metrics (DINOv2)
└── output/                   # Timestamped test run folders with annotated image outputs
```

---

## 📜 7. License & Acknowledgements
This project is licensed under the MIT License.
Special thanks to the authors of the **PlantDoc** dataset (*Singh et al., IIT Gandhinagar*) and the open-source computer vision communities behind **Ultralytics YOLO** and **Meta AI DINOv2**.
