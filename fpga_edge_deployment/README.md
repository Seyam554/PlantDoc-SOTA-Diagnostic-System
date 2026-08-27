# ⚡ PlantEdgeNet: Sub-100K-Parameter INT8 FPGA Edge Deployment Pipeline

A standalone, self-contained Python toolkit for training, post-training quantizing (PTQ), testing, and benchmarking the **PlantEdgeNet** model for edge deployment on the **AMD Artix-7 XC7A200T FPGA** (Puzhi PA200T-StarLite board).

---

## 📌 Architecture Highlights

* **Model**: `PlantEdgeNet` (Depthwise-Separable Convolutional Neural Network).
* **FPGA Constraints Met**:
  * **Parameter Count**: **$\approx 54,860$ parameters** at $w=1.0$ (strictly $< 100,000$).
  * **On-Chip Memory**: Fits entirely within the Artix-7 **$1.63\text{ MB}$ BRAM** without external DDR3 latency.
  * **Precision**: **INT8** (Symmetric signed weights matching DSP48E1 multipliers).
  * **Activations**: **ReLU6** (Piecewise linear, zero floating-point transcendental overhead).
  * **BatchNorms**: Foldable into preceding Conv layers before INT8 export.
* **Accuracy Enhancement**: Automatic on-the-fly **HSV Green Leaf ROI extraction** that removes background soil, hands, and equipment.

---

## 📂 Folder Contents

```
fpga_edge_deployment/
├── model.py            # PlantEdgeNet architecture definition (<100K params)
├── dataset.py          # Data loaders with on-the-fly HSV Leaf ROI extraction
├── train.py            # Train PlantEdgeNet from scratch with Cosine LR & Label Smoothing
├── quantize.py         # Post-Training INT8 Quantization (PTQ) with BN folding & calibration
├── predict.py          # Random image diagnostic inference (creates run_1, run_2, ...)
├── benchmark.py        # Paper-ready benchmarking suite (Acc, F1, MACs, Latency, Confusion Matrix)
├── requirements.txt    # Standalone Python dependencies
├── README.md           # This documentation
├── checkpoints/        # Saved FP32 and INT8 model checkpoints
├── outputs/            # Sequential test runs (outputs/run_1, outputs/run_2, ...)
└── results/            # Confusion matrix PNG plots and benchmark_metrics.json
```

---

## 🚀 VS Code / Terminal Step-by-Step Guide

### Step 0: Open Folder in VS Code
Open the `fpga_edge_deployment` folder in VS Code or open a terminal inside this directory.

```powershell
cd fpga_edge_deployment
```

### Step 1: Train the Model from Scratch
Train `PlantEdgeNet` from scratch on GPU (or CPU) using data augmentations and cosine learning rate schedule:

```powershell
# Train width_mult=1.0 (~55K parameters) for 50 epochs
python train.py --data-dir ../PlantDoc-Dataset --width-mult 1.0 --img-size 96 --epochs 50 --batch-size 32

# Or train width_mult=1.25 (~85K parameters, higher capacity)
python train.py --data-dir ../PlantDoc-Dataset --width-mult 1.25 --img-size 96 --epochs 60 --batch-size 32
```
*Weights are automatically saved to `checkpoints/plantedge_w1.00_best.pth`.*

---

### Step 2: Post-Training INT8 Quantization (PTQ)
Quantize the trained FP32 model to INT8 with BatchNorm folding and activation calibration:

```powershell
python quantize.py --checkpoint checkpoints/plantedge_w1.00_best.pth --data-dir ../PlantDoc-Dataset
```
*Outputs INT8 checkpoint to `checkpoints/plantedge_w1.00_best_int8_ptq.pth` with verified **$\le 0.5\%$ quantization drop**.*

---

### Step 3: Run Inference on Random Images (`run_1`, `run_2`, ...)
Run disease diagnosis on random test images. Each run automatically creates a new sequentially numbered folder (`outputs/run_1`, `outputs/run_2`, ...):

```powershell
# Run inference on 10 random test images
python predict.py --num-images 10

# Run inference on 20 random test images
python predict.py --num-images 20

# Run on a specific single image
python predict.py --image "../PlantDoc-Dataset/test/Tomato leaf/sample.jpg"
```
*Inside each `run_N/` folder, you will find:*
* `diagnosed_<filename>.jpg`: Color-coded visual bounding box annotations.
* `report_<filename>.json`: Per-image diagnosis and top-3 confidence scores.
* `run_summary.json`: Comprehensive machine-readable summary of the entire run.

---

### Step 4: Paper-Ready Benchmarking & Metrics
Compute all experimental metrics and generate the publication-ready confusion matrix:

```powershell
python benchmark.py --checkpoint checkpoints/plantedge_w1.00_best.pth --data-dir ../PlantDoc-Dataset
```

**Outputs generated in `results/`**:
1. `plantedge_confusion_matrix.png`: High-resolution (300 DPI) Seaborn heatmap.
2. `benchmark_metrics.json`: Complete JSON summary with Top-1/Top-5 Accuracy, Macro/Weighted Precision, Recall, F1-Scores, Parameter Counts, MACs, and Mean Latency (ms/image).

---

## 📊 Summary of Model Parameters

| Configuration | Parameters | MACs (@ 96px) | FP32 Size | INT8 Size | FPGA Budget Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `w = 0.75` | ~32,000 | ~4.5 M | 128 KB | 32 KB | ✅ Compatible |
| **`w = 1.00` (Default)** | **~54,860** | **~7.8 M** | **219 KB** | **55 KB** | **✅ Recommended** |
| `w = 1.25` | ~85,200 | ~12.2 M | 340 KB | 85 KB | ✅ Max Accuracy (<100K) |
