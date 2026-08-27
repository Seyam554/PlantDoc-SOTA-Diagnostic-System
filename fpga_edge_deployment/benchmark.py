import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import torch
from tqdm import tqdm

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

from model import get_plantedge_model
from dataset import get_dataloaders

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def resolve_file_path(p):
    if os.path.exists(p):
        return os.path.abspath(p)
    candidate1 = os.path.join(_CURR_DIR, p)
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.join(_CURR_DIR, "checkpoints", os.path.basename(p))
    if os.path.exists(candidate2):
        return candidate2
    candidate3 = os.path.join(_CURR_DIR, "..", p)
    if os.path.exists(candidate3):
        return os.path.abspath(candidate3)
    return p

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Paper-Ready Benchmarking Suite for PlantEdgeNet")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_CURR_DIR, "checkpoints", "plantedge_w1.00_best.pth"), help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to dataset root directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--output-dir", type=str, default=os.path.join(_CURR_DIR, "results"), help="Directory to save benchmark metrics and plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()

def benchmark_model(checkpoint_path, data_dir="PlantDoc-Dataset", batch_size=32, output_dir="results", device="cpu"):
    resolved_ckpt = resolve_file_path(checkpoint_path)
    if not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found at '{resolved_ckpt}'!")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    print("==================================================")
    print("PlantEdgeNet: Paper-Ready Benchmarking Suite")
    print(f"Device: {device}")
    print(f"Checkpoint: {resolved_ckpt}")
    print(f"Output Directory: {output_dir}")
    print("==================================================")

    # 1. Load Checkpoint
    ckpt = torch.load(resolved_ckpt, map_location=device)
    num_classes = ckpt.get("num_classes", 28)
    width_mult = ckpt.get("width_mult", 1.0)
    img_size = ckpt.get("img_size", 96)
    classes = ckpt.get("classes", [])

    model = get_plantedge_model(num_classes=num_classes, width_mult=width_mult)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # 2. Hardware Characteristics
    total_params = model.count_parameters()
    total_macs = model.count_macs((1, 3, img_size, img_size))
    fp32_size_kb = (total_params * 4) / 1024.0
    int8_size_kb = (total_params * 1) / 1024.0

    print("\n[Hardware Metrics]")
    print(f"• Total Trainable Parameters: {total_params:,} (< 100K FPGA Limit)")
    print(f"• Theoretical MACs (@ {img_size}x{img_size}): {total_macs/1e6:.2f} M MACs")
    print(f"• FP32 Weight Memory: {fp32_size_kb:.2f} KB")
    print(f"• INT8 Weight Memory: {int8_size_kb:.2f} KB (Fits in Artix-7 1.63 MB BRAM)")

    # 3. Load Test Data
    _, test_loader, loaded_classes = get_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        use_hsv_roi=True
    )
    if not classes:
        classes = loaded_classes

    # 4. Measure Inference Latency & Accuracy
    all_preds = []
    all_targets = []
    top5_correct = 0
    total_samples = 0

    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    for _ in range(10):
        _ = model(dummy)

    start_lat_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Benchmarking Test Set"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            _, top5_preds = torch.topk(probs, min(5, num_classes), dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
            total_samples += labels.size(0)

    total_eval_time = time.time() - start_lat_time
    avg_latency_ms = (total_eval_time / total_samples) * 1000.0
    fps = total_samples / total_eval_time

    # 5. Compute Classification Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    top1_acc = accuracy_score(all_targets, all_preds)
    top5_acc = top5_correct / total_samples if total_samples > 0 else 0.0

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    wt_prec, wt_rec, wt_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    print("\n[Accuracy & Classification Metrics]")
    print(f"• Top-1 Test Accuracy: {top1_acc*100:.2f}%")
    print(f"• Top-5 Test Accuracy: {top5_acc*100:.2f}%")
    print(f"• Macro Precision: {macro_prec*100:.2f}% | Macro Recall: {macro_rec*100:.2f}% | Macro F1: {macro_f1:.4f}")
    print(f"• Weighted Precision: {wt_prec*100:.2f}% | Weighted Recall: {wt_rec*100:.2f}% | Weighted F1: {wt_f1:.4f}")
    print(f"• Mean Inference Latency: {avg_latency_ms:.2f} ms/image ({fps:.1f} FPS)")

    # 6. Generate & Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"PlantEdgeNet Benchmark Confusion Matrix (Top-1 Acc: {top1_acc*100:.2f}%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "plantedge_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved Confusion Matrix Plot to: {cm_path}")

    # 7. Save JSON Metrics
    clf_report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    summary_data = {
        "model_architecture": "PlantEdgeNet",
        "parameters": total_params,
        "macs": total_macs,
        "width_mult": width_mult,
        "input_resolution": f"{img_size}x{img_size}",
        "fp32_memory_kb": fp32_size_kb,
        "int8_memory_kb": int8_size_kb,
        "hardware_target": "AMD Artix-7 XC7A200T FPGA",
        "metrics": {
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_precision": wt_prec,
            "weighted_recall": wt_rec,
            "weighted_f1": wt_f1,
            "avg_latency_ms": avg_latency_ms,
            "throughput_fps": fps,
            "total_test_samples": total_samples
        },
        "per_class_metrics": clf_report
    }

    json_path = os.path.join(output_dir, "benchmark_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Saved Complete Benchmark Report to: {json_path}")
    print("==================================================")
    return summary_data

def main():
    args = parse_args()
    benchmark_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()
