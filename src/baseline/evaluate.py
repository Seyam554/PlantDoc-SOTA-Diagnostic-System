import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from dataset import get_dataloaders
from models import get_model

# Ensure safe printing on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PlantDoc Classification Model")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to PlantDoc-Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vgg16_best.pth", help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Model architecture override")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save evaluation results")
    return parser.parse_args()

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"PlantDoc Evaluation & Benchmark Pipeline")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"==================================================")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model_name", "vgg16")
    classes = checkpoint.get("classes", None)

    # Load test dataloader
    _, test_loader, dataloader_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=2
    )

    if classes is None:
        classes = dataloader_classes

    num_classes = len(classes)
    print(f"Target classes ({num_classes}): {classes}")
    print(f"Total test samples: {len(test_loader.dataset)}")

    # Initialize model
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # Evaluate
    preds, targets, probs = evaluate(model, test_loader, device)

    # Metrics
    acc = accuracy_score(targets, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(targets, preds, average="weighted", zero_division=0)

    print(f"\n" + "="*50)
    print(f"OVERALL BENCHMARK RESULTS ({model_name.upper()})")
    print(f"="*50)
    print(f"Top-1 Accuracy:     {acc * 100:.2f}%")
    print(f"Macro F1-Score:     {macro_f1:.4f} (Precision: {macro_p:.4f}, Recall: {macro_r:.4f})")
    print(f"Weighted F1-Score:  {weighted_f1:.4f} (Precision: {weighted_p:.4f}, Recall: {weighted_r:.4f})")
    print(f"="*50)

    # Classification report
    report_dict = classification_report(
        targets, preds, target_names=classes, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        targets, preds, target_names=classes, digits=4, zero_division=0
    )
    print("\nDetailed Per-Class Classification Report:")
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"PlantDoc Confusion Matrix - {model_name.upper()} (Acc: {acc*100:.2f}%)", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(args.save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Save metrics JSON
    results = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "test_samples": len(test_loader.dataset),
        "num_classes": num_classes,
        "classification_report": report_dict
    }

    metrics_path = os.path.join(args.save_dir, f"{model_name}_benchmark_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
