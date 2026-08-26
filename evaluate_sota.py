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
from dataset_sota import get_sota_dataloaders
from models_sota import get_sota_model

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SOTA PlantDoc Models")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_sota/dinov2_vits14_best.pth", help="Checkpoint path")
    parser.add_argument("--model", type=str, default=None, help="Model architecture override")
    parser.add_argument("--img-size", type=int, default=518, help="Image resolution")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--use-tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--save-dir", type=str, default="results_sota", help="Results save directory")
    return parser.parse_args()

@torch.no_grad()
def evaluate_model(model, test_loader, device, use_tta=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in tqdm(test_loader, desc="Evaluating (TTA)" if use_tta else "Evaluating"):
        images = images.to(device, non_blocking=True)

        if use_tta:
            # Multi-view Test-Time Augmentation (Original + Horizontal Flip)
            images_flipped = torch.flip(images, dims=[-1])
            logits1 = model(images)
            logits2 = model(images_flipped)
            probs = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2.0
        else:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"SOTA PlantDoc Model Evaluation Pipeline")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"TTA Enabled: {args.use_tta}")
    print(f"==================================================")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model_name", "dinov2_vits14")
    classes = checkpoint.get("classes", None)

    # Load dataloaders
    _, test_loader, dataloader_classes = get_sota_dataloaders(
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
    model = get_sota_model(arch=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # Run evaluation
    preds, targets, probs = evaluate_model(model, test_loader, device, use_tta=args.use_tta)

    acc = accuracy_score(targets, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(targets, preds, average="weighted", zero_division=0)

    clean_name = model_name.replace("/", "_").replace(".", "_")

    print(f"\n" + "="*50)
    print(f"OVERALL SOTA BENCHMARK RESULTS ({model_name.upper()})")
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

    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.title(f"PlantDoc Confusion Matrix - {model_name.upper()} (Acc: {acc*100:.2f}%)", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(args.save_dir, f"{clean_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Save metrics JSON
    results = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "use_tta": args.use_tta,
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

    metrics_path = os.path.join(args.save_dir, f"{clean_name}_benchmark_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
