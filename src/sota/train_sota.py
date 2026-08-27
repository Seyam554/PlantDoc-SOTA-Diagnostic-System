import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset_sota import get_sota_dataloaders
from models_sota import get_sota_model

# Ensure safe printing on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SOTA PlantDoc Models")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Dataset directory")
    parser.add_argument("--model", type=str, default="dinov2_vits14", help="Model: dinov2_vits14, dinov2_vitb14, convnext_base.fb_in22k_ft_in1k, swin_base_patch4_window7_224.ms_in22k_ft_in1k")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--img-size", type=int, default=518, help="Image resolution (518 for DINOv2)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--save-dir", type=str, default="checkpoints_sota", help="Save directory")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = nn.functional.nll_loss(log_preds, target, reduction='mean')
        return (1.0 - self.smoothing) * nll + (self.smoothing / n_classes) * loss

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Val", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"SOTA Plant Disease Training (Target: >=92%)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model: {args.model} | Input Size: {args.img_size}x{args.img_size}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | Label Smoothing: {args.label_smoothing}")
    print(f"LR Backbone: {args.lr_backbone} | LR Head: {args.lr_head}")
    print(f"==================================================")

    # Dataloaders
    train_loader, test_loader, classes = get_sota_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers
    )
    num_classes = len(classes)
    print(f"Dataset: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples across {num_classes} classes.")

    # Save classes
    with open(os.path.join(args.save_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    # Initialize model
    model = get_sota_model(arch=args.model, num_classes=num_classes)
    model = model.to(device)

    # Differential parameter groups
    if hasattr(model, "backbone") and hasattr(model, "head"):
        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head}
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr_head}]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    scaler = GradScaler()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    clean_model_name = args.model.replace("/", "_").replace(".", "_")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "best_acc": best_val_acc,
            "classes": classes,
            "args": vars(args)
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"{clean_model_name}_last.pth"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(args.save_dir, f"{clean_model_name}_best.pth"))
            print(f"  --> Saved new best checkpoint with Val Acc: {best_val_acc*100:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    with open(os.path.join(args.save_dir, f"{clean_model_name}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
