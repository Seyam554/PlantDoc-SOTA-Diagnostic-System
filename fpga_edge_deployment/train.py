import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

from model import get_plantedge_model
from dataset import get_dataloaders

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Train PlantEdgeNet for FPGA Edge Deployment from Scratch")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to dataset root directory")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier (0.75=~32K, 1.0=~55K, 1.25=~85K)")
    parser.add_argument("--img-size", type=int, default=96, help="Input image resolution (96x96 recommended)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--use-hsv-roi", action="store_true", default=True, help="Extract leaf ROI to remove background")
    parser.add_argument("--teacher-weights", type=str, default=None, help="Optional teacher checkpoint for Knowledge Distillation")
    parser.add_argument("--save-dir", type=str, default=os.path.join(_CURR_DIR, "checkpoints"), help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device, teacher_model=None, kd_temp=4.0, kd_alpha=0.3):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Knowledge Distillation if teacher is provided
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            kd_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(outputs / kd_temp, dim=1),
                torch.softmax(teacher_outputs / kd_temp, dim=1)
            ) * (kd_temp ** 2)
            loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    args = parse_args()
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("==================================================")
    print("PlantEdgeNet: FPGA-Targeted Lightweight Training")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Width Multiplier: {args.width_mult} | Resolution: {args.img_size}x{args.img_size}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Save Directory: {save_dir}")
    print("==================================================")

    # 1. Load Data
    train_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        use_hsv_roi=args.use_hsv_roi
    )
    num_classes = len(class_names)

    # 2. Build Model
    model = get_plantedge_model(num_classes=num_classes, width_mult=args.width_mult)
    model = model.to(device)
    params = model.count_parameters()
    macs = model.count_macs((1, 3, args.img_size, args.img_size))
    print(f"Model Initialized: {params:,} Parameters (< 100K FPGA Limit Compliant)")
    print(f"Theoretical Compute: {macs/1e6:.2f} M MACs per inference\n")

    # 3. Optional Teacher Model
    teacher_model = None

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = -1.0
    best_ckpt_path = os.path.join(save_dir, f"plantedge_w{args.width_mult:.2f}_best.pth")
    last_ckpt_path = os.path.join(save_dir, f"plantedge_w{args.width_mult:.2f}_latest.pth")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, teacher_model)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_accuracy": val_acc,
            "train_accuracy": train_acc,
            "width_mult": args.width_mult,
            "img_size": args.img_size,
            "num_classes": num_classes,
            "classes": class_names,
            "params": params,
            "macs": macs
        }

        # Always save latest
        torch.save(save_dict, last_ckpt_path)

        is_best = val_acc >= best_acc
        if is_best:
            best_acc = val_acc
            save_dict["best_accuracy"] = best_acc
            torch.save(save_dict, best_ckpt_path)

        mark = "★ BEST" if is_best else ""
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% {mark}")

    total_time = time.time() - start_time
    print("\n==================================================")
    print(f"Training Complete in {total_time/60:.2f} minutes!")
    print(f"Peak Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Best Model Weights Saved to: {best_ckpt_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
