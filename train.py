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

from dataset import get_dataloaders
from models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train PlantDoc Classification Models")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to PlantDoc-Dataset")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg16", "resnet50", "mobilenet_v2", "inception_v3"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw", "adam"], help="Optimizer")
    parser.add_argument("--img-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet pretrained weights")
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, is_inception=False):
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
            if is_inception:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"PlantDoc Model Training Pipeline")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Image Size: {args.img_size}")
    print(f"Optimizer: {args.optimizer} | LR: {args.lr}")
    print(f"==================================================")

    # Load data
    train_loader, test_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers
    )
    num_classes = len(classes)
    print(f"Loaded {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples across {num_classes} classes.")

    # Save class names mapping
    with open(os.path.join(args.save_dir, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    # Initialize model
    is_inception = (args.model == "inception_v3")
    model = get_model(args.model, num_classes=num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, is_inception=is_inception)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "best_acc": best_val_acc,
            "classes": classes,
            "args": vars(args)
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"{args.model}_last.pth"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(args.save_dir, f"{args.model}_best.pth"))
            print(f"  --> Saved new best checkpoint with Val Acc: {best_val_acc*100:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation/Test Accuracy: {best_val_acc*100:.2f}%")

    with open(os.path.join(args.save_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
