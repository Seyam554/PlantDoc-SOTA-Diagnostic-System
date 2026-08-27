import os
import sys
import copy
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchvision import models

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURR_DIR, ".."))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model import get_plantedge_model
from dataset import get_dataloaders

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="High-Accuracy Knowledge Distillation & Training for PlantEdgeNet")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to dataset root directory")
    parser.add_argument("--arch-type", type=str, default="inverted_residual", choices=["inverted_residual", "depthwise_separable"], help="Architecture type")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier (0.75=~58K, 1.0=~94K)")
    parser.add_argument("--img-size", type=int, default=96, help="Input image resolution (96x96 recommended)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--mixup-prob", type=float, default=0.3, help="Probability of applying MixUp augmentation")
    parser.add_argument("--teacher-weights", type=str, default=None, help="Path to teacher checkpoint for Knowledge Distillation (e.g. checkpoints_sota/dinov2_vits14_best.pth)")
    parser.add_argument("--kd-temp", type=float, default=4.0, help="Distillation temperature (higher = softer targets)")
    parser.add_argument("--kd-alpha", type=float, default=0.6, help="Weight of distillation loss vs hard target CE loss")
    parser.add_argument("--use-hsv-roi", action="store_true", default=True, help="Extract leaf ROI to remove background")
    parser.add_argument("--save-dir", type=str, default=os.path.join(_CURR_DIR, "checkpoints"), help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()

class TeacherWrapper(nn.Module):
    """
    Wraps large SOTA vision models (DINOv2 ViT, ResNet-50) for Knowledge Distillation.
    Automatically handles input resolution upsampling and frozen inference.
    """
    def __init__(self, weights_path, num_classes=28, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        resolved_path = os.path.abspath(weights_path) if os.path.exists(weights_path) else os.path.join(_PROJECT_ROOT, weights_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Teacher checkpoint '{weights_path}' not found at '{resolved_path}'!")

        print(f"Loading Teacher Model from: {resolved_path}")
        ckpt = torch.load(resolved_path, map_location="cpu")

        if "dinov2" in weights_path.lower():
            self.model_type = "dinov2"
            self.input_size = 518
            try:
                from src.sota.models_sota import get_sota_model
                self.teacher = get_sota_model(arch="dinov2_vits14", num_classes=num_classes)
            except Exception:
                # Fallback direct hub load
                from models_sota import get_sota_model
                self.teacher = get_sota_model(arch="dinov2_vits14", num_classes=num_classes)
            
            state_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
            self.teacher.load_state_dict(ckpt[state_key], strict=False)
        else:
            self.model_type = "resnet50"
            self.input_size = 224
            self.teacher = models.resnet50(weights=None)
            self.teacher.fc = nn.Linear(self.teacher.fc.in_features, num_classes)
            state_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
            self.teacher.load_state_dict(ckpt[state_key], strict=False)

        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # Interpolate x to teacher's input resolution
        if x.shape[-1] != self.input_size:
            x_up = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        else:
            x_up = x
        return self.teacher(x_up)


def apply_mixup(images, labels, alpha=0.2):
    """Applies MixUp data augmentation to batch."""
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam


def train_epoch(model, dataloader, criterion, optimizer, device, teacher_wrapper=None, kd_temp=4.0, kd_alpha=0.6, mixup_prob=0.3):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        use_mixup = (mixup_prob > 0) and (np.random.rand() < mixup_prob) and (teacher_wrapper is None)
        if use_mixup:
            images, labels_a, labels_b, lam = apply_mixup(images, labels, alpha=0.2)

        outputs = model(images)

        # 1. Base Task Loss
        if use_mixup:
            loss_task = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss_task = criterion(outputs, labels)

        # 2. Knowledge Distillation Loss
        if teacher_wrapper is not None:
            with torch.no_grad():
                teacher_logits = teacher_wrapper(images)
            
            # KL Divergence on Temperature-Softened Probabilities
            loss_kd = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(outputs / kd_temp, dim=1),
                F.softmax(teacher_logits / kd_temp, dim=1)
            ) * (kd_temp ** 2)

            loss = (1 - kd_alpha) * loss_task + kd_alpha * loss_kd
        else:
            loss = loss_task

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == (labels_a if use_mixup else labels).data).item()
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
    print("PlantEdgeNet: SOTA Edge Training & Knowledge Distillation")
    print(f"Architecture: {args.arch_type} | Width Multiplier: {args.width_mult}")
    print(f"Input Resolution: {args.img_size}x{args.img_size} | Device: {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Teacher Distillation: {args.teacher_weights or 'None (From Scratch)'}")
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
    model = get_plantedge_model(
        num_classes=num_classes,
        width_mult=args.width_mult,
        arch_type=args.arch_type
    )
    model = model.to(device)
    params = model.count_parameters()
    macs = model.count_macs((1, 3, args.img_size, args.img_size))
    print(f"Model Initialized: {params:,} Parameters (< 100K FPGA Limit: {params < 100000})")
    print(f"Theoretical Compute: {macs/1e6:.2f} M MACs per inference\n")

    # 3. Load Teacher if provided
    teacher_wrapper = None
    if args.teacher_weights:
        teacher_wrapper = TeacherWrapper(
            weights_path=args.teacher_weights,
            num_classes=num_classes,
            device=args.device
        )
        print("Teacher-Student Knowledge Distillation active!\n")

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = -1.0
    best_ckpt_path = os.path.join(save_dir, f"plantedge_w{args.width_mult:.2f}_best.pth")
    last_ckpt_path = os.path.join(save_dir, f"plantedge_w{args.width_mult:.2f}_latest.pth")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_wrapper=teacher_wrapper,
            kd_temp=args.kd_temp,
            kd_alpha=args.kd_alpha,
            mixup_prob=args.mixup_prob
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_accuracy": val_acc,
            "train_accuracy": train_acc,
            "width_mult": args.width_mult,
            "arch_type": args.arch_type,
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
