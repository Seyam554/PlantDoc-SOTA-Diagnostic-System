"""Shared dataloaders + metrics for the FPGA student pipeline.

Reuses the repo's PlantDoc-Dataset ImageFolder layout but at 64x64 (the
accelerator input size). Normalization is ImageNet stats during training;
at export we fold it into the first conv (see export_onnx.py).
"""

import os
import json
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _valid(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_transforms(img_size=64, train=True, strong=True):
    if train:
        ops = [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        ]
        if strong:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(data_dir="PlantDoc-Dataset", img_size=64, batch_size=64, workers=2, strong_aug=True):
    tr = datasets.ImageFolder(os.path.join(data_dir, "train"),
                              transform=build_transforms(img_size, True, strong_aug),
                              is_valid_file=_valid)
    te = datasets.ImageFolder(os.path.join(data_dir, "test"),
                              transform=build_transforms(img_size, False),
                              is_valid_file=_valid)
    pin = torch.cuda.is_available()
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True),
        DataLoader(te, batch_size, shuffle=False, num_workers=workers, pin_memory=pin),
        tr.classes,
    )


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    preds, tgts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if tta:
            out = out.softmax(1) + model(torch.flip(x, dims=[-1])).softmax(1)
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.numpy())
    preds, tgts = np.array(preds), np.array(tgts)
    acc = accuracy_score(tgts, preds)
    mp, mr, mf1, _ = precision_recall_fscore_support(tgts, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_precision": float(mp),
            "macro_recall": float(mr), "macro_f1": float(mf1)}, preds, tgts


def save_report(out_dir, tag, metrics, preds, tgts, classes):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(tgts, preds)
        plt.figure(figsize=(18, 14))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=classes, yticklabels=classes)
        plt.title(f"{tag}  acc={metrics['accuracy']*100:.2f}%")
        plt.xlabel("pred"); plt.ylabel("true")
        plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_confusion_matrix.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[warn] confusion matrix skipped: {e}")
