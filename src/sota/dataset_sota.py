import os
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_sota_transforms(img_size=518):
    # Training transforms with RandAugment and photometric adjustments
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def is_valid_image(path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_exts:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_sota_dataloaders(data_dir="PlantDoc-Dataset", batch_size=16, img_size=518, num_workers=2):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Train/Test directories not found under {data_dir}")

    train_transform, test_transform = get_sota_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
        is_valid_file=is_valid_image
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform,
        is_valid_file=is_valid_image
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, train_dataset.classes
