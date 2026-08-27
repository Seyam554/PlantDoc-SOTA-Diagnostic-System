import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def extract_hsv_leaf_crop(img, min_area_ratio=0.05):
    """
    FPGA-friendly leaf ROI extraction using HSV green thresholding.
    Finds the primary leaf bounding box to eliminate background soil/hands.
    """
    img_np = np.array(img)
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return img

    hsv = np.array(img.convert('HSV'))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Broad green/yellow/brown vegetation mask in HSV
    # Green/Yellow: H in [25, 95] (0-255 scale in PIL), S > 30, V > 25
    mask = (h >= 20) & (h <= 110) & (s >= 30) & (v >= 25)

    coords = np.argwhere(mask)
    if len(coords) < (img_np.shape[0] * img_np.shape[1] * min_area_ratio):
        # Fallback to center 80% crop if insufficient green pixels
        w, h = img.size
        return img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add 5% padding margin
    pad_y = int((y_max - y_min) * 0.05)
    pad_x = int((x_max - x_min) * 0.05)
    
    w_orig, h_orig = img.size
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w_orig, x_max + pad_x)
    y_max = min(h_orig, y_max + pad_y)

    if (x_max - x_min) < 16 or (y_max - y_min) < 16:
        return img

    return img.crop((x_min, y_min, x_max, y_max))


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=96, use_hsv_roi=True, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.use_hsv_roi = use_hsv_roi
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory '{split_dir}' not found!")

        self.classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(split_dir, cls_name)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"):
                for img_path in glob.glob(os.path.join(cls_folder, ext)):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        print(f"Loaded {len(self.samples)} images across {len(self.classes)} classes for [{split}] split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Handle corrupt/truncated files gracefully
            image = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        if self.use_hsv_roi:
            image = extract_hsv_leaf_crop(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_fpga_transforms(img_size=96):
    """
    Standard transforms for training and testing.
    Uses standard ImageNet normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir="PlantDoc-Dataset", img_size=96, batch_size=32, num_workers=2, use_hsv_roi=True):
    train_tf, eval_tf = get_fpga_transforms(img_size=img_size)

    # Check if dataset is in parent directory
    if not os.path.exists(data_dir):
        alt_path = os.path.join("..", data_dir)
        if os.path.exists(alt_path):
            data_dir = alt_path

    train_dataset = PlantDiseaseDataset(root_dir=data_dir, split="train", img_size=img_size, use_hsv_roi=use_hsv_roi, transform=train_tf)
    test_dataset = PlantDiseaseDataset(root_dir=data_dir, split="test", img_size=img_size, use_hsv_roi=use_hsv_roi, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, train_dataset.classes
