import os
import sys
import re
import glob
import json
import random
import argparse
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

from model import get_plantedge_model
from dataset import extract_hsv_leaf_crop

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

def get_next_run_dir(base_output_dir):
    """
    Finds existing run_1, run_2, ... in base_output_dir and returns the next folder path (e.g. run_3).
    """
    os.makedirs(base_output_dir, exist_ok=True)
    existing_runs = []
    
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path):
            match = re.match(r"^run_(\d+)$", item, re.IGNORECASE)
            if match:
                existing_runs.append(int(match.group(1)))

    next_num = max(existing_runs, default=0) + 1
    run_folder_name = f"run_{next_num}"
    run_dir = os.path.join(base_output_dir, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_folder_name

def parse_args():
    parser = argparse.ArgumentParser(description="PlantEdgeNet Random Image Inference & Diagnosis")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image (optional)")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to custom directory of images (optional)")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to dataset root for test image sampling")
    parser.add_argument("--num-images", type=int, default=10, help="Number of random test images to sample per run")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_CURR_DIR, "checkpoints", "plantedge_w1.00_best.pth"), help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-base", type=str, default=os.path.join(_CURR_DIR, "outputs"), help="Base directory for run outputs (e.g. outputs/run_1, outputs/run_2)")
    parser.add_argument("--img-size", type=int, default=96, help="Model input resolution")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()

class PlantDoctorEdge:
    def __init__(self, checkpoint_path, img_size=96, device="cpu"):
        self.device = torch.device(device)
        self.img_size = img_size

        resolved_ckpt = resolve_file_path(checkpoint_path)
        if not os.path.exists(resolved_ckpt):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found at '{resolved_ckpt}'!")

        ckpt = torch.load(resolved_ckpt, map_location=self.device)
        self.num_classes = ckpt.get("num_classes", 28)
        self.width_mult = ckpt.get("width_mult", 1.0)
        self.classes = ckpt.get("classes", [f"Class_{i}" for i in range(self.num_classes)])

        self.model = get_plantedge_model(num_classes=self.num_classes, width_mult=self.width_mult)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def diagnose(self, image_path, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        try:
            img_orig = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        w_orig, h_orig = img_orig.size

        # 1. Extract Leaf ROI
        leaf_roi = extract_hsv_leaf_crop(img_orig)

        # 2. Forward pass
        input_tensor = self.transform(leaf_roi).unsqueeze(0).to(self.device)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        # Top 3 predictions
        top_probs, top_indices = torch.topk(probs, min(3, self.num_classes))
        
        predictions = []
        for p, idx in zip(top_probs, top_indices):
            cls_name = self.classes[idx.item()] if idx.item() < len(self.classes) else f"Class_{idx.item()}"
            predictions.append({
                "class_name": cls_name,
                "confidence": float(p.item()),
                "confidence_percent": f"{p.item() * 100:.2f}%"
            })

        primary_pred = predictions[0]
        is_healthy = "healthy" in primary_pred["class_name"].lower() or primary_pred["class_name"].lower().endswith(" leaf")
        color = "#00FF00" if is_healthy else "#FF0000"

        # 3. Draw visual annotations
        draw = ImageDraw.Draw(img_orig)
        label_text = f"{primary_pred['class_name']} ({primary_pred['confidence_percent']})"
        
        for offset in range(3):
            draw.rectangle([offset, offset, w_orig - offset, h_orig - offset], outline=color)
        draw.text((10, 10), label_text, fill=color)

        clean_base = os.path.splitext(os.path.basename(image_path))[0].replace(" ", "_").replace("?", "_")
        annotated_path = os.path.join(output_dir, f"diagnosed_{clean_base}.jpg")
        img_orig.save(annotated_path, quality=95)

        report = {
            "image_file": os.path.basename(image_path),
            "source_path": os.path.abspath(image_path),
            "primary_diagnosis": primary_pred["class_name"],
            "primary_confidence": primary_pred["confidence_percent"],
            "confidence_value": primary_pred["confidence"],
            "is_healthy": is_healthy,
            "top_3_predictions": predictions,
            "annotated_image": os.path.basename(annotated_path)
        }

        json_path = os.path.join(output_dir, f"report_{clean_base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # 1. Create sequential run folder: run_1, run_2, run_3, ...
    base_output_dir = os.path.abspath(args.output_base)
    run_dir, run_folder_name = get_next_run_dir(base_output_dir)

    print("==================================================")
    print("PlantEdgeNet: FPGA Diagnostic Inference Engine")
    print(f"Created Execution Run: {run_folder_name}")
    print(f"Output Directory: {run_dir}")
    print(f"Device: {args.device}")
    print("==================================================")

    # 2. Gather candidate images
    image_paths = []
    if args.image:
        if os.path.exists(args.image):
            image_paths.append(args.image)
        else:
            print(f"Error: Specified image '{args.image}' not found!")
            return
    elif args.image_dir:
        resolved_img_dir = resolve_file_path(args.image_dir)
        if os.path.exists(resolved_img_dir):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                image_paths.extend(glob.glob(os.path.join(resolved_img_dir, ext)))
        else:
            print(f"Error: Specified image directory '{args.image_dir}' not found!")
            return
    else:
        # Sample random test images from dataset test split
        resolved_data_dir = resolve_file_path(args.data_dir)
        test_dir = os.path.join(resolved_data_dir, "test")
        
        if not os.path.exists(test_dir):
            # Check parent directory fallback
            test_dir = os.path.join(_CURR_DIR, "..", "PlantDoc-Dataset", "test")

        if os.path.exists(test_dir):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"):
                image_paths.extend(glob.glob(os.path.join(test_dir, "*", ext)))
        
        if not image_paths:
            # Fallback to any images in dataset
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                image_paths.extend(glob.glob(os.path.join(resolved_data_dir, "**", ext), recursive=True))

    if not image_paths:
        print("Error: No test images found in dataset directories!")
        return

    # Randomly sample N images
    num_to_sample = min(args.num_images, len(image_paths))
    selected_images = random.sample(image_paths, num_to_sample) if not args.image else image_paths
    print(f"Randomly selected {len(selected_images)} test images for this run from {len(image_paths)} total candidates.")

    # 3. Load Model
    doctor = PlantDoctorEdge(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        device=args.device
    )

    # 4. Execute Predictions
    all_reports = []
    print("\n" + "="*80)
    print(f"{'#':<3} | {'Image File':<35} | {'Primary Diagnosis':<25} | {'Confidence':<12}")
    print("="*80)

    for i, img_path in enumerate(selected_images, 1):
        report = doctor.diagnose(img_path, output_dir=run_dir)
        if report is not None:
            all_reports.append(report)
            filename = report["image_file"]
            diagnosis = report["primary_diagnosis"]
            conf = report["primary_confidence"]
            print(f"{i:<3} | {filename[:33]:<35} | {diagnosis[:23]:<25} | {conf:<12}")

    # 5. Save Comprehensive Run Summary JSON
    summary_data = {
        "run_folder": run_folder_name,
        "run_directory": os.path.abspath(run_dir),
        "timestamp": datetime.now().isoformat(),
        "checkpoint": os.path.abspath(resolve_file_path(args.checkpoint)),
        "total_images_processed": len(all_reports),
        "results": all_reports
    }

    summary_json_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("="*80)
    print(f"\nAll {len(all_reports)} images diagnosed and saved to: {run_folder_name}")
    print(f"Run Summary Report: {os.path.abspath(summary_json_path)}\n")

if __name__ == "__main__":
    main()
