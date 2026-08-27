import os
import sys
import glob
import json
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="PlantEdgeNet Image Inference & Diagnosis")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to directory of input images")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_CURR_DIR, "checkpoints", "plantedge_w1.00_best.pth"), help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(_CURR_DIR, "outputs"), help="Directory to save diagnosis outputs")
    parser.add_argument("--img-size", type=int, default=96, help="Model input resolution")
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
        img_orig = Image.open(image_path).convert("RGB")
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
        
        # Border
        for offset in range(3):
            draw.rectangle([offset, offset, w_orig - offset, h_orig - offset], outline=color)
        draw.text((10, 10), label_text, fill=color)

        clean_base = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(output_dir, f"diagnosed_{clean_base}.jpg")
        img_orig.save(annotated_path, quality=95)

        report = {
            "image_file": os.path.basename(image_path),
            "primary_diagnosis": primary_pred["class_name"],
            "primary_confidence": primary_pred["confidence_percent"],
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
    doctor = PlantDoctorEdge(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        device=args.device
    )

    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    else:
        # Check standard test images
        image_paths = glob.glob("PlantDoc-Dataset/test/*/*.jpg")[:5]
        if not image_paths:
            image_paths = glob.glob("../PlantDoc-Dataset/test/*/*.jpg")[:5]

    if not image_paths:
        print("No test images found. Pass --image path/to/image.jpg")
        return

    print("==================================================")
    print("PlantEdgeNet: Diagnostic Inference on Test Images")
    print(f"Total Images: {len(image_paths)} | Output: {args.output_dir}")
    print("==================================================")

    for i, p in enumerate(image_paths, 1):
        report = doctor.diagnose(p, output_dir=args.output_dir)
        print(f"[{i:02d}/{len(image_paths):02d}] {report['image_file'][:30]:<32} -> {report['primary_diagnosis']} ({report['primary_confidence']})")

    print("\nInference Complete! Reports saved to:", os.path.abspath(args.output_dir))

if __name__ == "__main__":
    main()
