import os
import sys
import glob
import json
import random
import argparse
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from ultralytics import YOLO

from models_sota import get_sota_model

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def draw_high_visibility_label(draw, img_size, xmin, ymin, xmax, ymax, label_text, is_healthy=False):
    """
    Renders high-visibility, large, readable badges with solid backgrounds and thick bounding boxes.
    """
    w_orig, h_orig = img_size
    
    # 1. Determine dynamic font size and border thickness based on image resolution
    min_dim = min(w_orig, h_orig)
    font_size = max(18, min(48, int(min_dim * 0.045)))
    border_width = max(4, min(10, int(min_dim * 0.009)))
    
    # Load bold TrueType font
    font = None
    for fname in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "calibri.ttf", "segoeui.ttf"]:
        try:
            font = ImageFont.truetype(fname, font_size)
            break
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)
        except Exception:
            font = ImageFont.load_default()

    # Colors
    badge_bg = "#1B5E20" if is_healthy else "#B71C1C"      # Solid deep green / deep red
    box_border = "#00E676" if is_healthy else "#FF1744"    # Vibrant neon green / neon red
    text_color = "#FFFFFF"                                 # High contrast white

    # 2. Draw thick bounding box
    for offset in range(border_width):
        draw.rectangle(
            [xmin + offset, ymin + offset, xmax - offset, ymax - offset],
            outline=box_border
        )

    # 3. Compute text size
    pad_x = max(10, int(font_size * 0.45))
    pad_y = max(6, int(font_size * 0.25))

    try:
        t_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = t_bbox[2] - t_bbox[0]
        text_h = t_bbox[3] - t_bbox[1]
    except Exception:
        text_w = font_size * len(label_text) * 0.6
        text_h = font_size

    # Position badge at top of box or inside if too close to top edge
    badge_x1 = max(0, xmin)
    badge_y1 = max(0, ymin - text_h - pad_y * 2)
    if badge_y1 == 0 and ymin < (text_h + pad_y * 2):
        badge_y1 = min(h_orig - text_h - pad_y * 2, ymin + border_width)

    badge_x2 = min(w_orig, badge_x1 + text_w + pad_x * 2)
    badge_y2 = min(h_orig, badge_y1 + text_h + pad_y * 2)

    # Draw solid filled badge with white border
    draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=badge_bg, outline="#FFFFFF", width=2)

    # Draw bold white text inside the badge
    text_x = badge_x1 + pad_x
    text_y = badge_y1 + pad_y
    draw.text((text_x, text_y), label_text, fill=text_color, font=font)

def parse_args():
    parser = argparse.ArgumentParser(description="Run 3-Stage Plant Disease Model on Random Test Images")
    parser.add_argument("--num-images", type=int, default=10, help="Number of random test images to sample")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLO leaf detection confidence threshold")
    parser.add_argument("--detector-weights", type=str, default="runs/detect/runs_stage1_yolo/plantdoc_detector/weights/best.pt", help="Path to YOLOv11 detector weights")
    parser.add_argument("--classifier-weights", type=str, default="checkpoints_sota/dinov2_vits14_best.pth", help="Path to DINOv2 classifier weights")
    parser.add_argument("--output-base", type=str, default="output", help="Base directory for outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    return parser.parse_args()

class EndToEndPlantDoctor:
    def __init__(self, detector_path, classifier_path, img_size=518, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        print("==================================================")
        print("Initializing 3-Stage Plant Disease Diagnostic System")
        print(f"Device: {self.device}")
        print(f"Detector (Stage 1): {detector_path}")
        print(f"Classifier (Stage 2): {classifier_path}")
        print("==================================================")

        # Stage 1: Load YOLO Detector
        self.detector = YOLO(detector_path)

        # Stage 2: Load DINOv2 Classifier
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.num_classes = len(self.classes)

        self.classifier = get_sota_model(arch="dinov2_vits14", num_classes=self.num_classes)
        self.classifier.load_state_dict(checkpoint["state_dict"])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_crop(self, pil_crop):
        tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        return self.classes[pred_idx.item()], conf.item()

    def process_image(self, image_path, output_dir, conf_threshold=0.25):
        img_orig = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img_orig.size

        # Stage 1: Leaf Detection
        results = self.detector(image_path, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        draw = ImageDraw.Draw(img_orig)
        diagnoses = []

        if len(boxes) == 0:
            # Fallback to whole image classification
            pred_class, confidence = self.classify_crop(img_orig)
            is_healthy = "healthy" in pred_class.lower() or pred_class.lower().endswith(" leaf")
            diagnoses.append({
                "region_id": 1,
                "bbox": [0, 0, w_orig, h_orig],
                "disease": pred_class,
                "confidence": confidence,
                "is_fallback_full_scene": True
            })
            draw_high_visibility_label(
                draw=draw,
                img_size=(w_orig, h_orig),
                xmin=0,
                ymin=0,
                xmax=w_orig,
                ymax=h_orig,
                label_text=f" Full Scene: {pred_class} | {confidence*100:.1f}% ",
                is_healthy=is_healthy
            )
        else:
            for idx, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = coords
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w_orig, xmax)
                ymax = min(h_orig, ymax)

                if (xmax - xmin) < 10 or (ymax - ymin) < 10:
                    continue

                # Stage 2: Crop & Classify with DINOv2
                leaf_crop = img_orig.crop((xmin, ymin, xmax, ymax))
                pred_disease, disease_conf = self.classify_crop(leaf_crop)

                diagnoses.append({
                    "region_id": idx + 1,
                    "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "detector_conf": float(box.conf[0].item()),
                    "disease": pred_disease,
                    "confidence": float(disease_conf)
                })

                # Draw high-visibility color-coded diagnosis
                is_healthy = "healthy" in pred_disease.lower() or pred_disease.lower().endswith(" leaf")
                label_text = f" [{idx+1}] {pred_disease} | {disease_conf*100:.1f}% "
                
                draw_high_visibility_label(
                    draw=draw,
                    img_size=(w_orig, h_orig),
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    label_text=label_text,
                    is_healthy=is_healthy
                )

        clean_base = os.path.splitext(os.path.basename(image_path))[0].replace(" ", "_").replace("?", "_")
        annotated_path = os.path.join(output_dir, f"diagnosed_{clean_base}.jpg")
        img_orig.save(annotated_path, quality=95)

        return diagnoses, annotated_path

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 1. Create a unique, timestamped run folder for each execution
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_base, f"run_{timestamp_str}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nCreated new output run folder: {run_dir}")

    # 2. Gather candidate test images
    test_image_paths = []
    
    # Priority 1: Object detection full-scene test images
    obj_det_test = glob.glob("PlantDoc-Object-Detection-Dataset/TEST/*.*") + glob.glob("PlantDoc-Object-Detection-Dataset/test/*.*")
    test_image_paths.extend([f for f in obj_det_test if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Priority 2: YOLO converted validation images
    if not test_image_paths:
        yolo_val = glob.glob("dataset_yolo/images/val/*.*")
        test_image_paths.extend([f for f in yolo_val if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Priority 3: PlantDoc test classification dataset
    if not test_image_paths:
        cls_test = glob.glob("PlantDoc-Dataset/test/*/*.*")
        test_image_paths.extend([f for f in cls_test if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not test_image_paths:
        print("Error: No test images found in dataset directories!")
        return

    num_samples = min(args.num_images, len(test_image_paths))
    selected_images = random.sample(test_image_paths, num_samples)
    print(f"Selected {num_samples} random images from {len(test_image_paths)} total candidates.")

    # 3. Load Pipeline
    doctor = EndToEndPlantDoctor(
        detector_path=args.detector_weights,
        classifier_path=args.classifier_weights,
        img_size=518
    )

    # 4. Run Diagnoses
    run_summary = []
    print("\n" + "="*80)
    print(f"{'#':<3} | {'Image File':<32} | {'Leaves Found':<12} | {'Top Diagnosis':<26}")
    print("="*80)

    for i, img_path in enumerate(selected_images, 1):
        filename = os.path.basename(img_path)
        diagnoses, annotated_img = doctor.process_image(
            image_path=img_path,
            output_dir=run_dir,
            conf_threshold=args.conf_threshold
        )

        top_disease = diagnoses[0]["disease"] if diagnoses else "None"
        top_conf = diagnoses[0]["confidence"] if diagnoses else 0.0
        num_leaves = len(diagnoses)

        print(f"{i:<3} | {filename[:30]:<32} | {num_leaves:<12} | {top_disease[:20]} ({top_conf*100:.1f}%)")

        run_summary.append({
            "image_file": filename,
            "source_path": os.path.abspath(img_path),
            "num_leaves_detected": num_leaves,
            "diagnoses": diagnoses,
            "annotated_image": os.path.basename(annotated_img)
        })

    # 5. Save Run Report JSON
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_timestamp": timestamp_str,
            "total_images_processed": len(selected_images),
            "confidence_threshold": args.conf_threshold,
            "results": run_summary
        }, f, indent=2)

    print("="*80)
    print(f"\nAll {len(selected_images)} random images diagnosed and saved to: {run_dir}")
    print(f"Summary JSON: {summary_path}\n")

if __name__ == "__main__":
    main()
