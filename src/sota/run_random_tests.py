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

    @torch.no_grad()
    def classify_crop(self, crop_img):
        tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
        tensor_flipped = torch.flip(tensor, dims=[-1])
        logits1 = self.classifier(tensor)
        logits2 = self.classifier(tensor_flipped)
        probs = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2.0

        top_prob, top_idx = torch.max(probs, 1)
        pred_class = self.classes[top_idx.item()]
        return pred_class, top_prob.item()

    def process_image(self, image_path, conf_thresh=0.25, output_dir="output"):
        img_orig = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img_orig.size

        # Stage 1: Leaf Localization
        yolo_results = self.detector.predict(image_path, conf=conf_thresh, verbose=False)
        boxes = yolo_results[0].boxes

        draw = ImageDraw.Draw(img_orig)
        diagnoses = []

        if len(boxes) == 0:
            # Fallback to full scene
            pred_class, confidence = self.classify_crop(img_orig)
            diagnoses.append({
                "region_id": 1,
                "bbox": [0, 0, w_orig, h_orig],
                "detector_conf": 1.0,
                "disease": pred_class,
                "confidence": confidence,
                "is_fallback_full_scene": True
            })
            draw.rectangle([0, 0, w_orig, h_orig], outline="blue", width=3)
            draw.text((10, 10), f"Full Scene: {pred_class} ({confidence*100:.1f}%)", fill="blue")
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

                # Draw color-coded diagnosis
                is_healthy = "healthy" in pred_disease.lower() or pred_disease.lower().endswith(" leaf")
                color = "#00FF00" if is_healthy else "#FF0000"
                
                # Draw thick bounding box
                for width_offset in range(3):
                    draw.rectangle(
                        [xmin - width_offset, ymin - width_offset, xmax + width_offset, ymax + width_offset],
                        outline=color
                    )

                label_text = f"[{idx+1}] {pred_disease} ({disease_conf*100:.1f}%)"
                draw.text((xmin + 4, max(4, ymin - 16)), label_text, fill=color)

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

    # Priority 3: PlantDoc-Dataset test images
    if not test_image_paths:
        pd_test = glob.glob("PlantDoc-Dataset/test/*/*.*")
        test_image_paths.extend([f for f in pd_test if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not test_image_paths:
        print("Error: No test images found in dataset directories!")
        return

    # Randomly select sample images
    num_to_sample = min(args.num_images, len(test_image_paths))
    selected_images = random.sample(test_image_paths, num_to_sample)
    print(f"Randomly selected {num_to_sample} test images from {len(test_image_paths)} total test images.")

    # 3. Load model and run inference
    doctor = EndToEndPlantDoctor(
        detector_path=args.detector_weights,
        classifier_path=args.classifier_weights
    )

    all_run_results = []
    print("\n" + "="*80)
    print(f"{'#':<3} | {'Image File':<35} | {'Detections':<12} | {'Primary Diagnosis':<30}")
    print("="*80)

    for i, img_path in enumerate(selected_images, 1):
        filename = os.path.basename(img_path)
        diagnoses, out_img_path = doctor.process_image(
            image_path=img_path,
            conf_thresh=args.conf_threshold,
            output_dir=run_dir
        )

        primary_diagnosis = diagnoses[0]["disease"] if diagnoses else "No leaf detected"
        primary_conf = diagnoses[0]["confidence"] if diagnoses else 0.0
        diag_summary = f"{primary_diagnosis} ({primary_conf*100:.1f}%)" if diagnoses else "N/A"

        print(f"{i:<3} | {filename[:33]:<35} | {len(diagnoses):<12} | {diag_summary:<30}")

        all_run_results.append({
            "image_index": i,
            "source_path": img_path,
            "filename": filename,
            "annotated_image": os.path.basename(out_img_path),
            "num_regions_detected": len(diagnoses),
            "diagnoses": diagnoses
        })

    # 4. Save comprehensive run summary report
    summary_json_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_timestamp": timestamp_str,
            "run_directory": run_dir,
            "total_images_processed": num_to_sample,
            "detector_model": args.detector_weights,
            "classifier_model": args.classifier_weights,
            "confidence_threshold": args.conf_threshold,
            "results": all_run_results
        }, f, indent=2)

    print("="*80)
    print(f"\nAll {num_to_sample} images diagnosed and saved successfully!")
    print(f"Output folder: {os.path.abspath(run_dir)}")
    print(f"Summary JSON: {os.path.abspath(summary_json_path)}\n")

if __name__ == "__main__":
    main()
