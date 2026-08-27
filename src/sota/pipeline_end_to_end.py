import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from ultralytics import YOLO

from models_sota import get_sota_model

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End 3-Stage Plant Disease Diagnostic Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to input raw image")
    parser.add_argument("--detector-weights", type=str, default="runs/detect/runs_stage1_yolo/plantdoc_detector/weights/best.pt", help="Path to YOLOv11 detector weights")
    parser.add_argument("--classifier-weights", type=str, default="checkpoints_sota/dinov2_vits14_best.pth", help="Path to DINOv2 classifier weights")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--img-size", type=int, default=518, help="DINOv2 input resolution")
    parser.add_argument("--save-dir", type=str, default="pipeline_outputs", help="Output directory")
    return parser.parse_args()

class EndToEndPlantDoctor:
    def __init__(self, detector_path, classifier_path, img_size=518, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        print("==================================================")
        print("Loading End-to-End 3-Stage Plant Disease Diagnostic System")
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

        # DINOv2 Transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def classify_crop(self, crop_img):
        tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
        # TTA: original + horizontal flip
        tensor_flipped = torch.flip(tensor, dims=[-1])
        logits1 = self.classifier(tensor)
        logits2 = self.classifier(tensor_flipped)
        probs = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2.0

        top_prob, top_idx = torch.max(probs, 1)
        pred_class = self.classes[top_idx.item()]
        return pred_class, top_prob.item()

    def process_image(self, image_path, conf_thresh=0.25, output_dir="pipeline_outputs"):
        os.makedirs(output_dir, exist_ok=True)
        img_orig = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img_orig.size

        # Stage 1: Detect Leaves
        yolo_results = self.detector.predict(image_path, conf=conf_thresh, verbose=False)
        boxes = yolo_results[0].boxes

        draw = ImageDraw.Draw(img_orig)
        diagnoses = []

        print(f"\nProcessing '{os.path.basename(image_path)}': Detected {len(boxes)} candidate leaf regions.")

        if len(boxes) == 0:
            # Fallback: full image classification
            pred_class, confidence = self.classify_crop(img_orig)
            diagnoses.append({
                "box": [0, 0, w_orig, h_orig],
                "yolo_conf": 1.0,
                "disease_class": pred_class,
                "disease_confidence": confidence
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

                # Stage 2: Crop leaf & Classify with DINOv2
                leaf_crop = img_orig.crop((xmin, ymin, xmax, ymax))
                pred_disease, disease_conf = self.classify_crop(leaf_crop)

                diagnoses.append({
                    "region_id": idx + 1,
                    "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "detector_conf": float(box.conf[0].item()),
                    "disease": pred_disease,
                    "confidence": float(disease_conf)
                })

                # Draw bounding box
                color = "green" if "healthy" in pred_disease.lower() or "leaf" == pred_disease.lower().split()[-1] else "red"
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                label_text = f"{pred_disease} ({disease_conf*100:.1f}%)"
                draw.text((xmin + 4, max(0, ymin - 15)), label_text, fill=color)

        output_img_path = os.path.join(output_dir, f"diagnosed_{os.path.basename(image_path)}")
        img_orig.save(output_img_path)

        output_json_path = os.path.join(output_dir, f"report_{os.path.splitext(os.path.basename(image_path))[0]}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(diagnoses, f, indent=2)

        print(f"Saved diagnostic visualization: {output_img_path}")
        print(f"Saved diagnostic report: {output_json_path}")
        return diagnoses, output_img_path

def main():
    args = parse_args()
    doctor = EndToEndPlantDoctor(
        detector_path=args.detector_weights,
        classifier_path=args.classifier_weights,
        img_size=args.img_size
    )

    if args.image:
        doctor.process_image(args.image, conf_thresh=args.conf_threshold, output_dir=args.save_dir)
    else:
        # Run demonstration on sample test images
        import glob
        test_images = glob.glob("PlantDoc-Object-Detection-Dataset/TEST/*.jpg") + glob.glob("PlantDoc-Object-Detection-Dataset/test/*.jpg")
        if not test_images:
            test_images = glob.glob("dataset_yolo/images/val/*.jpg")

        print(f"Running End-to-End demonstration on {min(5, len(test_images))} test images...")
        for img_path in test_images[:5]:
            doctor.process_image(img_path, conf_thresh=args.conf_threshold, output_dir=args.save_dir)

if __name__ == "__main__":
    main()
