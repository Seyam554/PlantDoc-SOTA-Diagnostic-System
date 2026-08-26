import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 YOLOv11 Leaf Detector")
    parser.add_argument("--data", type=str, default="dataset_yolo/plantdoc_yolo.yaml", help="Path to YOLO yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained YOLO model weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU device index")
    parser.add_argument("--project", type=str, default="runs_stage1_yolo", help="Save project directory")
    return parser.parse_args()

def main():
    args = parse_args()
    print("==================================================")
    print("Stage 1: Training YOLOv11 Leaf & Disease Detector")
    print(f"Data: {args.data}")
    print(f"Model: {args.model} | Epochs: {args.epochs} | ImgSz: {args.imgsz}")
    print("==================================================")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="plantdoc_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save=True,
        val=True
    )

    print("\nStage 1 YOLOv11 Training Completed!")
    print(f"Model weights saved to {args.project}/plantdoc_detector/weights/best.pt")

if __name__ == "__main__":
    main()
