import os
import glob
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
from tqdm import tqdm

def convert_voc_to_yolo(voc_dir="PlantDoc-Object-Detection-Dataset", yolo_dir="dataset_yolo"):
    os.makedirs(os.path.join(yolo_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "val"), exist_ok=True)

    class_map = {}
    class_list = []

    splits = ["TRAIN", "TEST"] if os.path.exists(os.path.join(voc_dir, "TRAIN")) else ["train", "test"]

    for split in splits:
        split_dir = os.path.join(voc_dir, split)
        if not os.path.exists(split_dir):
            continue

        target_split = "val" if split.lower() == "test" else "train"
        xml_files = glob.glob(os.path.join(split_dir, "*.xml"))
        print(f"Processing {len(xml_files)} XML annotations for split '{split}' -> '{target_split}'...")

        for xml_file in tqdm(xml_files, desc=f"Converting {split}"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                filename_elem = root.find("filename")
                if filename_elem is not None and filename_elem.text:
                    img_name = filename_elem.text
                else:
                    img_name = os.path.splitext(os.path.basename(xml_file))[0] + ".jpg"

                img_path = os.path.join(split_dir, img_name)
                if not os.path.exists(img_path):
                    base = os.path.splitext(os.path.basename(xml_file))[0]
                    found = False
                    for ext in [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]:
                        candidate = os.path.join(split_dir, base + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            img_name = base + ext
                            found = True
                            break
                    if not found:
                        continue

                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception:
                    size_elem = root.find("size")
                    if size_elem is not None:
                        width = int(size_elem.find("width").text)
                        height = int(size_elem.find("height").text)
                    else:
                        continue

                if width <= 0 or height <= 0:
                    continue

                yolo_labels = []
                for obj in root.findall("object"):
                    name = obj.find("name").text.strip()
                    if name not in class_map:
                        class_map[name] = len(class_list)
                        class_list.append(name)
                    cls_id = class_map[name]

                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)

                    xmin = max(0.0, min(float(width), xmin))
                    xmax = max(0.0, min(float(width), xmax))
                    ymin = max(0.0, min(float(height), ymin))
                    ymax = max(0.0, min(float(height), ymax))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    x_center = ((xmin + xmax) / 2.0) / width
                    y_center = ((ymin + ymax) / 2.0) / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                if not yolo_labels:
                    continue

                clean_base = os.path.splitext(os.path.basename(img_name))[0].replace(" ", "_").replace("?", "_")
                clean_img_name = clean_base + os.path.splitext(img_name)[1]

                dest_img_path = os.path.join(yolo_dir, "images", target_split, clean_img_name)
                dest_txt_path = os.path.join(yolo_dir, "labels", target_split, clean_base + ".txt")

                shutil.copy(img_path, dest_img_path)
                with open(dest_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_labels))

            except Exception as e:
                continue

    abs_yolo_dir = os.path.abspath(yolo_dir).replace(os.sep, "/")
    yaml_content = f"path: {abs_yolo_dir}\ntrain: images/train\nval: images/val\n\nnames:\n"
    for idx, name in enumerate(class_list):
        yaml_content += f"  {idx}: '{name}'\n"

    yaml_path = os.path.join(yolo_dir, "plantdoc_yolo.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nYOLO dataset conversion complete!")
    print(f"Classes ({len(class_list)}): {class_list}")
    print(f"Configuration written to: {yaml_path}")
    return yaml_path, class_list

if __name__ == "__main__":
    convert_voc_to_yolo()
