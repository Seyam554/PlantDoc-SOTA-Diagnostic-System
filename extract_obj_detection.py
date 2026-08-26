import os
import subprocess
import re
from tqdm import tqdm

def sanitize_filename(name):
    return re.sub(r'[\\:*?"<>|]', '_', name)

def main():
    repo_dir = "PlantDoc-Object-Detection-Dataset"
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        print("Git repo not found!")
        return

    print("Listing git tree...")
    res = subprocess.run(
        ["git", "-C", repo_dir, "ls-tree", "-r", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    lines = res.stdout.strip().split("\n")
    print(f"Total objects: {len(lines)}")

    count = 0
    for line in tqdm(lines, desc="Extracting object detection files"):
        if not line.strip():
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        mode, obj_type, blob_hash, filepath = parts
        if obj_type != "blob":
            continue

        parts = filepath.split("/")
        sanitized_parts = [sanitize_filename(p) for p in parts]
        target_path = os.path.join(repo_dir, *sanitized_parts)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
            show_proc = subprocess.run(
                ["git", "-C", repo_dir, "cat-file", "-p", blob_hash],
                capture_output=True
            )
            with open(target_path, "wb") as f:
                f.write(show_proc.stdout)
            count += 1

    print(f"Extracted {count} files into {repo_dir}.")

if __name__ == "__main__":
    main()
