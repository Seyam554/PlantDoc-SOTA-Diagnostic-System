import os
import subprocess
import re
from pathlib import Path
from tqdm import tqdm

def sanitize_filename(name):
    # Remove query strings and invalid windows characters: \ / : * ? " < > |
    name = re.sub(r'[\\:*?"<>|]', '_', name)
    return name

def main():
    repo_dir = "PlantDoc-Dataset"
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        print("Git repository not found!")
        return

    print("Retrieving git tree object list...")
    result = subprocess.run(
        ["git", "-C", repo_dir, "ls-tree", "-r", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    lines = result.stdout.strip().split("\n")
    print(f"Total objects in git repository: {len(lines)}")

    extracted_count = 0
    for line in tqdm(lines, desc="Extracting files"):
        if not line.strip():
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        mode, obj_type, blob_hash, filepath = parts
        if obj_type != "blob":
            continue

        # Sanitize filepath for Windows
        path_parts = filepath.split("/")
        sanitized_parts = [sanitize_filename(p) for p in path_parts]
        target_path = os.path.join(repo_dir, *sanitized_parts)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
            # Extract blob content
            show_proc = subprocess.run(
                ["git", "-C", repo_dir, "cat-file", "-p", blob_hash],
                capture_output=True
            )
            with open(target_path, "wb") as f:
                f.write(show_proc.stdout)
            extracted_count += 1

    print(f"Successfully checked and extracted {extracted_count} files into {repo_dir}.")

if __name__ == "__main__":
    main()
