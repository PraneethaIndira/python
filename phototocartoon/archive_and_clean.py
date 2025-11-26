import os
import shutil
from datetime import datetime

# Define paths
checkpoint_dir = "checkpoints"
output_dir = "outputs"
metrics_file = "metrics.txt"
archive_root = "archive"

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def archive_path(name):
    return os.path.join(archive_root, f"{name}_{timestamp()}")

def archive_and_clean_dir(path):
    if os.path.exists(path):
        dest = archive_path(os.path.basename(path))
        shutil.move(path, dest)
        print(f"📦 Archived: {path} → {dest}")
    os.makedirs(path)
    print(f"📁 Recreated: {path}")

def archive_and_clean_file(path):
    if os.path.exists(path):
        dest = archive_path(os.path.splitext(os.path.basename(path))[0] + ".txt")
        os.makedirs(archive_root, exist_ok=True)
        shutil.move(path, dest)
        print(f"📦 Archived: {path} → {dest}")
    else:
        print(f"✅ No file to archive: {path}")

def main():
    print("🧼 Archiving and cleaning training environment...")
    os.makedirs(archive_root, exist_ok=True)
    archive_and_clean_dir(checkpoint_dir)
    archive_and_clean_dir(output_dir)
    archive_and_clean_file(metrics_file)
    print("✅ Environment cleaned and archived. Ready for fresh training!")

if __name__ == "__main__":
    main()