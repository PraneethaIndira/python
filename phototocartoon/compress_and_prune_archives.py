import os
import shutil
import zipfile
from datetime import datetime

ARCHIVE_DIR = "archive"
MAX_ARCHIVES = 5  # Keep only the 5 most recent archives

def zip_folder(folder_path):
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)
    print(f"🗜️ Compressed: {folder_path} → {zip_path}")
    shutil.rmtree(folder_path)
    print(f"🧹 Removed original folder: {folder_path}")

def prune_old_archives(max_keep=MAX_ARCHIVES):
    all_archives = sorted(
        [f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".zip")],
        key=lambda x: os.path.getmtime(os.path.join(ARCHIVE_DIR, x))
    )
    if len(all_archives) > max_keep:
        to_delete = all_archives[:-max_keep]
        for file in to_delete:
            path = os.path.join(ARCHIVE_DIR, file)
            os.remove(path)
            print(f"🗑️ Deleted old archive: {path}")

def main():
    print("📦 Compressing and pruning training archives...")
    if not os.path.exists(ARCHIVE_DIR):
        print("✅ No archive directory found. Nothing to compress or prune.")
        return

    for item in os.listdir(ARCHIVE_DIR):
        full_path = os.path.join(ARCHIVE_DIR, item)
        if os.path.isdir(full_path):
            zip_folder(full_path)

    prune_old_archives()
    print("✅ Archive compression and cleanup complete.")

if __name__ == "__main__":
    main()