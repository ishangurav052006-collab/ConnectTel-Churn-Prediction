"""Package the ConnectTel Churn Prediction project into a submission-ready ZIP archive."""

import os
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_OUTPUT_PATH = os.path.join(BASE_DIR, "ConnectTel_Churn_Submission.zip")

# Explicit directories and files to include
INCLUDE_ITEMS = [
    "ConnectTel_Churn_Analysis.ipynb",
    "app.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "data",
    "src",
    "models",
    "notebooks",
    "outputs"
]

# Strict exclusions
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".venv",
    "venv",
    "ENV",
    "env",
    ".git",
    ".ipynb_checkpoints",
    "ConnectTel_Churn_Submission.zip",
    ".DS_Store",
    "Thumbs.db"
]


def should_exclude(rel_path: str) -> bool:
    """Check if the given relative path matches any exclusion pattern."""
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDE_PATTERNS:
            return True
        if part.endswith(".pyc") or part.endswith(".pyo") or part.endswith(".zip"):
            return True
    return False


def create_submission_zip():
    print("=" * 80)
    print("CREATING SUBMISSION ARCHIVE: ConnectTel_Churn_Submission.zip")
    print("=" * 80)

    # Remove existing zip if present to avoid self-inclusion or corruption
    if os.path.exists(ZIP_OUTPUT_PATH):
        os.remove(ZIP_OUTPUT_PATH)
        print(f"Removed previous archive: {ZIP_OUTPUT_PATH}")

    archived_files = []
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for item in INCLUDE_ITEMS:
            item_path = os.path.join(BASE_DIR, item)
            if not os.path.exists(item_path):
                print(f"[WARNING] Item not found: {item_path}")
                continue

            if os.path.isfile(item_path):
                rel_path = item
                if not should_exclude(rel_path):
                    zipf.write(item_path, arcname=rel_path)
                    file_size = os.path.getsize(item_path)
                    total_uncompressed_bytes += file_size
                    archived_files.append((rel_path, file_size))
            elif os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path):
                    # Filter dirs in-place to prevent traversing excluded directories
                    dirs[:] = [d for d in dirs if not should_exclude(d)]
                    for file in files:
                        file_full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_full_path, BASE_DIR)
                        if not should_exclude(rel_path):
                            zipf.write(file_full_path, arcname=rel_path)
                            file_size = os.path.getsize(file_full_path)
                            total_uncompressed_bytes += file_size
                            archived_files.append((rel_path, file_size))

    # Sort archived files by path
    archived_files.sort(key=lambda x: x[0])

    print("\n--- ARCHIVED FILE MANIFEST ---")
    print(f"{'No.':<4} {'Relative Path':<60} {'Uncompressed Size':>14}")
    print("-" * 80)
    for idx, (path, size) in enumerate(archived_files, start=1):
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.2f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"{idx:<4} {path:<60} {size_str:>14}")

    final_zip_size = os.path.getsize(ZIP_OUTPUT_PATH)
    compression_ratio = (1.0 - (final_zip_size / total_uncompressed_bytes)) * 100 if total_uncompressed_bytes > 0 else 0

    print("-" * 80)
    print(f"Total Files Archived:           {len(archived_files)}")
    print(f"Total Uncompressed Size:        {total_uncompressed_bytes / (1024 * 1024):.2f} MB ({total_uncompressed_bytes:,} bytes)")
    print(f"Final Compressed ZIP Size:      {final_zip_size / (1024 * 1024):.2f} MB ({final_zip_size:,} bytes)")
    print(f"Overall Space Savings:          {compression_ratio:.1f}%")
    print(f"Archive Output Destination:     {ZIP_OUTPUT_PATH}")
    print("=" * 80)
    print("SUBMISSION ZIP CREATION COMPLETE - READY FOR PORTAL UPLOAD!")
    print("=" * 80)


if __name__ == "__main__":
    create_submission_zip()

