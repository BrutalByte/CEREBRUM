import os
import re
from datetime import datetime

# CONFIGURATION
TARGET_VERSION = "v2.24.0"
TARGET_STATUS = "Phase 111 (Active Inference) COMPLETE"
REVIEW_DATE = "April 21, 2026"
DOCS_DIR = "docs"

# Patterns to find and replace
STATUS_PATTERN = re.compile(r"\*\*Status\*\*:.*", re.IGNORECASE)
PHASE_PATTERN = re.compile(r"Phase \d+ COMPLETE", re.IGNORECASE)
VERSION_PATTERN = re.compile(r"v\d+\.\d+\.\d+", re.IGNORECASE)

def update_file(file_path):
    # Skip non-markdown files and archive/old folders
    if not file_path.endswith(".md"): return
    if "archive" in file_path.lower() or "old docs" in file_path.lower(): return

    print(f"Updating {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update existing Status lines
    new_content = STATUS_PATTERN.sub(f"**Status**: {TARGET_VERSION} ({TARGET_STATUS})", content)
    
    # 2. Update standalone version strings if they look like system versions
    # Only if not already replaced by status
    if new_content == content:
         new_content = VERSION_PATTERN.sub(TARGET_VERSION, content)

    # 3. Add "Reviewed on" note at the bottom or after status if not present
    review_note = f"\n\n---\n**Reviewed on**: {REVIEW_DATE} for version {TARGET_VERSION}\n"
    if "Reviewed on" not in new_content:
        # Append to the end of the file
        new_content = new_content.strip() + review_note

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

def walk_and_update(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            update_file(os.path.join(root, file))

if __name__ == "__main__":
    walk_and_update(DOCS_DIR)
    print("Done.")
