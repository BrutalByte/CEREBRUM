import os

dir_path = "docs/arxiv"
old_status = "v2.52.0 (Phase 172 (Sleep-Phase Consolidation) COMPLETE)"
new_status = "v2.52.0 (Phase 172 (STRB) COMPLETE)"
old_date = "April 21, 2026 for version v2.52.0"
new_date = "May 2, 2026 for version v2.52.0"
old_month = "April 2026"
new_month = "May 2026"

for filename in os.listdir(dir_path):
    if filename.startswith("PAPER_") and filename.endswith(".md"):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        updated_content = content.replace(old_status, new_status)
        updated_content = updated_content.replace(old_date, new_date)
        updated_content = updated_content.replace(old_month, new_month)
        
        # Also update generic v2.52.0 if it's in the text and not part of a phase description
        # But be careful. 
        # Actually, let's just do the headers and footers first.
        
        if updated_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            print(f"Updated {filename}")
