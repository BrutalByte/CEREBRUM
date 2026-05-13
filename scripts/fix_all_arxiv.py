import os
import re

dir_path = "docs/arxiv"
new_status = "**Status**: v2.52.0 (Phase 172 (STRB) COMPLETE)"
new_date = "**Date**: May 2, 2026"
new_review = "**Reviewed on**: May 2, 2026 for version v2.52.0"

for filename in os.listdir(dir_path):
    if filename.startswith("PAPER_") and filename.endswith(".md"):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        changed = False
        
        # 1. Update title if it contains v2.52.0
        if "v2.52.0" in lines[0]:
            lines[0] = lines[0].replace("v2.52.0", "v2.52.0")
            changed = True
            
        # 2. Find or Insert Status and Date
        status_idx = -1
        date_idx = -1
        for i, line in enumerate(lines[:15]):
            if "**Status**" in line or "Status:" in line:
                status_idx = i
            if "**Date**" in line or "Date:" in line:
                date_idx = i
        
        if status_idx != -1:
            if new_status not in lines[status_idx]:
                lines[status_idx] = new_status + "\n"
                changed = True
        else:
            # Insert after authors or affiliations
            insert_idx = 1
            for i, line in enumerate(lines[:10]):
                if "**Authors**" in line or "**Affiliations**" in line or "**Series**" in line:
                    insert_idx = i + 1
            lines.insert(insert_idx, new_status + "\n")
            changed = True
            if date_idx != -1: date_idx += 1 # adjust
            
        if date_idx != -1:
            if new_date not in lines[date_idx]:
                lines[date_idx] = new_date + "\n"
                changed = True
        else:
            # Insert after Status
            for i, line in enumerate(lines[:15]):
                if "**Status**" in line:
                    lines.insert(i+1, new_date + "\n")
                    changed = True
                    break

        # 3. Update Review footer
        last_lines = lines[-10:]
        found_review = False
        for i, line in enumerate(last_lines):
            if "**Reviewed on**" in line:
                if new_review not in line:
                    last_lines[i] = new_review + "\n"
                    changed = True
                found_review = True
                break
        
        if not found_review:
            # Append it
            lines.append("\n---\n" + new_review + "\n")
            changed = True
        else:
            lines = lines[:-10] + last_lines

        if changed:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"Updated {filename}")
