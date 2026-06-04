from typing import Type
import os
import re
import requests
import urllib3
from pathlib import Path

# Suppress InsecureRequestWarning for institutional repositories
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_sources():
    bib_path = Path("docs/latex/references.bib")
    output_dir = Path("docs/sources")
    output_dir.mkdir(exist_ok=True)
    
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple regex to find entries with URLs
    # Matches: @type{key, ... url={url} ... }
    entries = re.findall(r"@.*\{(.*),[\s\S]*?url=\{(.*)\}", content)
    
    print(f"Found {len(entries)} entries with URLs.")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    report = []
    
    for key, url in entries:
        filename = f"{key}.pdf"
        target_path = output_dir / filename
        
        if target_path.exists():
            print(f"Skipping {filename} (already exists).")
            continue
            
        print(f"Downloading {key} from {url}...")
        try:
            # Handle SSL mismatch for institutional repositories and smarter PDF check
            response = requests.get(url, headers=headers, timeout=15, stream=True, verify=False)
            
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type or url.endswith(".pdf") or "pdf" in url.lower()

            if response.status_code == 200 and is_pdf:
                with open(target_path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
                print(f"  [+] Success: {filename}")
                report.append(f"SUCCESS: {key} ({url})")
            elif response.status_code == 200:
                print(f"  [!] Skipping non-PDF Content-Type: {content_type}")
                report.append(f"FAILED: {key} (Non-PDF Content: {content_type})")
            else:
                print(f"  [-] Failed: Status {response.status_code}")
                report.append(f"FAILED: {key} (Status {response.status_code})")
        except Exception as e:
            print(f"  [-] Error: {str(e)}")
            report.append(f"ERROR: {key} ({str(e)})")
            
    with open("docs/sources/download_report.txt", "w", encoding="utf-8") as rep_file:
        rep_file.write("\n".join(report))
    
    print("\nDownload complete. See docs/sources/download_report.txt for details.")

if __name__ == "__main__":
    fetch_sources()
