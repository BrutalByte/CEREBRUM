import os
import re
import subprocess

# CEREBRUM: Professional Batch PDF Generation Engine (v1.2.4)
# Orchestrates the transformation of the entire technical library into conference-ready PDFs.

CSS_PATH = 'e:/Development/Parallax/docs/assets/premium_guide.css'
HERO_IMAGE = 'file:///C:/Users/bryan/.gemini/antigravity/brain/77bb37a0-e733-41be-824d-b07e7cce5a6f/cerebrum_hero_v77bb37a0_e730_41be_824d_b07e7cce5a6f_png_1774651388694.png'
OUTPUT_DIR = 'e:/Development/Parallax/docs/PDF'

# Targets
DIRS = ['e:/Development/Parallax/docs/arxiv', 'e:/Development/Parallax/docs']
SKIP_FILES = ['README.md', 'CONTRIBUTING.md', 'LICENSE', 'CEREBRUM_EXPLAINED.md', 'Parallax_Plain_Language_Guide.md', 'Parallax_Plain_Language_Guide_Professional.md']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_cover_html(title, subtitle="Technical Specification"):
    return f"""<div class="cover">
    <img src="{HERO_IMAGE}" alt="CEREBRUM Visionary Hero">
    <h1 class="title">{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="meta">
        <strong>Version 1.2.4 (Hardened Edition)</strong><br>
        March 2026 — Bryan Alexander Buchorn / Independent Researcher
    </div>
</div>

---
"""

def process_file(file_path):
    name = os.path.basename(file_path)
    if name in SKIP_FILES: return
    
    print(f"Processing: {name}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 1. Surgical Metadata Sync (v1.2.4 + Bryan)
    # Ensure current author tagging
    content = re.sub(r'Authors?: .*', 'Author: Bryan Alexander Buchorn / Independent Researcher', content)
    content = re.sub(r'Version: .*', 'Version: 1.2.4 Hardened', content)
    
    # 2. Inject Cover
    title = name.replace('.md', '').replace('_', ' ').title()
    if 'PAPER' in title:
        title = title.replace('Paper ', 'CEREBRUM Module: ')
    
    full_content = get_cover_html(title) + content
    
    temp_md = file_path.replace('.md', '_temp_batch.md')
    with open(temp_md, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    # 3. Convert
    output_pdf = os.path.join(OUTPUT_DIR, name.replace('.md', '.pdf'))
    
    # Using npx direct with stylesheet (Most reliable in shell)
    cmd = f'npx -y md-to-pdf@latest "{temp_md}" --stylesheet "{CSS_PATH}" --output "{output_pdf}"'
    
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print(f"   Success: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"   Failed: {name} - {e.stderr.decode()}")
    finally:
        if os.path.exists(temp_md):
            os.remove(temp_md)

# Main Loop
for d in DIRS:
    for f in os.listdir(d):
        if f.endswith('.md'):
            process_file(os.path.join(d, f))

print("\nBatch generation complete.")
