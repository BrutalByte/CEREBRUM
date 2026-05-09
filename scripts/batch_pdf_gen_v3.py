import os
import re
import subprocess
import time

# CEREBRUM: Professional Batch PDF Generation Engine v3 (v1.2.4)
# Orchestrates the transformation of the entire library using the CLI + temp-file trick for 100% stability.

MANUSCRIPT_ROOT = os.getcwd()
CSS_PATH = os.path.join(MANUSCRIPT_ROOT, 'docs', 'assets', 'premium_guide.css')
HERO_IMAGE = 'file:///C:/Users/bryan/.gemini/antigravity/brain/77bb37a0-e733-41be-824d-b07e7cce5a6f/cerebrum_hero_v77bb37a0_e730_41be_824d_b07e7cce5a6f_png_1774651388694.png'
OUTPUT_DIR = os.path.join(MANUSCRIPT_ROOT, 'docs', 'PDF')

# Targets
DIRS = [os.path.join(MANUSCRIPT_ROOT, 'docs', 'arxiv'), os.path.join(MANUSCRIPT_ROOT, 'docs')]
SKIP_FILES = ['README.md', 'CONTRIBUTING.md', 'LICENSE', 'CEREBRUM_EXPLAINED.md', 'Parallax_Plain_Language_Guide.md', 'Parallax_Plain_Language_Guide_Professional.md', 'Parallax_Plain_Language_Guide_Professional.pdf', 'CEREBRUM_EXPLAINED.pdf', 'md-pdf-config.json', 'Parallax_Plain_Language_Guide_v3.md', 'Parallax_Plain_Language_Guide_v3.pdf', 'md-pdf-professional-config.json', 'PAPER.md']

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

def convert_file(file_path):
    name = os.path.basename(file_path)
    if name in SKIP_FILES: return
    
    # Safety Check: Verify source file exists and is not empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"   ❌ Skipping: {file_path} (Missing or empty)")
        return

    print(f"Orchestrating: {name}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Surgical Cleanup
    content = re.sub(r'Authors?: .*', 'Author: Bryan Alexander Buchorn / Independent Researcher', content)
    content = re.sub(r'Version: .*', 'Version: 1.2.4 Hardened', content)
    
    title = name.replace('.md', '').replace('_', ' ').replace('PAPER', 'Module').title()
    
    full_content = get_cover_html(title) + content
    
    # 1. Create temp MD in the PDF folder for direct conversion
    temp_md_path = os.path.join(OUTPUT_DIR, "__" + name)
    with open(temp_md_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    # 2. Run CLI
    cmd = f'npx -y md-to-pdf@latest "{temp_md_path}" --stylesheet "{CSS_PATH}"'
    
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        
        # 3. Rename resulting PDF to final name
        temp_pdf = temp_md_path.replace('.md', '.pdf')
        final_pdf = os.path.join(OUTPUT_DIR, name.replace('.md', '.pdf'))
        
        if os.path.exists(temp_pdf):
            if os.path.exists(final_pdf): os.remove(final_pdf)
            os.rename(temp_pdf, final_pdf)
            print(f"   Success: {os.path.basename(final_pdf)}")
        
    except subprocess.CalledProcessError as e:
        print(f"   Failed: {name} - {e.stderr.decode()}")
    finally:
        if os.path.exists(temp_md_path):
            os.remove(temp_md_path)

# Execute
for d in DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith('.md'):
                convert_file(os.path.join(d, f))

print("\nBatch generation complete.")
