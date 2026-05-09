import os
import re
import subprocess
import json

# CEREBRUM: Professional Batch PDF Generation Engine v2 (v1.2.4)
# Orchestrates the transformation of the entire library using the Node API for maximum stability.

MANUSCRIPT_ROOT = os.getcwd()
CSS_PATH = os.path.join(MANUSCRIPT_ROOT, 'docs', 'assets', 'premium_guide.css')
HERO_IMAGE = 'file:///C:/Users/bryan/.gemini/antigravity/brain/77bb37a0-e733-41be-824d-b07e7cce5a6f/cerebrum_hero_v77bb37a0_e730_41be_824d_b07e7cce5a6f_png_1774651388694.png'
OUTPUT_DIR = os.path.join(MANUSCRIPT_ROOT, 'docs', 'PDF')

# Targets
DIRS = [os.path.join(MANUSCRIPT_ROOT, 'docs', 'arxiv'), os.path.join(MANUSCRIPT_ROOT, 'docs')]
SKIP_FILES = ['README.md', 'CONTRIBUTING.md', 'LICENSE', 'CEREBRUM_EXPLAINED.md', 'Parallax_Plain_Language_Guide.md', 'Parallax_Plain_Language_Guide_Professional.md', 'Parallax_Plain_Language_Guide_Professional.pdf', 'CEREBRUM_EXPLAINED.pdf', 'md-pdf-config.json', 'Parallax_Plain_Language_Guide_v3.md', 'Parallax_Plain_Language_Guide_v3.pdf', 'md-pdf-professional-config.json']

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

# Create a master Node conversion script
conversion_targets = []

def prepare_for_conversion(file_path):
    name = os.path.basename(file_path)
    if name in SKIP_FILES: return
    
    # Safety Check: Verify source file exists and is not empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"   ❌ Skipping: {file_path} (Missing or empty)")
        return

    print(f"Preparing: {name}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Surgical Cleanup
    content = re.sub(r'Authors?: .*', 'Author: Bryan Alexander Buchorn / Independent Researcher', content)
    content = re.sub(r'Version: .*', 'Version: 1.2.4 Hardened', content)
    
    title = name.replace('.md', '').replace('_', ' ').title()
    if 'PAPER' in title:
        title = title.replace('Paper ', 'CEREBRUM Module: ')
    
    full_content = get_cover_html(title) + content
    
    # Create temp MD in the docs/PDF folder to avoid relative path mess
    temp_md_name = f"__temp_{name}"
    temp_md_path = os.path.join(OUTPUT_DIR, temp_md_name)
    
    with open(temp_md_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    output_pdf = os.path.join(OUTPUT_DIR, name.replace('.md', '.pdf'))
    
    conversion_targets.append({
        "md": temp_md_path,
        "pdf": output_pdf
    })

# Gather all targets
for d in DIRS:
    for f in os.listdir(d):
        if f.endswith('.md'):
            prepare_for_conversion(os.path.join(d, f))

# Write the Master Node Script
node_script_path = os.path.join(MANUSCRIPT_ROOT, 'tmp', 'batch_converter.js')
targets_json = json.dumps(conversion_targets)

node_script_content = f"""
const {{ mdToPdf }} = require('md-to-pdf');
const fs = require('fs');

const targets = {targets_json};
const cssPath = '{CSS_PATH.replace('\\', '/')}';

async function convertAll() {{
    for (const target of targets) {{
        console.log(`Converting ${{target.md}} -> ${{target.pdf}}...`);
        try {{
            const pdf = await mdToPdf({{ path: target.md }}, {{
                stylesheet: cssPath,
                pdf_options: {{
                    format: 'A4',
                    margin: '0mm',
                    printBackground: true
                }},
                launch_options: {{
                    args: ['--no-sandbox', '--disable-setuid-sandbox', '--allow-file-access-from-files']
                }}
            }});
            if (pdf) {{
                fs.writeFileSync(target.pdf, pdf.content);
                // fs.unlinkSync(target.md);
                console.log("   Done.");
            }}
        }} catch (err) {{
            console.error(`   Failed: ${{err}}`);
        }}
    }}
}}

convertAll();
"""

with open(node_script_path, 'w', encoding='utf-8') as f:
    f.write(node_script_content)

print(f"\nCreated batch converter: {node_script_path}")
print(f"Executing conversion for {len(conversion_targets)} files...")

# Run Node
cmd = f'npx -y -p md-to-pdf node "{node_script_path}"'
subprocess.run(cmd, shell=True)

# Cleanup temp MDs
for target in conversion_targets:
    if os.path.exists(target["md"]):
        os.remove(target["md"])

print("\nBatch generation complete.")
