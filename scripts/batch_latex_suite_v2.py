import os
import re
import subprocess
import shutil

# CEREBRUM: Global LaTeX Publication Engine v2 (v1.2.4)
# Orchestrates the high-precision transition of the entire library using the MiKTeX engine.

MANUSCRIPT_ROOT = os.getcwd()
ACADEMIC_TEMPLATE = os.path.join(MANUSCRIPT_ROOT, 'templates', 'academic_v1.tex')
BROCHURE_TEMPLATE = os.path.join(MANUSCRIPT_ROOT, 'templates', 'brochure_v1.tex')
LATEX_BUILD_DIR = os.path.join(MANUSCRIPT_ROOT, 'docs', 'latex', 'batch_build')
OUTPUT_DIR = os.path.join(MANUSCRIPT_ROOT, 'docs', 'PDF')

# MiKTeX Binary Paths (Full paths for reliability)
PDFLATEX_BIN = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
BIBTEX_BIN = r"C:\Program Files\MiKTeX\miktex\bin\x64\bibtex.exe"

# Import conversion logic from build_arxiv
import sys
sys.path.append(os.path.join(MANUSCRIPT_ROOT, 'scripts'))
import build_arxiv

# Targets
ARXIV_DIR = os.path.join(MANUSCRIPT_ROOT, 'docs', 'arxiv')
ROOT_DOCS = [
    'PAPER.md',
    'CEREBRUM_EXPLAINED.md',
    'Parallax_Plain_Language_Guide_Professional.md',
    'NOVEL_CONTRIBUTIONS.md',
    'API_REFERENCE.md',
    'GLOSSARY.md',
    'DEPLOYMENT.md',
    'REASONING_STUDIO_GUIDE.md'
]

os.makedirs(LATEX_BUILD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pdf(name, md_path, template_path, title, subtitle):
    print(f"--- Processing: {name} ---")
    
    # Safety Check: Verify source file exists and is not empty
    if not os.path.exists(md_path) or os.path.getsize(md_path) == 0:
        print(f"   ❌ Skipping: {md_path} (Missing or empty)")
        return

    if not os.path.exists(template_path) or os.path.getsize(template_path) == 0:
        print(f"   ❌ Skipping: Template {template_path} (Missing or empty)")
        return
    
    with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
        md_content = f.read()

    # 1. Convert Markdown -> LaTeX Snippet
    try:
        tex_snippet = build_arxiv.convert_markdown_to_tex(md_content)
    except Exception as e:
        print(f"   ❌ Conversion failed for {name}: {e}")
        return
    
    # 2. Wrap in Template
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    full_tex = template.replace('[[TITLE]]', title)
    full_tex = full_tex.replace('[[SUBTITLE]]', subtitle)
    full_tex = full_tex.replace('[[CONTENT]]', tex_snippet)
    
    job_name = name.replace('.md', '')
    tex_file = os.path.join(LATEX_BUILD_DIR, f"{job_name}.tex")
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(full_tex)
    
    # 3. Compile (4-pass build for BibTeX)
    print(f"   Compiling LaTeX ({job_name})...")
    
    def run_cmd(cmd_list):
        subprocess.run(cmd_list, cwd=LATEX_BUILD_DIR, capture_output=True, check=False)

    # Clean previous aux files
    for ext in ['.aux', '.log', '.out', '.bbl', '.blg', '.pdf', '.toc']:
        f_path = os.path.join(LATEX_BUILD_DIR, f"{job_name}{ext}")
        if os.path.exists(f_path): os.remove(f_path)

    run_cmd([PDFLATEX_BIN, '-interaction=nonstopmode', f"{job_name}.tex"])
    
    if 'academic' in template_path:
        run_cmd([BIBTEX_BIN, job_name])
        run_cmd([PDFLATEX_BIN, '-interaction=nonstopmode', f"{job_name}.tex"])
    
    run_cmd([PDFLATEX_BIN, '-interaction=nonstopmode', f"{job_name}.tex"])
    
    # 4. Move Result
    final_pdf = os.path.join(LATEX_BUILD_DIR, f"{job_name}.pdf")
    if os.path.exists(final_pdf):
        dest_pdf = os.path.join(OUTPUT_DIR, f"{job_name}.pdf")
        shutil.copy2(final_pdf, dest_pdf)
        print(f"   ✅ Success: {dest_pdf}")
    else:
        print(f"   ❌ Failed to generate PDF for {name}")

# --- Execution ---

# 1. ArXiv Modules
for f in sorted(os.listdir(ARXIV_DIR)):
    if f.endswith('.md'):
        title = f.replace('.md', '').replace('_', ' ').replace('PAPER ', 'Module ').title()
        generate_pdf(f, os.path.join(ARXIV_DIR, f), ACADEMIC_TEMPLATE, title, f"Advanced Graph Attention Analysis ({f.split('_')[1]})")

# 2. Root Docs
for f in ROOT_DOCS:
    f_path = os.path.join(MANUSCRIPT_ROOT, 'docs', f)
    if os.path.exists(f_path):
        is_academic = 'PAPER.md' in f or 'CONTRIBUTIONS' in f
        tmpl = ACADEMIC_TEMPLATE if is_academic else BROCHURE_TEMPLATE
        title = f.replace('.md', '').replace('_', ' ').replace('Parallax ', '').title()
        generate_pdf(f, f_path, tmpl, title, "Framework Implementation Guide")

print("\n--- Global LaTeX Suite Generation Complete ---")
