import os
import re
import subprocess
import shutil
import sys

# CEREBRUM: Global Publication Rebuild Engine (v1.2.4)
# Orchestrates the high-precision transition of the entire library using the MiKTeX engine.
# Final PDFs are moved to the docs/ directory, overwriting existing ones.

MANUSCRIPT_ROOT = 'e:/Development/Parallax'
ACADEMIC_TEMPLATE = f'{MANUSCRIPT_ROOT}/templates/academic_v1.tex'
BROCHURE_TEMPLATE = f'{MANUSCRIPT_ROOT}/templates/brochure_v1.tex'
LATEX_BUILD_DIR = f'{MANUSCRIPT_ROOT}/docs/latex/batch_build'
DOCS_DIR = f'{MANUSCRIPT_ROOT}/docs'
ARXIV_DIR = f'{DOCS_DIR}/arxiv'
PDF_STORAGE_DIR = f'{DOCS_DIR}/PDF'

# MiKTeX Binary Paths
PDFLATEX_BIN = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
BIBTEX_BIN = r"C:\Program Files\MiKTeX\miktex\bin\x64\bibtex.exe"

# Import conversion logic from build_arxiv
sys.path.append(f'{MANUSCRIPT_ROOT}/scripts')
import build_arxiv

# Ensure directories exist
os.makedirs(LATEX_BUILD_DIR, exist_ok=True)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

ACADEMIC_DOCS = [
    'PAPER.md',
    'NOVEL_CONTRIBUTIONS.md',
    'Parallax_White_Paper.md',
    'ARXIV_SUBMISSION_GUIDE.md',
    'CEREBRUM_EXPLAINED.md'
]

SKIP_FILES = [
    'README.md',
    'CONTRIBUTING.md',
    'LICENSE',
    'SECURITY.md',
    'TESTING.md',
    'CLAUDE.md',
    'GEMINI.md'
]

def generate_pdf(name, md_path, template_path, title, subtitle):
    print(f"--- Processing: {name} ---")
    
    try:
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
            md_content = f.read()

        # 1. Convert Markdown -> LaTeX Snippet
        tex_snippet = build_arxiv.convert_markdown_to_tex(md_content)
        
        # 2. Wrap in Template
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        full_tex = template.replace('[[TITLE]]', title)
        full_tex = full_tex.replace('[[SUBTITLE]]', subtitle)
        full_tex = full_tex.replace('[[CONTENT]]', tex_snippet)
        
        job_name = name.replace('.md', '')
        # Remove spaces and underscores for LaTeX job name
        job_name_tex = job_name.replace(' ', '_').replace('.', '_')
        tex_file = os.path.join(LATEX_BUILD_DIR, f"{job_name_tex}.tex")
        
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(full_tex)
        
        # 3. Compile
        print(f"   Compiling LaTeX ({job_name_tex})...")
        
        def run_cmd(cmd_list):
            return subprocess.run(cmd_list, cwd=LATEX_BUILD_DIR, capture_output=True, check=False)

        # Clean previous aux files
        for ext in ['.aux', '.log', '.out', '.bbl', '.blg', '.pdf', '.toc']:
            f_path = os.path.join(LATEX_BUILD_DIR, f"{job_name_tex}{ext}")
            if os.path.exists(f_path): os.remove(f_path)

        run_cmd([PDFLATEX_BIN, '-interaction=nonstopmode', f"{job_name_tex}.tex"])
        
        if template_path == ACADEMIC_TEMPLATE:
            # Check if bib file exists
            bib_path = os.path.join(DOCS_DIR, 'latex', 'references.bib')
            if os.path.exists(bib_path):
                run_cmd([BIBTEX_BIN, job_name_tex])
                run_cmd([PDFLATEX_BIN, '-interaction=nonstopmode', f"{job_name_tex}.tex"])
        
        run_cmd([PDFLATEX_BIN, '-interaction=nonstopmode', f"{job_name_tex}.tex"])
        
        # 4. Move Result
        generated_pdf = os.path.join(LATEX_BUILD_DIR, f"{job_name_tex}.pdf")
        if os.path.exists(generated_pdf):
            # Destination in docs/
            dest_pdf_root = os.path.join(DOCS_DIR, f"{job_name}.pdf")
            # Destination in docs/PDF/
            dest_pdf_storage = os.path.join(PDF_STORAGE_DIR, f"{job_name}.pdf")
            
            # Copy to both locations
            shutil.copy2(generated_pdf, dest_pdf_root)
            shutil.copy2(generated_pdf, dest_pdf_storage)
            print(f"   ✅ Success: {dest_pdf_root}")
        else:
            print(f"   ❌ Failed to generate PDF for {name}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    # 1. Process Root Docs
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith('.md') and f not in SKIP_FILES:
            md_path = os.path.join(DOCS_DIR, f)
            tmpl = ACADEMIC_TEMPLATE if f in ACADEMIC_DOCS else BROCHURE_TEMPLATE
            title = f.replace('.md', '').replace('_', ' ').replace('Parallax ', '').title()
            generate_pdf(f, md_path, tmpl, title, "Framework Implementation Guide")

    # 2. Process ArXiv Modules
    for f in sorted(os.listdir(ARXIV_DIR)):
        if f.endswith('.md'):
            md_path = os.path.join(ARXIV_DIR, f)
            title = f.replace('.md', '').replace('_', ' ').replace('PAPER ', 'Module ').title()
            generate_pdf(f, md_path, ACADEMIC_TEMPLATE, title, f"Advanced Graph Attention Analysis")

    print("\n--- Global Publication Rebuild Complete ---")

if __name__ == "__main__":
    main()
