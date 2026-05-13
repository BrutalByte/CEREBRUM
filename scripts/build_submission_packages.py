#!/usr/bin/env python3
"""Build arXiv submission zip packages for all 6 publication papers.

Creates research/papers/XX-*/arxiv_submission/ with:
  - main .tex file
  - cerebrum-macros.sty (shared macros)
  - notation.tex (shared notation)
  - author-block.tex (shared author info)
  - references.bib (bibliography)
  - README_SUBMISSION.txt

Usage:
    python scripts/build_submission_packages.py
    python scripts/build_submission_packages.py --paper 01-flagship
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "research" / "papers"
SHARED_DIR = PAPERS_DIR / "shared"
BIB_FILE   = REPO_ROOT / "docs" / "latex" / "references.bib"

PAPERS = {
    "00-technical-report": {
        "tex":     "cerebrum-v252-report.tex",
        "title":   "CEREBRUM v2.52: Complete Technical Specification",
        "arxiv":   "cs.AI",
        "notes":   "Submit first — other papers reference this arXiv ID.",
    },
    "01-flagship": {
        "tex":     "cerebrum-flagship.tex",
        "title":   "CEREBRUM: Training-Free KG Reasoning via Community-Structured Graph Attention",
        "arxiv":   "cs.AI",
        "notes":   "Submit second. Replace [CEREBRUM_REPORT_ID] with Technical Report arXiv ID first.",
    },
    "02-community-detection": {
        "tex":     "tsc-paper.tex",
        "title":   "Triple-Signal Consensus: Temperature-Annealed Community Detection",
        "arxiv":   "cs.SI",
        "notes":   "Fill in LFR benchmark values (run scripts/run_lfr_benchmark.py) before submitting.",
    },
    "03-graph-plasticity": {
        "tex":     "plasticity-paper.tex",
        "title":   "Experience-Dependent Structural Plasticity in Knowledge Graphs",
        "arxiv":   "cs.AI",
        "notes":   "Submit after flagship.",
    },
    "04-federated": {
        "tex":     "holographic-indexing.tex",
        "title":   "Holographic Indexing: Privacy-Preserving Discovery in Federated KG Networks",
        "arxiv":   "cs.DC",
        "notes":   "Submit after flagship.",
    },
    "05-production": {
        "tex":     "production-kg.tex",
        "title":   "Production Knowledge Graph Reasoning: Fault Tolerance, Streaming, Maintenance",
        "arxiv":   "cs.SE",
        "notes":   "Submit after flagship.",
    },
}


def build_package(paper_dir_name: str) -> None:
    info = PAPERS[paper_dir_name]
    paper_dir = PAPERS_DIR / paper_dir_name
    sub_dir   = paper_dir / "arxiv_submission"

    if not paper_dir.exists():
        print(f"SKIP  {paper_dir_name}: directory not found")
        return

    sub_dir.mkdir(exist_ok=True)

    # Copy main .tex
    tex_src = paper_dir / info["tex"]
    if tex_src.exists():
        shutil.copy2(tex_src, sub_dir / info["tex"])
    else:
        print(f"WARN  {paper_dir_name}: {info['tex']} not found")

    # Copy shared files (flattened into submission dir so \usepackage{cerebrum-macros} works)
    for shared_file in ["cerebrum-macros.sty", "notation.tex", "author-block.tex"]:
        src = SHARED_DIR / shared_file
        if src.exists():
            shutil.copy2(src, sub_dir / shared_file)

    # Copy bibliography
    if BIB_FILE.exists():
        shutil.copy2(BIB_FILE, sub_dir / "references.bib")

    # For the technical report, regenerate sections/ if source .md files are newer
    md_sections_dir = paper_dir / "sections"
    tex_sections_dir = sub_dir / "sections"
    if md_sections_dir.exists():
        md_files = list(md_sections_dir.glob("*.md"))
        tex_files = list(tex_sections_dir.glob("*.tex")) if tex_sections_dir.exists() else []
        if len(tex_files) < len(md_files):
            print(f"  running section converter for {paper_dir_name}...")
            import subprocess
            subprocess.run(
                ["python", str(REPO_ROOT / "scripts" / "convert_sections_to_tex.py")],
                check=True,
            )

    # Write README
    readme = sub_dir / "README_SUBMISSION.txt"
    readme.write_text(
        f"arXiv Submission Package\n"
        f"========================\n\n"
        f"Paper:    {info['title']}\n"
        f"Category: {info['arxiv']}\n"
        f"Main file: {info['tex']}\n\n"
        f"Build:\n"
        f"  pdflatex {info['tex']}\n"
        f"  bibtex {info['tex'].replace('.tex', '')}\n"
        f"  pdflatex {info['tex']}\n"
        f"  pdflatex {info['tex']}\n\n"
        f"Pre-submission notes:\n"
        f"  {info['notes']}\n\n"
        f"After building, update shared/ imports in the .tex file to local paths:\n"
        f"  \\usepackage{{cerebrum-macros}} (already in submission dir)\n"
        f"  \\input{{notation}} (already in submission dir)\n"
        f"  \\input{{author-block}} (already in submission dir)\n\n"
        f"Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with\n"
        f"the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX\n",
        encoding="utf-8",
    )

    print(f"  built {sub_dir.relative_to(REPO_ROOT)}")


def fix_tex_paths(paper_dir_name: str) -> None:
    """Rewrite ../shared/ paths to local paths in the submission copy."""
    info = PAPERS[paper_dir_name]
    sub_dir = PAPERS_DIR / paper_dir_name / "arxiv_submission"
    tex_path = sub_dir / info["tex"]

    if not tex_path.exists():
        return

    text = tex_path.read_text(encoding="utf-8")
    text = text.replace("\\usepackage{../shared/cerebrum-macros}", "\\usepackage{cerebrum-macros}")
    text = text.replace("\\input{../shared/notation}", "\\input{notation}")
    text = text.replace("\\input{../shared/author-block}", "\\input{author-block}")
    # Also fix the bibliography path for the technical report
    text = text.replace("\\bibliography{../../docs/latex/references}", "\\bibliography{references}")
    tex_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", help="Build only this paper (e.g. 01-flagship)")
    args = parser.parse_args()

    targets = [args.paper] if args.paper else list(PAPERS.keys())
    print(f"Building {len(targets)} submission package(s)...")
    for name in targets:
        if name not in PAPERS:
            print(f"Unknown paper: {name}. Valid: {list(PAPERS.keys())}")
            continue
        build_package(name)
        fix_tex_paths(name)
    print("Done.")


if __name__ == "__main__":
    main()
