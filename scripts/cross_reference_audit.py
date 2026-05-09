#!/usr/bin/env python3
"""Cross-reference audit for CEREBRUM publication papers.

Verifies that:
  1. Every \\cite{key} in the 6 publication .tex files has a matching entry in references.bib
  2. Every \\ref{label} or \\eqref{label} has a corresponding \\label{label} somewhere
  3. All \\input{...} files referenced in the technical report exist
  4. The master manuscript's \\input{compiled/PAPER_*.tex} targets all exist
  5. No duplicate BibTeX keys across references.bib

Usage:
    python scripts/cross_reference_audit.py
    python scripts/cross_reference_audit.py --paper 01-flagship
    python scripts/cross_reference_audit.py --bib-only
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
PAPERS_DIR  = REPO_ROOT / "research" / "papers"
LATEX_DIR   = REPO_ROOT / "docs" / "latex"
BIB_FILE    = LATEX_DIR / "references.bib"
MASTER_TEX  = LATEX_DIR / "cerebrum_master.tex"

PAPER_TEX = {
    "00-technical-report": "cerebrum-v251-report.tex",
    "01-flagship":          "cerebrum-flagship.tex",
    "02-community-detection": "tsc-paper.tex",
    "03-graph-plasticity":  "plasticity-paper.tex",
    "04-federated":         "holographic-indexing.tex",
    "05-production":        "production-kg.tex",
}


def _fail(msg: str) -> str:
    return f"  FAIL  {msg}"


def _warn(msg: str) -> str:
    return f"  WARN  {msg}"


def _ok(msg: str) -> str:
    return f"  OK    {msg}"


# -- BibTeX key extraction -----------------------------------------------------

def load_bib_keys(bib_path: Path) -> tuple[set[str], list[str]]:
    """Returns (set_of_keys, list_of_duplicate_warnings)."""
    if not bib_path.exists():
        return set(), [_fail(f"{bib_path.relative_to(REPO_ROOT)}: file not found")]
    text = bib_path.read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{([^,\s]+)\s*,", text)
    seen: set[str] = set()
    dupes: list[str] = []
    for k in keys:
        if k in seen:
            dupes.append(_fail(f"references.bib: duplicate key '{k}'"))
        seen.add(k)
    return seen, dupes


# -- Citation → BibTeX resolution ---------------------------------------------

def check_citations(tex_path: Path, bib_keys: set[str]) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    # \cite{key}, \citep{key,key2}, \citet{key}
    cited_raw = re.findall(r"\\cite[pt]?\{([^}]+)\}", text)
    cited: set[str] = set()
    for group in cited_raw:
        for k in group.split(","):
            cited.add(k.strip())

    issues = []
    missing = cited - bib_keys
    if missing:
        for k in sorted(missing):
            issues.append(_fail(f"{tex_path.name}: \\cite{{{k}}} — key not in references.bib"))
    else:
        issues.append(_ok(f"{tex_path.name}: all {len(cited)} citation(s) resolved"))
    return issues


# -- \\ref / \\eqref cross-check -----------------------------------------------

def check_refs(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs   = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text))
    missing = refs - labels
    issues = []
    if missing:
        for r in sorted(missing):
            issues.append(_warn(f"{tex_path.name}: \\ref{{{r}}} — no matching \\label (may be in another file)"))
    else:
        issues.append(_ok(f"{tex_path.name}: all {len(refs)} \\ref(s) have local \\label"))
    return issues


# -- \\input{} target existence ------------------------------------------------

def check_inputs(tex_path: Path, search_root: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{([^}]+)\}", text)
    issues = []
    for inp in inputs:
        # Try relative to tex file's dir, then to search_root
        candidates = [
            tex_path.parent / inp,
            tex_path.parent / (inp + ".tex"),
            search_root / inp,
            search_root / (inp + ".tex"),
        ]
        if not any(c.exists() for c in candidates):
            issues.append(_warn(f"{tex_path.name}: \\input{{{inp}}} — target not found"))
    if not issues:
        issues.append(_ok(f"{tex_path.name}: all {len(inputs)} \\input(s) resolvable"))
    return issues


# -- Master .tex \\input coverage ---------------------------------------------

def check_master_tex() -> list[str]:
    if not MASTER_TEX.exists():
        return [_warn(f"{MASTER_TEX.relative_to(REPO_ROOT)}: not found")]
    text = MASTER_TEX.read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{compiled/(PAPER_[^}]+)\}", text)
    compiled_dir = LATEX_DIR / "compiled"
    issues = []
    for inp in inputs:
        target = compiled_dir / (inp if inp.endswith(".tex") else inp + ".tex")
        if not target.exists():
            issues.append(_fail(f"cerebrum_master.tex: \\input{{compiled/{inp}}} — target not found"))
    if not issues:
        issues.append(_ok(f"cerebrum_master.tex: all {len(inputs)} \\input(compiled/...) targets exist"))
    return issues


# -- Main ----------------------------------------------------------------------

def run_audit(paper_dirs: list[str], bib_only: bool = False) -> bool:
    print(f"CEREBRUM Cross-Reference Audit\n")

    all_lines: list[str] = []
    fail_count = 0

    print("-- BibTeX: key uniqueness ------------------------------------------")
    bib_keys, dupe_lines = load_bib_keys(BIB_FILE)
    for line in dupe_lines:
        print(line)
    if not dupe_lines:
        print(f"  OK    references.bib: {len(bib_keys)} unique key(s), no duplicates")
    all_lines.extend(dupe_lines)

    if bib_only:
        fail_count = sum(1 for l in all_lines if "FAIL" in l)
        print(f"\n{'='*60}")
        print(f"BIB-ONLY mode: {fail_count} failure(s)")
        return fail_count == 0

    print("\n-- Citations → references.bib --------------------------------------")
    for name in paper_dirs:
        tex_name = PAPER_TEX.get(name)
        if not tex_name:
            continue
        # Check both the source and submission copy
        for sub in ["", "arxiv_submission/"]:
            tex_path = PAPERS_DIR / name / sub / tex_name if sub else PAPERS_DIR / name / tex_name
            if tex_path.exists():
                r = check_citations(tex_path, bib_keys)
                all_lines.extend(r)
                for line in r:
                    print(line)

    print("\n-- \\ref / \\eqref completeness --------------------------------------")
    for name in paper_dirs:
        tex_name = PAPER_TEX.get(name)
        if not tex_name:
            continue
        tex_path = PAPERS_DIR / name / tex_name
        if tex_path.exists():
            r = check_refs(tex_path)
            all_lines.extend(r)
            for line in r:
                print(line)

    print("\n-- \\input{} target existence ----------------------------------------")
    for name in paper_dirs:
        tex_name = PAPER_TEX.get(name)
        if not tex_name:
            continue
        tex_path = PAPERS_DIR / name / tex_name
        if tex_path.exists():
            r = check_inputs(tex_path, PAPERS_DIR / name)
            all_lines.extend(r)
            for line in r:
                print(line)

    print("\n-- cerebrum_master.tex \\input(compiled/) coverage ------------------")
    r = check_master_tex()
    all_lines.extend(r)
    for line in r:
        print(line)

    fail_count = sum(1 for l in all_lines if "FAIL" in l)
    warn_count = sum(1 for l in all_lines if "WARN" in l)

    print(f"\n{'='*60}")
    if fail_count == 0:
        print(f"AUDIT PASSED  ({warn_count} warning(s))")
        print("\nNote: \\ref warnings for cross-file labels are expected — verify by compiling with pdflatex.")
    else:
        print(f"AUDIT FAILED  {fail_count} error(s), {warn_count} warning(s)")

    return fail_count == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", help="Audit only this paper (e.g. 01-flagship)")
    parser.add_argument("--bib-only", action="store_true", help="Only check references.bib for duplicates")
    args = parser.parse_args()

    targets = [args.paper] if args.paper else list(PAPER_TEX.keys())
    ok = run_audit(targets, bib_only=args.bib_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
