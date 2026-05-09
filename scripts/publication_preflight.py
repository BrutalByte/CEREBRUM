#!/usr/bin/env python3
"""Pre-submission preflight check for CEREBRUM arXiv publication packages.

Checks:
  1. Zero "Claude Sonnet" / "Research Collaborator" co-authorship references
  2. Zero SPEC_xxx.md / PARALLAX.md references (should be [CEREBRUM_REPORT_PLACEHOLDER])
  3. All 6 paper abstracts ≤ 1,920 characters
  4. Canonical benchmark numbers match docs/BENCHMARK_CANONICAL.md
  5. All 6 arxiv_submission/ directories exist and contain required files
  6. All 6 flagship .tex files have exactly one \\author{} block

Usage:
    python scripts/publication_preflight.py
    python scripts/publication_preflight.py --paper 01-flagship
    python scripts/publication_preflight.py --fix-placeholders  # show placeholder counts
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "research" / "papers"
ARXIV_DIR  = REPO_ROOT / "docs" / "arxiv"

# -- Canonical benchmark values (from docs/BENCHMARK_CANONICAL.md) ------------
CANONICAL = {
    "metaqa_1hop_h1":  "46.1%",
    "metaqa_2hop_h1":  "30.0%",
    "metaqa_3hop_h1":  "12.5%",
    "metaqa_1hop_h10": "96.6%",
    "metaqa_2hop_h10": "86.3%",
    "metaqa_3hop_h10": "50.3%",
    "webqsp_h1":        "7.5%",
    "webqsp_h10":      "17.5%",
    "hetionet_h1":       "61%",
    "hetionet_h10":      "85%",
    "hetionet_mrr":      "0.72",
    "ikgwq_50pct_auc":   "0.89",
}

# Numbers that appear in papers but must match canonical values
CANONICAL_PATTERNS = [
    (r"46\.1\s*%",  "metaqa_1hop_h1"),
    (r"30\.0\s*%",  "metaqa_2hop_h1"),
    (r"12\.5\s*%",  "metaqa_3hop_h1"),
    (r"96\.6\s*%",  "metaqa_1hop_h10"),
    (r"86\.3\s*%",  "metaqa_2hop_h10"),
    (r"50\.3\s*%",  "metaqa_3hop_h10"),
]

PAPER_DIRS = [
    "00-technical-report",
    "01-flagship",
    "02-community-detection",
    "03-graph-plasticity",
    "04-federated",
    "05-production",
]

PAPER_TEX = {
    "00-technical-report": "cerebrum-v251-report.tex",
    "01-flagship":          "cerebrum-flagship.tex",
    "02-community-detection": "tsc-paper.tex",
    "03-graph-plasticity":  "plasticity-paper.tex",
    "04-federated":         "holographic-indexing.tex",
    "05-production":        "production-kg.tex",
}

REQUIRED_SUBMISSION_FILES = [
    "cerebrum-macros.sty",
    "notation.tex",
    "author-block.tex",
    "references.bib",
    "README_SUBMISSION.txt",
]

COAUTHOR_PATTERNS = [
    r"Claude\s+Sonnet",
    r"Research\s+Collaborator",
    r"AI\s+Co[-\s]?author",
    r"Anthropic\s+Claude",
]

SPEC_REF_PATTERN   = re.compile(r"SPEC_\d{3}_\w+\.md")
PARX_REF_PATTERN   = re.compile(r"PARALLAX\.md")
PLACEHOLDER_PATTERN = re.compile(r"\[CEREBRUM_REPORT_PLACEHOLDER\]|\[CEREBRUM(?:\\_)?REPORT(?:\\_)?ID\]")


def _fail(msg: str) -> str:
    return f"  FAIL  {msg}"


def _warn(msg: str) -> str:
    return f"  WARN  {msg}"


def _ok(msg: str) -> str:
    return f"  OK    {msg}"


# -- Check 1: no co-authorship references -------------------------------------

def check_coauthorship(targets: list[Path]) -> list[str]:
    issues = []
    pattern = re.compile("|".join(COAUTHOR_PATTERNS), re.IGNORECASE)
    for path in targets:
        text = path.read_text(encoding="utf-8")
        hits = pattern.findall(text)
        if hits:
            issues.append(_fail(f"{path.relative_to(REPO_ROOT)}: co-authorship ref: {hits[:3]}"))
    return issues


# -- Check 2: no bare SPEC_xxx.md references -----------------------------------

def check_spec_refs(targets: list[Path]) -> list[str]:
    issues = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        spec_hits = SPEC_REF_PATTERN.findall(text)
        parx_hits = PARX_REF_PATTERN.findall(text)
        if spec_hits:
            issues.append(_fail(f"{path.relative_to(REPO_ROOT)}: bare SPEC refs: {spec_hits[:3]}"))
        if parx_hits:
            issues.append(_fail(f"{path.relative_to(REPO_ROOT)}: bare PARALLAX.md ref"))
    return issues


# -- Check 3: abstract length ≤ 1,920 chars -----------------------------------

def check_abstract_length(paper_dirs: list[str]) -> list[str]:
    issues = []
    for name in paper_dirs:
        tex_name = PAPER_TEX.get(name)
        if not tex_name:
            continue
        tex_path = PAPERS_DIR / name / tex_name
        if not tex_path.exists():
            issues.append(_warn(f"{name}: {tex_name} not found — cannot check abstract"))
            continue
        text = tex_path.read_text(encoding="utf-8")
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
        if not m:
            issues.append(_warn(f"{name}: no \\begin{{abstract}} found"))
            continue
        abstract = m.group(1).strip()
        # Strip LaTeX commands for a rough character count
        stripped = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", "", abstract)
        stripped = re.sub(r"[{}%]", "", stripped).strip()
        char_count = len(stripped)
        if char_count > 1920:
            issues.append(_fail(f"{name}: abstract {char_count} chars (limit 1920)"))
        else:
            issues.append(_ok(f"{name}: abstract {char_count}/1920 chars"))
    return issues


# -- Check 4: canonical benchmark numbers -------------------------------------

def check_canonical_numbers(targets: list[Path]) -> list[str]:
    issues = []
    # Look for wrong versions of numbers that should match canonical
    # e.g., someone writing 12.4% instead of 12.5% for 3-hop H@1
    wrong_3hop = re.compile(r"(?:3[-–]hop|three[-–]hop)[^%\n]{0,40}H@1[^%\n]{0,20}(1[0-2]\.\d)(?!\s*%\s*(?:of|the))")
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for m in wrong_3hop.finditer(text):
            val = m.group(1)
            if val != "12.5":
                issues.append(_fail(
                    f"{path.relative_to(REPO_ROOT)}: 3-hop H@1 should be 12.5%, found {val}%"
                ))
    return issues


# -- Check 5: submission directory completeness -------------------------------

def check_submission_dirs(paper_dirs: list[str]) -> list[str]:
    results = []
    for name in paper_dirs:
        sub_dir = PAPERS_DIR / name / "arxiv_submission"
        if not sub_dir.exists():
            results.append(_fail(f"{name}/arxiv_submission/: directory missing"))
            continue
        tex_name = PAPER_TEX.get(name)
        files_needed = REQUIRED_SUBMISSION_FILES[:]
        if tex_name:
            files_needed.append(tex_name)
        missing = [f for f in files_needed if not (sub_dir / f).exists()]
        if missing:
            results.append(_fail(f"{name}/arxiv_submission/: missing {missing}"))
        else:
            results.append(_ok(f"{name}/arxiv_submission/: all required files present"))
    return results


# -- Check 6: single \\author{} block per submission .tex ---------------------

def check_author_blocks(paper_dirs: list[str]) -> list[str]:
    results = []
    for name in paper_dirs:
        tex_name = PAPER_TEX.get(name)
        if not tex_name:
            continue
        tex_path = PAPERS_DIR / name / "arxiv_submission" / tex_name
        if not tex_path.exists():
            results.append(_warn(f"{name}: submission {tex_name} not found"))
            continue
        text = tex_path.read_text(encoding="utf-8")
        author_count = len(re.findall(r"\\author\{", text))
        input_author = len(re.findall(r"\\input\{author-block\}", text))
        if author_count == 0 and input_author == 0:
            results.append(_fail(f"{name}: no \\author{{}} or \\input{{author-block}} found"))
        elif author_count > 1:
            results.append(_fail(f"{name}: {author_count} \\author{{}} blocks (expected 1)"))
        else:
            results.append(_ok(f"{name}: author block present"))

        # Also verify no ../shared/ paths leaked into submission copy
        if "\\usepackage{../shared/" in text or "\\input{../shared/" in text:
            results.append(_fail(f"{name}: submission .tex still has ../shared/ paths"))
        else:
            results.append(_ok(f"{name}: no ../shared/ paths in submission copy"))
    return results


# -- Check 7: placeholder count report ----------------------------------------

def report_placeholders(paper_dirs: list[str]) -> list[str]:
    results = []
    total = 0
    for name in paper_dirs:
        tex_name = PAPER_TEX.get(name)
        if not tex_name:
            continue
        tex_path = PAPERS_DIR / name / "arxiv_submission" / tex_name
        if not tex_path.exists():
            continue
        text = tex_path.read_text(encoding="utf-8")
        count = len(PLACEHOLDER_PATTERN.findall(text))
        total += count
        if count > 0:
            results.append(_warn(f"{name}: {count} [CEREBRUM_REPORT_ID] placeholder(s) — replace after Technical Report submission"))
        else:
            results.append(_ok(f"{name}: no report-ID placeholders"))

    also_check = list(ARXIV_DIR.glob("PAPER_*.md"))
    md_count = sum(
        len(re.findall(r"\[CEREBRUM_REPORT_PLACEHOLDER\]", p.read_text(encoding="utf-8")))
        for p in also_check
    )
    if md_count > 0:
        results.append(_warn(f"docs/arxiv/: {md_count} [CEREBRUM_REPORT_PLACEHOLDER] across {len(also_check)} papers"))
    results.append(f"  INFO  Total placeholders pending arXiv ID: {total + md_count}")
    return results


# -- Main ----------------------------------------------------------------------

def run_preflight(paper_dirs: list[str], show_placeholders: bool = False) -> bool:
    print(f"CEREBRUM Publication Preflight — checking {len(paper_dirs)} paper(s)\n")

    all_issues: list[str] = []
    fail_count = 0

    # Collect all .tex and .md files across active docs
    active_docs = (
        list(ARXIV_DIR.glob("PAPER_*.md"))
        + [PAPERS_DIR / n / PAPER_TEX[n] for n in paper_dirs if (PAPERS_DIR / n / PAPER_TEX[n]).exists()]
        + [PAPERS_DIR / n / "arxiv_submission" / PAPER_TEX[n]
           for n in paper_dirs
           if (PAPERS_DIR / n / "arxiv_submission" / PAPER_TEX[n]).exists()]
    )

    print("-- Check 1: Co-authorship references ------------------------------")
    r = check_coauthorship(active_docs)
    all_issues.extend(r)
    for line in r:
        print(line)
    if not r:
        print("  OK    no co-authorship references found")

    print("\n-- Check 2: Bare SPEC_xxx.md / PARALLAX.md references -------------")
    submission_tex = [
        PAPERS_DIR / n / "arxiv_submission" / PAPER_TEX[n]
        for n in paper_dirs
        if (PAPERS_DIR / n / "arxiv_submission" / PAPER_TEX[n]).exists()
    ]
    r2 = check_spec_refs(submission_tex)
    all_issues.extend(r2)
    for line in r2:
        print(line)
    if not r2:
        print("  OK    no bare SPEC refs in submission .tex files")

    print("\n-- Check 3: Abstract length ≤ 1,920 characters --------------------")
    r3 = check_abstract_length(paper_dirs)
    all_issues.extend(r3)
    for line in r3:
        print(line)

    print("\n-- Check 4: Canonical benchmark numbers ----------------------------")
    r4 = check_canonical_numbers(active_docs)
    all_issues.extend(r4)
    for line in r4:
        print(line)
    if not r4:
        print("  OK    no obviously wrong benchmark numbers detected")

    print("\n-- Check 5: Submission directory completeness ----------------------")
    r5 = check_submission_dirs(paper_dirs)
    all_issues.extend(r5)
    for line in r5:
        print(line)

    print("\n-- Check 6: Author blocks ------------------------------------------")
    r6 = check_author_blocks(paper_dirs)
    all_issues.extend(r6)
    for line in r6:
        print(line)

    if show_placeholders:
        print("\n-- Check 7: Placeholder status (informational) ---------------------")
        r7 = report_placeholders(paper_dirs)
        for line in r7:
            print(line)

    fail_count = sum(1 for line in all_issues if "FAIL" in line)
    warn_count = sum(1 for line in all_issues if "WARN" in line)

    print(f"\n{'='*60}")
    if fail_count == 0:
        print(f"PREFLIGHT PASSED  ({warn_count} warning(s))")
        print("\nSubmission order:")
        print("  1. Submit 00-technical-report → get arXiv ID")
        print("  2. python scripts/update_arxiv_papers.py --report-id YYYY.NNNNN")
        print("  3. Submit 01-flagship")
        print("  4. Submit 02–05 in any order")
    else:
        print(f"PREFLIGHT FAILED  {fail_count} error(s), {warn_count} warning(s)")
        print("\nFix all FAIL items before submitting.")

    return fail_count == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", help="Check only this paper dir (e.g. 01-flagship)")
    parser.add_argument("--fix-placeholders", action="store_true",
                        help="Include placeholder count report (informational)")
    args = parser.parse_args()

    targets = [args.paper] if args.paper else PAPER_DIRS
    for t in targets:
        if t not in PAPER_DIRS and t not in {d: d for d in PAPER_DIRS}:
            print(f"Unknown paper: {t}. Valid: {PAPER_DIRS}", file=sys.stderr)
            sys.exit(1)

    ok = run_preflight(targets, show_placeholders=args.fix_placeholders)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
