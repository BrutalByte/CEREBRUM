#!/usr/bin/env python3
"""Replace [CEREBRUM_REPORT_PLACEHOLDER] and [CEREBRUM_REPORT_ID] with the
actual arXiv ID once the Technical Report has been submitted.

Usage:
    python scripts/update_arxiv_papers.py --report-id 2026.XXXXX
    python scripts/update_arxiv_papers.py --report-id 2026.XXXXX --dry-run

Targets:
    - docs/arxiv/PAPER_*.md
    - docs/latex/compiled/PAPER_*.tex
    - research/papers/**/*.tex
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PLACEHOLDER_MD   = "[CEREBRUM_REPORT_PLACEHOLDER]"
PLACEHOLDER_TEX  = "[CEREBRUM\\_REPORT\\_ID]"          # LaTeX-escaped version
PLACEHOLDER_TEX2 = "[CEREBRUM_REPORT_ID]"              # unescaped version in .tex

TARGETS = [
    ("docs/arxiv",             "*.md",   PLACEHOLDER_MD),
    ("docs/latex/compiled",    "*.tex",  PLACEHOLDER_TEX),
    ("docs/latex/compiled",    "*.tex",  PLACEHOLDER_TEX2),
    ("research/papers",        "**/*.tex", PLACEHOLDER_TEX),
    ("research/papers",        "**/*.tex", PLACEHOLDER_TEX2),
]


def make_arxiv_citation_md(arxiv_id: str) -> str:
    return f"arXiv:{arxiv_id} [cs.AI]"


def make_arxiv_citation_tex(arxiv_id: str) -> str:
    return f"arXiv:{arxiv_id}"


def update_files(arxiv_id: str, dry_run: bool) -> None:
    changed = []

    for rel_dir, pattern, placeholder in TARGETS:
        target_dir = REPO_ROOT / rel_dir
        if not target_dir.exists():
            continue

        for path in sorted(target_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            if placeholder not in text:
                continue

            if ".md" in path.suffix:
                replacement = make_arxiv_citation_md(arxiv_id)
            else:
                replacement = make_arxiv_citation_tex(arxiv_id)

            new_text = text.replace(placeholder, replacement)
            if dry_run:
                count = text.count(placeholder)
                print(f"DRY RUN  {path.relative_to(REPO_ROOT)}  ({count} replacements)")
            else:
                path.write_text(new_text, encoding="utf-8")
                changed.append(path.relative_to(REPO_ROOT))

    if not dry_run:
        print(f"Updated {len(changed)} files with arXiv ID {arxiv_id}:")
        for p in changed:
            print(f"  {p}")
    elif not changed and not dry_run:
        print("No files contained placeholder text.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-id",
        required=True,
        metavar="ARXIV_ID",
        help="arXiv ID of the Technical Report (e.g. 2026.12345)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files",
    )
    args = parser.parse_args()

    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", args.report_id):
        print(f"ERROR: '{args.report_id}' does not look like an arXiv ID (expected YYYY.NNNNN)",
              file=sys.stderr)
        sys.exit(1)

    update_files(args.report_id, args.dry_run)


if __name__ == "__main__":
    main()
