#!/usr/bin/env python3
"""
Fix authorship, internal citations, and add acknowledgments across all
CEREBRUM documentation files targeted for arXiv/external distribution.

Changes made:
  1. "Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)"
     → "Bryan Alexander Buchorn"
  2. "Independent Researcher · Anthropic"
     → "Independent Researcher, Las Vegas, NV, USA"
  3. Ref-list entries: "Buchorn, B. A., & Sonnet, C. (2026). ... SPEC_xxx.md."
     → "Buchorn, B. A. (2026). CEREBRUM v2.51: Complete Technical Specification. [CEREBRUM_REPORT_PLACEHOLDER]"
  4. Inline body citations: (SPEC_xxx) or PARALLAX.md
     → [Buchorn, 2026]
  5. Adds Acknowledgments section before ## References in each paper file
  6. Fixes LaTeX author block in cerebrum_master.tex
  7. Updates version string in academic_v1.tex

Run: python scripts/fix_authorship.py
"""
import re
import os
import glob
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUTHOR_OLD = r'\*\*Authors\*\*: Bryan Alexander Buchorn · Claude Sonnet 4\.6 \(Research Collaborator\)\s*'
AUTHOR_NEW = '**Author**: Bryan Alexander Buchorn  \n'

AFFIL_OLD = r'\*\*Affiliations\*\*: Independent Researcher · Anthropic\s*'
AFFIL_NEW = '**Affiliation**: Independent Researcher, Las Vegas, NV, USA  \n'

# Ref-list entries with SPEC_xxx.md or PARALLAX.md
REF_SPEC_OLD = re.compile(
    r'Buchorn, B\. A\., & Sonnet, C\. \(2026\)\. (.+?)\. (?:SPEC_\d+\.md|PARALLAX\.md)\.'
)
REF_SPEC_NEW = (
    r'Buchorn, B. A. (2026). CEREBRUM v2.51: Complete Technical Specification '
    r'for Autonomous Knowledge Graph Reasoning. [CEREBRUM_REPORT_PLACEHOLDER].'
)

# Inline body references like (SPEC_001) or (SPEC_005)
INLINE_SPEC_OLD = re.compile(r'\(SPEC_\d{3}\)')
INLINE_SPEC_NEW = '[Buchorn, 2026]'

# Also replace bare SPEC_xxx in text not wrapped in parens
BARE_SPEC_OLD = re.compile(r'\bSPEC_\d{3}(?:\.md)?\b')
BARE_SPEC_NEW = '[Buchorn, 2026]'

ACKNOWLEDGMENTS = """\n## Acknowledgments

The author gratefully acknowledges the use of Claude (Anthropic) as a research assistant \
throughout this work. Claude assisted with mathematical formalization, code generation, \
manuscript preparation, and technical writing. All conceptual contributions, architectural \
decisions, experimental design, and intellectual claims are solely the author's.

"""

def fix_markdown_file(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()

    original = text

    # 1. Fix author line
    text = re.sub(AUTHOR_OLD, AUTHOR_NEW, text)

    # 2. Fix affiliation line
    text = re.sub(AFFIL_OLD, AFFIL_NEW, text)

    # 3. Fix reference list entries citing SPEC files
    text = REF_SPEC_OLD.sub(REF_SPEC_NEW, text)

    # 4. Fix inline (SPEC_xxx) citations
    text = INLINE_SPEC_OLD.sub(INLINE_SPEC_NEW, text)

    # 5. Fix bare SPEC_xxx mentions
    text = BARE_SPEC_OLD.sub(BARE_SPEC_NEW, text)

    # 6. Fix PARALLAX.md references in text
    text = text.replace('PARALLAX.md', '[Buchorn, 2026]')

    # 7. Add acknowledgments section if not already present
    if '## Acknowledgments' not in text and '## References' in text:
        text = text.replace('## References', ACKNOWLEDGMENTS + '## References', 1)

    if text != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    return False


def fix_latex_file(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()

    original = text

    # Remove Claude author block (IEEEtran format)
    text = re.sub(
        r'\s*\\and\s*\n\s*\\IEEEauthorblockN\{Claude Sonnet[^}]*\}\s*\n'
        r'\s*\\IEEEauthorblockA\{[^}]*\}\s*',
        '\n',
        text
    )

    # Also fix any \and Claude blocks in article format
    text = re.sub(
        r'\s*\\and\s*Claude\s+Sonnet[^\n]*\n?',
        '\n',
        text
    )

    # Fix affiliation if it still shows Anthropic
    text = re.sub(
        r'\\IEEEauthorblockA\{\\textit\{Independent Research\}[^}]*\}',
        r'\\IEEEauthorblockA{Independent Researcher, Las Vegas, NV, USA \\\\\n'
        r'        \\texttt{bryan.buchorn@gmail.com}}',
        text
    )

    # Fix plain \author{} block if present
    text = re.sub(
        r'\\author\{[^}]*Claude Sonnet[^}]*\}',
        r'\\author{Bryan Alexander Buchorn \\\\ Independent Researcher, Las Vegas, NV, USA \\\\ \\texttt{bryan.buchorn@gmail.com}}',
        text
    )

    if text != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    return False


def fix_template_version(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    original = text
    # Update stale version strings
    text = re.sub(r'v1\.\d+\.\d+ · \w+ 2026', 'v2.51.1 · May 2026', text)
    text = re.sub(r'v2\.\d+\.\d+ · \w+ 2026', 'v2.51.1 · May 2026', text)
    if text != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    return False


def main():
    changed = 0
    skipped = 0

    # --- Markdown files ---
    md_targets = [
        os.path.join(ROOT, 'docs', 'CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md'),
        os.path.join(ROOT, 'docs', 'CEREBRUM_BENCHMARK_COMPARISON_PAPER.md'),
    ]

    # All arxiv papers
    md_targets += glob.glob(os.path.join(ROOT, 'docs', 'arxiv', '*.md'))

    # All spec files
    md_targets += glob.glob(os.path.join(ROOT, 'docs', 'specifications', 'SPEC_*.md'))

    for path in md_targets:
        if not os.path.exists(path):
            print(f'  SKIP (not found): {os.path.relpath(path, ROOT)}')
            skipped += 1
            continue
        if fix_markdown_file(path):
            print(f'  FIXED: {os.path.relpath(path, ROOT)}')
            changed += 1
        else:
            print(f'  ok   : {os.path.relpath(path, ROOT)}')

    # --- LaTeX files ---
    latex_targets = [
        os.path.join(ROOT, 'docs', 'latex', 'cerebrum_master.tex'),
    ]
    for path in latex_targets:
        if not os.path.exists(path):
            print(f'  SKIP (not found): {os.path.relpath(path, ROOT)}')
            skipped += 1
            continue
        if fix_latex_file(path):
            print(f'  FIXED: {os.path.relpath(path, ROOT)}')
            changed += 1
        else:
            print(f'  ok   : {os.path.relpath(path, ROOT)}')

    # --- Template version strings ---
    template_targets = glob.glob(os.path.join(ROOT, 'docs', 'latex', 'templates', '*.tex'))
    for path in template_targets:
        if fix_template_version(path):
            print(f'  FIXED (version): {os.path.relpath(path, ROOT)}')
            changed += 1

    print()
    print(f'Done. {changed} files modified, {skipped} skipped.')

    # --- Final verification ---
    print()
    print('=== Post-fix verification ===')
    sonnet_count = 0
    spec_count = 0
    for root_dir, dirs, files in os.walk(os.path.join(ROOT, 'docs')):
        # Skip archive
        dirs[:] = [d for d in dirs if d != 'archive']
        for fname in files:
            if not (fname.endswith('.md') or fname.endswith('.tex')):
                continue
            fpath = os.path.join(root_dir, fname)
            try:
                content = open(fpath, encoding='utf-8', errors='ignore').read()
                if 'Claude Sonnet' in content or 'Research Collaborator' in content:
                    sonnet_count += 1
                    print(f'  STILL HAS Claude: {os.path.relpath(fpath, ROOT)}')
                if re.search(r'SPEC_\d{3}\.md', content):
                    spec_count += 1
                    print(f'  STILL HAS SPEC ref: {os.path.relpath(fpath, ROOT)}')
            except Exception:
                pass

    if sonnet_count == 0:
        print('  PASS: Zero "Claude Sonnet" occurrences in docs/')
    else:
        print(f'  FAIL: {sonnet_count} files still have Claude authorship')

    if spec_count == 0:
        print('  PASS: Zero SPEC_xxx.md references in docs/')
    else:
        print(f'  FAIL: {spec_count} files still have internal SPEC citations')


if __name__ == '__main__':
    main()
