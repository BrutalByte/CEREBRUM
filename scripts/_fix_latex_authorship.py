#!/usr/bin/env python3
"""Fix Claude authorship in compiled/batch_build LaTeX files and ARXIV_SUBMISSION_GUIDE."""
import sys, io, re, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
changed = 0

targets = (
    glob.glob(os.path.join(ROOT, 'docs', 'latex', 'compiled', '*.tex')) +
    glob.glob(os.path.join(ROOT, 'docs', 'latex', 'batch_build', '*.tex')) +
    [os.path.join(ROOT, 'docs', 'ARXIV_SUBMISSION_GUIDE.md')]
)

LATEX_CLAUDE_BLOCK = re.compile(
    r'\\and\s*\n\s*\\IEEEauthorblockN\{Claude Sonnet[^}]*\}\s*\n\s*\\IEEEauthorblockA\{[^}]*\}',
    re.MULTILINE
)

for path in targets:
    if not os.path.exists(path):
        continue
    try:
        text = open(path, encoding='utf-8', errors='replace').read()
    except Exception as e:
        print(f'  ERROR reading {path}: {e}')
        continue
    orig = text

    if path.endswith('.tex'):
        text = LATEX_CLAUDE_BLOCK.sub('', text)
        # Handle the compiled LaTeX format:
        # \textbf{Authors}: Bryan Alexander Buchorn $\cdot$ Claude Sonnet 4.6 (Research Collab\-orator)
        # Use literal string matching with a broad regex on the Claude part
        for variant in ['4.6', '4.5', '4.7', '3', '3.5', '4']:
            old = f' $\\cdot$ Claude Sonnet {variant} (Research Collab\\-orator)'
            text = text.replace(old, '')
            old2 = f' $\\cdot$ Claude Sonnet {variant} (Research Collaborator)'
            text = text.replace(old2, '')
        # Fix affiliations line
        text = text.replace(
            'Independent Researcher $\\cdot$ Anthropic',
            'Independent Researcher, Las Vegas, NV, USA'
        )
        # Catch any remaining generic form
        text = re.sub(r' \$\\cdot\$ Claude Sonnet [^\n(]*\([^)]*\)', '', text)
        text = text.replace('Claude Sonnet 4.6 (Research Collaborator)', '')
        text = text.replace('Claude Sonnet 4.5 (Research Collaborator)', '')
    else:
        # Markdown ARXIV_SUBMISSION_GUIDE
        text = re.sub(
            r'\*\*Authors?\*\*: Bryan Alexander Buchorn [·•] Claude Sonnet [0-9.]+ \(Research Collaborator\)\s*',
            '**Author**: Bryan Alexander Buchorn  \n', text
        )
        text = re.sub(
            r'\*\*Affiliations?\*\*: Independent Researcher [·•] Anthropic\s*',
            '**Affiliation**: Independent Researcher, Las Vegas, NV, USA  \n', text
        )
        # The guide likely just mentions Claude as a tool — only strip co-author claims
        text = re.sub(
            r'Bryan Alexander Buchorn · Claude Sonnet [0-9.]+',
            'Bryan Alexander Buchorn',
            text
        )

    if text != orig:
        open(path, 'w', encoding='utf-8').write(text)
        print(f'  FIXED: {os.path.relpath(path, ROOT)}')
        changed += 1

print(f'\nDone: {changed} files fixed')
