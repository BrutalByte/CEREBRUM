#!/usr/bin/env python3
"""
Audit CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.docx for unconverted LaTeX / math syntax.
"""
import re
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
doc  = Document(os.path.join(ROOT, 'docs', 'CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.docx'))

def is_code_para(p):
    runs_with_text = [r for r in p.runs if r.text.strip()]
    return runs_with_text and all(r.font.name == 'Courier New' for r in runs_with_text)

backslash_cmd  = re.compile(r'\\[a-zA-Z]+')   # \anything
dollar_sign    = re.compile(r'\$')             # bare $ not in code
raw_brace_grp  = re.compile(r'(?<![a-zA-Z0-9\'])\{[^}]*\}')  # {content} outside LaTeX
lone_caret     = re.compile(r'\^\{[^}]+\}|_\{[^}]+\}')       # ^{...} or _{...} not processed

categories = {
    'backslash_cmd': [],
    'dollar_sign':   [],
    'raw_braces':    [],
}

def para_non_code_text(p):
    """Return paragraph text with Courier New run content replaced by spaces."""
    parts = []
    for r in p.runs:
        if r.font.name == 'Courier New':
            parts.append(' ' * len(r.text))
        else:
            parts.append(r.text)
    return ''.join(parts)

for i, p in enumerate(doc.paragraphs):
    t = p.text
    if not t.strip():
        continue
    if is_code_para(p):
        continue
    # Use only non-code-span text for pattern matching
    t = para_non_code_text(p)
    if not t.strip():
        continue

    if backslash_cmd.search(t):
        for m in backslash_cmd.finditer(t):
            categories['backslash_cmd'].append((i, m.group(), t[:110]))
            break  # one entry per paragraph

    if dollar_sign.search(t):
        categories['dollar_sign'].append((i, '$', t[:110]))

    if lone_caret.search(t):
        for m in lone_caret.finditer(t):
            categories['raw_braces'].append((i, m.group(), t[:110]))
            break

total = sum(len(v) for v in categories.values())
print(f"Total non-code paragraphs with residual LaTeX: {total}")
print()

for cat, items in categories.items():
    if items:
        print(f"=== {cat}: {len(items)} paragraphs ===")
        for idx, match, text in items[:20]:
            print(f"  [{idx:4d}] match={match!r:20s}  text={text}")
        print()

if total == 0:
    print("PASS — no unconverted LaTeX syntax found outside code blocks.")
