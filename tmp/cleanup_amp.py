import os
import re

# Comprehensive global removal of "AMP" callsign
# Regex strategy: Match "(AMP)", " · AMP", " / AMP", and variants, excluding "AMPA"

patterns = [
    (r' \(AMP\)', ''),
    (r' · AMP', ''),
    (r'·  AMP', ''),
    (r' / AMP', ''),
    (r' /AMP', ''),
    (r'AMP / ', '/ '),
    (r'AMP Independent Research', 'Independent Research'),
    (r'Bryan Alexander Buchorn  ·  AMP', 'Bryan Alexander Buchorn'),
    (r'Bryan Alexander Buchorn  · AMP', 'Bryan Alexander Buchorn'),
]

# Target directories
MANUSCRIPT_ROOT = os.getcwd()
dirs = [
    os.path.join(MANUSCRIPT_ROOT, 'docs'),
    os.path.join(MANUSCRIPT_ROOT, 'scripts')
]

file_exts = ('.md', '.tex', '.bib', '.py')

def clean_file(path):
    # Safety Check: Verify source file exists and is not empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original = content
    for p, r in patterns:
        content = re.sub(p, r, content)
    
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned: {path}")

for d in dirs:
    if os.path.exists(d):
        for root, _, files in os.walk(d):
            for name in files:
                if name.endswith(file_exts):
                    clean_file(os.path.join(root, name))

print("Cleanup complete.")
