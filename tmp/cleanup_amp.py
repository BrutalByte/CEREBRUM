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

def clean_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original = content
    for p, r in patterns:
        content = re.sub(p, r, content)
    
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned: {path}")

# Target directories
dirs = [
    'e:/Development/Parallax/docs',
    'e:/Development/Parallax/scripts'
]

file_exts = ('.md', '.tex', '.bib', '.py')

for d in dirs:
    for root, _, files in os.walk(d):
        for name in files:
            if name.endswith(file_exts):
                clean_file(os.path.join(root, name))

print("Cleanup complete.")
