import os
import re

def sync_files():
    replacements = [
        # Standard vX.Y.Z
        (r'v2\.51(\.\d+)?', 'v2.52.0'),
        (r'v2\.24(\.\d+)?', 'v2.52.0'),
        # Bare X.Y.Z in assignments
        (r'(version\s*[:=]\s*")2\.24(\.\d+)?(")', r'\g<1>2.52.0\g<3>'),
        (r'(version\s*[:=]\s*")1\.3(\.\d+)?(")', r'\g<1>2.52.0\g<3>'),
        (r'(version\s*[:=]\s*")2\.4(\.\d+)?(")', r'\g<1>2.52.0\g<3>'),
        (r'(version\s*[:=]\s*")2\.51(\.\d+)?(")', r'\g<1>2.52.0\g<3>'),
        # Phase mappings
        (r'Phase 16[0-7]', 'Phase 172'),
        (r'Phase 172', 'Phase 172'),
        # Test counts
        (r'217[0-6]\+ tests', '2177+ tests'),
        (r'2177+ tests', '2177+ tests'),
        (r'1,490\+ passing tests', '2177+ tests'),
    ]

    # Progression fixes
    progression_patterns = [
        # Headers
        (r'v2\.52\.0\s*(->|→)\s*v2\.52\.0', 'v2.51.1 -> v2.52.0'),
        # Table comparison headers
        (r'\(v2\.52\.0\)(?=.*\(v2\.52\.0 Phase 172\))', '(v2.52.0)'),
        # Textual progression
        (r'between v2\.52\.0 and v2\.52\.0', 'between v2.51.1 and v2.52.0'),
        (r'Prior to v2\.52\.0', 'Prior to v2.51.1'),
        (r'at v2\.52\.0', 'at v2.51.1'),
        (r'since v2\.52\.0', 'since v2.51.1'),
        (r'Since v2\.52\.0', 'Since v2.51.1'),
        (r'at Phase 167', 'at Phase 167'),
        (r'from Phase 167', 'from Phase 167'),
    ]

    files_to_process = []
    # Broad search for relevant file types
    extensions = ('.md', '.tex', '.sty', '.toml', '.py', '.txt', '.sh', '.json')
    ignore_dirs = {'.git', '.mypy_cache', '.pytest_cache', '.ruff_cache', '__pycache__', 'tmp'}

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if file.endswith(extensions):
                files_to_process.append(os.path.join(root, file))

    updated_count = 0
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        new_content = content
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, new_content)
        
        # After initial replacements, check for broken progression
        for pattern, replacement in progression_patterns:
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            updated_count += 1
            print(f"Updated: {file_path}")

    print(f"\nTotal files updated: {updated_count}")

if __name__ == "__main__":
    sync_files()
