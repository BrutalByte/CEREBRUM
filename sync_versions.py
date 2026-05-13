import os
import re

def sync_files():
    replacements = [
        # Standard vX.Y.Z
        (r'v2\.51\.[01]', 'v2.52.0'),
        (r'v2\.24\.0', 'v2.52.0'),
        # Bare X.Y.Z in assignments
        (r'(version\s*[:=]\s*")2\.24\.0(")', r'\g<1>2.52.0\g<2>'),
        (r'(version\s*[:=]\s*")1\.3\.0(")', r'\g<1>2.52.0\g<2>'),
        (r'(version\s*[:=]\s*")2\.4\.0(")', r'\g<1>2.52.0\g<2>'),
        # Phase mappings
        (r'Phase 167', 'Phase 172'),
        (r'Phase 112', 'Phase 172'),
        # Test counts
        (r'2175\+ tests', '2177+ tests'),
        (r'1,357 passing tests', '2177+ tests'),
    ]

    # Progression fixes
    progression_patterns = [
        # Headers
        (r'v2\.52\.0\s*(->|→)\s*v2\.52\.0', 'v2.51.1 -> v2.52.0'),
        # Table comparison headers (e.g. | H@1 (v2.52.0) | H@1 (v2.52.0 Phase 172) |)
        (r'\(v2\.52\.0\)(?=.*\(v2\.52\.0 Phase 172\))', '(v2.51.1)'),
        # Textual progression
        (r'between v2\.52\.0 and v2\.52\.0', 'between v2.51.1 and v2.52.0'),
        (r'Prior to v2\.52\.0', 'Prior to v2.51.1'),
        (r'at v2\.52\.0', 'at v2.51.1'),
        (r'since v2\.52\.0', 'since v2.51.1'),
        (r'Since v2\.52\.0', 'Since v2.51.1'),
        (r'at Phase 172', 'at Phase 167'),
        (r'from Phase 172', 'from Phase 167'),
    ]

    files_to_process = []
    # Directories to search
    target_dirs = ['docs', 'research/papers', 'templates', 'core', 'api']
    
    for target in target_dirs:
        if not os.path.exists(target):
            continue
        for root, dirs, files in os.walk(target):
            for file in files:
                if file.endswith(('.md', '.tex', '.sty')):
                    files_to_process.append(os.path.join(root, file))
    
    if os.path.exists('README.md'):
        files_to_process.append('README.md')
    if os.path.exists('GEMINI.md'):
        files_to_process.append('GEMINI.md')
    if os.path.exists('pyproject.toml'):
        files_to_process.append('pyproject.toml')
    if os.path.exists('api/server.py'):
        files_to_process.append('api/server.py')
    if os.path.exists('core/telemetry.py'):
        files_to_process.append('core/telemetry.py')

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
