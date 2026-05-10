import os
import re

def sync_files():
    replacements = [
        (r'v2\.24\.0', 'v2.51.0'),
        (r'Phase 112', 'Phase 167'),
        (r'1978\+ tests', '2175+ tests'),
        (r'1,357 passing tests', '2175+ tests'),
    ]

    # Progression fixes
    progression_patterns = [
        # Headers
        (r'v2\.51\.0\s*(->|→)\s*v2\.51\.0', 'v2.24.0 -> v2.51.0'),
        # Table comparison headers (e.g. | H@1 (v2.51.0) | H@1 (v2.51.0 Phase 167) |)
        (r'\(v2\.51\.0\)(?=.*\(v2\.51\.0 Phase 167\))', '(v2.24.0)'),
        # Textual progression
        (r'between v2\.51\.0 and v2\.51\.0', 'between v2.24.0 and v2.51.0'),
        (r'Prior to v2\.51\.0', 'Prior to v2.24.0'),
        (r'at v2\.51\.0', 'at v2.24.0'),
        (r'since v2\.51\.0', 'since v2.24.0'),
        (r'Since v2\.51\.0', 'Since v2.24.0'),
        (r'at Phase 167', 'at Phase 112'),
        (r'from Phase 167', 'from Phase 112'),
    ]

    files_to_process = []
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith('.md'):
                files_to_process.append(os.path.join(root, file))
    
    if os.path.exists('README.md'):
        files_to_process.append('README.md')

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
