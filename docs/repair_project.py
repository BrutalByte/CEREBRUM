import os
import re

def repair_file(file_path):
    print(f"Repairing {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # 1. Fix the "date injected before every char" issue (mostly for CHANGELOG.md)
        # Pattern: b'2026-04-28' followed by any byte
        if b'2026-04-28' in content:
            # We want to replace b'2026-04-28' + byte with just the byte
            # But wait, some might be \r\n
            new_content = bytearray()
            i = 0
            while i < len(content):
                if content[i:i+10] == b'2026-04-28':
                    i += 10
                    if i < len(content):
                        new_content.append(content[i])
                        i += 1
                else:
                    new_content.append(content[i])
                    i += 1
            content = bytes(new_content)

        # 2. Fix mangled UTF-8 characters (Interpret as UTF-8, but handle double-encoded cases)
        # Common manglings:
        # â€” (e2 80 94) -> Interpreted as Latin-1: \xe2\x80\x94
        # If it's already UTF-8, but showing as â€” in a Latin-1 tool, we don't need to change the bytes.
        # But if the bytes themselves are mangled (e.g. e2 80 94 became something else), we fix them.
        
        # Actually, let's just decode as UTF-8 and replace known bad sequences with ASCII for safety,
        # or just normalize to clean UTF-8.
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            # If it fails, try Latin-1 and then encode back to UTF-8
            text = content.decode('latin-1')
        
        # Standardize dashes and arrows
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('→', '->')
        text = text.replace('↔', '<->')
        text = text.replace('α', 'alpha')
        text = text.replace('β', 'beta')
        text = text.replace('²', '^2')
        text = text.replace('×', 'x')
        
        # Also handle the double-encoded UTF-8 lookalikes if any
        text = text.replace('â€”', '-')
        text = text.replace('â†’', '->')
        text = text.replace('â†”', '<->')
        text = text.replace('Ã¢Â€Â”', '-')
        text = text.replace('Ã¢Â†Â’', '->')
        text = text.replace('Ã¢Â†Â”', '<->')
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            f.write(text)
        print(f"Successfully repaired {file_path}")
    except Exception as e:
        print(f"Error repairing {file_path}: {e}")

files_to_repair = [
    '../reasoning/traversal.py',
    '../core/cerebrum.py',
    '../CHANGELOG.md',
    '../core/insight_validator.py',
    'CEREBRUM_MASTER_PAPER.md',
    'NOVEL_CONTRIBUTIONS.md'
]

for fp in files_to_repair:
    if os.path.exists(fp):
        repair_file(fp)
    else:
        print(f"File not found: {fp}")
