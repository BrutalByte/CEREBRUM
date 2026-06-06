"""PostToolUse hook: check arXiv abstract length after editing PAPER_*.md files."""
import sys
import json
import os
import re

data = json.load(sys.stdin)
path = data.get("file_path", "").replace("\\", "/")

if not re.search(r"PAPER_\d+.*\.md$", path):
    sys.exit(0)

if not os.path.isfile(path):
    sys.exit(0)

with open(path, encoding="utf-8") as f:
    content = f.read()

# Extract text between ### Abstract and the next ### heading
match = re.search(r"#{1,3}\s+Abstract\s*\n(.*?)(?=\n#{1,3}\s|\Z)", content, re.DOTALL)
if not match:
    sys.exit(0)

abstract = match.group(1).strip()
char_count = len(abstract)
LIMIT = 1920

if char_count > LIMIT:
    print(
        f"[arxiv_abstract_check] WARNING: Abstract in {os.path.basename(path)} "
        f"is {char_count} chars — exceeds arXiv limit of {LIMIT} "
        f"(over by {char_count - LIMIT}).",
        file=sys.stderr,
    )
    sys.exit(1)
else:
    print(
        f"[arxiv_abstract_check] Abstract OK: {char_count}/{LIMIT} chars "
        f"({LIMIT - char_count} remaining).",
        file=sys.stderr,
    )
