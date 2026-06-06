"""PostToolUse hook: syntax-check Python files after Edit/Write."""
import sys
import json
import py_compile
import os

data = json.load(sys.stdin)
path = data.get("file_path", "").replace("\\", "/")

if not path.endswith(".py") or not os.path.isfile(path):
    sys.exit(0)

try:
    py_compile.compile(path, doraise=True)
except py_compile.PyCompileError as e:
    print(f"[syntax_check] SyntaxError in {path}:\n{e}", file=sys.stderr)
    sys.exit(1)
