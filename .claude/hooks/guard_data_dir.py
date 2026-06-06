"""PreToolUse hook: block Write operations targeting data/cerebrum/ live data files."""
import sys
import json

PROTECTED = [
    "data/cerebrum/engram_cache.json",
    "data/cerebrum/api_keys.json",
    "data/cerebrum/graph_wal.ndjson",
    "data/cerebrum/query_log.ndjson",
]

data = json.load(sys.stdin)
path = data.get("file_path", "").replace("\\", "/")

for protected in PROTECTED:
    if path.endswith(protected) or f"/{protected}" in path:
        print(
            f"[guard_data_dir] BLOCKED: {path} is a live data file.\n"
            "Edit data/cerebrum/ files manually or via the API, not via Claude tools.",
            file=sys.stderr,
        )
        sys.exit(1)
