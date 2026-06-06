"""PreToolUse hook: warn before destructive git commands (push, reset --hard, clean -f, branch -D)."""
import sys
import json
import re

data = json.load(sys.stdin)
command = data.get("command", "")

DESTRUCTIVE = [
    (r"\bgit\s+push\b", "git push — pushes commits to remote. Confirm you've reviewed the diff."),
    (r"\bgit\s+push\s+.*--force\b", "git push --force — FORCE PUSH. This can overwrite upstream history."),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard — discards all uncommitted changes permanently."),
    (r"\bgit\s+clean\s+.*-f\b", "git clean -f — permanently deletes untracked files."),
    (r"\bgit\s+branch\s+.*-[Dd]\b", "git branch -D — permanently deletes a local branch."),
    (r"\bgit\s+checkout\s+--\s", "git checkout -- — discards working-tree changes for listed files."),
]

for pattern, description in DESTRUCTIVE:
    if re.search(pattern, command):
        print(
            f"[guard_git] Destructive git command detected:\n  {description}\n"
            f"  Command: {command}\n"
            "Approve this tool call to proceed.",
            file=sys.stderr,
        )
        sys.exit(2)  # exit code 2 = block and show to user
