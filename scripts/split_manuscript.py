#!/usr/bin/env python3
"""Split CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md into per-section files.

Writes to research/papers/00-technical-report/sections/.
Replaces SPEC_xxx.md references with [CEREBRUM_REPORT_PLACEHOLDER].
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = REPO_ROOT / "docs" / "CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md"
OUT_DIR = REPO_ROOT / "research" / "papers" / "00-technical-report" / "sections"

# Known section heading prefixes → output filenames (order matters: first match wins)
# Use distinctive substrings from the actual headings, avoiding common words like "Autonomous"
SECTION_MAP = [
    ("DSCF/TSC",                        "01-dscf-tsc.md"),
    ("CSA: Community-Structured",       "02-csa.md"),
    ("Bridge Twin Engine",              "03-bridge-twin.md"),
    ("Spike-Timing-Dependent",          "04-stdp-causal.md"),
    ("Holographic Indexing",            "05-holographic-indexing.md"),
    ("Bayesian Beam Search",            "06-bayesian-beam.md"),
    ("REM Cycle",                       "07-rem-cycle.md"),
    ("Orthogonal Procrustes",           "08-signal-encoder.md"),
    ("THALAMUS:",                       "09-thalamus.md"),
    ("Inference Validator:",            "10-inference-validator.md"),
    ("Contradiction Materialization",   "11-contradiction.md"),
    ("Glass-Box Reasoning",             "12-glass-box.md"),
    ("Streaming Knowledge Graph",       "13-streaming-engine.md"),
    ("Metacognitive Verification",      "14-metacognitive-verification.md"),
    ("Algorithmic Depth",               "15-algorithmic-depth.md"),
    ("Structural Hole Patching",        "16-structural-holes.md"),
    ("Inference-Time GraphSAGE",        "17-graphsage-smoothing.md"),
    ("Engram-Steered Traversal",        "18-engram-traversal.md"),
    ("TemporalCalibrator:",             "19-temporal-calibrator.md"),
    ("Five Fault-Tolerance",            "20-fault-tolerance.md"),
    ("PAPER 021",                       "21-speedtalk.md"),
    ("PAPER 022",                       "22-looped-beam.md"),
    ("PAPER 023",                       "23-predictive-coding.md"),
    ("PAPER 024",                       "24-autoapprover.md"),
    ("PAPER 025",                       "25-triangulation.md"),
    ("PAPER 026",                       "26-discovery-calibration.md"),
    ("PAPER 027",                       "27-autonomous-loop.md"),
    ("PAPER 028",                       "28-studio-v2.md"),
    ("PAPER 029",                       "29-provenance-ledger.md"),
    ("PAPER 030",                       "30-feature-impact.md"),
    ("PAPER 031",                       "31-loop-provenance-recovery.md"),
    ("PAPER 032",                       "32-graph-adapter-protocol.md"),
    ("PAPER 033",                       "33-graph-snapshot.md"),
    ("PAPER 034",                       "34-adaptive-loop-tuning.md"),
    ("Neural Visualization Bridge",     "35-ue5-visualization.md"),
    ("Conclusion:",                     "36-conclusion.md"),
]

# Internal reference patterns to replace
SPEC_RE = re.compile(r'SPEC_\d{3}_\w+\.md', re.IGNORECASE)
PARALLAX_RE = re.compile(r'PARALLAX\.md', re.IGNORECASE)


def heading_to_filename(heading: str) -> str | None:
    """Return the target filename for a top-level heading, or None."""
    for prefix, fname in SECTION_MAP:
        if prefix.lower() in heading.lower():
            return fname
    return None


def replace_internal_refs(text: str) -> str:
    text = SPEC_RE.sub("[CEREBRUM_REPORT_PLACEHOLDER]", text)
    text = PARALLAX_RE.sub("[CEREBRUM_REPORT_PLACEHOLDER]", text)
    return text


def split(dry_run: bool = False) -> None:
    lines = MANUSCRIPT.read_text(encoding="utf-8").splitlines(keepends=True)

    sections: list[tuple[str, list[str]]] = []  # (filename, lines)
    current_fname: str | None = None
    current_lines: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.rstrip("\n")

        # Track code fences to avoid treating `# comment` as a heading
        if stripped.startswith("```"):
            in_fence = not in_fence

        is_section_heading = (
            not in_fence
            and stripped.startswith("# ")
            and len(stripped) > 2
            and stripped[2].isupper()
        )

        if is_section_heading:
            heading = stripped[2:].strip()
            fname = heading_to_filename(heading)

            if fname is not None:
                # Save previous section
                if current_fname is not None:
                    sections.append((current_fname, current_lines))
                # Start new section
                current_fname = fname
                current_lines = [line]
                continue
            # Unrecognised top-level heading — warn but keep appending
            print(f"WARNING: unrecognised top-level heading: {heading!r}", file=sys.stderr)

        if current_fname is not None:
            current_lines.append(line)

    # Flush last section
    if current_fname is not None:
        sections.append((current_fname, current_lines))

    # Deduplicate: if same filename appears more than once, merge
    merged: dict[str, list[str]] = {}
    for fname, sec_lines in sections:
        if fname in merged:
            merged[fname].extend(sec_lines)
        else:
            merged[fname] = list(sec_lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for fname, sec_lines in merged.items():
        text = replace_internal_refs("".join(sec_lines))
        out_path = OUT_DIR / fname
        if dry_run:
            print(f"DRY RUN  {out_path} ({len(sec_lines)} lines)")
        else:
            out_path.write_text(text, encoding="utf-8")
            written.append((fname, len(sec_lines)))

    if not dry_run:
        for fname, n in sorted(written):
            print(f"  wrote {fname:45s}  ({n:5d} lines)")
        print(f"\n{len(written)} sections written to {OUT_DIR}")

    # Report any expected sections that weren't found
    expected = {v for _, v in SECTION_MAP}
    missing = expected - set(merged.keys())
    if missing:
        print("\nMISSING expected sections:", file=sys.stderr)
        for m in sorted(missing):
            print(f"  {m}", file=sys.stderr)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    split(dry_run=dry_run)
