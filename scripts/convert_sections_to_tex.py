#!/usr/bin/env python3
"""Convert technical report sections from Markdown to LaTeX.

Reads research/papers/00-technical-report/sections/*.md and writes
corresponding .tex files to arxiv_submission/sections/.

Each section is \input{} inside a \chapter*{} in cerebrum-v251-report.tex,
so section files provide body content only â€” no \documentclass or \section
at the outermost level.

Heading mapping (chapter title is set by main tex, so # is skipped):
  #    -> skip  (chapter title already in main .tex)
  ##   -> \section*{...}
  ###  -> \subsection*{...}
  #### -> \subsubsection*{...}
"""
from __future__ import annotations
from typing import Match

import re
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
SECTION_SRC = REPO_ROOT / "research" / "papers" / "00-technical-report" / "sections"
SECTION_DST = REPO_ROOT / "research" / "papers" / "00-technical-report" / "arxiv_submission" / "sections"

# ---------------------------------------------------------------------------
# Unicode â†’ LaTeX mapping (outside math mode)
# ---------------------------------------------------------------------------
UNICODE_MAP = {
    "\u2014": "---",          # em dash
    "\u2013": "--",           # en dash
    "\u2019": "'",            # right single quote
    "\u2018": "`",            # left single quote
    "\u201c": "``",           # left double quote
    "\u201d": "''",           # right double quote
    "\u00e9": r"\'{e}",
    "\u00e8": r"\`{e}",
    "\u00e0": r"\`{a}",
    "\u00fc": r'\"u',
    "\u00f6": r'\"o',
    "\u00e4": r'\"a',
    "\u00d7": r"$\times$",
    "\u2265": r"$\geq$",
    "\u2264": r"$\leq$",
    "\u2248": r"$\approx$",
    "\u2192": r"$\to$",
    "\u2190": r"$\gets$",
    "\u00b1": r"$\pm$",
    "\u00b7": r"$\cdot$",
    "\u03bc": r"$\mu$",
    "\u03c4": r"$\tau$",
    "\u03bb": r"$\lambda$",
    "\u0394": r"$\Delta$",
    "\u03a3": r"$\Sigma$",
    "\u03c1": r"$\rho$",
    "\u03b7": r"$\eta$",
    "\u03b8": r"$\theta$",
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u03b3": r"$\gamma$",
    "\u03b4": r"$\delta$",
    "\u03b5": r"$\epsilon$",
    "\u03c0": r"$\pi$",
    "\u03c6": r"$\phi$",
    "\u03c9": r"$\omega$",
    "\u03c3": r"$\sigma$",
    "\u03ba": r"$\kappa$",
    "\u2212": r"$-$",         # minus sign
    "\u221e": r"$\infty$",
    "\u2208": r"$\in$",
    "\u2209": r"$\notin$",
    "\u2282": r"$\subset$",
    "\u2286": r"$\subseteq$",
    "\u222a": r"$\cup$",
    "\u2229": r"$\cap$",
    "\u2200": r"$\forall$",
    "\u2203": r"$\exists$",
    "\u00d7": r"$\times$",
    "\u00f7": r"$\div$",
    "\u2260": r"$\neq$",
    "\u2261": r"$\equiv$",
    "\u00ae": r"\textregistered{}",
    "\u00a9": r"\textcopyright{}",
    "\u2122": r"\texttrademark{}",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_unicode(text: str) -> str:
    for char, replacement in UNICODE_MAP.items():
        text = text.replace(char, replacement)
    return text


def escape_text(text: str) -> str:
    """Escape LaTeX special chars in plain text segments.

    Does NOT escape text inside:
      - $...$ / $$...$$ (math, already handled by segment splitting)
      - \\command{...} prefixes (already LaTeX)
    This is called on pre-split non-math, non-code segments only.
    """
    # Percent (not already escaped)
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    # Ampersand outside tabular (caller skips this in table rows)
    text = re.sub(r"(?<!\\)&", r"\\&", text)
    # Tilde (non-breaking space already handled; stray ~ in text)
    # Leave ~ alone â€” LaTeX uses it as non-breaking space which is fine
    return text


def protect_math(line: str) -> tuple[str, list[str]]:
    """Replace all $...$ spans with MATHTOKEN_N placeholders.

    Returns (protected_line, list_of_original_spans).
    Display math ($$...$$) is handled at block level; this handles inline only.
    Uses a simple [^$\n]+ pattern to avoid catastrophic backtracking.
    """
    spans: list[str] = []

    def _replace(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"MATHTOKEN_{len(spans)-1}_"

    # First protect $$...$$ so they are not split by the inline pattern
    def _replace_display(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"MATHTOKEN_{len(spans)-1}_"

    protected = re.sub(r"\$\$[^\n]+?\$\$", _replace_display, line)
    # Now replace remaining single-dollar inline math: $...$
    # [^$\n]+ ensures no nested $ and no newlines â€” safe, no backtracking
    protected = re.sub(r"\$[^$\n]+?\$", _replace, protected)
    return protected, spans


def restore_math(line: str, spans: list[str]) -> str:
    for i, span in enumerate(spans):
        line = line.replace(f"MATHTOKEN_{i}_", span)
    return line


def convert_inline(text: str, in_table: bool = False) -> str:
    """Convert inline Markdown to LaTeX in a single line of text."""
    # Protect math spans before any substitution
    text, math_spans = protect_math(text)

    # Bold before italic (longer match first)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)

    # Inline code  `` `code` `` â†’ \texttt{code}
    def _inline_code(m: re.Match) -> str:
        code = m.group(1)
        # Escape chars inside \texttt
        code = code.replace("\\", r"\textbackslash{}")
        code = code.replace("_", r"\_")
        code = code.replace("#", r"\#")
        code = code.replace("%", r"\%")
        code = code.replace("&", r"\&")
        code = code.replace("{", r"\{")
        code = code.replace("}", r"\}")
        code = code.replace("^", r"\^{}")
        code = code.replace("~", r"\textasciitilde{}")
        return r"\texttt{" + code + "}"
    text = re.sub(r"`([^`]+)`", _inline_code, text)

    # Escape text-mode special chars (not inside LaTeX commands or math)
    if not in_table:
        text = escape_text(text)

    # Apply unicode substitutions
    text = apply_unicode(text)

    # Restore math spans
    text = restore_math(text, math_spans)
    return text


def heading_text(raw: str) -> str:
    """Strip 'PAPER NNN: ' prefix and trailing punctuation from heading."""
    raw = re.sub(r"^PAPER\s+\d+[:\.\s]+", "", raw).strip()
    return raw


# ---------------------------------------------------------------------------
# Block-level conversion
# ---------------------------------------------------------------------------

def convert_table(lines: list[str]) -> str:
    """Convert a Markdown table (list of raw lines) to LaTeX tabular."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("|---") or re.match(r"^\|[-| :]+\|$", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return ""

    ncols = max(len(r) for r in rows)
    col_spec = "l" * ncols

    tex_lines = [
        r"\begin{table}[h]",
        r"\small\centering",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    for i, row in enumerate(rows):
        # Pad to ncols
        while len(row) < ncols:
            row.append("")
        cells_tex = [convert_inline(c, in_table=True) for c in row]
        tex_lines.append(" & ".join(cells_tex) + r" \\")
        if i == 0:
            tex_lines.append(r"\midrule")

    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(tex_lines)


def convert_list_block(items: list[tuple[str, str]]) -> str:
    """Convert a list of (kind, text) tuples.

    kind is 'bullet' or 'ordered'.
    """
    if not items:
        return ""
    env = "itemize" if items[0][0] == "bullet" else "enumerate"
    lines = [r"\begin{" + env + "}"]
    for _, text in items:
        lines.append(r"  \item " + convert_inline(text))
    lines.append(r"\end{" + env + "}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Footer / header detection
# ---------------------------------------------------------------------------

_FOOTER_SIGNALS = re.compile(
    r"^\*\*(References|Copyright|Reviewed on|Manuscript Finalized)",
    re.IGNORECASE,
)

_META_SIGNALS = re.compile(
    r"^\*\*(Author|Affiliation|Status|Date|CEREBRUM Phase)",
    re.IGNORECASE,
)


def strip_header_and_footer(lines: list[str]) -> list[str]:
    """Remove metadata header (up to first --- ) and trailing footer."""
    # --- strip header ---
    # Find first standalone '---' line and drop everything up to + including it
    header_end = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            header_end = i
            break

    if header_end is not None:
        lines = lines[header_end + 1:]

    # --- strip footer ---
    # Walk from the end; strip trailing:
    #   **Reviewed on** / **Copyright** / **Manuscript Finalized**
    #   numbered reference list (lines matching r"^\d+\. ...")
    #   **References** heading
    #   standalone --- lines

    footer_start = len(lines)
    i = len(lines) - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped == "---":
            footer_start = i
            i -= 1
            continue
        if _FOOTER_SIGNALS.match(stripped):
            footer_start = i
            i -= 1
            continue
        # Numbered reference entry: "1. Author, ..." OR plain number lines
        if re.match(r"^\d+\.\s+\S", stripped):
            footer_start = i
            i -= 1
            continue
        if stripped == "":
            i -= 1
            continue
        # Non-footer content encountered â€” stop
        break

    lines = lines[:footer_start]

    # Drop trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    return lines


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_section(md_text: str) -> str:
    raw_lines = md_text.splitlines()
    lines = strip_header_and_footer(raw_lines)

    output: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- fenced code block ----
        if stripped.startswith("```"):
            lang_match = re.match(r"^```(\w*)", stripped)
            lang = lang_match.group(1).capitalize() if lang_match and lang_match.group(1) else ""
            lang_opt = f"[language={lang}]" if lang else ""
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            output.append(f"\\begin{{lstlisting}}{lang_opt}")
            output.extend(code_lines)
            output.append(r"\end{lstlisting}")
            i += 1  # skip closing ```
            continue

        # ---- display math $$...$$ (single-line or multi-line) ----
        if stripped.startswith("$$"):
            # Collect until closing $$
            math_lines = [line]
            if stripped.count("$$") >= 2 and stripped.endswith("$$") and len(stripped) > 4:
                # single-line $$...$$  â†’ keep as-is (LaTeX handles it)
                output.append(stripped)
                i += 1
                continue
            i += 1
            while i < len(lines):
                math_lines.append(lines[i])
                if lines[i].strip().endswith("$$"):
                    break
                i += 1
            output.extend(math_lines)
            i += 1
            continue

        # ---- markdown table ----
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            output.append(convert_table(table_lines))
            output.append("")
            continue

        # ---- headings ----
        h4 = re.match(r"^#### (.+)", stripped)
        h3 = re.match(r"^### (.+)", stripped)
        h2 = re.match(r"^## (.+)", stripped)
        h1 = re.match(r"^# (.+)", stripped)

        if h4:
            text = convert_inline(heading_text(h4.group(1)))
            output.append(f"\\subsubsection*{{{text}}}")
            i += 1
            continue
        if h3:
            text = convert_inline(heading_text(h3.group(1)))
            output.append(f"\\subsection*{{{text}}}")
            i += 1
            continue
        if h2:
            text = convert_inline(heading_text(h2.group(1)))
            output.append(f"\\section*{{{text}}}")
            i += 1
            continue
        if h1:
            # Chapter title â€” skip (set by main tex \chapter*{})
            i += 1
            continue

        # ---- metadata lines (if they somehow survived header strip) ----
        if _META_SIGNALS.match(stripped):
            i += 1
            continue

        # ---- horizontal rule ----
        if stripped == "---":
            # Inline divider within content â€” render as small vertical space
            output.append(r"\medskip\noindent\rule{\linewidth}{0.4pt}\medskip")
            i += 1
            continue

        # ---- bullet list ----
        if re.match(r"^[-*] ", stripped):
            items = []
            while i < len(lines) and re.match(r"^[-*] ", lines[i].strip()):
                items.append(("bullet", lines[i].strip()[2:].strip()))
                i += 1
            output.append(convert_list_block(items))
            output.append("")
            continue

        # ---- ordered list ----
        if re.match(r"^\d+\. ", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i].strip()):
                text = re.sub(r"^\d+\. ", "", lines[i].strip())
                items.append(("ordered", text))
                i += 1
            output.append(convert_list_block(items))
            output.append("")
            continue

        # ---- blank line ----
        if not stripped:
            output.append("")
            i += 1
            continue

        # ---- normal paragraph line ----
        output.append(convert_inline(stripped))
        i += 1

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not SECTION_SRC.exists():
        print(f"ERROR: source directory not found: {SECTION_SRC}", file=sys.stderr)
        sys.exit(1)

    SECTION_DST.mkdir(parents=True, exist_ok=True)

    md_files = sorted(SECTION_SRC.glob("*.md"))
    if not md_files:
        print("No .md files found in sections/", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(md_files)} section(s) -> {SECTION_DST.relative_to(REPO_ROOT)}")

    errors = 0
    for md_path in md_files:
        tex_path = SECTION_DST / (md_path.stem + ".tex")
        try:
            md_text = md_path.read_text(encoding="utf-8")
            tex_text = convert_section(md_text)
            tex_path.write_text(tex_text, encoding="utf-8")
            lines_in  = md_text.count("\n")
            lines_out = tex_text.count("\n")
            print(f"  {md_path.name:<45} -> {tex_path.name}  ({lines_in} -> {lines_out} lines)")
        except Exception as exc:
            print(f"  ERROR {md_path.name}: {exc}")
            errors += 1

    print(f"\nDone. {len(md_files) - errors} converted, {errors} error(s).")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
