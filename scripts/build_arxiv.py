"""
Phase 25 — CEREBRUM arXiv LaTeX Compilation Pipeline (Professional Upgrade).
Transforms the docs/arxiv/*.md files into professionally formatted .tex documents
with robust handling of math blocks, tables, and lists.
"""
import os
import re
from glob import glob

def sanitize_tex(text: str) -> str:
    """Escapes special LaTeX characters in raw text."""
    # Temporarily hide soft-hyphens
    text = text.replace(r"\-", "SOFTHYPHENMARKER")
    
    text = text.replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")
    text = text.replace("<", "\\textless ").replace(">", "\\textgreater ")
    
    # Restore soft-hyphens
    text = text.replace("SOFTHYPHENMARKER", r"\-")
    
    # Bold/Italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    return text

def map_unicode_symbols(text: str) -> str:
    """Replaces Unicode symbols with LaTeX equivalents (math-safe)."""
    mapping = {
        "≥": r"$\ge$", "≤": r"$\le$", "≈": r"$\approx$", "→": r"$\to$", "←": r"$\gets$",
        "·": r"$\cdot$", "−": r"$-$", "±": r"$\pm$", "μ": r"$\mu$", "τ": r"$\tau$",
        "λ": r"$\lambda$", "ℒ": r"$\mathcal{L}$", "Δ": r"$\Delta$", "Σ": r"$\Sigma$",
        "ρ": r"$\rho$", "η": r"$\eta$", "θ": r"$\theta$", "×": r"$\times$",
        "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
        "ε": r"$\epsilon$", "ζ": r"$\zeta$", "π": r"$\pi$", "φ": r"$\phi$",
        "ω": r"$\omega$", "σ": r"$\sigma$", "κ": r"$\kappa$"
    }
    for char, replacement in mapping.items():
        text = text.replace(char, replacement)
    return text

def split_table_row(line: str) -> list[str]:
    """Splits a Markdown table row into cells, ignoring pipes inside inline math."""
    # Hide pipes inside math
    temp_line = ""
    in_math = False
    for char in line:
        if char == '$':
            in_math = not in_math
        if char == '|' and in_math:
            temp_line += "TEMP_PIPE_MARKER"
        else:
            temp_line += char
    
    # Split by actual pipes
    cells = temp_line.split("|")
    
    # Clean up and restore pipes
    res = []
    # If the line starts and ends with |, the first and last split will be empty
    start = 1 if line.strip().startswith("|") else 0
    end = -1 if line.strip().endswith("|") else None
    
    for cell in cells[start:end]:
        res.append(cell.strip().replace("TEMP_PIPE_MARKER", "|"))
    return res

def process_inline_text(text: str) -> str:
    """Handles protection of code/math and sanitation of text."""
    protected = []
    
    # Inline Code `...` -> escape all backslashes inside to ensure it works in \texttt
    def protect_code(m):
        idx = len(protected)
        # Specifically handle soft-hyphen \- as \allowbreak in monospaced
        content = m.group(1).replace("\\-", "\\allowbreak ").replace("\\", "\\textbackslash ").replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("<", "\\textless ").replace(">", "\\textgreater ")
        protected.append(f"\\texttt{{{content}}}")
        return f"MARKERPROTECT{idx}X"
    text = re.sub(r'`(.*?)`', protect_code, text)

    # Inline Math $...$
    def protect_math(m):
        idx = len(protected)
        protected.append(m.group(0))
        return f"MARKERPROTECT{idx}X"
    text = re.sub(r'\$(.*?)\$', protect_math, text)

    # LaTeX Commands like \cite{...}
    def protect_cmds(m):
        idx = len(protected)
        protected.append(m.group(0))
        return f"MARKERPROTECT{idx}X"
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', protect_cmds, text)

    # Markdown Links [text](url) -> \href{url}{text}
    # ... (rest of protect_links)
    def protect_links(m):
        idx = len(protected)
        lbl = sanitize_tex(m.group(1))
        url = m.group(2).strip()
        protected.append(f"\\href{{{url}}}{{{lbl}}}")
        return f"MARKERPROTECT{idx}X"
    text = re.sub(r'\[(.*?)\]\((http.*?)\)', protect_links, text)

    # Surgical Hyphenation for long citation strings in tables (e.g. miller2019explainability)
    # Match long alphabetic/numeric strings and allow breaks at logical boundaries
    def hyphenate_ids(m):
        full = m.group(0)
        # Insert soft hyphens after author and year parts
        res = re.sub(r'([a-zA-Z]{3,})(\d{2,})', r'\1\\-\2\\-', full)
        # Also break very long trailing words
        # (e.g. explainability -> explain\-ability)
        res = re.sub(r'([a-z]{5})([a-z]{5})', r'\1\\-\2', res)
        return res
    # Find words longer than 12 chars that are probably IDs or technical terms
    text = re.sub(r'\b[a-zA-Z0-9]{12,}\b', hyphenate_ids, text)

    # Sanitize & Map Unicode
    text = sanitize_tex(text)
    text = map_unicode_symbols(text)
    
    # Restore
    for i, val in enumerate(protected):
        text = text.replace(f"MARKERPROTECT{i}X", val)
        
    return text

def convert_markdown_to_tex(md_content: str) -> str:
    """Line-by-line state machine parser for professional LaTeX output."""
    lines = md_content.split("\n")
    tex_lines = []
    
    # States
    in_math_block = False
    in_code_block = False
    in_mermaid_block = False
    in_table = False
    active_list = None # 'itemize' or 'enumerate'

    for i, line in enumerate(lines):
        clean = line.strip()

        # 1. PRIMARY BLOCK MARKERS (These override everything else)
        
        # Mermaid
        if clean.startswith("```mermaid"):
            in_mermaid_block = True
            tex_lines.append("\\begin{figure*}[!t]\n\\centering")
            tex_lines.append("\\includegraphics[width=0.8\\textwidth]{placeholder_mermaid.pdf}")
            continue
        if in_mermaid_block:
            if clean.startswith("```"):
                in_mermaid_block = False
                tex_lines.append("\\caption{Mermaid diagram placeholder}\n\\end{figure*}")
            continue
            
        # Code Blocks
        if clean.startswith("```"):
            if in_code_block:
                in_code_block = False
                tex_lines.append("\\end{lstlisting}")
            else:
                in_code_block = True
                tex_lines.append("\\begin{lstlisting}")
            continue
        if in_code_block:
            tex_lines.append(line)
            continue

        # Math Blocks
        # Check for start/end of math block. 
        # Support both '$$' on its own and '$$content$$' on its own.
        if clean == "$$":
            if in_math_block:
                in_math_block = False
                tex_lines.append("\\end{equation}")
            else:
                in_math_block = True
                tex_lines.append("\\begin{equation}")
            continue
        if clean.startswith("$$") and clean.endswith("$$") and len(clean) > 4:
            content = clean[2:-2].strip()
            tex_lines.append(f"\\begin{{equation}}\n{content}\n\\end{{equation}}")
            continue
        if in_math_block:
            tex_lines.append(line) # No processing inside math blocks
            continue

        # 2. STATEFUL BLOCK TRANSLATIONS (Tables & Lists)

        # TABLES
        is_table_row = clean.startswith("|") and clean.endswith("|")
        # Ignore separators like |---|---| or |:---|:---|
        is_table_sep = is_table_row and all(c in '| -: ' for c in clean)
        
        if is_table_row:
            if not in_table:
                in_table = True
                table_rows = []
            if not is_table_sep:
                table_rows.append(split_table_row(clean))
            continue
        elif not is_table_row and in_table:
            # Process buffered table
            in_table = False
            if table_rows:
                cols = len(table_rows[0])
                # Calculate max length per column to decide on X vs l
                col_max_len = [0] * cols
                for row in table_rows:
                    for idx, cell in enumerate(row):
                        if idx < cols:
                            col_max_len[idx] = max(col_max_len[idx], len(cell))
                
                # Heuristic: if max_len > 20, use X. 
                # If no columns are long, use l.
                # If all are long, use X.
                # Heuristic: in a 2-column paper, almost EVERYTHING should wrap if > 4 chars
                col_specs = []
                has_x = False
                for length in col_max_len:
                    if length > 4:
                        col_specs.append(">{\\raggedright\\arraybackslash}X")
                        has_x = True
                    else:
                        col_specs.append("l")
                
                # If no X was found, make first one X
                if not has_x and cols > 0:
                    col_specs[0] = ">{\\raggedright\\arraybackslash}X"

                # Final assembly
                spec_str = "|" + "|".join(col_specs) + "|"
                
                tex_lines.append("\\begin{center}")
                # For long tables like the Appendix, use small font
                if cols > 2:
                    tex_lines.append("{\\footnotesize")
                tex_lines.append(f"\\begin{{tabularx}}{{\\columnwidth}}{{{spec_str}}}")
                tex_lines.append("\\hline")
                for r_idx, row in enumerate(table_rows):
                    tex_lines.append(" & ".join([process_inline_text(c) for c in row]) + " \\\\ \\hline")
                tex_lines.append("\\end{tabularx}")
                if cols > 2:
                    tex_lines.append("}")
                tex_lines.append("\\end{center}")
                table_rows = [] # CRITICAL: Clear for next table
            
            # Fall through to process this line as normal text

        # LISTS
        is_unordered = clean.startswith("- ") or clean.startswith("* ")
        is_ordered = bool(re.match(r'^\d+\. ', clean))
        
        if (is_unordered or is_ordered) and active_list is None:
            active_list = 'itemize' if is_unordered else 'enumerate'
            tex_lines.append(f"\\begin{{{active_list}}}")
        elif not (is_unordered or is_ordered) and active_list is not None and clean != "":
            # Close list if line is non-blank but not a list item
            tex_lines.append(f"\\end{{{active_list}}}")
            active_list = None

        if is_unordered:
            content = clean[2:].strip()
            tex_lines.append(f"\\item {process_inline_text(content)}")
            continue
        if is_ordered:
            content = re.sub(r'^\d+\. ', '', clean).strip()
            tex_lines.append(f"\\item {process_inline_text(content)}")
            continue

        # 3. PLAIN TEXT & HEADERS (Processed)

        if not clean:
            tex_lines.append("")
            continue

        # Headers
        if line.startswith("# "):
            tex_lines.append(f"\\section{{{process_inline_text(line[2:].strip())}}}")
            continue
        if line.startswith("## "):
            tex_lines.append(f"\\subsection{{{process_inline_text(line[3:].strip())}}}")
            continue
        if line.startswith("### "):
            tex_lines.append(f"\\subsubsection{{{process_inline_text(line[4:].strip())}}}")
            continue
        if line.startswith("#### "):
            tex_lines.append(f"\\paragraph{{{process_inline_text(line[5:].strip())}}}")
            continue
            
        # Standard Line
        tex_lines.append(process_inline_text(line))

    # Cleanup open states
    if in_table:
        tex_lines.append("\\end{tabularx}\n\\end{center}")
    if active_list:
        tex_lines.append(f"\\end{{{active_list}}}")
        
    return "\n".join(tex_lines)

def build_arxiv():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arxiv_dir = os.path.join(base_dir, "docs", "arxiv")
    latex_dir = os.path.join(base_dir, "docs", "latex")
    compiled_dir = os.path.join(latex_dir, "compiled")
    os.makedirs(compiled_dir, exist_ok=True)
    
    for md_file in sorted(glob(os.path.join(arxiv_dir, "*.md"))):
        print(f"Compiling {os.path.basename(md_file)}...")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Normalize soft hyphens if they are literal in MD
            content = content.replace("\\-", "\\-") 
            content = convert_markdown_to_tex(content)
        with open(os.path.join(compiled_dir, os.path.basename(md_file).replace(".md", ".tex")), "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    build_arxiv()
