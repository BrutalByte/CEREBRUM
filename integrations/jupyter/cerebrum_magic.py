"""
CEREBRUM Jupyter Magic Commands.

Load this extension to get %%cerebrum and %cerebrum_load magic commands
for interactive knowledge-graph queries in Jupyter notebooks.

    %load_ext integrations.jupyter.cerebrum_magic

    # Load a KB
    %cerebrum_load kb.csv

    # Ask a question — inline trace visualization
    %%cerebrum
    Who directed Inception?

    # Or as a line magic
    %cerebrum Who directed Inception?
"""
from __future__ import annotations

import sys
import os
from typing import Optional

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "sdk", "python") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "sdk", "python"))

_CEREBRUM_INSTANCE = None  # module-level singleton


def _get_ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


def _require_ipython():
    ip = _get_ipython()
    if ip is None:
        raise RuntimeError("CEREBRUM magic commands require IPython/Jupyter.")
    return ip


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _render_result_html(result, query: str) -> str:
    """Build a styled HTML card showing the reasoning result and trace."""
    trace_html = ""
    nodes = []
    for step in result.trace_path:
        nodes.append(f"<span style='background:#1c2b3a;color:#79c0ff;border:1px solid #30363d;"
                     f"border-radius:4px;padding:2px 8px;font-family:monospace;font-size:0.9em'>"
                     f"{step.entity}</span>")
        nodes.append(f"<span style='color:#bc8cff;margin:0 4px;font-size:0.85em'>"
                     f"&mdash;[{step.relation}]&rarr;</span>")
    if result.answer:
        nodes.append(f"<span style='background:#1c3b2a;color:#3fb950;border:1px solid #30363d;"
                     f"border-radius:4px;padding:2px 8px;font-family:monospace;font-size:0.9em;"
                     f"font-weight:bold'>{result.answer}</span>")
    trace_html = "".join(nodes) if nodes else "<em style='color:#888'>No trace</em>"

    top_k_rows = ""
    for i, cand in enumerate(result.top_k[:5]):
        bar_w = int(cand["confidence"] * 100)
        row_bg = "#1c3b2a" if i == 0 else "transparent"
        top_k_rows += f"""
        <tr style='background:{row_bg}'>
            <td style='padding:3px 8px;color:#c9d1d9;font-family:monospace'>{cand['entity']}</td>
            <td style='padding:3px 8px;width:120px'>
                <div style='background:#30363d;height:8px;border-radius:4px;'>
                    <div style='background:#58a6ff;width:{bar_w}%;height:100%;border-radius:4px'></div>
                </div>
            </td>
            <td style='padding:3px 8px;color:#79c0ff;font-family:monospace;font-size:0.85em'>{cand['confidence']:.4f}</td>
        </tr>"""

    return f"""
<div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;
            padding:16px;margin:8px 0;font-family:sans-serif;color:#c9d1d9'>
    <div style='color:#888;font-size:0.8em;margin-bottom:4px'>CEREBRUM &mdash; crystal-box reasoning</div>
    <div style='margin-bottom:10px'>
        <span style='color:#888'>Query:</span>
        <strong style='color:#e6edf3;margin-left:6px'>{query}</strong>
    </div>
    <div style='margin-bottom:12px;line-height:2.2em'>{trace_html}</div>
    <table style='border-collapse:collapse;width:100%;margin-bottom:8px'>
        <tr style='border-bottom:1px solid #21262d'>
            <th style='padding:3px 8px;text-align:left;color:#888;font-weight:normal;font-size:0.85em'>Entity</th>
            <th style='padding:3px 8px;text-align:left;color:#888;font-weight:normal;font-size:0.85em'>Confidence</th>
            <th style='padding:3px 8px;color:#888;font-weight:normal;font-size:0.85em'></th>
        </tr>
        {top_k_rows}
    </table>
    <div style='color:#888;font-size:0.75em'>{result.elapsed_ms:.1f} ms</div>
</div>"""


def display_trace(result, query: str = "") -> None:
    """
    Display a CEREBRUM Result as an inline notebook visualization.

    Parameters
    ----------
    result : Result object from Cerebrum.ask() or Cerebrum.query()
    query  : Optional query string for the header label
    """
    try:
        from IPython.display import HTML, display
        display(HTML(_render_result_html(result, query or result.answer)))
    except ImportError:
        print(f"Answer: {result.answer} (conf={result.confidence:.4f})")
        for step in result.trace_path:
            print(f"  {step.entity} --[{step.relation}]-->")


# ---------------------------------------------------------------------------
# Magic command implementation
# ---------------------------------------------------------------------------

def _run_query(query: str) -> None:
    global _CEREBRUM_INSTANCE
    if _CEREBRUM_INSTANCE is None:
        try:
            from IPython.display import HTML, display
            display(HTML(
                "<div style='color:#f85149;padding:8px;'>No KB loaded. "
                "Run <code>%cerebrum_load path/to/kb.csv</code> first.</div>"
            ))
        except ImportError:
            print("No KB loaded. Run %cerebrum_load path/to/kb.csv first.")
        return

    query = query.strip()
    if not query:
        return

    result = _CEREBRUM_INSTANCE.ask(query)

    try:
        from IPython.display import HTML, display
        display(HTML(_render_result_html(result, query)))
    except ImportError:
        print(f"Answer: {result.answer} (conf={result.confidence:.4f})")


# ---------------------------------------------------------------------------
# IPython magic registration
# ---------------------------------------------------------------------------

def load_ipython_extension(ipython):
    """Called by %load_ext to register the magic commands."""
    from IPython.core.magic import register_line_magic, register_cell_magic, register_line_cell_magic

    @register_line_magic("cerebrum_load")
    def cerebrum_load(line):
        """Load a knowledge base from a CSV, KB, or triples file.

        Usage:
            %cerebrum_load path/to/kb.csv
            %cerebrum_load path/to/kb.csv --embeddings sentence
            %cerebrum_load path/to/metaqa.txt --format kb
        """
        global _CEREBRUM_INSTANCE
        import shlex
        parts = shlex.split(line.strip())
        if not parts:
            print("Usage: %cerebrum_load path/to/file [--embeddings random|sentence] [--format csv|kb]")
            return

        path = parts[0]
        embeddings = "random"
        fmt = "csv"
        i = 1
        while i < len(parts):
            if parts[i] == "--embeddings" and i + 1 < len(parts):
                embeddings = parts[i + 1]; i += 2
            elif parts[i] == "--format" and i + 1 < len(parts):
                fmt = parts[i + 1]; i += 2
            else:
                i += 1

        try:
            from IPython.display import HTML, display
            display(HTML(f"<div style='color:#888'>Loading <code>{path}</code>...</div>"))
        except ImportError:
            print(f"Loading {path}...")

        from cerebrum_sdk import Cerebrum
        if fmt == "kb":
            _CEREBRUM_INSTANCE = Cerebrum.from_kb(path, embeddings=embeddings)
        else:
            _CEREBRUM_INSTANCE = Cerebrum.from_csv(path, embeddings=embeddings)

        stats = _CEREBRUM_INSTANCE.stats
        try:
            from IPython.display import HTML, display
            display(HTML(
                f"<div style='background:#0d1117;border:1px solid #3fb950;border-radius:6px;"
                f"padding:10px;font-family:monospace;color:#3fb950'>"
                f"KB loaded: {stats['entities']:,} entities | "
                f"{stats['relations']:,} relations | "
                f"{stats['communities']} communities</div>"
            ))
        except ImportError:
            print(f"KB loaded: {stats}")

    @register_line_cell_magic("cerebrum")
    def cerebrum_magic(line, cell=None):
        """Query the loaded knowledge base.

        Line magic:  %cerebrum Who directed Inception?
        Cell magic:  %%cerebrum
                     Who directed Inception?
        """
        query = cell.strip() if cell else line.strip()
        _run_query(query)

    print("CEREBRUM magic loaded. Use %cerebrum_load to load a KB, then %%cerebrum to query.")


def unload_ipython_extension(ipython):
    global _CEREBRUM_INSTANCE
    _CEREBRUM_INSTANCE = None
