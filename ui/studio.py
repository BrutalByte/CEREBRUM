"""
Parallax Studio -- Interactive Reasoning Interface.

A Gradio-based UI for exploring Knowledge Graphs using Community-Structured Attention.
Allows users to run reasoning queries, inspect paths, and visualize the attention mechanism.

Usage:
    pip install -r ui/requirements.txt
    python ui/studio.py
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

import networkx as nx
import gradio as gr
from pyvis.network import Network
import tempfile

# Windows: ensure stdout/stderr use utf-8 so emoji don't crash the terminal.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ParallaxStudio")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.file_adapter import load_file_adapter
from adapters.stream_adapter import StreamAdapter, PythonCallbackSource, FileTailSource, HTTPPollingSource
from core.embedding_engine import RandomEngine, SentenceEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract
from core.community_engine import best_of_n_dscf, merge_small_communities
from core.discretizer import ThresholdDiscretizer, BinningDiscretizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { background-color: #0b0f19 !important; }
.parallax-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 255, 255, 0.1);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    transition: all 0.2s;
}
.parallax-card:hover { border-color: #00ffff; background: rgba(0, 255, 255, 0.05); }
.score-bar-bg { background: #333; border-radius: 4px; height: 8px; width: 100%; margin: 8px 0; }
.score-bar-fill { background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%); height: 100%; border-radius: 4px; }
.path-text { font-family: 'Fira Code', monospace; color: #aaa; font-size: 0.9em; }
.entity-id { color: #00ffff; font-weight: bold; font-size: 1.1em; }
"""

SUPPORTED_FORMATS = [
    ".csv", ".tsv", ".json", ".jsonl", ".graphml", ".gexf", ".gml", ".parquet", ".xlsx", ".xls"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = str((PROJECT_ROOT / "tests" / "fixtures" / "toy_graph.csv").resolve())

# Global state
STATE = {
    "adapter": None,
    "csa": None,
    "embeddings": None,
    "graph_loaded": False,
    "communities": None,
    "n_nodes": 0,
    # Phase 11 — stream state
    "stream_adapter": None,
    "stream_running": False,
    "stream_event_log": [],   # list of dicts for display
}

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_graph(file_obj, csv_path_text, embedding_type):
    """
    Load graph from either an uploaded file or a typed path.
    Yields (status_text, community_summary) tuples for streaming progress.
    """
    # Resolve path: uploaded file takes precedence over typed path
    if file_obj is not None:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    elif csv_path_text and csv_path_text.strip():
        path = csv_path_text.strip()
    else:
        yield "ERROR: No file provided.", "N/A"
        return

    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        yield f"ERROR: Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}", "N/A"
        return

    logger.info("Loading graph from %s with %s embeddings", path, embedding_type)
    try:
        t0 = time.time()
        yield "Step 1/5: Initializing Adapter...", None

        adapter = load_file_adapter(path)
        G = adapter.to_networkx()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        STATE["adapter"] = adapter
        STATE["n_nodes"] = n_nodes

        yield f"Step 2/5: Generating Embeddings... ({n_nodes} nodes, {n_edges} edges)", None

        if embedding_type == "Sentence (all-MiniLM-L6-v2)":
            try:
                engine = SentenceEngine()
            except ImportError:
                logger.warning("sentence-transformers not installed; falling back to RandomEngine")
                engine = RandomEngine(dim=64)
        else:
            engine = RandomEngine(dim=64)

        labels = {n: n for n in G.nodes()}
        embeddings = engine.encode_entities(labels)
        adapter.embeddings = embeddings
        STATE["embeddings"] = embeddings

        yield "Step 3/5: Detecting Communities (TSC/DSCF)...", None

        # Scale resolution and trials with graph size
        resolution = adapter.adaptive_resolution()
        n_trials = 1 if n_nodes < 500 else (3 if n_nodes < 5000 else 1)
        parts = best_of_n_dscf(G, n_trials=n_trials, resolution=resolution, max_iter=30,
                                use_multiprocessing=(n_trials > 1))
        community_map = {node: i for i, members in enumerate(parts) for node in members}

        # Merge tiny communities on large graphs to prevent over-splitting
        min_size = max(2, n_nodes // 200) if n_nodes > 500 else 2
        if len(parts) > 50 and min_size > 2:
            community_map = merge_small_communities(community_map, G, min_size=min_size)
            # Rebuild parts list from merged map
            merged: dict = {}
            for node, cid in community_map.items():
                merged.setdefault(cid, []).append(node)
            parts = [frozenset(v) for v in merged.values()]

        adapter.community_map = community_map
        STATE["communities"] = parts

        comm_summary = f"{len(parts)} communities (resolution={resolution:.2f}, min_size={min_size}). "
        comm_summary += ", ".join([f"C{i}:{len(m)}" for i, m in enumerate(parts[:5])])
        if len(parts) > 5:
            comm_summary += f"... (+{len(parts) - 5} more)"

        yield "Step 4/5: Building Community Distance Index...", comm_summary

        dist = build_community_distance_matrix(G, community_map)
        adj = adjacent_community_pairs(G, community_map)

        yield "Step 5/5: Initializing CSA Engine...", comm_summary

        csa = CSAEngine(adapter=adapter)
        csa.set_community_graph(dist, adj)
        STATE["csa"] = csa
        STATE["graph_loaded"] = True

        total = time.time() - t0
        logger.info("System ready in %.2fs", total)
        yield f"[OK] System Ready ({total:.2f}s) | {n_nodes} nodes | {n_edges} edges", comm_summary

    except Exception as e:
        logger.error("Error loading graph: %s", str(e), exc_info=True)
        STATE["graph_loaded"] = False
        yield f"[ERROR] {type(e).__name__}: {str(e)}", "N/A"


def format_path_html(answers):
    """Create HTML cards for reasoning results."""
    if not answers:
        return "<div style='color:#888;padding:20px;'>No paths found. Try increasing beam width or max hops.</div>"

    html = "<div class='results-container'>"
    for ans in answers:
        path_str = " &rarr; ".join(f"<code>{n}</code>" for n in ans.best_path.nodes)
        score_pct = int(min(ans.score, 1.0) * 100)
        html += f"""
        <div class='parallax-card'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='entity-id'>{ans.entity_id}</span>
                <span style='color:#00ffff;font-family:monospace;'>{ans.score:.4f}</span>
            </div>
            <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{score_pct}%;'></div>
            </div>
            <div class='path-text'>{path_str}</div>
            <div style='font-size:0.8em;color:#666;margin-top:5px;'>
                Comm: {ans.score_breakdown.get('community', 0):.2f} |
                Sim: {ans.score_breakdown.get('semantic', 0):.2f} |
                Edge: {ans.score_breakdown.get('edge', 0):.2f}
            </div>
        </div>"""
    html += "</div>"
    return html


def run_reasoning(query, beam_width, max_hop, top_k):
    """Execute the reasoning pipeline."""
    if not STATE["graph_loaded"]:
        return "<h3 style='color:#ff4444;'>Please load a graph first.</h3>", None
    if not query or not query.strip():
        return "<h3 style='color:#ff4444;'>Enter a query entity.</h3>", None

    try:
        adapter = STATE["adapter"]
        csa = STATE["csa"]
        seeds = [e.id for e in adapter.find_entities(query.strip(), top_k=1) if e]
        if not seeds:
            return (
                f"<h3 style='color:#ff8844;'>No entity found matching <code>{query}</code>. "
                f"Try a different name or check the graph.</h3>",
                None,
            )

        seed = seeds[0]
        traversal = BeamTraversal(
            adapter=adapter, csa_engine=csa, beam_width=int(beam_width), max_hop=int(max_hop)
        )
        paths = traversal.traverse([seed])
        answers = extract(paths, top_k=int(top_k))

        html = (
            f"<div style='margin-bottom:16px;'>"
            f"<b>Seed Entity</b>: <code style='color:#00ffff;'>{seed}</code> "
            f"&nbsp;&middot;&nbsp; {len(answers)} answer(s) found"
            f"</div>"
        )
        html += format_path_html(answers)

        structured = [
            {
                "rank": i + 1,
                "answer": ans.entity_id,
                "score": ans.score,
                "path": ans.best_path.nodes,
                "breakdown": ans.score_breakdown,
            }
            for i, ans in enumerate(answers)
        ]
        return html, structured

    except Exception as e:
        logger.error("Reasoning error: %s", str(e), exc_info=True)
        return f"<h3 style='color:#ff4444;'>[ERROR] {type(e).__name__}: {e}</h3>", None


def generate_graph_viz():
    """Generate interactive graph visualization using pyvis."""
    if not STATE["graph_loaded"]:
        return "<div style='color:#888;padding:20px;'>Load a graph first.</div>"

    adapter = STATE["adapter"]
    G = adapter.to_networkx()
    n_nodes = G.number_of_nodes()

    # Cap visualization at 500 nodes to keep the browser responsive
    if n_nodes > 500:
        import random
        sample_nodes = random.sample(list(G.nodes()), 500)
        G = G.subgraph(sample_nodes)

    community_map = adapter.community_map
    colors = [
        "#FF5733", "#33FF57", "#3357FF", "#F333FF", "#FF33A1",
        "#33FFF6", "#F6FF33", "#FF8C33", "#8C33FF", "#33FF8C",
        "#FFB347", "#87CEEB", "#DDA0DD", "#98FB98", "#F0E68C",
    ]

    try:
        net = Network(height="600px", width="100%", bgcolor="#0b0f19", font_color="white")
        net.force_atlas_2based()

        for node in G.nodes():
            comm_id = community_map.get(node, 0)
            color = colors[comm_id % len(colors)]
            net.add_node(
                node,
                label=str(node),
                color=color,
                title=f"<b>{node}</b><br>Community: {comm_id}",
                size=18,
            )

        for source, target, data in G.edges(data=True):
            rel = data.get("relation", "")
            net.add_edge(source, target, color="rgba(255,255,255,0.15)", width=1, title=rel)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
            net.save_graph(tmp.name)
            html_content = Path(tmp.name).read_text(encoding="utf-8")

        note = ""
        if STATE["n_nodes"] > 500:
            note = (
                f"<div style='color:#888;font-size:0.85em;margin-bottom:8px;'>"
                f"Showing 500 of {STATE['n_nodes']} nodes (sampled for performance).</div>"
            )

        escaped = html_content.replace("&", "&amp;").replace('"', "&quot;")
        return note + f'<iframe style="width:100%;height:600px;border:none;" srcdoc="{escaped}"></iframe>'

    except Exception as e:
        logger.error("Viz error: %s", str(e), exc_info=True)
        return f"<div style='color:#ff4444;'>[ERROR] {type(e).__name__}: {e}</div>"


def get_graph_stats():
    """Return a markdown summary of the loaded graph."""
    if not STATE["graph_loaded"]:
        return "No graph loaded."
    adapter = STATE["adapter"]
    G = adapter.to_networkx()
    parts = STATE["communities"] or []
    edge_types = adapter.get_edge_types() if hasattr(adapter, "get_edge_types") else []
    lines = [
        f"**Nodes**: {G.number_of_nodes()}",
        f"**Edges**: {G.number_of_edges()}",
        f"**Communities**: {len(parts)}",
        f"**Directed**: {G.is_directed()}",
        f"**Edge types**: {len(edge_types)} — {', '.join(edge_types[:10])}{'...' if len(edge_types) > 10 else ''}",
        f"**Avg degree**: {(2 * G.number_of_edges() / max(G.number_of_nodes(), 1)):.2f}",
    ]
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 11 — Live Feed functions
# ---------------------------------------------------------------------------

def _make_stream_adapter(window_seconds: float, max_edges: int) -> StreamAdapter:
    """Create and configure a fresh StreamAdapter."""
    adapter = StreamAdapter(
        time_window_seconds=window_seconds,
        max_edges=int(max_edges),
        min_events_before_update=5,
        directed=True,
    )
    adapter.community_map = {}
    adapter.embeddings = {}
    return adapter


def start_stream(source_type, file_path_or_url, window_seconds, max_edges):
    """
    Start a live stream from the selected source.
    Returns status text and initial stats.
    """
    if STATE["stream_running"]:
        return "Stream already running. Stop it first.", _stream_stats_text()

    adapter = _make_stream_adapter(float(window_seconds), int(max_edges))
    STATE["stream_adapter"] = adapter
    STATE["stream_event_log"] = []

    # Register a mutation listener to populate the event log
    def on_mutation(action, event):
        entry = {
            "action": action,
            "source": event.source,
            "relation": event.relation,
            "target": event.target,
            "ts": round(event.timestamp, 2),
        }
        STATE["stream_event_log"].append(entry)
        if len(STATE["stream_event_log"]) > 200:
            STATE["stream_event_log"] = STATE["stream_event_log"][-200:]

    adapter.add_mutation_listener(on_mutation)

    # Wire the selected source
    try:
        if source_type == "File Tail (CSV/JSON Lines)":
            if not file_path_or_url or not file_path_or_url.strip():
                return "ERROR: Provide a file path to tail.", ""
            path = file_path_or_url.strip()
            if not Path(path).exists():
                return f"ERROR: File not found: {path}", ""
            source = FileTailSource(path)
            adapter.add_source(source)

        elif source_type == "HTTP Polling":
            if not file_path_or_url or not file_path_or_url.strip():
                return "ERROR: Provide an HTTP URL to poll.", ""
            source = HTTPPollingSource(file_path_or_url.strip(), poll_interval=1.0)
            adapter.add_source(source)

        elif source_type == "Simulated Sensor (demo)":
            # Built-in demo: simulates a temperature sensor + pressure sensor
            import random
            temp_disc = ThresholdDiscretizer("temp_sensor", low=15.0, high=35.0, spike=50.0)
            press_disc = ThresholdDiscretizer("pressure_sensor", low=900.0, high=1050.0, spike=1100.0)

            def sensor_callback():
                events = []
                events += temp_disc.process(random.gauss(25, 12))
                events += press_disc.process(random.gauss(1013, 30))
                return events

            source = PythonCallbackSource(sensor_callback, poll_interval=0.5)
            adapter.add_source(source)

        elif source_type == "Push (API only)":
            pass  # No background source — events come via ingest() calls

        else:
            return f"ERROR: Unknown source type: {source_type}", ""

        adapter.start()
        STATE["stream_running"] = True
        return f"[OK] Stream started — source: {source_type}", _stream_stats_text()

    except Exception as e:
        logger.error("Stream start error: %s", e, exc_info=True)
        return f"[ERROR] {type(e).__name__}: {e}", ""


def stop_stream():
    """Stop the running stream."""
    if not STATE["stream_running"]:
        return "No stream running.", ""
    adapter = STATE["stream_adapter"]
    if adapter:
        adapter.stop()
    STATE["stream_running"] = False
    return "[OK] Stream stopped.", _stream_stats_text()


def _stream_stats_text() -> str:
    adapter = STATE["stream_adapter"]
    if adapter is None:
        return "No stream active."
    s = adapter.live_stats()
    return (
        f"Nodes: {s['nodes']} | Edges: {s['edges']} | "
        f"Communities: {s['communities']} | "
        f"Events/s: {s['events_per_second']:.1f} | "
        f"Total: {s['total_ingested']} ingested, {s['total_evicted']} evicted"
    )


def refresh_stream_status():
    """Poll current stream stats for the UI."""
    return _stream_stats_text()


def get_stream_event_log():
    """Return the recent event log as HTML."""
    log = STATE["stream_event_log"]
    if not log:
        return "<div style='color:#888;padding:10px;'>No events yet. Start a stream to see live graph mutations.</div>"

    rows = ""
    for entry in reversed(log[-50:]):
        action_color = "#33FF57" if entry["action"] == "add" else "#FF5733"
        rows += (
            f"<tr>"
            f"<td style='color:{action_color};padding:2px 8px;'>{entry['action']}</td>"
            f"<td style='padding:2px 8px;color:#00ffff;'>{entry['source']}</td>"
            f"<td style='padding:2px 8px;color:#aaa;'>{entry['relation']}</td>"
            f"<td style='padding:2px 8px;color:#00ffff;'>{entry['target']}</td>"
            f"<td style='padding:2px 8px;color:#666;font-size:0.8em;'>{entry['ts']}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:0.9em;'>"
        "<thead><tr style='color:#555;'>"
        "<th style='padding:2px 8px;text-align:left;'>Action</th>"
        "<th style='padding:2px 8px;text-align:left;'>Source</th>"
        "<th style='padding:2px 8px;text-align:left;'>Relation</th>"
        "<th style='padding:2px 8px;text-align:left;'>Target</th>"
        "<th style='padding:2px 8px;text-align:left;'>Timestamp</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def run_stream_query(query, beam_width, max_hop, top_k):
    """Run a reasoning query against the live StreamAdapter graph."""
    adapter = STATE["stream_adapter"]
    if adapter is None or adapter.node_count() == 0:
        return "<div style='color:#888;'>Start a stream and wait for events before querying.</div>", None

    # Build a fresh CSA engine from current live graph state
    try:
        G = adapter.to_networkx()
        if G.number_of_nodes() == 0:
            return "<div style='color:#888;'>Graph is empty — no events ingested yet.</div>", None

        from core.attention_engine import CSAEngine
        from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
        from reasoning.answer_extractor import extract

        dist = build_community_distance_matrix(G, adapter.community_map)
        adj = adjacent_community_pairs(G, adapter.community_map)
        csa = CSAEngine(adapter=adapter)
        csa.set_community_graph(dist, adj)

        seeds = [e.id for e in adapter.find_entities(query.strip(), top_k=1) if e]
        if not seeds:
            return f"<div style='color:#ff8844;'>No entity found matching <code>{query}</code>.</div>", None

        traversal = BeamTraversal(adapter=adapter, csa_engine=csa,
                                  beam_width=int(beam_width), max_hop=int(max_hop))
        paths = traversal.traverse(seeds)
        answers = extract(paths, top_k=int(top_k))

        html = (
            f"<div style='margin-bottom:12px;'><b>Seed</b>: "
            f"<code style='color:#00ffff;'>{seeds[0]}</code> "
            f"({len(answers)} answer(s))</div>"
        )
        html += format_path_html(answers)
        structured = [
            {"rank": i + 1, "answer": a.entity_id, "score": a.score,
             "path": a.best_path.nodes, "breakdown": a.score_breakdown}
            for i, a in enumerate(answers)
        ]
        return html, structured
    except Exception as e:
        logger.error("Stream query error: %s", e, exc_info=True)
        return f"<div style='color:#ff4444;'>[ERROR] {type(e).__name__}: {e}</div>", None


def generate_stream_viz():
    """Visualize the current live StreamAdapter graph."""
    adapter = STATE["stream_adapter"]
    if adapter is None or adapter.node_count() == 0:
        return "<div style='color:#888;padding:10px;'>No live graph data yet.</div>"

    # Temporarily set community_map on the global STATE adapter so the existing viz works
    orig_adapter = STATE["adapter"]
    STATE["adapter"] = adapter
    STATE["graph_loaded"] = True
    STATE["n_nodes"] = adapter.node_count()
    result = generate_graph_viz()
    STATE["adapter"] = orig_adapter
    STATE["graph_loaded"] = orig_adapter is not None
    return result


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Parallax Studio") as demo:
    gr.Markdown("# Parallax Studio")
    gr.Markdown("*Glass-Box Community-Structured Reasoning Engine*")

    with gr.Row():
        # ── Left sidebar ────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            with gr.Group():
                gr.Markdown("### Graph Setup")

                file_upload = gr.File(
                    label="Upload graph file",
                    file_types=SUPPORTED_FORMATS,
                    file_count="single",
                )
                gr.Markdown(
                    "<small>Supported: CSV, TSV, JSON, JSONL, GraphML, GEXF, GML, Parquet, Excel</small>",
                )
                csv_input = gr.Textbox(
                    label="Or paste a file path",
                    value=DEFAULT_CSV,
                    placeholder="/path/to/graph.csv",
                )
                emb_input = gr.Dropdown(
                    choices=["Random (Fast)", "Sentence (all-MiniLM-L6-v2)"],
                    value="Random (Fast)",
                    label="Embedding Model",
                )
                load_btn = gr.Button("Load Engine", variant="primary")

            with gr.Accordion("Advanced Parameters", open=True):
                beam_slider = gr.Slider(1, 100, value=20, step=1, label="Beam Width")
                hop_slider  = gr.Slider(1, 10,  value=3,  step=1, label="Max Hops")
                k_slider    = gr.Slider(1, 50,  value=10, step=1, label="Top K Answers")

            with gr.Group():
                gr.Markdown("### Status")
                status_output = gr.Textbox(label="System Status", interactive=False)
                comm_output   = gr.Textbox(label="DSCF Community Info", interactive=False, lines=3)

        # ── Right main area ─────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Reasoning"):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Query Entity",
                            placeholder="Type a node name (e.g. 'newton')...",
                            scale=4,
                        )
                        reason_btn = gr.Button("Run Reasoning", variant="primary", scale=1)
                    chat_output = gr.HTML(label="Reasoning Trace")

                with gr.Tab("Interactive Graph"):
                    viz_btn    = gr.Button("Refresh Graph Visualization")
                    viz_output = gr.HTML(label="Knowledge Graph Explorer")

                with gr.Tab("Graph Stats"):
                    stats_btn    = gr.Button("Show Stats")
                    stats_output = gr.Markdown()

                with gr.Tab("Structured Data"):
                    json_output = gr.JSON(label="Raw Path Metadata")

                # ── Phase 11: Live Feed tab ──────────────────────────────
                with gr.Tab("Live Feed"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=240):
                            gr.Markdown("### Stream Source")
                            stream_source_type = gr.Dropdown(
                                choices=[
                                    "Simulated Sensor (demo)",
                                    "File Tail (CSV/JSON Lines)",
                                    "HTTP Polling",
                                    "Push (API only)",
                                ],
                                value="Simulated Sensor (demo)",
                                label="Source Type",
                            )
                            stream_path_input = gr.Textbox(
                                label="File Path / URL",
                                placeholder="/path/to/stream.jsonl  or  http://host/events",
                            )
                            gr.Markdown("### Window Settings")
                            stream_window = gr.Slider(5, 300, value=60, step=5, label="Window (seconds)")
                            stream_max_edges = gr.Slider(100, 50000, value=5000, step=100, label="Max Edges")

                            with gr.Row():
                                stream_start_btn = gr.Button("Start", variant="primary")
                                stream_stop_btn  = gr.Button("Stop",  variant="stop")

                            stream_status_out = gr.Textbox(label="Stream Status", interactive=False, lines=2)

                        with gr.Column(scale=3):
                            with gr.Tabs():
                                with gr.Tab("Event Log"):
                                    stream_refresh_btn = gr.Button("Refresh Log")
                                    stream_log_out = gr.HTML()

                                with gr.Tab("Live Graph"):
                                    stream_viz_btn = gr.Button("Refresh Visualization")
                                    stream_viz_out = gr.HTML()

                                with gr.Tab("Query Live Graph"):
                                    with gr.Row():
                                        stream_query_input = gr.Textbox(
                                            label="Query Entity",
                                            placeholder="e.g. temp_sensor",
                                            scale=4,
                                        )
                                        stream_query_btn = gr.Button("Run", variant="primary", scale=1)
                                    stream_query_out  = gr.HTML()
                                    stream_query_json = gr.JSON(label="Structured Results")

    # ── Event wiring ────────────────────────────────────────────────────────

    load_btn.click(
        fn=load_graph,
        inputs=[file_upload, csv_input, emb_input],
        outputs=[status_output, comm_output],
    ).then(
        fn=generate_graph_viz,
        inputs=[],
        outputs=viz_output,
    ).then(
        fn=get_graph_stats,
        inputs=[],
        outputs=stats_output,
    )

    viz_btn.click(fn=generate_graph_viz, inputs=[], outputs=viz_output)
    stats_btn.click(fn=get_graph_stats, inputs=[], outputs=stats_output)

    reason_btn.click(
        fn=run_reasoning,
        inputs=[query_input, beam_slider, hop_slider, k_slider],
        outputs=[chat_output, json_output],
    )
    query_input.submit(
        fn=run_reasoning,
        inputs=[query_input, beam_slider, hop_slider, k_slider],
        outputs=[chat_output, json_output],
    )

    # ── Stream event wiring ─────────────────────────────────────────────────

    stream_start_btn.click(
        fn=start_stream,
        inputs=[stream_source_type, stream_path_input, stream_window, stream_max_edges],
        outputs=[stream_status_out, stream_status_out],
    )
    stream_stop_btn.click(
        fn=stop_stream,
        inputs=[],
        outputs=[stream_status_out, stream_status_out],
    )
    stream_refresh_btn.click(
        fn=lambda: (get_stream_event_log(), refresh_stream_status()),
        inputs=[],
        outputs=[stream_log_out, stream_status_out],
    )
    stream_viz_btn.click(
        fn=generate_stream_viz,
        inputs=[],
        outputs=stream_viz_out,
    )
    stream_query_btn.click(
        fn=run_stream_query,
        inputs=[stream_query_input, beam_slider, hop_slider, k_slider],
        outputs=[stream_query_out, stream_query_json],
    )
    stream_query_input.submit(
        fn=run_stream_query,
        inputs=[stream_query_input, beam_slider, hop_slider, k_slider],
        outputs=[stream_query_out, stream_query_json],
    )


# ---------------------------------------------------------------------------
# Test harness (python ui/studio.py --test)
# ---------------------------------------------------------------------------

def test_logic():
    print("Testing UI Logic...")

    # 1. Load
    print("  Loading graph...")
    result = None
    for status, comm in load_graph(None, DEFAULT_CSV, "Random (Fast)"):
        print(f"    {status}")
        result = (status, comm)

    if not STATE["graph_loaded"]:
        print("  FAIL: graph load failed")
        return False

    # 2. Reasoning
    print("  Testing reasoning ('newton')...")
    text, data = run_reasoning("newton", 10, 2, 5)
    count = len(data) if data else 0
    print(f"    Found {count} answer(s).")
    if count == 0:
        print("  FAIL: no answers returned")
        return False

    # 3. Visualization
    print("  Testing graph visualization...")
    html = generate_graph_viz()
    if "<iframe" not in html and "ERROR" in html:
        print(f"  FAIL: viz returned error: {html[:200]}")
        return False
    print(f"    HTML length: {len(html)}")

    # 4. Stats
    print("  Testing graph stats...")
    stats = get_graph_stats()
    print(f"    {stats[:120]}")

    print("Logic test PASSED.")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.exit(0 if test_logic() else 1)
    else:
        demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft(), inbrowser=True)
