"""
CEREBRUM Studio -- Interactive Reasoning Interface.

A Gradio-based UI for exploring Knowledge Graphs using Community-Structured Attention.
Allows users to run reasoning queries, inspect paths, and visualize the attention mechanism.

Usage:
    python ui/studio.py
"""
import sys
import time
import logging
import pickle
import json
import random
import tempfile
from pathlib import Path
from typing import Union, Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network

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
logger = logging.getLogger("CEREBRUMStudio")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.file_adapter import load_file_adapter
from adapters.stream_adapter import StreamAdapter, PythonCallbackSource, FileTailSource, HTTPPollingSource
from core.cerebrum import CerebrumGraph
from core.graph_bridge import GraphBridgeEngine
from core.embedding_engine import RandomEngine, SentenceEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs, build_community_graph
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract
from core.community_engine import best_of_n_dscf
from core.discretizer import ThresholdDiscretizer
from core.rem_engine import REMEngine
from core.insight_engine import InsightEngine
from core.insight_validator import InsightValidator
from core.resource_governor import ResourceGovernor

# ---------------------------------------------------------------------------
# Constants & State
# ---------------------------------------------------------------------------

PARAM_PRESETS = {
    "Standard (Balanced)": {"beam": 20, "hop": 3, "k": 10, "mem": 90},
    "Deep Reasoning (High Precision)": {"beam": 50, "hop": 6, "k": 5, "mem": 85},
    "Fast Discovery (Broad Scan)": {"beam": 10, "hop": 2, "k": 20, "mem": 95},
    "Stress Test (Aggressive)": {"beam": 100, "hop": 10, "k": 10, "mem": 99},
}

SUPPORTED_FORMATS = [
    ".csv", ".tsv", ".json", ".jsonl", ".graphml", ".gexf", ".gml", ".parquet", ".xlsx", ".xls"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = str((PROJECT_ROOT / "tests" / "fixtures" / "toy_graph.csv").resolve())

STATE: Dict[str, Any] = {
    "graph_obj": None,
    "adapter": None,
    "csa": None,
    "graph_loaded": False,
    "communities": None,
    "n_nodes": 0,
    "stream_adapter": None,
    "stream_running": False,
    "stream_event_log": [],
    "rem": None,
    "insight": None,
    "validator": None,
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def generate_param_radar(alpha, beta, gamma, delta, epsilon, zeta, eta, iota, theta):
    """Generate a radar chart for the current parameter configuration."""
    categories = [
        'Semantic (Î±)', 'Community (Î²)', 'Edge Type (Î³)', 'Distance (Î´)', 
        'Hop Decay (Îµ)', 'PageRank (Î¶)', 'Temp Edge (Î·)', 'Node Recency (Î¹)', 'Grounding (Î¸)'
    ]
    r_values = [alpha, beta, gamma, abs(delta), epsilon, zeta, eta, iota, theta]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_values, theta=categories, fill='toself', name='Parameter Profile', line_color='#bc8cff'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d"),
            angularaxis=dict(gridcolor="#30363d")
        ),
        showlegend=False, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=20), height=300
    )
    return fig

def generate_attention_radar(feature_tuple):
    """Generate a radar chart for a 9-element feature vector."""
    if not feature_tuple or len(feature_tuple) < 9:
        return go.Figure()

    categories = [
        'Similarity', 'Community', 'Edge Weight', 'Distance (inv)', 
        'Hop Decay', 'PageRank', 'Temp Decay', 'Node Recency', 'Grounding'
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=feature_tuple, theta=categories, fill='toself', name='Step Attention', line_color='#58a6ff'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d"),
            angularaxis=dict(gridcolor="#30363d")
        ),
        showlegend=False, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=20), height=300
    )
    return fig

def format_path_html(answers):
    """Create HTML cards for reasoning results."""
    if not answers:
        return "<div style='color:#888;padding:20px;'>No paths found. Try increasing beam width or max hops.</div>"

    html = "<div class='results-container'>"
    for ans in answers:
        path_str = " &rarr; ".join(f"<code>{n}</code>" for n in ans.best_path.nodes)
        score_pct = int(min(ans.score, 1.0) * 100)
        html += f"""
        <div class='cerebrum-card'>
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

# ---------------------------------------------------------------------------
# Core Callbacks
# ---------------------------------------------------------------------------

def load_graph(file_obj, csv_path_text, embedding_type, progress=gr.Progress()):
    if file_obj is not None:
        path = file_obj.name
        source_name = Path(path).name
    elif csv_path_text and csv_path_text.strip():
        path = csv_path_text.strip()
        source_name = path
    else:
        return "ERROR: No file provided.", "N/A"

    logger.info("Loading graph from %s", path)
    try:
        t0 = time.time()
        progress(0, desc=f"Initializing {source_name}...")
        emb_mode = "sentence" if "Sentence" in embedding_type else "random"
        graph = CerebrumGraph.from_kb(path, embeddings=emb_mode)
        
        progress(0.3, desc=f"Building Index... ({graph.node_count} nodes)")
        graph.build(seed=42)
        
        progress(0.7, desc="Finalizing Engines...")
        STATE.update({
            "graph_obj": graph, "adapter": graph.adapter, "csa": graph._csa,
            "communities": graph.communities, "n_nodes": graph.node_count, "graph_loaded": True,
            "rem": REMEngine(graph.adapter), "insight": InsightEngine(graph.adapter),
            "validator": InsightValidator(graph.adapter)
        })

        comm_summary = f"{len(graph.communities)} communities detected via DSCF."
        total = time.time() - t0
        status = f"[OK] Loaded {source_name} ({total:.2f}s) | {graph.node_count} nodes"
        return status, comm_summary
    except Exception as e:
        logger.error("Load error: %s", e, exc_info=True)
        return f"[ERROR] {type(e).__name__}: {e}", "N/A"

def run_reasoning(query, beam_width, max_hop, top_k, mem_threshold, governor=None):
    if not STATE["graph_loaded"] or not STATE["graph_obj"]:
        return "Load a graph first.", None, go.Figure()
    if not query or not query.strip():
        return "Enter a query entity.", None, go.Figure()

    try:
        graph = STATE["graph_obj"]
        seeds = [e.id for e in graph.adapter.find_entities(query.strip(), top_k=1) if e]
        if not seeds:
            return f"No entity matching '{query}'", None, go.Figure()

        answers = graph.query([seeds[0]], top_k=int(top_k), max_hop=int(max_hop), 
                              beam_width=int(beam_width), memory_threshold_pct=float(mem_threshold))

        best_path_nodes = []
        if answers:
            best_path_nodes = answers[0].best_path.nodes

        radar_fig = go.Figure()
        if answers and hasattr(answers[0].best_path, 'edge_features') and answers[0].best_path.edge_features:
            mean_feats = np.mean(np.array(answers[0].best_path.edge_features), axis=0)
            radar_fig = generate_attention_radar(mean_feats.tolist())

        html = f"<div style='margin-bottom:12px;'><b>Seed</b>: <code style='color:#00ffff;'>{seeds[0]}</code></div>" + format_path_html(answers)
        structured = [{"answer": a.entity_id, "score": a.score, "path": a.best_path.nodes} for a in answers]
        return html, structured, radar_fig, best_path_nodes
    except Exception as e:
        logger.error("Reasoning error: %s", e, exc_info=True)
        return f"[ERROR] {e}", None, go.Figure()

def get_graph_stats():
    if not STATE["graph_loaded"]: return go.Figure(), "No graph loaded."
    G = STATE["adapter"].to_networkx()
    
    # Degree plot
    degrees = [d for n, d in G.degree()]
    fig_deg = px.histogram(pd.DataFrame({"Degree": degrees}), x="Degree", title="Degree Distribution", template="plotly_dark")
    
    # Community plot
    comm_sizes = [len(c) for c in (STATE["communities"] or [])]
    fig_comm = px.pie(pd.DataFrame({"Size": comm_sizes, "ID": [f"C{i}" for i in range(len(comm_sizes))]}), 
                      values="Size", names="ID", title="Community Composition", hole=0.4, template="plotly_dark")

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]])
    for t in fig_deg.data: fig.add_trace(t, row=1, col=1)
    for t in fig_comm.data: fig.add_trace(t, row=1, col=2)
    fig.update_layout(template="plotly_dark", height=400, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    summary = f"**Nodes**: {G.number_of_nodes()} | **Edges**: {G.number_of_edges()} | **Density**: {nx.density(G):.4f}"
    return fig, summary

def commit_params(a, b, g, d, e, z, eta, iota, theta):
    if not STATE["csa"]: return "ERROR: Load graph first."
    try:
        STATE["csa"].alpha, STATE["csa"].beta, STATE["csa"].gamma = a, b, g
        STATE["csa"].delta, STATE["csa"].epsilon, STATE["csa"].zeta = d, e, z
        STATE["csa"].eta, STATE["csa"].iota = eta, iota
        return f"Success: Weights updated (Î±={a:.2f}, Î²={b:.2f}...)"
    except Exception as err: return f"ERROR: {err}"

def run_rem_cycle(dry_run=True):
    if not STATE["rem"]: return "ERROR: REM not initialized."
    try:
        rep = STATE["rem"].run(dry_run=dry_run)
        return f"REM Complete. Pruned: {rep.pruned_count}, Synthesized: {rep.synthesized_count}"
    except Exception as e: return f"ERROR: {e}"

def get_insight_log():
    if not STATE["insight"]: return "Insight Engine not initialized."
    events = STATE["insight"].recent_events(n=10)
    if not events: return "<div style='color:#888;padding:20px;'>No significant structural insights discovered yet.</div>"
    html = "<div class='results-container'>"
    for ev in events:
        badge_class = "insight-surprise" if ev.insight_score > 0.3 else "insight-high"
        path_str = " &rarr; ".join(ev.path.nodes) if ev.path else f"{ev.source} &rarr; {ev.target} (cold)"
        html += f"""
        <div class='cerebrum-card'>
            <div style='display:flex;justify-content:space-between;'>
                <span class='entity-id'>{ev.bridging_node}</span>
                <span class='insight-badge {badge_class}'>INSIGHT</span>
            </div>
            <div style='margin-top:10px;font-size:0.9em;'>
                <strong>Score:</strong> {ev.insight_score:.4f} | <strong>Status:</strong> {ev.validation_status}
            </div>
            <div class='path-text' style='margin-top:8px;'>{path_str}</div>
        </div>"""
    html += "</div>"
    return html

def run_validation():
    if not STATE["validator"] or not STATE["insight"]:
        return "Validator or Insight Engine not initialized."
    try:
        events = STATE["insight"].recent_events(n=100)
        results = STATE["validator"].validate_all(events)
        return f"Validation Complete. {len(results)} insights processed."
    except Exception as e:
        return f"ERROR in Validation: {e}"

# ---------------------------------------------------------------------------
# Phase 11 â€” Live Feed functions
# ---------------------------------------------------------------------------

def start_stream(source_type, file_path_or_url, window_seconds, max_edges):
    if STATE["stream_running"]: return "Stream already running.", ""
    adapter = StreamAdapter(time_window_seconds=float(window_seconds), max_edges=int(max_edges))
    STATE["stream_adapter"] = adapter
    STATE["stream_event_log"] = []
    
    def on_mutation(action, event):
        STATE["stream_event_log"].append({"action": action, "source": event.source, "relation": event.relation, "target": event.target, "ts": event.timestamp})
        if len(STATE["stream_event_log"]) > 200: STATE["stream_event_log"].pop(0)
    
    adapter.add_mutation_listener(on_mutation)
    try:
        if "File" in source_type: adapter.add_source(FileTailSource(file_path_or_url.strip()))
        elif "HTTP" in source_type: adapter.add_source(HTTPPollingSource(file_path_or_url.strip()))
        elif "Simulated" in source_type:
            temp_disc = ThresholdDiscretizer("temp", low=15, high=35)
            adapter.add_source(PythonCallbackSource(lambda: temp_disc.process(random.gauss(25,5)), poll_interval=1.0))
        adapter.start()
        STATE["stream_running"] = True
        return f"Stream started: {source_type}", "Active"
    except Exception as e: return f"Error: {e}", "Failed"

def stop_stream():
    if STATE["stream_adapter"]: STATE["stream_adapter"].stop()
    STATE["stream_running"] = False
    return "Stream stopped", "Inactive"

def get_stream_event_log():
    rows = "".join([f"<tr><td>{e['action']}</td><td>{e['source']}</td><td>{e['relation']}</td><td>{e['target']}</td></tr>" for e in reversed(STATE["stream_event_log"])])
    return f"<table>{rows}</table>"

def generate_graph_viz():
    if not STATE["graph_loaded"]: return "Load a graph first."
    G = STATE["adapter"].to_networkx()
    if G.number_of_nodes() > 500: return "Graph too large for 2D (use 3D Explorer)."
    
    net = Network(height="600px", width="100%", bgcolor="#0d1117", font_color="#c9d1d9")
    for n in G.nodes():
        cid = STATE["adapter"].get_community(n)
        net.add_node(str(n), label=str(n), color=("#58a6ff" if cid < 0 else px.colors.qualitative.Safe[cid % 10]))
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        net.add_edge(str(u), str(v), title=rel, dashes=("rem_synthesized" in rel), color=("#ff8844" if "wormhole" in rel else "#8b949e"))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        return f"""<iframe srcdoc='{Path(f.name).read_text(encoding="utf-8").replace("'", "&apos;")}' style='width:100%;height:600px;border:none;'></iframe>"""

def generate_3d_viz(highlight_nodes=None):
    if not STATE["graph_loaded"]: return "Load a graph first."
    G = STATE["adapter"].to_networkx()
    
    path_set = set(highlight_nodes) if highlight_nodes else set()
    
    nodes_data = []
    for n in G.nodes():
        is_on_path = str(n) in path_set
        nodes_data.append({
            "id": str(n),
            "group": STATE["adapter"].get_community(n),
            "size": 12 if is_on_path else 4,
            "color": "#00ffff" if is_on_path else None # Auto-color if not on path
        })

    links_data = []
    for u, v, d in G.edges(data=True):
        u_str, v_str = str(u), str(v)
        rel = d.get("relation", "")
        is_on_path = u_str in path_set and v_str in path_set
        
        # Determine color
        if is_on_path:
            color = "#00ffff"
        elif "wormhole" in rel:
            color = "#ff8844"
        elif "rem_synthesized" in rel:
            color = "#8b949e"
        else:
            color = "#444444"

        links_data.append({
            "source": u_str,
            "target": v_str,
            "rel": rel,
            "color": color,
            "width": 3 if is_on_path else 1,
            "is_on_path": is_on_path,
            "is_wormhole": "wormhole" in rel
        })
    
    # Use a more robust CDN and handle large JSON by separating it
    graph_json = json.dumps({"nodes": nodes_data, "links": links_data})
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/3d-force-graph"></script>
        <style> body {{ margin: 0; background: #0d1117; overflow: hidden; }} </style>
    </head>
    <body>
        <div id="3d-graph"></div>
        <script>
            const data = {graph_json};
            const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
                .backgroundColor('#0d1117')
                .nodeLabel(node => `<span style="color:#00ffff">${{node.id}}</span>`)
                .nodeAutoColorBy('group')
                .graphData(data)
                .linkDirectionalArrowLength(3.5)
                .linkDirectionalArrowRelPos(1)
                .linkColor(link => link.color)
                .linkWidth(link => link.width)
                .linkCurvature(link => link.is_wormhole ? 0.2 : 0)
                .linkDirectionalParticles(link => link.is_on_path ? 10 : (link.width > 1 ? 2 : 0))
                .linkDirectionalParticleSpeed(link => link.is_on_path ? 0.02 : 0.005)
                .linkDirectionalParticleWidth(link => link.is_on_path ? 4 : 2);

            // If a path exists, adjust camera after a delay
            if ({'true' if highlight_nodes else 'false'}) {{
                setTimeout(() => {{
                    const pathNodes = data.nodes.filter(n => {list(path_set)}.includes(n.id));
                    if (pathNodes.length > 0) {{
                        Graph.cameraPosition({{ x: 200, y: 200, z: 200 }}, pathNodes[0], 2000);
                    }}
                }}, 1000);
            }}
        </script>
    </body>
    </html>
    """
    return f"""<iframe srcdoc='{html.replace("'", "&apos;")}' style='width:100%;height:750px;border:none;border-radius:8px;'></iframe>"""

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

CUSTOM_CSS = ".gradio-container { background-color: #0d1117 !important; color: #c9d1d9 !important; } .cerebrum-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; } .score-bar-bg { background: #30363d; height: 8px; border-radius: 4px; } .score-bar-fill { background: #58a6ff; height: 100%; border-radius: 4px; }"

with gr.Blocks(title="CEREBRUM Studio", css=CUSTOM_CSS) as demo:
    gr.Markdown("# CEREBRUM STUDIO Pro")
    best_path_state = gr.State([]) # Hidden state for 3D highlighting
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                file_in = gr.File(label="Upload Graph (CSV/JSON)")
                path_in = gr.Textbox(
                    label="Or paste a local file path", 
                    placeholder="E:\\path\\to\\your\\graph.csv",
                    value=""
                )
                emb_in = gr.Dropdown(["Random (Fast)", "Sentence (SBERT)"], value="Random (Fast)", label="Embeddings")
                load_btn = gr.Button("Load Engine", variant="primary")
            
            with gr.Accordion("Advanced Parameters", open=True):
                beam_sl = gr.Slider(1, 100, 20, label="Beam Width")
                hop_sl = gr.Slider(1, 10, 3, label="Max Hops")
                k_sl = gr.Slider(1, 50, 10, label="Top K")
                mem_sl = gr.Slider(50, 99, 90, label="Memory %")
                
                with gr.Accordion("CSA Weight Profiler", open=False):
                    p_radar = gr.Plot()
                    weights = [gr.Slider(0, 1, v, label=l) for v, l in [(0.4,"Î±"), (0.4,"Î²"), (0.1,"Î³"), (0.05,"Î´"), (0.05,"Îµ"), (0.1,"Î¶"), (0.1,"Î·"), (0.05,"Î¹"), (1.0,"Î¸")]]
                    commit_btn = gr.Button("Commit Weights")

            status_out = gr.Textbox(label="Status")
            comm_output = gr.Textbox(label="Community Summary", visible=True)

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Reasoning"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            q_in = gr.Textbox(label="Query Entity", placeholder="e.g. newton")
                            q_btn = gr.Button("Run", variant="primary")
                            q_html = gr.HTML()
                        with gr.Column(scale=1):
                            attn_radar = gr.Plot(label="Attention Analysis")
                
                with gr.Tab("Structural Analytics"):
                    s_plot = gr.Plot()
                    s_md = gr.Markdown()
                    s_btn = gr.Button("Refresh Analytics")

                with gr.Tab("Interactive Graph"):
                    v2_btn = gr.Button("Refresh 2D")
                    v2_out = gr.HTML()

                with gr.Tab("3D Explorer"):
                    v3_btn = gr.Button("Refresh 3D")
                    v3_out = gr.HTML()

                with gr.Tab("Insight & Maintenance"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### REM Cycle")
                            rem_dry_btn = gr.Button("Run Dry-Run")
                            rem_run_btn = gr.Button("Run Commit", variant="stop")
                            rem_out = gr.Textbox(label="REM Report")
                        with gr.Column(scale=2):
                            gr.Markdown("### Insights")
                            ins_btn = gr.Button("Refresh Insights")
                            val_btn = gr.Button("Validate All", variant="primary")
                            ins_out = gr.HTML()

                with gr.Tab("Live Feed"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            st_type = gr.Dropdown(["Simulated", "File Tail", "HTTP Polling"], value="Simulated", label="Source")
                            st_path = gr.Textbox(label="Path/URL")
                            st_win = gr.Slider(5, 300, 60, label="Window (s)")
                            st_start = gr.Button("Start", variant="primary")
                            st_stop = gr.Button("Stop", variant="stop")
                            st_status = gr.Textbox(label="Stream Status")
                        with gr.Column(scale=2):
                            st_log = gr.HTML(label="Mutation Log")
                            st_ref = gr.Button("Refresh Log")

    # Wiring
    # Mutual clearing for Graph Setup
    file_in.upload(lambda: "", None, path_in)
    file_in.clear(lambda: "", None, path_in)
    path_in.change(lambda x: None if x and x.strip() else gr.update(), inputs=[path_in], outputs=[file_in])

    for w in weights:
        w.change(generate_param_radar, weights, p_radar, api_name="update_param_profile")
    
    commit_btn.click(commit_params, weights, status_out, api_name="commit_weights")
    
    load_btn.click(
        load_graph, 
        [file_in, path_in, emb_in], 
        [status_out, comm_output],
        api_name="load_graph"
    ).then(
        fn=lambda *args: (generate_param_radar(*args), *get_graph_stats()),
        inputs=weights,
        outputs=[p_radar, s_plot, s_md]
    )
    
    q_btn.click(
        run_reasoning, 
        [q_in, beam_sl, hop_sl, k_sl, mem_sl], 
        [q_html, gr.JSON(visible=False), attn_radar, best_path_state], 
        api_name="run_reasoning"
    ).then(
        fn=generate_3d_viz,
        inputs=[best_path_state],
        outputs=v3_out
    )
    s_btn.click(get_graph_stats, [], [s_plot, s_md], api_name="refresh_analytics")
    v2_btn.click(generate_graph_viz, [], v2_out, api_name="get_2d_viz")
    v3_btn.click(generate_3d_viz, [], v3_out, api_name="get_3d_viz")

    # Insight & REM Wiring
    rem_dry_btn.click(run_rem_cycle, [gr.State(True)], rem_out, api_name="rem_dry_run")
    rem_run_btn.click(run_rem_cycle, [gr.State(False)], rem_out, api_name="rem_commit")
    ins_btn.click(get_insight_log, [], ins_out, api_name="get_insights")
    val_btn.click(run_validation, [], rem_out, api_name="validate_insights").then(get_insight_log, [], ins_out)

    # Stream Wiring
    st_start.click(start_stream, [st_type, st_path, st_win, gr.State(5000)], [status_out, st_status], api_name="start_stream")
    st_stop.click(stop_stream, [], st_status, api_name="stop_stream")
    st_ref.click(get_stream_event_log, [], st_log, api_name="get_stream_log")

if __name__ == "__main__":
    if "--test" in sys.argv:
        print("Testing...")
        # Basic logic test would go here
        sys.exit(0)
    demo.launch(inbrowser=True)
