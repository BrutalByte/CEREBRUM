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
        'Semantic (α)', 'Community (β)', 'Edge Type (γ)', 'Distance (δ)', 
        'Hop Decay (ε)', 'PageRank (ζ)', 'Temp Edge (η)', 'Node Recency (ι)', 'Grounding (θ)'
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
        html += f\"\"\"
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
        </div>\"\"\"
    html += "</div>"
    return html

# ---------------------------------------------------------------------------
# Core Callbacks
# ---------------------------------------------------------------------------

def load_graph(file_obj, csv_path_text, embedding_type):
    if file_obj is not None:
        path = file_obj.name
    elif csv_path_text and csv_path_text.strip():
        path = csv_path_text.strip()
    else:
        yield "ERROR: No file provided.", "N/A"
        return

    logger.info("Loading graph from %s", path)
    try:
        t0 = time.time()
        yield "Step 1/4: Initializing Unified Pipeline...", None
        emb_mode = "sentence" if "Sentence" in embedding_type else "random"
        graph = CerebrumGraph.from_kb(path, embeddings=emb_mode)
        
        yield f"Step 2/4: Building Index... ({graph.node_count} nodes)", None
        graph.build(seed=42)
        
        STATE.update({
            "graph_obj": graph, "adapter": graph.adapter, "csa": graph._csa,
            "communities": graph.communities, "n_nodes": graph.node_count, "graph_loaded": True,
            "rem": REMEngine(graph.adapter), "insight": InsightEngine(graph.adapter),
            "validator": InsightValidator(graph.adapter)
        })

        comm_summary = f"{len(graph.communities)} communities detected via DSCF."
        total = time.time() - t0
        yield f"[OK] System Ready ({total:.2f}s) | {graph.node_count} nodes", comm_summary
    except Exception as e:
        logger.error("Load error: %s", e, exc_info=True)
        yield f"[ERROR] {type(e).__name__}: {e}", "N/A"

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

        radar_fig = go.Figure()
        if answers and hasattr(answers[0].best_path, 'edge_features') and answers[0].best_path.edge_features:
            mean_feats = np.mean(np.array(answers[0].best_path.edge_features), axis=0)
            radar_fig = generate_attention_radar(mean_feats.tolist())

        html = f"<div style='margin-bottom:12px;'><b>Seed</b>: <code style='color:#00ffff;'>{seeds[0]}</code></div>" + format_path_html(answers)
        structured = [{"answer": a.entity_id, "score": a.score, "path": a.best_path.nodes} for a in answers]
        return html, structured, radar_fig
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
        return f"Success: Weights updated (α={a:.2f}, β={b:.2f}...)"
    except Exception as err: return f"ERROR: {err}"

def run_rem_cycle(dry_run=True):
    if not STATE["rem"]: return "ERROR: REM not initialized."
    try:
        rep = STATE["rem"].run(dry_run=dry_run)
        return f"REM Complete. Pruned: {rep.pruned_count}, Synthesized: {rep.synthesized_count}"
    except Exception as e: return f"ERROR: {e}"

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
        return f"<iframe srcdoc='{Path(f.name).read_text(encoding=\"utf-8\").replace(\"'\", \"&apos;\")}' style='width:100%;height:600px;border:none;'></iframe>"

def generate_3d_viz():
    if not STATE["graph_loaded"]: return "Load a graph first."
    G = STATE["adapter"].to_networkx()
    nodes = [{"id": str(n), "group": STATE["adapter"].get_community(n)} for n in G.nodes()]
    links = [{"source": str(u), "target": str(v), "rel": d.get("relation",""), 
              "color": ("#ff8844" if "wormhole" in d.get("relation","") else "#8b949e"),
              "is_wormhole": ("wormhole" in d.get("relation",""))} for u, v, d in G.edges(data=True)]
    
    return f\"\"\"<iframe srcdoc='<html><head><script src=\"https://unpkg.com/3d-force-graph\"></script></head><body style=\"margin:0;background:#0d1117;\"><div id=\"3d-graph\"></div><script>
        const g = ForceGraph3D()(document.getElementById(\"3d-graph\")).backgroundColor(\"#0d1117\").nodeLabel(\"id\").nodeAutoColorBy(\"group\")
        .linkColor(l=>l.color).linkCurvature(l=>l.is_wormhole?0.2:0).graphData({{\"nodes\":{json.dumps(nodes)},\"links\":{json.dumps(links)}}});
    </script></body></html>' style='width:100%;height:750px;border:none;'></iframe>\"\"\"

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

CUSTOM_CSS = ".gradio-container { background-color: #0d1117 !important; color: #c9d1d9 !important; } .cerebrum-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; } .score-bar-bg { background: #30363d; height: 8px; border-radius: 4px; } .score-bar-fill { background: #58a6ff; height: 100%; border-radius: 4px; }"

with gr.Blocks(title="CEREBRUM Studio", css=CUSTOM_CSS) as demo:
    gr.Markdown("# CEREBRUM STUDIO Pro")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                file_in = gr.File(label="Upload Graph")
                path_in = gr.Textbox(label="Local Path", value=DEFAULT_CSV)
                emb_in = gr.Dropdown(["Random (Fast)", "Sentence (SBERT)"], value="Random (Fast)", label="Embeddings")
                load_btn = gr.Button("Load Engine", variant="primary")
            
            with gr.Accordion("Advanced Parameters", open=True):
                beam_sl = gr.Slider(1, 100, 20, label="Beam Width")
                hop_sl = gr.Slider(1, 10, 3, label="Max Hops")
                k_sl = gr.Slider(1, 50, 10, label="Top K")
                mem_sl = gr.Slider(50, 99, 90, label="Memory %")
                
                with gr.Accordion("CSA Weight Profiler", open=False):
                    p_radar = gr.Plot()
                    weights = [gr.Slider(0, 1, v, label=l) for v, l in [(0.4,"α"), (0.4,"β"), (0.1,"γ"), (0.05,"δ"), (0.05,"ε"), (0.1,"ζ"), (0.1,"η"), (0.05,"ι"), (1.0,"θ")]]
                    commit_btn = gr.Button("Commit Weights")

            status_out = gr.Textbox(label="Status")

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

    # Wiring
    for w in weights:
        w.change(generate_param_radar, weights, p_radar)
    
    commit_btn.click(commit_params, weights, status_out)
    
    load_btn.click(load_graph, [file_in, path_in, emb_in], [status_out, status_out]).then(
        generate_param_radar, weights, p_radar).then(get_graph_stats, [], [s_plot, s_md])
    
    q_btn.click(run_reasoning, [q_in, beam_sl, hop_sl, k_sl, mem_sl], [q_html, gr.JSON(visible=False), attn_radar])
    s_btn.click(get_graph_stats, [], [s_plot, s_md])
    v2_btn.click(generate_graph_viz, [], v2_out)
    v3_btn.click(generate_3d_viz, [], v3_out)

if __name__ == "__main__":
    if "--test" in sys.argv:
        print("Testing...")
        # Basic logic test would go here
        sys.exit(0)
    demo.launch(inbrowser=True)
