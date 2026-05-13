"""
CEREBRUM Studio -- Interactive Reasoning Interface.

A Gradio-based UI for exploring Knowledge Graphs using Community-Structured Attention.
All business logic lives in core.studio_engine.StudioEngine.  This file is
responsible only for layout, wiring, and pure presentation helpers.

Usage:
    python ui/studio.py
"""
import sys
import logging
import argparse
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.studio_engine import StudioEngine

# ---------------------------------------------------------------------------
# Singleton engine — owns all graph state
# ---------------------------------------------------------------------------

_engine = StudioEngine()

# ---------------------------------------------------------------------------
# Pure presentation helpers (no state)
# ---------------------------------------------------------------------------

# 10-param CSA weight labels in canonical order:
#   alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta
_WEIGHT_DEFS = [
    (0.40, "α — Semantic"),
    (0.40, "β — Community"),
    (0.10, "γ — Edge Type"),
    (0.05, "δ — Distance"),
    (0.05, "ε — Hop Decay"),
    (0.10, "ζ — PageRank"),
    (0.10, "η — Temp Decay"),
    (0.05, "ι — Node Recency"),
    (0.10, "μ — Synth Density"),
    (1.00, "θ — Grounding"),
]

_PARAM_CATEGORIES = [label for _, label in _WEIGHT_DEFS]


def generate_param_radar(
    alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta
) -> go.Figure:
    """Radar chart for the current 10-parameter CSA weight configuration."""
    r_values = [alpha, beta, gamma, abs(delta), epsilon, zeta, eta, iota, mu, theta]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_values,
        theta=_PARAM_CATEGORIES,
        fill="toself",
        name="Parameter Profile",
        line_color="#bc8cff",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d"),
            angularaxis=dict(gridcolor="#30363d"),
        ),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio file-upload adapter
# ---------------------------------------------------------------------------

def _load_graph_cb(file_obj, csv_path_text, embedding_type, progress=gr.Progress()):
    """Bridge between Gradio file-upload and engine.load_graph(path, emb_type)."""
    if file_obj is not None:
        path = file_obj.name
    elif csv_path_text and csv_path_text.strip():
        path = csv_path_text.strip()
    else:
        return "ERROR: No file provided.", "N/A"
    
    # engine.load_graph is now a generator
    res = ("[Starting...]", "N/A")
    for status, comms in _engine.load_graph(path, embedding_type, progress=progress):
        res = (status, comms)
    return res


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

PARAM_PRESETS = {
    "Standard (Balanced)":        {"beam": 20,  "hop": 3,  "k": 10, "mem": 90},
    "Deep Reasoning (High Prec)": {"beam": 50,  "hop": 6,  "k":  5, "mem": 85},
    "Fast Discovery (Broad Scan)":{"beam": 10,  "hop": 2,  "k": 20, "mem": 95},
    "Stress Test (Aggressive)":   {"beam": 100, "hop": 10, "k": 10, "mem": 99},
}

CUSTOM_CSS = (
    ".gradio-container{background-color:#0d1117!important;color:#c9d1d9!important;}"
    ".cerebrum-card{background:#161b22;border:1px solid #30363d;border-radius:10px;"
    "padding:15px;margin-bottom:10px;}"
    ".score-bar-bg{background:#30363d;height:8px;border-radius:4px;}"
    ".score-bar-fill{background:#58a6ff;height:100%;border-radius:4px;}"
)

with gr.Blocks(title="CEREBRUM Studio") as demo:
    gr.Markdown("# CEREBRUM STUDIO Pro")
    best_path_state = gr.State([])   # hidden state for 3D path highlighting

    with gr.Row():
        # ── Left panel: controls ──────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Group():
                file_in = gr.File(label="Upload Graph (CSV/JSON/GraphML/…)")
                path_in = gr.Textbox(
                    label="Or paste a local file path",
                    placeholder=r"E:\path\to\your\graph.csv",
                    value="",
                )
                with gr.Row():
                    history_in = gr.Dropdown(
                        label="Recent Graphs",
                        choices=_engine.get_recent_paths(),
                        interactive=True,
                    )
                    history_ref_btn = gr.Button("🔄", scale=0)
                emb_in  = gr.Dropdown(
                    ["Random (Fast)", "Sentence (BGE)", "Sentence + GraphSAGE"],
                    value="Random (Fast)",
                    label="Embeddings",
                )
                load_btn = gr.Button("Load Engine", variant="primary")

            with gr.Accordion("System Health & Backups", open=False):
                with gr.Group():
                    gr.Markdown("### Backup & Restore")
                    with gr.Row():
                        backup_btn = gr.Button("Create Backup")
                        restore_refresh_btn = gr.Button("List Backups")
                    backup_list = gr.Dropdown(
                        label="Available Backups",
                        choices=_engine.list_backups(),
                    )
                    restore_btn = gr.Button("Restore Selected Backup", variant="stop")

            with gr.Accordion("Advanced Parameters", open=True):
                beam_sl = gr.Slider(1,   100, 20, label="Beam Width")
                hop_sl  = gr.Slider(1,    10,  3, label="Max Hops")
                k_sl    = gr.Slider(1,    50, 10, label="Top K")
                mem_sl  = gr.Slider(50,   99, 90, label="Memory %")

                with gr.Accordion("CSA Weight Profiler", open=False):
                    p_radar = gr.Plot()
                    weights = [
                        gr.Slider(0, 1, default_val, label=label)
                        for default_val, label in _WEIGHT_DEFS
                    ]
                    commit_btn = gr.Button("Commit Weights")

            status_out  = gr.Textbox(label="Status")
            comm_output = gr.Textbox(label="Community Summary", visible=True)

        # ── Right panel: results ──────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Reasoning"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            q_in  = gr.Textbox(label="Query Entity", placeholder="e.g. newton")
                            q_btn = gr.Button("Run", variant="primary")
                            q_html = gr.HTML()
                        with gr.Column(scale=1):
                            attn_radar = gr.Plot(label="Attention Analysis")

                with gr.Tab("Structural Analytics"):
                    s_plot = gr.Plot()
                    s_md   = gr.Markdown()
                    s_btn  = gr.Button("Refresh Analytics")

                with gr.Tab("Interactive Graph"):
                    v2_btn = gr.Button("Refresh 2D")
                    v2_out = gr.HTML()

                with gr.Tab("3D Explorer"):
                    v3_btn = gr.Button("Refresh 3D")
                    v3_out = gr.HTML()

                with gr.Tab("Settings"):
                    gr.Markdown("### SSD & Storage Manager")
                    disk_dropdown = gr.Dropdown(
                        label="Select NVMe SSD",
                        choices=[d['mountpoint'] for d in _engine.get_storage_disks()],
                    )
                    refresh_disks_btn = gr.Button("Refresh Disks")
                    init_disk_btn = gr.Button("Initialize Selected SSD")
                    storage_status = gr.Textbox(label="Storage Status")
                    
                    gr.Markdown("### Live I/O Performance")
                    io_plot = gr.LinePlot(label="Read/Write Throughput (MB/s)")

                with gr.Tab("Insight & Maintenance"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### REM Cycle")
                            rem_dry_btn = gr.Button("Run Dry-Run")
                            rem_run_btn = gr.Button("Run Commit", variant="stop")
                            rem_out     = gr.Textbox(label="REM Report")
                        with gr.Column(scale=2):
                            gr.Markdown("### Insights")
                            ins_btn = gr.Button("Refresh Insights")
                            val_btn = gr.Button("Validate All", variant="primary")
                            ins_out = gr.HTML()

                with gr.Tab("Live Feed"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            st_type  = gr.Dropdown(
                                ["Simulated", "File Tail", "HTTP Polling"],
                                value="Simulated", label="Source",
                            )
                            st_path   = gr.Textbox(label="Path/URL")
                            st_win    = gr.Slider(5, 300, 60, label="Window (s)")
                            st_start  = gr.Button("Start", variant="primary")
                            st_stop   = gr.Button("Stop",  variant="stop")
                            st_status = gr.Textbox(label="Stream Status")
                        with gr.Column(scale=2):
                            st_log = gr.HTML(label="Mutation Log")
                            st_ref = gr.Button("Refresh Log")

    # ── Wiring ────────────────────────────────────────────────────────

    # Disk Management
    refresh_disks_btn.click(
        lambda: gr.update(choices=[d['mountpoint'] for d in _engine.get_storage_disks()]),
        None, disk_dropdown
    )
    init_disk_btn.click(_engine.init_storage, [disk_dropdown], [storage_status])
    
    # Simple I/O monitor
    # In a real app, use a timer to poll every ~1s; here we trigger on load/click
    # Placeholder for live gauge:
    # io_plot.change(...) 
    file_in.clear( lambda: "",   None,         path_in)
    path_in.change(lambda x: None if x and x.strip() else gr.update(),
                   inputs=[path_in], outputs=[file_in])

    history_in.change(lambda x: x, history_in, path_in)
    history_ref_btn.click(lambda: gr.update(choices=_engine.get_recent_paths()), None, history_in)

    # Health & Backups
    backup_btn.click(_engine.save_current_state, [], status_out).then(
        lambda: gr.update(choices=_engine.list_backups()), None, backup_list
    )
    restore_refresh_btn.click(lambda: gr.update(choices=_engine.list_backups()), None, backup_list)
    restore_btn.click(_engine.restore_state, backup_list, status_out).then(
        fn=lambda: (generate_param_radar(*[0.4, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0]), *_engine.get_graph_stats()),
        outputs=[p_radar, s_plot, s_md],
    )

    # CSA weight profiler — live radar + commit (10 params)
    for w in weights:
        w.change(generate_param_radar, weights, p_radar, api_name="update_param_profile")
    commit_btn.click(_engine.commit_params, weights, status_out, api_name="commit_weights")

    # Graph load → also refresh radar and analytics
    load_btn.click(
        _load_graph_cb,
        [file_in, path_in, emb_in],
        [status_out, comm_output],
        api_name="load_graph",
    ).then(
        fn=lambda *args: (generate_param_radar(*args), *_engine.get_graph_stats()),
        inputs=weights,
        outputs=[p_radar, s_plot, s_md],
    ).then(
        lambda: gr.update(choices=_engine.get_recent_paths()), None, history_in
    )

    # Reasoning → also update 3D path highlight
    q_btn.click(
        _engine.run_reasoning,
        [q_in, beam_sl, hop_sl, k_sl, mem_sl],
        [q_html, gr.JSON(visible=False), attn_radar, best_path_state],
        api_name="run_reasoning",
    ).then(
        fn=_engine.generate_3d_viz,
        inputs=[best_path_state],
        outputs=v3_out,
    )

    # Analytics / visualisation
    s_btn.click(_engine.get_graph_stats,  [], [s_plot, s_md],     api_name="refresh_analytics")
    v2_btn.click(_engine.generate_graph_viz, [], v2_out,          api_name="get_2d_viz")
    v3_btn.click(_engine.generate_3d_viz,    [], v3_out,          api_name="get_3d_viz")

    # REM / Insight
    rem_dry_btn.click(_engine.run_rem_cycle, [gr.State(True)],  rem_out, api_name="rem_dry_run")
    rem_run_btn.click(_engine.run_rem_cycle, [gr.State(False)], rem_out, api_name="rem_commit")
    ins_btn.click(_engine.get_insight_log, [], ins_out,              api_name="get_insights")
    val_btn.click(_engine.run_validation,  [], rem_out,              api_name="validate_insights"
                  ).then(_engine.get_insight_log, [], ins_out)

    # Live feed
    st_start.click(
        _engine.start_stream,
        [st_type, st_path, st_win, gr.State(5000)],
        [status_out, st_status],
        api_name="start_stream",
    )
    st_stop.click(_engine.stop_stream, [], [status_out, st_status], api_name="stop_stream")
    st_ref.click(_engine.get_stream_event_log, [], st_log,          api_name="get_stream_log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEREBRUM Studio")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--reload", action="store_true", help="Use 'gradio ui/studio.py' for real reload")
    args = parser.parse_args()

    if args.reload:
        print("Starting in Gradio Dev Mode...")
        # Note: True reload requires running 'gradio ui/studio.py' from shell
        demo.launch(server_port=args.port, share=False, css=CUSTOM_CSS)
    else:
        demo.launch(server_port=args.port, share=False, css=CUSTOM_CSS)
