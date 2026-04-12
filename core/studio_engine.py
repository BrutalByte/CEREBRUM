"""
StudioEngine — pure business-logic layer for CEREBRUM Studio.

Extracted from ui/studio.py so every callback is independently importable
and unit-testable without a running Gradio server.

ui/studio.py imports this module and wires StudioEngine methods to Gradio
components as event handlers.

Parameter order for all 10-param CSA calls:
    alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta
"""
import json
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adapters.file_adapter import load_file_adapter
from adapters.stream_adapter import (
    StreamAdapter,
    PythonCallbackSource,
    FileTailSource,
    HTTPPollingSource,
)
from core.cerebrum import CerebrumGraph
from core.discretizer import ThresholdDiscretizer
from core.insight_engine import InsightEngine
from core.insight_validator import InsightValidator
from core.rem_engine import REMEngine
from core.persistence import save_state, load_state, is_state_cached
from core.studio_history import StudioHistory

log = logging.getLogger("cerebrum.studio")

# ---------------------------------------------------------------------------
# StudioEngine
# ---------------------------------------------------------------------------

class StudioEngine:
    """
    Stateful business-logic engine for CEREBRUM Studio.

    Owns graph state and exposes plain-Python methods that Gradio callbacks
    can delegate to directly::

        engine = StudioEngine()

        # In Gradio wiring:
        load_btn.click(engine.load_graph, [path_in, emb_in], [status_out, comm_out])
        q_btn.click(engine.run_reasoning, [...], [...])

    All methods are callable without Gradio — pass plain Python values.
    """

    def __init__(self) -> None:
        self.graph_obj: Optional[CerebrumGraph] = None
        self.adapter = None
        self.csa = None
        self.graph_loaded: bool = False
        self.communities: Optional[List] = None
        self.n_nodes: int = 0
        self.rem: Optional[REMEngine] = None
        self.insight: Optional[InsightEngine] = None
        self.validator: Optional[InsightValidator] = None
        self.stream_adapter: Optional[StreamAdapter] = None
        self.stream_running: bool = False
        self.stream_event_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    def load_graph(self, path: str, embedding_type: str, progress=None) -> tuple:
        """
        Load a knowledge graph from *path* and build the CEREBRUM index.
        """
        if not path or not path.strip():
            yield "ERROR: No file path provided.", "N/A"
            return

        path = path.strip()
        source_name = Path(path).name
        log.info("Loading graph from %s", path)
        
        # Track history
        StudioHistory.add_to_history(path)

        try:
            t0 = time.time()
            if progress: progress(0, desc="Initializing Thalamus Ingestor...")
            
            use_graphsage = "GraphSAGE" in embedding_type
            emb_mode = "sentence" if "Sentence" in embedding_type else "random"
            graph = CerebrumGraph.from_kb(path, embeddings=emb_mode)

            # Use a wrapper for progress if provided
            def _build_callback(p, s):
                if progress:
                    progress(p, desc=s)

            if use_graphsage and progress:
                progress(0.1, desc="Building graph index...")
            graph.build(seed=42, callback=_build_callback, use_graphsage=use_graphsage)
            if use_graphsage and progress:
                progress(0.85, desc="GraphSAGE neighbourhood smoothing applied.")

            if progress: progress(0.9, desc="Finalizing Engines...")
            self.graph_obj = graph
            self.adapter = graph.adapter
            self.csa = graph._csa
            self.communities = graph.communities
            self.n_nodes = graph.node_count
            self.graph_loaded = True
            self.rem = REMEngine(graph.adapter)
            self.insight = InsightEngine(graph.adapter)
            self.validator = InsightValidator(graph.adapter)

            comm_summary = f"{len(graph.communities)} communities detected via DSCF."
            elapsed = time.time() - t0
            status = f"[OK] Loaded {source_name} ({elapsed:.2f}s) | {graph.node_count} nodes"
            yield status, comm_summary
        except Exception as exc:
            log.error("Load error: %s", exc, exc_info=True)
            yield f"[ERROR] {type(exc).__name__}: {exc}", "N/A"

    def save_current_state(self, backup_name: Optional[str] = None) -> str:
        """
        Save the current graph state as a binary backup for restoration.
        """
        if not self.graph_loaded or not self.graph_obj:
            return "ERROR: No graph loaded to save."
        
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            name = backup_name if backup_name else f"backup_{ts}.pkl"
            
            # Ensure path is relative to sandbox as per persistence.py requirements
            save_path = f"backups/{name}"
            
            save_state(
                file_path=save_path,
                adapter=self.adapter,
                community_map=self.graph_obj.communities_map if hasattr(self.graph_obj, 'communities_map') else {},
                embeddings=self.graph_obj._embedding_engine.embeddings if hasattr(self.graph_obj._embedding_engine, 'embeddings') else {},
                csa_metadata={
                    "node_count": self.n_nodes,
                    "csa_params": {
                        "alpha": self.csa.alpha,
                        "beta": self.csa.beta,
                        "gamma": self.csa.gamma,
                    } if self.csa else {}
                }
            )
            return f"[OK] State backed up as '{name}'"
        except Exception as exc:
            return f"[ERROR] Backup failed: {exc}"

    def restore_state(self, backup_name: str) -> str:
        """
        Restore the graph state from a binary backup.
        """
        try:
            state = load_state(f"backups/{backup_name}")
            
            # Reconstruct minimal Graph/Cerebrum objects from state
            self.adapter = state["adapter"]
            self.communities = list(state["community_map"].keys()) if "community_map" in state else []
            self.n_nodes = state["csa_metadata"]["node_count"]
            self.graph_loaded = True
            
            # Re-init engines
            self.rem = REMEngine(self.adapter)
            self.insight = InsightEngine(self.adapter)
            self.validator = InsightValidator(self.adapter)
            
            return f"[OK] State restored from '{backup_name}'"
        except Exception as exc:
            return f"[ERROR] Restore failed: {exc}"

    def list_backups(self) -> List[str]:
        """List available backup files."""
        backup_dir = Path("data/cerebrum/backups")
        if not backup_dir.exists():
            return []
        return [f.name for f in backup_dir.glob("*.pkl")]

    def get_recent_paths(self) -> List[str]:
        """Expose history to the UI."""
        return StudioHistory.get_history()

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def run_reasoning(
        self,
        query: str,
        beam_width: int,
        max_hop: int,
        top_k: int,
        mem_threshold: float,
    ) -> tuple:
        """
        Run beam-search reasoning from the *query* entity.

        Returns
        -------
        (html_str, structured_list, attention_radar_fig, best_path_nodes)
        """
        if not self.graph_loaded or not self.graph_obj:
            return "Load a graph first.", None, go.Figure(), []
        if not query or not query.strip():
            return "Enter a query entity.", None, go.Figure(), []

        try:
            empty_fig = go.Figure()
            graph = self.graph_obj
            # Use BGE instruction-enhanced encoding for query
            if hasattr(graph._embedding_engine, 'encode_query'):
                query_vec = graph._embedding_engine.encode_query([query.strip()])[0]
            else:
                query_vec = graph._embedding_engine.encode([query.strip()])[0]
            
            seeds = [e.id for e in graph.adapter.find_entities(query.strip(), top_k=1) if e]
            if not seeds:
                return f"No entity matching '{query}'", None, empty_fig, []

            from reasoning.trace import ReasoningTrace
            trace = ReasoningTrace(query=query.strip(), seeds=[seeds[0]])

            answers = graph.query(
                [seeds[0]],
                top_k=int(top_k),
                max_hop=int(max_hop),
                beam_width=int(beam_width),
                memory_threshold_pct=float(mem_threshold),
                trace_info=trace,
            )

            best_path_nodes: List[str] = []
            if answers:
                best_path_nodes = answers[0].best_path.nodes

            radar_fig = empty_fig
            if (
                answers
                and hasattr(answers[0].best_path, "edge_features")
                and answers[0].best_path.edge_features
            ):
                mean_feats = np.mean(np.array(answers[0].best_path.edge_features), axis=0)
                radar_fig = self._attention_radar(mean_feats.tolist())

            from core.verbalizer import EngramVerbalizer
            _engram_verb = EngramVerbalizer()
            engram_trace = _engram_verb.verbalize(answers, self.adapter)

            # Format ERT HTML
            ert_html = self._format_trace_html(trace)

            html = (
                f"<div style='margin-bottom:12px;'><b>Seed</b>: "
                f"<code style='color:#00ffff;'>{seeds[0]}</code></div>"
                f"<div class='engram-block' style='background:#1c2128;border:1px solid #30363d;"
                f"padding:10px;margin-bottom:15px;border-radius:6px;font-family:monospace;font-size:0.9em;color:#79c0ff;'>"
                f"<strong>Engram Trace:</strong> {engram_trace}</div>"
                f"<details style='margin-bottom:15px;'><summary style='cursor:pointer;color:#bc8cff;'><b>Explainable Reasoning Trace (ERT)</b></summary>"
                f"<div style='margin-top:10px;'>{ert_html}</div></details>"
                + self._format_path_html(answers)
            )
            structured = [
                {"answer": a.entity_id, "score": a.score, "path": a.best_path.nodes}
                for a in answers
            ]
            return html, structured, radar_fig, best_path_nodes
        except Exception as exc:
            log.error("Reasoning error: %s", exc, exc_info=True)
            return f"[ERROR] {exc}", None, empty_fig, []

    # ------------------------------------------------------------------
    # Structural analytics
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> tuple:
        """
        Return (plotly_figure, markdown_summary) with degree + community charts.
        """
        if not self.graph_loaded:
            return go.Figure(), "No graph loaded."

        G = self.adapter.to_networkx()
        degrees = [d for _, d in G.degree()]
        fig_deg = px.histogram(
            pd.DataFrame({"Degree": degrees}),
            x="Degree",
            title="Degree Distribution",
            template="plotly_dark",
        )
        comm_sizes = [len(c) for c in (self.communities or [])]
        fig_comm = px.pie(
            pd.DataFrame({
                "Size": comm_sizes,
                "ID": [f"C{i}" for i in range(len(comm_sizes))],
            }),
            values="Size",
            names="ID",
            title="Community Composition",
            hole=0.4,
            template="plotly_dark",
        )
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]])
        for t in fig_deg.data:
            fig.add_trace(t, row=1, col=1)
        for t in fig_comm.data:
            fig.add_trace(t, row=1, col=2)
        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        summary = (
            f"**Nodes**: {G.number_of_nodes()} | "
            f"**Edges**: {G.number_of_edges()} | "
            f"**Density**: {nx.density(G):.4f}"
        )
        return fig, summary

    # ------------------------------------------------------------------
    # Parameter management  (10-param CSA vector)
    # ------------------------------------------------------------------

    def commit_params(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        epsilon: float,
        zeta: float,
        eta: float,
        iota: float,
        mu: float,
        theta: float,
    ) -> str:
        """
        Push an updated 10-parameter CSA weight vector to the live CSAEngine.

        Parameters match the canonical order:
        alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta
        """
        if not self.csa:
            return "ERROR: Load graph first."
        try:
            self.csa.alpha   = alpha
            self.csa.beta    = beta
            self.csa.gamma   = gamma
            self.csa.delta   = delta
            self.csa.epsilon = epsilon
            self.csa.zeta    = zeta
            self.csa.eta     = eta
            self.csa.iota    = iota
            self.csa.mu      = mu
            self.csa.theta   = theta
            return (
                f"Success: Weights updated "
                f"(α={alpha:.2f}, β={beta:.2f}, μ={mu:.2f}, θ={theta:.2f})"
            )
        except Exception as err:
            return f"ERROR: {err}"

    # ------------------------------------------------------------------
    # REM / Insight
    # ------------------------------------------------------------------

    def run_rem_cycle(self, dry_run: bool = True) -> str:
        """Run a REM prune/synthesize cycle and return a summary string."""
        if not self.rem:
            return "ERROR: REM not initialized."
        try:
            rep = self.rem.run(dry_run=dry_run)
            return (
                f"REM Complete. "
                f"Pruned: {rep.pruned_count}, "
                f"Synthesized: {rep.synthesized_count}"
            )
        except Exception as exc:
            return f"ERROR: {exc}"

    def get_insight_log(self) -> str:
        """Return recent InsightEngine events as an HTML string."""
        if not self.insight:
            return "Insight Engine not initialized."
        events = self.insight.recent_events(n=10)
        if not events:
            return (
                "<div style='color:#888;padding:20px;'>"
                "No significant structural insights discovered yet."
                "</div>"
            )
        html = "<div class='results-container'>"
        for ev in events:
            badge_class = "insight-surprise" if ev.insight_score > 0.3 else "insight-high"
            path_str = (
                " &rarr; ".join(ev.path.nodes)
                if ev.path
                else f"{ev.source} &rarr; {ev.target} (cold)"
            )
            html += f"""
            <div class='cerebrum-card'>
                <div style='display:flex;justify-content:space-between;'>
                    <span class='entity-id'>{ev.bridging_node}</span>
                    <span class='insight-badge {badge_class}'>INSIGHT</span>
                </div>
                <div style='margin-top:10px;font-size:0.9em;'>
                    <strong>Score:</strong> {ev.insight_score:.4f} |
                    <strong>Status:</strong> {ev.validation_status}
                </div>
                <div class='path-text' style='margin-top:8px;'>{path_str}</div>
            </div>"""
        html += "</div>"
        return html

    def run_validation(self) -> str:
        """Validate all recent insight events and return a summary."""
        if not self.validator or not self.insight:
            return "Validator or Insight Engine not initialized."
        try:
            events = self.insight.recent_events(n=100)
            results = self.validator.validate_all(events)
            return f"Validation Complete. {len(results)} insights processed."
        except Exception as exc:
            return f"ERROR in Validation: {exc}"

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_stream(
        self,
        source_type: str,
        file_path_or_url: str,
        window_seconds: float,
        max_edges: int,
    ) -> tuple:
        """
        Start a live-feed stream source.

        Returns
        -------
        (status_msg, stream_status_label)
        """
        if self.stream_running:
            return "Stream already running.", ""
        adapter = StreamAdapter(
            time_window_seconds=float(window_seconds),
            max_edges=int(max_edges),
        )
        self.stream_adapter = adapter
        self.stream_event_log = []

        def on_mutation(action, event):
            self.stream_event_log.append({
                "action":   action,
                "source":   event.source,
                "relation": event.relation,
                "target":   event.target,
                "ts":       event.timestamp,
            })
            if len(self.stream_event_log) > 200:
                self.stream_event_log.pop(0)

        adapter.add_mutation_listener(on_mutation)
        try:
            if "File" in source_type:
                adapter.add_source(FileTailSource(file_path_or_url.strip()))
            elif "HTTP" in source_type:
                adapter.add_source(HTTPPollingSource(file_path_or_url.strip()))
            elif "Simulated" in source_type:
                disc = ThresholdDiscretizer("temp", low=15, high=35)
                adapter.add_source(
                    PythonCallbackSource(
                        lambda: disc.process(random.gauss(25, 5)),
                        poll_interval=1.0,
                    )
                )
            adapter.start()
            self.stream_running = True
            return f"Stream started: {source_type}", "Active"
        except Exception as exc:
            return f"Error: {exc}", "Failed"

    def stop_stream(self) -> tuple:
        """
        Stop the active stream.

        Returns
        -------
        (status_msg, stream_status_label)
        """
        if self.stream_adapter:
            self.stream_adapter.stop()
        self.stream_running = False
        return "Stream stopped", "Inactive"

    def get_stream_event_log(self) -> str:
        """Return recent stream mutation events as an HTML table."""
        rows = "".join(
            f"<tr><td>{e['action']}</td><td>{e['source']}</td>"
            f"<td>{e['relation']}</td><td>{e['target']}</td></tr>"
            for e in reversed(self.stream_event_log)
        )
        return f"<table>{rows}</table>"

    # ------------------------------------------------------------------
    # Visualisation (graph-state-dependent)
    # ------------------------------------------------------------------

    def generate_graph_viz(self) -> str:
        """Render an interactive 2D PyVis graph as an HTML iframe string."""
        if not self.graph_loaded:
            return "Load a graph first."
        G = self.adapter.to_networkx()
        if G.number_of_nodes() > 500:
            return "Graph too large for 2D (use 3D Explorer)."
        from pyvis.network import Network

        net = Network(height="600px", width="100%", bgcolor="#0d1117", font_color="#c9d1d9")
        for n in G.nodes():
            cid = self.adapter.get_community(n)
            net.add_node(
                str(n),
                label=str(n),
                color="#58a6ff" if cid < 0 else px.colors.qualitative.Safe[cid % 10],
            )
        for u, v, data in G.edges(data=True):
            rel = data.get("relation", "")
            net.add_edge(
                str(u), str(v),
                title=rel,
                dashes="rem_synthesized" in rel,
                color="#ff8844" if "wormhole" in rel else "#8b949e",
            )
        import base64
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            html_content = Path(f.name).read_text(encoding="utf-8")
            b64_content = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            return (
                f"<iframe src='data:text/html;base64,{b64_content}'"
                " style='width:100%;height:600px;border:none;border-radius:8px;'></iframe>"
            )

    def generate_3d_viz(self, highlight_nodes: Optional[List[str]] = None) -> str:
        """Render a 3D force-graph as an HTML iframe string."""
        if not self.graph_loaded:
            return "Load a graph first."
        G = self.adapter.to_networkx()
        path_set = set(highlight_nodes) if highlight_nodes else set()

        nodes_data = [
            {
                "id":    str(n),
                "group": self.adapter.get_community(n),
                "size":  12 if str(n) in path_set else 4,
                "color": "#00ffff" if str(n) in path_set else None,
            }
            for n in G.nodes()
        ]
        links_data = []
        for u, v, d in G.edges(data=True):
            u_str, v_str = str(u), str(v)
            rel = d.get("relation", "")
            on_path = u_str in path_set and v_str in path_set
            color = (
                "#00ffff"   if on_path           else
                "#ff8844"   if "wormhole" in rel  else
                "#8b949e"   if "rem_synthesized" in rel else
                "#444444"
            )
            links_data.append({
                "source":      u_str,
                "target":      v_str,
                "rel":         rel,
                "color":       color,
                "width":       3 if on_path else 1,
                "is_on_path":  on_path,
                "is_wormhole": "wormhole" in rel,
            })

        graph_json     = json.dumps({"nodes": nodes_data, "links": links_data})
        path_nodes_js  = json.dumps(list(path_set))
        has_path       = "true" if highlight_nodes else "false"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/3d-force-graph@1.73.3/dist/3d-force-graph.min.js"></script>
    <style>
        body{{margin:0;background:#0d1117;overflow:hidden;}}
        #3d-graph{{width:100vw;height:100vh;display:block;}}
    </style>
</head>
<body>
    <div id="3d-graph"></div>
    <script>
        const data = {graph_json};
        const Graph = ForceGraph3D({{
            rendererConfig: {{ antialias: true, powerPreference: 'high-performance' }}
        }})(document.getElementById('3d-graph'))
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
        if ({has_path}) {{
            setTimeout(() => {{
                const pathNodes = data.nodes.filter(n => {path_nodes_js}.includes(n.id));
                if (pathNodes.length > 0) {{
                    Graph.cameraPosition({{x:200,y:200,z:200}}, pathNodes[0], 2000);
                }}
            }}, 1000);
        }}
    </script>
</body>
</html>"""
        import base64
        b64_content = base64.b64encode(html.encode('utf-8')).decode('utf-8')
        return (
            f"<iframe src='data:text/html;base64,{b64_content}'"
            " style='width:100%;height:750px;border:none;border-radius:8px;'></iframe>"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_trace_html(trace) -> str:
        """Format a ReasoningTrace as HTML."""
        if not trace or not trace.hops:
            return "No trace data available."
        
        html = "<div class='trace-container' style='font-size:0.85em;'>"
        for hop in trace.hops:
            html += f"<div style='margin-bottom:8px;border-left:3px solid #bc8cff;padding-left:10px;'>"
            html += f"<div style='color:#bc8cff;font-weight:bold;'>Hop {hop.hop}</div>"
            html += f"<div style='color:#888;'>Candidates: {hop.total_candidates} | Beam Width: {hop.beam_width}</div>"
            
            # Winners
            if hop.winners:
                html += "<div style='margin-top:4px;'><b>Winners:</b> "
                for w in hop.winners[:3]: # Show top 3 winners
                    html += f"<span style='color:#00ffff;margin-right:8px;'>{w['tail']} ({w['score']:.3f})</span>"
                if len(hop.winners) > 3:
                    html += f"<span style='color:#666;'>+{len(hop.winners)-3} more</span>"
                html += "</div>"
            
            # Competitors (Pruned)
            if hop.competitors:
                html += "<div style='margin-top:2px;'><b>Pruned Competitors:</b> "
                for c in hop.competitors[:3]:
                    html += f"<span style='color:#ff7b72;margin-right:8px;'>{c['tail']} ({c['score']:.3f})</span>"
                if len(hop.competitors) > 3:
                    html += f"<span style='color:#666;'>+{len(hop.competitors)-3} more</span>"
                html += "</div>"
            
            html += "</div>"
        html += "</div>"
        return html

    @staticmethod
    def _attention_radar(feature_tuple) -> go.Figure:
        """10-element attention feature radar chart."""
        if not feature_tuple or len(feature_tuple) < 10:
            return go.Figure()
        categories = [
            "Similarity", "Community", "Edge Weight", "Distance (inv)",
            "Hop Decay", "PageRank", "Temp Decay", "Node Recency",
            "Synth Density", "Grounding",
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(feature_tuple[:10]),
            theta=categories,
            fill="toself",
            name="Step Attention",
            line_color="#58a6ff",
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

    @staticmethod
    def _format_path_html(answers) -> str:
        """HTML cards for reasoning results."""
        if not answers:
            return (
                "<div style='color:#888;padding:20px;'>"
                "No paths found. Try increasing beam width or max hops."
                "</div>"
            )
        html = "<div class='results-container'>"
        for ans in answers:
            path_str  = " &rarr; ".join(f"<code>{n}</code>" for n in ans.best_path.nodes)
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
