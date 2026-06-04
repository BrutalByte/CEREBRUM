"""
StudioEngine â€” pure business-logic layer for CEREBRUM Studio.

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
from typing import Any, Dict, List, Optional, Type

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
from core.federated_registry import FederatedGraphRegistry
from core.trifecta import TrifectaEngine
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

    All methods are callable without Gradio â€” pass plain Python values.
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
        # Phase 75 â€” optional v2 dashboard attachments
        self._research_agent = None
        self._modulator = None
        self._loop = None
        self._provenance_ledger = None  # Phase 78

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

    def load_graph_with_columns(
        self,
        path: str,
        src_col: str,
        tgt_col: str,
        rel_col: str,
        embedding_type: str,
        progress=None,
    ):
        """KB Builder entry-point: loads a CSV with user-specified column names."""
        from adapters.csv_adapter import load_csv_adapter
        from core.cerebrum import CerebrumGraph
        from core.embedding_engine import RandomEngine, SentenceEngine
        from reasoning.traversal import BeamTraversal
        from core.rem_engine import REMEngine
        from core.insight_engine import InsightEngine
        from core.insight_validator import InsightValidator

        if not path or not path.strip():
            yield "ERROR: No file path provided.", "N/A"
            return

        path = path.strip()
        source_name = Path(path).name
        try:
            t0 = time.time()
            if progress:
                progress(0.1, desc="Reading columns...")

            adapter = load_csv_adapter(
                path,
                source_col=src_col or "source",
                target_col=tgt_col or "target",
                relation_col=rel_col or "relation",
            )

            if progress:
                progress(0.4, desc="Building index...")
            emb_mode = "sentence" if "Sentence" in embedding_type else "random"
            graph = CerebrumGraph.from_adapter(adapter, embeddings=emb_mode)
            graph.build(seed=42)

            if progress:
                progress(0.9, desc="Finalizing...")
            self.graph_obj = graph
            self.adapter = graph.adapter
            self.csa = graph._csa
            self.communities = graph.communities
            self.n_nodes = graph.node_count
            self.graph_loaded = True
            self.rem = REMEngine(graph.adapter)
            self.insight = InsightEngine(graph.adapter)
            self.validator = InsightValidator(graph.adapter)

            elapsed = time.time() - t0
            comm_summary = f"{len(graph.communities)} communities detected via DSCF."
            status = f"[OK] Built KB from {source_name} ({elapsed:.2f}s) | {graph.node_count} nodes"
            yield status, comm_summary
        except Exception as exc:
            log.error("KB build error: %s", exc, exc_info=True)
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

    def get_storage_disks(self) -> List[Dict[str, Any]]:
        """Return available disk mount points for NVMe management."""
        try:
            import psutil
            return [{"mountpoint": p.mountpoint, "device": p.device} for p in psutil.disk_partitions()]
        except Exception:
            return [{"mountpoint": "/", "device": "default"}]

    def init_storage(self, mountpoint: str) -> str:
        """Initialize CEREBRUM data directory on the selected disk."""
        if not mountpoint:
            return "No disk selected."
        try:
            data_dir = Path(mountpoint) / "cerebrum_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            return f"Initialized storage at {data_dir}"
        except Exception as exc:
            return f"Error: {exc}"

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
            self._last_answers = answers  # cache for audit export
            structured = [
                {"answer": a.entity_id, "score": a.score, "path": a.best_path.nodes}
                for a in answers
            ]
            return html, structured, radar_fig, best_path_nodes
        except Exception as exc:
            log.error("Reasoning error: %s", exc, exc_info=True)
            return f"[ERROR] {exc}", None, empty_fig, []

    def generate_audit_json_from_last(self) -> str:
        """Export the last reasoning result as a crystal-box JSON audit report."""
        answers = getattr(self, "_last_answers", [])
        if not answers:
            return '{"error": "No reasoning result available. Run a query first."}'
        return self.generate_audit_json(answers)

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
                f"(Î±={alpha:.2f}, Î²={beta:.2f}, Î¼={mu:.2f}, Î¸={theta:.2f})"
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
    # Emergency Management (Phase 178)
    # ------------------------------------------------------------------

    def emergency_snapshot(self) -> str:
        """Atomically persist all reasoning state to a 'panics/' snapshot."""
        try:
            snapshot_dir = Path("panics") / f"snapshot_{int(time.time())}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Persist Engram and Graph State
            if self.engram:
                self.engram.save_if_path(str(snapshot_dir / "engram.json"))
            
            # Save Graph Topology
            if self.graph_obj:
                # We can't easily save the memmap state, but we save the mappings
                with open(snapshot_dir / "nodes.map", "w") as f:
                    for node, i in self.graph_obj.adapter.id_map.items():
                        f.write(f"{node}\t{i}\n")
                        
            return f"DON'T PANIC. State saved to {snapshot_dir}"
        except Exception as e:
            return f"PANIC: Snapshot failed: {e}"

    def switch_graph(self, path: str, embedding_type: str) -> tuple:
        """Tear down current graph and load a new one."""
        # 1. Cleanup
        if self.graph_obj:
            self.graph_obj.adapter.close()
        
        # 2. Reload
        return next(self.load_graph(path, embedding_type))

    def enable_adaptive_reasoning(self, enabled: bool) -> str:
        """Enable or disable autonomous parameter tuning."""
        if not self.csa:
            return "ERROR: Load graph first."
        
        if enabled:
            from core.parameter_learner import MetaParameterLearner
            self.meta_learner = MetaParameterLearner(self.csa)
            return "Adaptive reasoning enabled."
        else:
            self.meta_learner = None
            return "Adaptive reasoning disabled."

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
                color="#ff8844" if "SynapticBridge" in rel else "#8b949e",
            )
        import base64
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        ) as f:
            tmp_path = Path(f.name)
        net.save_graph(str(tmp_path))
        try:
            html_content = tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)
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
                "#ff8844"   if "SynapticBridge" in rel  else
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
                "is_SynapticBridge": "SynapticBridge" in rel,
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
            .linkCurvature(link => link.is_SynapticBridge ? 0.2 : 0)
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

    # ------------------------------------------------------------------
    # Phase 75 â€” v2 Dashboard: optional engine attachments
    # ------------------------------------------------------------------

    def attach_research_agent(self, agent) -> None:
        """Attach a live ResearchAgent for AutoApprover audit and revision queue panels."""
        self._research_agent = agent

    def attach_modulator(self, modulator) -> None:
        """Attach a live ChemicalModulator for the blood panel."""
        self._modulator = modulator

    def attach_loop(self, loop) -> None:
        """Attach a live AutonomousDiscoveryLoop for the loop status panel."""
        self._loop = loop

    def attach_provenance_ledger(self, ledger) -> None:
        """Attach a live ProvenanceLedger for the provenance panel (Phase 78)."""
        self._provenance_ledger = ledger

    # ------------------------------------------------------------------
    # Phase 75 â€” AutoApprover audit log
    # ------------------------------------------------------------------

    def get_auto_approver_audit(self, n: int = 20) -> str:
        """
        Return the last *n* AutoApprover decisions as an HTML table.
        Gracefully returns a notice if no AutoApprover is attached.
        """
        agent = self._research_agent
        aa = getattr(agent, "_auto_approver", None) if agent is not None else None
        if aa is None:
            return "<p style='color:#888;'>No AutoApprover attached to ResearchAgent.</p>"

        records = list(aa.audit_log)[-n:]
        if not records:
            return "<p style='color:#888;'>AutoApprover audit log is empty.</p>"

        ACTION_COLOR = {"approve": "#3fb950", "reject": "#f85149", "review": "#d29922"}
        rows = ""
        for d in reversed(records):
            action = getattr(d, "action", "?")
            color = ACTION_COLOR.get(action, "#ccc")
            conf = getattr(d, "confidence", 0.0)
            reason = getattr(d, "reason", "")
            fid = getattr(d, "finding_id", "")
            rows += (
                f"<tr>"
                f"<td style='color:{color};font-weight:bold;'>{action.upper()}</td>"
                f"<td style='color:#79c0ff;font-family:monospace;font-size:0.85em;'>{fid[:24]}</td>"
                f"<td style='color:#e6edf3;'>{conf:.3f}</td>"
                f"<td style='color:#8b949e;font-size:0.85em;'>{reason[:60]}</td>"
                f"</tr>"
            )

        stats = aa.stats()
        summary = (
            f"<div style='margin-bottom:8px;color:#8b949e;font-size:0.85em;'>"
            f"Trained: <b style='color:#e6edf3;'>{stats['n_trained']}</b> &nbsp;|&nbsp; "
            f"Approve: <b style='color:#3fb950;'>{stats['n_approve']}</b> &nbsp;|&nbsp; "
            f"Reject: <b style='color:#f85149;'>{stats['n_reject']}</b> &nbsp;|&nbsp; "
            f"Review: <b style='color:#d29922;'>{stats['n_review']}</b>"
            f"</div>"
        )
        table = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='color:#8b949e;border-bottom:1px solid #30363d;'>"
            "<th style='text-align:left;padding:4px 8px;'>Action</th>"
            "<th style='text-align:left;padding:4px 8px;'>Finding ID</th>"
            "<th style='text-align:left;padding:4px 8px;'>Confidence</th>"
            "<th style='text-align:left;padding:4px 8px;'>Reason</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        return summary + table

    # ------------------------------------------------------------------
    # Phase 75 â€” ContradictionResolver revision queue
    # ------------------------------------------------------------------

    def get_revision_queue(self) -> str:
        """
        Return the ContradictionResolver revision queue as an HTML list.
        These are findings where proposed evidence outweighed the contradiction
        score â€” the existing graph edges may be stale.
        """
        agent = self._research_agent
        if agent is None:
            return "<p style='color:#888;'>No ResearchAgent attached.</p>"

        queue = list(getattr(agent, "_revision_candidates", []))
        if not queue:
            return "<p style='color:#3fb950;'>Revision queue is empty â€” no contested findings.</p>"

        items = ""
        for rec in reversed(queue):
            fid = getattr(rec, "finding_id", str(rec))
            net = getattr(rec, "net_evidence_score", None)
            rw = getattr(rec, "revision_weight", None)
            net_str = f"{net:.3f}" if net is not None else "?"
            rw_str = f"{rw:.2f}Ã—" if rw is not None else "?"
            items += (
                f"<div style='border:1px solid #30363d;border-radius:4px;padding:6px 10px;"
                f"margin-bottom:6px;background:#161b22;'>"
                f"<span style='color:#79c0ff;font-family:monospace;font-size:0.85em;'>{fid[:32]}</span>"
                f"<span style='float:right;color:#d29922;font-size:0.85em;'>"
                f"net={net_str} &nbsp; weight={rw_str}</span>"
                f"</div>"
            )
        header = (
            f"<div style='color:#d29922;margin-bottom:8px;font-size:0.9em;'>"
            f"<b>{len(queue)}</b> finding(s) queued for revision review</div>"
        )
        return header + items

    # ------------------------------------------------------------------
    # Phase 75 â€” DiscoveryCalibrator community heatmap
    # ------------------------------------------------------------------

    def get_discovery_heatmap(self) -> go.Figure:
        """
        Return a Plotly heatmap of per-community discovery weights from
        the DiscoveryCalibrator.  Returns an empty figure if not attached.
        """
        agent = self._research_agent
        calibrator = getattr(agent, "_calibrator", None) if agent is not None else None
        if calibrator is None:
            fig = go.Figure()
            fig.update_layout(
                title="DiscoveryCalibrator not attached",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
            )
            return fig

        stats = calibrator.stats()
        communities = stats.get("communities", {})
        if not communities:
            fig = go.Figure()
            fig.update_layout(
                title="No community data yet",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
            )
            return fig

        cids = sorted(communities.keys())
        weights = [communities[c]["weight"] for c in cids]
        rates = [communities[c]["rate"] for c in cids]

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Sampling Weight", "Discovery Rate"])
        fig.add_trace(
            go.Bar(
                x=[str(c) for c in cids],
                y=weights,
                name="Weight",
                marker_color=[
                    f"rgb({int(255*(1-w/max(weights+[1])))},100,{int(255*w/max(weights+[1]))})"
                    for w in weights
                ],
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[str(c) for c in cids],
                y=rates,
                name="Rate",
                marker_color="#58a6ff",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            title=f"DiscoveryCalibrator â€” {len(cids)} communities tracked",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            showlegend=False,
            height=300,
        )
        return fig

    # ------------------------------------------------------------------
    # Phase 75 â€” ChemicalModulator blood panel
    # ------------------------------------------------------------------

    def get_chemical_panel(self) -> go.Figure:
        """
        Return a Plotly radar/bar chart of the 5 ChemicalModulator scalars.
        Returns an empty figure if modulator is not attached.
        """
        modulator = self._modulator
        if modulator is None:
            fig = go.Figure()
            fig.update_layout(
                title="ChemicalModulator not attached",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
            )
            return fig

        state = getattr(modulator, "state", {})
        LABELS = {
            "reinforcement": "Reinforcement\n(Dopamine)",
            "arousal":       "Arousal\n(Norepinephrine)",
            "novelty":       "Novelty\n(Acetylcholine)",
            "cohesion":      "Cohesion\n(Oxytocin)",
            "persistence":   "Persistence\n(Vasopressin)",
        }
        baseline_val = getattr(modulator, "baseline", 1.0)
        keys = list(LABELS.keys())
        values = [state.get(k, 0.0) for k in keys]
        baselines = [baseline_val for _ in keys]

        colors = []
        for v, b in zip(values, baselines):
            if v > b * 1.15:
                colors.append("#f85149")   # elevated â€” red
            elif v < b * 0.85:
                colors.append("#58a6ff")   # suppressed â€” blue
            else:
                colors.append("#3fb950")   # near-baseline â€” green

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[LABELS[k] for k in keys],
            y=values,
            marker_color=colors,
            name="Current",
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=[LABELS[k] for k in keys],
            y=baselines,
            mode="lines+markers",
            name="Baseline",
            line=dict(color="#8b949e", dash="dash"),
            marker=dict(symbol="line-ew", size=8, color="#8b949e"),
        ))
        fig.update_layout(
            title="ChemicalModulator â€” Metabolic State",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            yaxis=dict(range=[0, max(values + baselines) * 1.25]),
            legend=dict(bgcolor="#161b22"),
            height=320,
        )
        return fig

    # ------------------------------------------------------------------
    # Phase 75 â€” Autonomous Loop status panel
    # ------------------------------------------------------------------

    def get_loop_panel(self) -> tuple:
        """
        Return (status_html: str, cycle_fig: go.Figure) for the loop dashboard.
        """
        loop = self._loop
        if loop is None:
            empty = go.Figure()
            empty.update_layout(
                title="AutonomousDiscoveryLoop not attached",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
            )
            return "<p style='color:#888;'>Loop not attached.</p>", empty

        s = loop.status()
        running_color = "#3fb950" if s["running"] else "#f85149"
        cb_color = "#f85149" if s["circuit_breaker_tripped"] else "#3fb950"
        rate = s["current_approval_rate"]
        rate_str = f"{rate:.1%}" if rate is not None else "n/a"

        status_html = (
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;"
            f"font-family:monospace;font-size:0.9em;margin-bottom:12px;'>"
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>STATUS</div>"
            f"<div style='color:{running_color};font-weight:bold;'>"
            f"{'RUNNING' if s['running'] else 'STOPPED'}</div></div>"
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>CIRCUIT BREAKER</div>"
            f"<div style='color:{cb_color};font-weight:bold;'>"
            f"{'TRIPPED' if s['circuit_breaker_tripped'] else 'OPEN'}</div></div>"
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>APPROVAL RATE</div>"
            f"<div style='color:#79c0ff;font-weight:bold;'>{rate_str}</div></div>"
            f"</div>"
            f"<div style='color:#8b949e;font-size:0.85em;'>"
            f"Cycles: <b style='color:#e6edf3;'>{s['total_cycles']}</b> &nbsp;|&nbsp; "
            f"Approved: <b style='color:#3fb950;'>{s['total_approved']}</b> &nbsp;|&nbsp; "
            f"Rejected: <b style='color:#f85149;'>{s['total_rejected']}</b> &nbsp;|&nbsp; "
            f"Review: <b style='color:#d29922;'>{s['total_review']}</b> &nbsp;|&nbsp; "
            f"Edges: <b style='color:#58a6ff;'>{s['total_edges_added']}</b>"
            f"</div>"
        )

        # Cycle history chart
        cycles = s.get("recent_cycles", [])
        if not cycles:
            fig = go.Figure()
            fig.update_layout(
                title="No cycles run yet",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
            )
            return status_html, fig

        nums = [c["cycle_number"] for c in cycles]
        approved = [c["auto_approved"] for c in cycles]
        rejected = [c["auto_rejected"] for c in cycles]
        review = [c["sent_to_review"] for c in cycles]
        edges = [c["edges_added"] for c in cycles]

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Decisions per Cycle", "Edges Added per Cycle"],
                            shared_xaxes=True, vertical_spacing=0.12)
        fig.add_trace(go.Bar(x=nums, y=approved, name="Approved",
                             marker_color="#3fb950"), row=1, col=1)
        fig.add_trace(go.Bar(x=nums, y=rejected, name="Rejected",
                             marker_color="#f85149"), row=1, col=1)
        fig.add_trace(go.Bar(x=nums, y=review, name="Review",
                             marker_color="#d29922"), row=1, col=1)
        fig.add_trace(go.Scatter(x=nums, y=edges, mode="lines+markers",
                                 name="Edges Added", line=dict(color="#58a6ff")),
                      row=2, col=1)
        fig.update_layout(
            title="Autonomous Loop â€” Cycle History",
            barmode="stack",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            legend=dict(bgcolor="#161b22"),
            height=420,
        )
        return status_html, fig

    # ------------------------------------------------------------------
    # Phase 78 â€” ProvenanceLedger panel
    # ------------------------------------------------------------------

    def get_provenance_panel(self, n: int = 20) -> tuple:
        """
        Return (stats_html: str, batch_fig: go.Figure, timeline_fig: go.Figure).

        - stats_html   : 4-card summary row (totals + rollback count).
        - batch_fig    : horizontal bar chart of recent batches, coloured by
                         rollback status.
        - timeline_fig : line chart of edges added per cycle_number (groups
                         batches that share the same cycle).

        All three degrade gracefully when no ledger is attached.
        """
        _EMPTY = "<p style='color:#888;'>ProvenanceLedger not attached.</p>"
        _DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                     font=dict(color="#8b949e"))

        ledger = self._provenance_ledger
        if ledger is None:
            empty = go.Figure()
            empty.update_layout(title="ProvenanceLedger not attached", **_DARK)
            return _EMPTY, empty, empty

        s = ledger.stats()
        total_b = s["total_batches"]
        total_e = s["total_edges_recorded"]
        rolled  = s["batches_rolled_back"]
        n_cycles = len(s.get("cycles_seen", []))

        stats_html = (
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;"
            f"font-family:monospace;font-size:0.9em;margin-bottom:12px;'>"
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>BATCHES</div>"
            f"<div style='color:#e6edf3;font-weight:bold;'>{total_b}</div></div>"
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>EDGES RECORDED</div>"
            f"<div style='color:#58a6ff;font-weight:bold;'>{total_e}</div></div>"
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>ROLLED BACK</div>"
            f"<div style='color:{'#f85149' if rolled else '#3fb950'};"
            f"font-weight:bold;'>{rolled}</div></div>"
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:6px;padding:8px;'>"
            f"<div style='color:#8b949e;font-size:0.8em;'>CYCLES SEEN</div>"
            f"<div style='color:#d29922;font-weight:bold;'>{n_cycles}</div></div>"
            f"</div>"
        )

        batches = ledger.list_batches(n=n)
        if not batches:
            empty = go.Figure()
            empty.update_layout(title="No materialization batches yet",
                                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                font=dict(color="#8b949e"))
            return stats_html, empty, empty

        # --- batch bar chart (newest-first order, bottom-to-top on horizontal bar) ---
        b_ids     = [b.batch_id[:20] for b in reversed(batches)]
        b_counts  = [len(b.edges) for b in reversed(batches)]
        b_colors  = ["#f85149" if b.rolled_back else "#3fb950"
                     for b in reversed(batches)]

        batch_fig = go.Figure(go.Bar(
            x=b_counts,
            y=b_ids,
            orientation="h",
            marker_color=b_colors,
            text=b_counts,
            textposition="outside",
        ))
        batch_fig.update_layout(
            title=f"Recent Materializations (last {len(batches)})",
            xaxis_title="Edges",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            height=max(260, len(batches) * 24 + 80),
            margin=dict(l=160, r=40, t=40, b=40),
        )

        # --- timeline: edges per cycle ---
        cycle_edges: Dict[int, int] = {}
        no_cycle_total = 0
        for b in batches:
            if b.cycle_number is not None:
                cycle_edges[b.cycle_number] = cycle_edges.get(b.cycle_number, 0) + len(b.edges)
            else:
                no_cycle_total += len(b.edges)

        if not cycle_edges:
            timeline_fig = go.Figure()
            timeline_fig.update_layout(
                title="No cycle-tagged batches yet",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#8b949e"),
            )
        else:
            sorted_cycles = sorted(cycle_edges.keys())
            cycle_vals    = [cycle_edges[c] for c in sorted_cycles]
            cumulative    = []
            running = 0
            for v in cycle_vals:
                running += v
                cumulative.append(running)

            timeline_fig = go.Figure()
            timeline_fig.add_trace(go.Bar(
                x=sorted_cycles, y=cycle_vals,
                name="Edges (cycle)", marker_color="#58a6ff",
            ))
            timeline_fig.add_trace(go.Scatter(
                x=sorted_cycles, y=cumulative,
                mode="lines+markers", name="Cumulative",
                line=dict(color="#d29922", dash="dot"),
                yaxis="y2",
            ))
            timeline_fig.update_layout(
                title="Edges Materialized per Cycle",
                xaxis_title="Cycle",
                yaxis=dict(title="Edges (cycle)"),
                yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                legend=dict(bgcolor="#161b22"),
                height=320,
                barmode="group",
            )

        return stats_html, batch_fig, timeline_fig

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
        """HTML result cards with visual hop-by-hop path flowcharts."""
        if not answers:
            return (
                "<div style='color:#888;padding:20px;'>"
                "No paths found. Try increasing beam width or max hops."
                "</div>"
            )

        # Colours
        ENTITY_BG   = "#1c2b3a"
        ENTITY_FG   = "#79c0ff"
        RELATION_FG = "#bc8cff"
        BEST_BORDER = "#00ffff"
        OTHER_BORDER= "#30363d"

        def _path_flow(nodes):
            """Render alternating [entity] â†’ [relation] â†’ [entity] nodes."""
            parts = []
            for i, node in enumerate(nodes):
                if i % 2 == 0:
                    # Entity node
                    parts.append(
                        f"<span style='background:{ENTITY_BG};color:{ENTITY_FG};"
                        f"border:1px solid #30363d;border-radius:4px;padding:2px 7px;"
                        f"font-family:monospace;font-size:0.85em;'>{node}</span>"
                    )
                else:
                    # Relation arrow
                    parts.append(
                        f"<span style='color:{RELATION_FG};margin:0 4px;font-size:0.8em;'>"
                        f"&mdash;[{node}]&rarr;</span>"
                    )
            return "".join(parts)

        html = "<div class='results-container'>"
        for rank, ans in enumerate(answers):
            score_pct  = int(min(ans.score, 1.0) * 100)
            border_col = BEST_BORDER if rank == 0 else OTHER_BORDER
            rank_badge = (
                "<span style='background:#00ffff;color:#000;border-radius:3px;"
                "padding:1px 6px;font-size:0.75em;margin-left:6px;'>BEST</span>"
                if rank == 0 else f"<span style='color:#666;font-size:0.8em;margin-left:4px;'>#{rank+1}</span>"
            )
            path_flow = _path_flow(ans.best_path.nodes) if ans.best_path else ""
            html += f"""
            <div style='background:#161b22;border:1px solid {border_col};border-radius:8px;
                        padding:12px 16px;margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                    <span style='color:#c9d1d9;font-weight:bold;font-size:1.05em;'>{ans.entity_id}{rank_badge}</span>
                    <span style='color:#00ffff;font-family:monospace;font-size:1.1em;'>{ans.score:.4f}</span>
                </div>
                <div style='background:#30363d;height:6px;border-radius:3px;margin-bottom:8px;'>
                    <div style='background:#58a6ff;width:{score_pct}%;height:100%;border-radius:3px;'></div>
                </div>
                <div style='margin-bottom:8px;line-height:2em;'>{path_flow}</div>
                <div style='font-size:0.78em;color:#8b949e;border-top:1px solid #21262d;padding-top:5px;'>
                    Community&nbsp;{ans.score_breakdown.get('community', 0):.3f} &nbsp;|&nbsp;
                    Semantic&nbsp;{ans.score_breakdown.get('semantic', 0):.3f} &nbsp;|&nbsp;
                    Edge&nbsp;{ans.score_breakdown.get('edge', 0):.3f} &nbsp;|&nbsp;
                    Hops&nbsp;{ans.best_path.hop_depth if ans.best_path else '?'}
                </div>
            </div>"""
        html += "</div>"
        return html

    def generate_audit_json(self, answers) -> str:
        """Return the reasoning results as a JSON audit report (crystal-box export)."""
        import json as _json
        records = []
        for rank, ans in enumerate(answers):
            nodes = ans.best_path.nodes if ans.best_path else []
            entities = nodes[0::2]
            relations = nodes[1::2]
            records.append({
                "rank": rank + 1,
                "answer": ans.entity_id,
                "confidence": round(float(ans.score), 6),
                "hop_depth": ans.best_path.hop_depth if ans.best_path else None,
                "path": {
                    "entities": entities,
                    "relations": relations,
                    "full_nodes": nodes,
                },
                "score_breakdown": {k: round(float(v), 6) for k, v in (ans.score_breakdown or {}).items()},
            })
        return _json.dumps({"cerebrum_audit": records, "n_candidates": len(records)}, indent=2)

    # ------------------------------------------------------------------
    # C1 â€” Explainability Dashboard
    # ------------------------------------------------------------------

    def explain_beam(self, answer_index: int = 0) -> str:
        """
        Return an HTML panel explaining WHY the chosen answer scored highest.

        Shows:
        - The winning path with per-feature score bars
        - Feature-by-feature delta vs the runner-up (green = winner advantage)
        - Top-10 candidate paths ranked by score
        """
        answers = getattr(self, "_last_answers", [])
        if not answers:
            return "<div style='color:#888;padding:16px;'>Run a query first.</div>"
        if answer_index >= len(answers):
            answer_index = 0

        winner = answers[answer_index]
        runner_up = answers[answer_index + 1] if answer_index + 1 < len(answers) else None

        FEATURE_LABELS = ["Semantic", "Community", "Edge-Type", "Distance", "Hop-Decay",
                          "PageRank", "Temporal", "Recency", "Synth-Density", "Grounding"]

        def _bar(value: float, max_val: float, color: str) -> str:
            pct = int(min(value / max(max_val, 1e-9), 1.0) * 100)
            return (
                f"<div style='display:inline-block;width:80px;height:8px;"
                f"background:#21262d;border-radius:4px;vertical-align:middle;margin-left:6px;'>"
                f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px;'></div>"
                f"</div>"
            )

        def _feature_row(label: str, w_val: float, r_val: float) -> str:
            delta = w_val - r_val
            delta_col = "#3fb950" if delta >= 0 else "#f85149"
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            return (
                f"<tr>"
                f"<td style='color:#8b949e;font-size:0.82em;padding:3px 8px;'>{label}</td>"
                f"<td style='color:#58a6ff;font-family:monospace;font-size:0.82em;'>"
                f"{w_val:.3f}{_bar(w_val, 1.0, '#58a6ff')}</td>"
                f"<td style='color:#6e7681;font-family:monospace;font-size:0.82em;'>"
                f"{r_val:.3f}{_bar(r_val, 1.0, '#6e7681')}</td>"
                f"<td style='color:{delta_col};font-family:monospace;font-size:0.82em;'>{delta_str}</td>"
                f"</tr>"
            )

        def _edge_feats(ans) -> list:
            if ans.best_path is None or not ans.best_path.edge_features:
                return [0.0] * 10
            import numpy as np
            arr = np.array(ans.best_path.edge_features)
            return list(np.mean(arr, axis=0))

        def _path_mini(nodes, score: float, rank: int) -> str:
            ENTITY_BG = "#1c2b3a"; ENTITY_FG = "#79c0ff"; REL_FG = "#bc8cff"
            parts = []
            for i, n in enumerate(nodes):
                if i % 2 == 0:
                    parts.append(
                        f"<span style='background:{ENTITY_BG};color:{ENTITY_FG};"
                        f"border:1px solid #30363d;border-radius:3px;padding:1px 5px;"
                        f"font-family:monospace;font-size:0.78em;'>{n}</span>"
                    )
                else:
                    parts.append(
                        f"<span style='color:{REL_FG};margin:0 3px;font-size:0.75em;'>"
                        f"â€”[{n}]â†’</span>"
                    )
            rank_badge = (
                "<span style='background:#00ffff;color:#000;border-radius:3px;"
                "padding:0 4px;font-size:0.72em;'>BEST</span> " if rank == 0
                else f"<span style='color:#6e7681;font-size:0.75em;'>#{rank+1}</span> "
            )
            score_pct = int(min(score, 1.0) * 100)
            return (
                f"<div style='padding:6px 0;border-bottom:1px solid #21262d;'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                f"<span>{rank_badge}{''.join(parts)}</span>"
                f"<span style='color:#00ffff;font-family:monospace;font-size:0.82em;'>{score:.4f}</span>"
                f"</div>"
                f"<div style='background:#21262d;height:4px;border-radius:2px;'>"
                f"<div style='width:{score_pct}%;height:100%;background:#58a6ff;border-radius:2px;'></div>"
                f"</div></div>"
            )

        # â”€â”€ Section 1: Winner explanation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        w_feats = _edge_feats(winner)
        r_feats = _edge_feats(runner_up) if runner_up else [0.0] * 10

        # Score breakdown (structured) â€” fall back to edge features
        w_bd = winner.score_breakdown or {}
        r_bd = (runner_up.score_breakdown or {}) if runner_up else {}

        # Feature comparison table (only show when runner-up exists)
        cmp_table = ""
        if runner_up:
            rows = ""
            for i, label in enumerate(FEATURE_LABELS):
                wv = float(w_feats[i]) if i < len(w_feats) else 0.0
                rv = float(r_feats[i]) if i < len(r_feats) else 0.0
                rows += _feature_row(label, wv, rv)
            ru_path = _path_mini(
                runner_up.best_path.nodes if runner_up.best_path else [],
                runner_up.score, 1
            )
            cmp_table = f"""
            <div style='margin-top:14px;'>
              <div style='color:#bc8cff;font-weight:bold;font-size:0.9em;margin-bottom:6px;'>
                Feature comparison vs runner-up ({runner_up.entity_id})
              </div>
              <table style='width:100%;border-collapse:collapse;'>
                <tr>
                  <th style='color:#6e7681;font-size:0.78em;text-align:left;padding:2px 8px;'>Feature</th>
                  <th style='color:#58a6ff;font-size:0.78em;text-align:left;'>Winner</th>
                  <th style='color:#6e7681;font-size:0.78em;text-align:left;'>Runner-up</th>
                  <th style='color:#8b949e;font-size:0.78em;text-align:left;'>Delta</th>
                </tr>
                {rows}
              </table>
              <div style='margin-top:8px;color:#8b949e;font-size:0.8em;'>Runner-up path:</div>
              {ru_path}
            </div>"""

        win_path_html = _path_mini(
            winner.best_path.nodes if winner.best_path else [],
            winner.score, answer_index
        )

        bd_items = " &nbsp;|&nbsp; ".join(
            f"<span style='color:#8b949e;'>{k}:</span> "
            f"<span style='color:#58a6ff;font-family:monospace;'>{v:.3f}</span>"
            for k, v in (w_bd or {}).items()
        )

        section1 = f"""
        <div style='background:#161b22;border:1px solid #00ffff;border-radius:8px;padding:14px 16px;margin-bottom:12px;'>
          <div style='color:#00ffff;font-weight:bold;margin-bottom:8px;'>
            âœ¦ Why <code style='background:#1c2b3a;padding:1px 5px;border-radius:3px;'>{winner.entity_id}</code>
            scored highest (rank #{answer_index+1})
          </div>
          {win_path_html}
          <div style='font-size:0.8em;margin-top:6px;'>{bd_items}</div>
          {cmp_table}
        </div>"""

        # â”€â”€ Section 2: All candidate paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        all_paths = "".join(
            _path_mini(
                a.best_path.nodes if a.best_path else [],
                a.score, rank
            )
            for rank, a in enumerate(answers[:10])
        )
        section2 = f"""
        <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px;'>
          <div style='color:#bc8cff;font-weight:bold;margin-bottom:8px;'>
            Top-{min(len(answers), 10)} candidate paths
          </div>
          {all_paths}
        </div>"""

        return (
            "<div style='font-family:system-ui,sans-serif;max-width:900px;'>"
            + section1 + section2
            + "</div>"
        )

    def explain_beam_from_last(self) -> str:
        """Explain why the top-1 answer from the last query scored highest."""
        return self.explain_beam(answer_index=0)

    def export_audit_pdf_html(self) -> str:
        """
        Return a self-contained HTML string suitable for browser print-to-PDF.

        The page includes the full crystal-box reasoning trace, score breakdown,
        and all candidate paths for the last query result.
        """
        answers = getattr(self, "_last_answers", [])
        if not answers:
            return "<html><body><p>No reasoning result available.</p></body></html>"

        import json as _json
        import time as _time

        records = _json.loads(self.generate_audit_json(answers))
        explain_html = self.explain_beam(answer_index=0)
        ts = _time.strftime("%Y-%m-%d %H:%M:%S UTC", _time.gmtime())

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CEREBRUM Crystal-Box Audit Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 40px; color: #24292f; }}
  h1   {{ color: #5a2d82; }}
  h2   {{ color: #6e40c9; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }}
  code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
  pre  {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.82em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; font-size: 0.88em; }}
  th   {{ background: #f6f8fa; }}
  .best {{ background: #dafbe1; font-weight: bold; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
</head>
<body>
<h1>CEREBRUM Crystal-Box Audit Report</h1>
<p><strong>Generated:</strong> {ts}<br>
<strong>System:</strong> CEREBRUM v2.58.0 â€” zero-hallucination KG traversal<br>
<strong>Candidates:</strong> {records['n_candidates']}</p>

<h2>Candidate Paths</h2>
<table>
<tr><th>Rank</th><th>Answer</th><th>Confidence</th><th>Hop Depth</th><th>Path (entities)</th><th>Path (relations)</th></tr>
{"".join(
    f"<tr class='{'best' if r['rank'] == 1 else ''}'>"
    f"<td>#{r['rank']}</td>"
    f"<td><code>{r['answer']}</code></td>"
    f"<td>{r['confidence']:.4f}</td>"
    f"<td>{r['hop_depth']}</td>"
    f"<td>{' â†’ '.join(r['path']['entities'])}</td>"
    f"<td>{' â†’ '.join(r['path']['relations'])}</td>"
    f"</tr>"
    for r in records['cerebrum_audit']
)}
</table>

<h2>Score Breakdown (Top Answer)</h2>
<table>
{"".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in records['cerebrum_audit'][0]['score_breakdown'].items())}
</table>

<h2>Full JSON Audit Record</h2>
<pre>{_json.dumps(records, indent=2)}</pre>

<hr>
<p style='font-size:0.8em;color:#57606a;'>
This report was generated by CEREBRUM's crystal-box reasoning engine.
Every answer is a verified path through knowledge graph edges â€” no neural inference,
no hallucination. The path shown IS the explanation.
</p>
</body>
</html>"""
