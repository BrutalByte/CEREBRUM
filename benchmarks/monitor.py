"""
CEREBRUM Benchmark Monitor — Streamlit live dashboard.

Usage:
    streamlit run benchmarks/monitor.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import streamlit as st

DEFAULT_LOG    = Path("C:/Users/bryan/AppData/Local/Temp/metaqa_run.log")
MLFLOW_URI     = "mlruns"
REFRESH_SECS   = 5

st.set_page_config(
    page_title="CEREBRUM Monitor",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> dict:
    s = {
        "running": False, "completed": 0, "total": 0, "elapsed_s": 0.0,
        "workers": 1, "hits1": None, "hits10": None, "mrr": None,
        "beam_width": None, "embeddings": None, "hop": None,
        "errors": [], "raw_tail": "",
    }
    if not log_path.exists():
        return s
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return s

    lines = text.splitlines()
    s["raw_tail"] = "\n".join(lines[-40:])

    for line in lines:
        m = re.search(r"([\d,]+)/([\d,]+)\s+questions\s+\(([0-9.]+)s elapsed\)", line)
        if m:
            s["running"]    = True
            s["completed"]  = int(m.group(1).replace(",", ""))
            s["total"]      = int(m.group(2).replace(",", ""))
            s["elapsed_s"]  = float(m.group(3))
        m = re.search(r"Spawning (\d+) worker", line)
        if m:
            s["workers"] = int(m.group(1))
        m = re.search(r"Beam width:\s+(\d+)", line)
        if m:
            s["beam_width"] = int(m.group(1))
        m = re.search(r"Embeddings:\s+(\w+)", line)
        if m:
            s["embeddings"] = m.group(1)
        m = re.search(r"---\s+(\d+)-hop evaluation", line)
        if m:
            s["hop"] = int(m.group(1))
        m = re.search(r"Hits@1\s*:\s*([0-9.]+)", line)
        if m:
            s["hits1"]   = float(m.group(1))
            s["running"] = False
        m = re.search(r"Hits@10\s*:\s*([0-9.]+)", line)
        if m:
            s["hits10"] = float(m.group(1))
        m = re.search(r"\bMRR\s*:\s*([0-9.]+)", line)
        if m:
            s["mrr"] = float(m.group(1))
        if any(kw in line for kw in ("Error", "Traceback", "CUDA error")):
            s["errors"].append(line.strip())
    return s


# ---------------------------------------------------------------------------
# MLflow run history
# ---------------------------------------------------------------------------

def load_mlflow_runs(uri: str):
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        exps   = client.search_experiments()
        rows   = []
        for exp in exps:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=50,
            )
            for r in runs:
                p = r.data.params
                m = r.data.metrics
                rows.append({
                    "Run":        r.info.run_name or r.info.run_id[:8],
                    "Status":     r.info.status,
                    "Started":    time.strftime("%m/%d %H:%M", time.localtime(r.info.start_time / 1000)),
                    "Hop":        p.get("hop", "—"),
                    "Beam":       p.get("beam_width", "—"),
                    "FHRB":       p.get("fhrb_factor", "—"),
                    "H@1 (3h)":   f"{float(m['hop3_hits1'])*100:.2f}%" if "hop3_hits1" in m else "—",
                    "H@10 (3h)":  f"{float(m['hop3_hits10'])*100:.2f}%" if "hop3_hits10" in m else "—",
                    "MRR (3h)":   f"{float(m['hop3_mrr']):.4f}" if "hop3_mrr" in m else "—",
                    "Time (3h)":  f"{float(m['hop3_elapsed'])/60:.1f}m" if "hop3_elapsed" in m else "—",
                })
        return rows
    except Exception as e:
        return [{"Error": str(e)}]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🧠 CEREBRUM Monitor")
log_path = Path(st.sidebar.text_input("Log file", str(DEFAULT_LOG)))
mlflow_uri = st.sidebar.text_input("MLflow URI", MLFLOW_URI)
refresh  = st.sidebar.slider("Refresh (s)", 2, 30, REFRESH_SECS)
st.sidebar.markdown("---")
st.sidebar.caption("Phase 182 — Multiprocessing benchmark")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_live, tab_history = st.tabs(["⏳ Live Run", "📊 Run History (MLflow)"])

placeholder_live    = tab_live.empty()
placeholder_history = tab_history.empty()

while True:
    # ---- Live tab ----
    s = parse_log(log_path)
    with placeholder_live.container():
        if s["hits1"] is not None:
            st.success("✅ Run complete")
        elif s["running"]:
            st.info(f"⏳ Running — {s['workers']} worker(s)")
        elif not log_path.exists():
            st.warning(f"Log not found: `{log_path}`")
        else:
            st.warning("Waiting for benchmark to start...")

        c1, c2, c3 = st.columns(3)
        c1.metric("Hop",        f"{s['hop']}-hop"   if s["hop"]        else "—")
        c2.metric("Beam width", s["beam_width"]      if s["beam_width"] else "—")
        c3.metric("Embeddings", s["embeddings"]      if s["embeddings"] else "—")

        st.markdown("---")

        if s["total"] > 0:
            pct = s["completed"] / s["total"]
            st.progress(pct, text=f"{s['completed']:,} / {s['total']:,}  ({pct*100:.1f}%)")
            elapsed = s["elapsed_s"]
            if 0 < s["completed"] < s["total"]:
                rate = elapsed / s["completed"]
                eta  = rate * (s["total"] - s["completed"])
                e1, e2, e3 = st.columns(3)
                e1.metric("Elapsed", f"{elapsed/60:.1f} min")
                e2.metric("Rate",    f"{rate:.2f} s/q")
                e3.metric("ETA",     f"{eta/60:.1f} min")
            elif s["completed"] == s["total"] and elapsed:
                st.metric("Total time", f"{elapsed/60:.1f} min")

        if s["hits1"] is not None:
            st.markdown("### Final Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("H@1",  f"{s['hits1']*100:.2f}%")
            r2.metric("H@10", f"{s['hits10']*100:.2f}%" if s["hits10"] else "—")
            r3.metric("MRR",  f"{s['mrr']:.4f}"         if s["mrr"]    else "—")

        if s["errors"]:
            with st.expander(f"⚠️ {len(s['errors'])} warning(s)", expanded=True):
                for e in s["errors"][-10:]:
                    st.code(e)

        with st.expander("Raw log (last 40 lines)"):
            st.code(s["raw_tail"], language=None)

        st.caption(f"Last updated: {time.strftime('%H:%M:%S')} · refreshes every {refresh}s")

    # ---- History tab ----
    with placeholder_history.container():
        st.subheader("MLflow Experiment Runs")
        st.caption(f"Tracking URI: `{mlflow_uri}`  |  Start runs with `--mlflow`")

        rows = load_mlflow_runs(mlflow_uri)
        if rows and "Error" not in rows[0]:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif rows and "Error" in rows[0]:
            st.info(f"No MLflow runs yet. {rows[0]['Error']}")
        else:
            st.info("No runs recorded yet. Add `--mlflow` to your next benchmark run.")

        if st.button("🔄 Refresh history", key="refresh_history"):
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**Launch MLflow UI** (full experiment browser):  \n"
            "`mlflow ui --backend-store-uri mlruns --port 5000`  \n"
            "Then open [localhost:5000](http://localhost:5000)"
        )

    time.sleep(refresh)
