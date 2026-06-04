"""
setup_graph_layout.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Query a live CEREBRUM REST API, compute 3D Fibonacci-sphere layout for every
entity grouped by community, and write Content/graph_layout.json that
ACerebrumBrain can load at startup for a richer initial visualisation.

Usage:
    python ue5_project/setup_graph_layout.py
    python ue5_project/setup_graph_layout.py --api http://localhost:8200 --token YOUR_JWT
    python ue5_project/setup_graph_layout.py --api http://localhost:8200 --out my_layout.json

The script requires no external deps beyond the Python standard library.
Run it from the repo root while CEREBRUM is running.
"""

from typing import Type
import argparse
import json
import math
import random
import sys
import urllib.request
import urllib.error
from pathlib import Path


# ---------------------------------------------------------------------------
# Layout constants (mirror ACerebrumBrain defaults)
# ---------------------------------------------------------------------------
COMMUNITY_ORBIT_RADIUS = 2500.0  # UE world units
NODE_CLUSTER_RADIUS    = 600.0
GOLDEN_ANGLE           = math.pi * (3.0 - math.sqrt(5.0))  # ~2.399 rad
GOLDEN_RATIO_CONJ      = 0.6180339887


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get_json(url: str, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {url}: {e.reason}")
        return {}
    except urllib.error.URLError as e:
        print(f"  Could not reach {url}: {e.reason}")
        return {}


# ---------------------------------------------------------------------------
# Spatial layout
# ---------------------------------------------------------------------------

def community_center(community_id: int, total_communities: int) -> tuple[float, float, float]:
    """Fibonacci sphere placement â€” stable for a given community_id."""
    i = community_id
    n = max(total_communities, 1)
    theta = GOLDEN_ANGLE * i
    y = 1.0 - (i / max(n - 1, 1)) * 2.0
    r = math.sqrt(max(1.0 - y * y, 0.0))
    return (
        r * math.cos(theta) * COMMUNITY_ORBIT_RADIUS,
        r * math.sin(theta) * COMMUNITY_ORBIT_RADIUS,
        y * COMMUNITY_ORBIT_RADIUS,
    )


def community_color(community_id: int) -> dict:
    """Golden-ratio hue wheel â†’ RGB (same algorithm as NeuronNodeActor.cpp)."""
    hue = (community_id * GOLDEN_RATIO_CONJ) % 1.0
    # HSVâ†’RGB (S=0.78, V=0.90)
    h6 = hue * 6.0
    i  = int(h6)
    f  = h6 - i
    s, v = 0.78, 0.90
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i % 6]
    return {"r": round(r, 4), "g": round(g, 4), "b": round(b, 4)}


def node_position(community_center: tuple, node_id: str) -> tuple[float, float, float]:
    """Deterministic random offset within the community cluster sphere."""
    rng = random.Random(hash(node_id) & 0xFFFFFFFF)
    az  = rng.uniform(0, 2 * math.pi)
    el  = rng.uniform(-math.pi / 2, math.pi / 2)
    r   = rng.uniform(NODE_CLUSTER_RADIUS * 0.2, NODE_CLUSTER_RADIUS)
    cx, cy, cz = community_center
    return (
        cx + r * math.cos(el) * math.cos(az),
        cy + r * math.cos(el) * math.sin(az),
        cz + r * math.sin(el),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate CEREBRUM UE5 graph layout JSON")
    parser.add_argument("--api",   default="http://localhost:8200",
                        help="CEREBRUM REST API base URL (default: http://localhost:8200)")
    parser.add_argument("--token", default="",
                        help="JWT bearer token for authenticated endpoints")
    parser.add_argument("--out",
                        default=str(Path(__file__).parent / "Content" / "graph_layout.json"),
                        help="Output path for layout JSON")
    parser.add_argument("--edge-limit", type=int, default=500,
                        help="Max edges to fetch from /graph/edges (default: 500)")
    args = parser.parse_args()

    token = args.token or None
    api   = args.api.rstrip("/")
    out   = Path(args.out)

    print(f"Querying CEREBRUM at {api} â€¦")

    # â”€â”€ Health check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    health = get_json(f"{api}/health", token)
    if not health:
        print("ERROR: /health failed â€” is CEREBRUM running?")
        sys.exit(1)
    node_count = health.get("node_count", "?")
    comm_count = health.get("community_count", "?")
    print(f"  Nodes: {node_count}  |  Communities: {comm_count}")

    # â”€â”€ Community map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("  Fetching /communities â€¦")
    comm_resp = get_json(f"{api}/communities", token)
    community_map: dict[str, int] = comm_resp.get("node_to_community", {})
    if not community_map:
        print("  No community data (node_to_community) returned. Layout will be empty.")

    # Build reverse map: community_id â†’ [node_ids]
    communities: dict[int, list[str]] = {}
    for node_id, cid in community_map.items():
        communities.setdefault(int(cid), []).append(node_id)

    total_communities = len(communities)
    print(f"  Found {total_communities} communities, {len(community_map)} nodes.")

    # â”€â”€ Compute community centres & colours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    centers: dict[int, tuple] = {}
    colors:  dict[int, dict]  = {}
    for cid in sorted(communities.keys()):
        centers[cid] = community_center(cid, total_communities)
        colors[cid]  = community_color(cid)

    # â”€â”€ Compute per-node positions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    nodes_out = []
    for cid, node_ids in sorted(communities.items()):
        center = centers[cid]
        color  = colors[cid]
        for node_id in sorted(node_ids):
            x, y, z = node_position(center, node_id)
            nodes_out.append({
                "node_id":       node_id,
                "label":         node_id.replace("_", " ").title(),
                "community_id":  cid,
                "community_color": color,
                "position":      {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)},
            })

    # â”€â”€ Community metadata list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    communities_out = []
    for cid in sorted(communities.keys()):
        x, y, z = centers[cid]
        communities_out.append({
            "community_id": cid,
            "color":        colors[cid],
            "center":       {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)},
            "node_count":   len(communities[cid]),
        })

    # â”€â”€ Fetch edges from /graph/edges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    edges_out = []
    if args.edge_limit > 0:
        print(f"  Fetching /graph/edges?limit={args.edge_limit} â€¦")
        edge_resp = get_json(f"{api}/graph/edges?limit={args.edge_limit}", token)
        raw_edges = edge_resp.get("edges", [])
        for e in raw_edges:
            src = e.get("source_id", "")
            tgt = e.get("target_id", "")
            rel = e.get("relation_type", "")
            w   = e.get("weight", 1.0)
            if src and tgt and rel:
                edges_out.append({
                    "source_id":    src,
                    "target_id":    tgt,
                    "relation_type": rel,
                    "weight":       round(float(w), 4),
                })
        print(f"  Got {len(edges_out)} edges.")

    # â”€â”€ Output JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    layout = {
        "cerebrum_layout_version": "1.1",
        "api_url":       api,
        "node_count":    len(nodes_out),
        "community_count": total_communities,
        "edge_count":    len(edges_out),
        "communities":   communities_out,
        "nodes":         nodes_out,
        "edges":         edges_out,
        "layout_params": {
            "community_orbit_radius": COMMUNITY_ORBIT_RADIUS,
            "node_cluster_radius":    NODE_CLUSTER_RADIUS,
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    print(f"\nWrote {len(nodes_out)} nodes, {len(edges_out)} edges "
          f"({total_communities} communities) â†’ {out}")
    print("Load this in UE5 via ACerebrumBrain or import as a DataAsset.")


if __name__ == "__main__":
    main()
