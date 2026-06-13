"""
Phase 260 — Build Freebase MID → readable-name TSV via Wikidata SPARQL.

Wikidata property P646 stores Freebase IDs. For each MID in freebase_2hop.txt
we query Wikidata in batches and write a mid_to_name.tsv file that
load_mid_name_map() in webqsp_param_eval.py can consume via --mid-name-file.

Defaults to QLever (https://qlever.cs.uni-freiburg.de/api/wikidata) which is
an independent high-performance SPARQL engine mirroring Wikidata — unaffected
by WDQS (query.wikidata.org) outages.

Usage:
    python benchmarks/build_mid_name_map.py
    python benchmarks/build_mid_name_map.py --batch-size 200 --delay 0.5
    python benchmarks/build_mid_name_map.py --resume          # skip already-done MIDs
    python benchmarks/build_mid_name_map.py --endpoint wdqs   # use official WDQS
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

ENDPOINTS = {
    "qlever": "https://qlever.cs.uni-freiburg.de/api/wikidata",
    "wdqs":   "https://query.wikidata.org/sparql",
}
USER_AGENT = "CEREBRUM-MIDMapper/2.93 (research; bryan.buchorn@gmail.com)"

WEBQSP_DIR = Path(__file__).parent / "data" / "webqsp"
FB_PATH    = WEBQSP_DIR / "freebase_2hop.txt"
OUT_PATH   = WEBQSP_DIR / "mid_to_name.tsv"

# QLever requires explicit prefixes; WDQS supports SERVICE wikibase:label shorthand
_QUERY_QLEVER = """\
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?mid ?itemLabel WHERE {{
  VALUES ?mid {{ {values} }}
  ?item wdt:P646 ?mid .
  ?item rdfs:label ?itemLabel .
  FILTER (lang(?itemLabel) = "en")
}}"""

_QUERY_WDQS = """\
SELECT ?mid ?itemLabel WHERE {{
  VALUES ?mid {{ {values} }}
  ?item wdt:P646 ?mid .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}"""


def collect_mids(fb_path: Path) -> list[str]:
    mids: set[str] = set()
    with fb_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, _, t = parts
            if h.startswith("/m/"):
                mids.add(h)
            if t.startswith("/m/"):
                mids.add(t)
    return sorted(mids)


def wikidata_batch(mids: list[str], endpoint_url: str, query_tmpl: str, timeout: int = 60) -> dict[str, str]:
    values = " ".join(f'"{m}"' for m in mids)
    query = query_tmpl.format(values=values)
    url = endpoint_url + "?query=" + urllib.parse.quote(query)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: dict[str, str] = {}
    for row in data.get("results", {}).get("bindings", []):
        mid   = row["mid"]["value"]
        label = row.get("itemLabel", {}).get("value", "")
        if label and not label.startswith("Q"):  # skip unlabeled Qxxx fallback
            results[mid] = label
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MID→name TSV from Wikidata SPARQL.")
    parser.add_argument("--fb-path",    default=str(FB_PATH),  help="Path to freebase_2hop.txt")
    parser.add_argument("--out-path",   default=str(OUT_PATH), help="Output TSV path")
    parser.add_argument("--batch-size", type=int,   default=250,  help="MIDs per SPARQL query")
    parser.add_argument("--delay",      type=float, default=0.5,  help="Base seconds between requests")
    parser.add_argument("--max-retries",type=int,   default=6,    help="Max retries per batch on 429/5xx")
    parser.add_argument("--endpoint",   choices=list(ENDPOINTS), default="qlever",
                        help="SPARQL endpoint: 'qlever' (default, independent mirror) or 'wdqs' (official).")
    parser.add_argument("--resume",     action="store_true",       help="Skip MIDs already in output file")
    args = parser.parse_args()

    endpoint_url = ENDPOINTS[args.endpoint]
    query_tmpl   = _QUERY_QLEVER if args.endpoint == "qlever" else _QUERY_WDQS

    fb_path  = Path(args.fb_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Phase260] Collecting MIDs from {fb_path}...")
    all_mids = collect_mids(fb_path)
    print(f"  {len(all_mids):,} unique MIDs found")
    print(f"  Endpoint: {endpoint_url}")

    already_done: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                mid = line.split("\t")[0].strip()
                if mid:
                    already_done.add(mid)
        print(f"  Resuming: {len(already_done):,} MIDs already resolved, skipping")

    pending = [m for m in all_mids if m not in already_done]
    print(f"  {len(pending):,} MIDs to query in batches of {args.batch_size}")

    batches       = [pending[i:i+args.batch_size] for i in range(0, len(pending), args.batch_size)]
    total_hits    = 0
    errors        = 0
    current_delay = args.delay

    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as out_f:
        for i, batch in enumerate(batches):
            hits: dict[str, str] = {}
            for attempt in range(args.max_retries):
                try:
                    hits = wikidata_batch(batch, endpoint_url, query_tmpl)
                    current_delay = max(args.delay, current_delay * 0.9)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:
                        backoff = min(120.0, current_delay * (2 ** (attempt + 1)))
                        print(f"  [429] Rate-limited batch {i+1}, retry {attempt+1}/{args.max_retries} in {backoff:.0f}s")
                        time.sleep(backoff)
                        current_delay = backoff
                    elif exc.code >= 500:
                        backoff = min(60.0, args.delay * (2 ** attempt))
                        print(f"  [{exc.code}] Server error batch {i+1}, retry {attempt+1}/{args.max_retries} in {backoff:.0f}s")
                        time.sleep(backoff)
                    else:
                        print(f"  [HTTP {exc.code}] batch {i+1} failed: {exc}")
                        errors += 1
                        break
                except Exception as exc:
                    backoff = min(30.0, args.delay * (2 ** attempt))
                    print(f"  [ERR] batch {i+1} attempt {attempt+1}: {type(exc).__name__}: {exc} — retry in {backoff:.0f}s")
                    time.sleep(backoff)
            else:
                print(f"  [SKIP] batch {i+1} exhausted {args.max_retries} retries")
                errors += 1

            for mid, label in hits.items():
                out_f.write(f"{mid}\t{label}\n")
            total_hits += len(hits)

            if (i + 1) % 100 == 0 or i == len(batches) - 1:
                pct = (i + 1) / len(batches) * 100
                print(f"  [{pct:5.1f}%] batch {i+1}/{len(batches)} — {total_hits:,} hits, {errors} errors")
                out_f.flush()

            time.sleep(current_delay)

    coverage = total_hits / len(all_mids) * 100 if all_mids else 0
    print(f"\n[Phase260] Done. {total_hits:,} MIDs resolved ({coverage:.1f}% of total {len(all_mids):,}).")
    print(f"  Output: {out_path}")
    if errors:
        print(f"  {errors} batches failed. Re-run with --resume to fill gaps.")
    print(f"\nNext step:")
    print(f"  python benchmarks/webqsp_param_eval.py --mid-name-file {out_path} [other args]")


if __name__ == "__main__":
    main()
